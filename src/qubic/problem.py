"""High-level task object: :class:`Problem` and the shared :class:`Result`.

A :class:`Problem` bundles the task data (library matrix ``A``, target
spectrum ``y``) and the shared settings (``k_max``, ``tau``); solver objects
(:mod:`qubic.solver`) bundle their own settings.  ``Problem.solve(solver)``
runs one solver on the task and returns a :class:`Result` -- uniformly for
QUBO-backed solvers and the alternative backend.

Lazy semantics ("don't pay for what you don't use"):
nothing is built at construction; a :class:`~qubic.solver.QuboSolver` builds
the QUBO inside its solve call only, and the built QUBO is memoized on the
problem, so solving one problem with several QUBO solvers builds it at most
once per (encoding, penalties) combination.  Alternative-backend solvers
never trigger any QUBO work.
"""
from __future__ import annotations

import numpy as np


class Result:
    """Uniform result of one solver run on a :class:`Problem`.

    Attributes
    ----------
    method : str
        Method registry name (e.g. ``'milp'``, ``'highs'``, ``'omp_bic'``).
    selected : tuple of int
        Selected component indices (possibly empty).
    concentrations : np.ndarray
        NNLS-refined concentrations aligned with ``selected``.
    rss : float
        Residual sum of squares ``||y - A[:, selected] @ c||^2`` of the fit.
    bic : float or None
        BIC score (lower is better); None when the method found nothing.
    top_K : list of dict
        Refined candidates sorted by ascending BIC; each entry has
        ``selected``, ``concentrations``, ``rss``, ``bic`` and
        ``qubo_energy`` (None for the alternative backend).
    stats : dict
        Run statistics (wall time, solver gap, encoding/penalties for
        QUBO-backed solvers, ...).

    The aliases ``best_support``, ``best_concentrations`` and ``best_score``
    expose the winning candidate as ``selected``/``concentrations``/``bic``.
    """

    def __init__(self, method, selected, concentrations, rss, bic,
                 top_K=None, stats=None):
        self.method = method
        self.selected = tuple(int(i) for i in selected)
        self.concentrations = np.asarray(concentrations, dtype=float).ravel()
        self.rss = float(rss)
        self.bic = None if bic is None else float(bic)
        self.top_K = list(top_K or [])
        self.stats = dict(stats or {})

    @property
    def best_support(self):
        """Alias of :attr:`selected`."""
        return self.selected

    @property
    def best_concentrations(self):
        """Alias of :attr:`concentrations`."""
        return self.concentrations

    @property
    def best_score(self):
        """Alias of :attr:`bic`."""
        return self.bic

    @property
    def k_pred(self):
        """Predicted number of components."""
        return len(self.selected)

    def __repr__(self):
        return (f"Result(method={self.method!r}, selected={self.selected}, "
                f"k_pred={self.k_pred}, bic={self.bic})")


class Problem:
    """An LCF identification task: find the components behind a target spectrum.

    Parameters
    ----------
    A : np.ndarray (m, N)
        Library matrix; column ``j`` is the reference spectrum of component
        ``j``.
    y : np.ndarray (m,)
        Target (mixture) spectrum.
    k_max : int or None
        Number of components assumed in the mixture (default None =
        unknown).  QUBO-backed solvers use it as the cardinality target and
        solver-side bound; alternative-backend methods use it as the maximum
        support size.  When None, the QUBO cardinality penalty is switched
        off and the solver-side bound falls back to 15.
    tau : float
        Concentration cutoff of the refinement stage (default 0.01):
        refitted concentrations <= ``tau`` are trimmed from the reported
        support.
    weights : np.ndarray (m,) or None
        Optional non-negative per-point weights; rows of ``A`` and entries
        of ``y`` are scaled by ``weights`` for all fitting.

    Usage::

        import qubic

        task = qubic.Problem(A, y, k_max=2, tau=0.01)
        res = task.solve(qubic.MILP(time_limit=30.0))   # QUBO-backed
        res = task.solve(qubic.SA(restarts=20, T0=1.0))  # QUBO-backed
        res = task.solve(qubic.OMP())                    # alternative backend

    The fit runs in the weighted space ``(Aw, yw)`` (identical to ``(A, y)``
    when ``weights`` is None); :attr:`A` and :attr:`y` expose the raw arrays.
    """

    def __init__(self, A, y, k_max=None, tau=0.01, weights=None):
        A = np.asarray(A, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if A.ndim != 2:
            raise ValueError(f"A must be 2-D (m, N); got shape {A.shape}")
        if A.shape[0] != y.shape[0]:
            raise ValueError(
                f"A has {A.shape[0]} rows but y has {y.shape[0]} entries"
            )
        if k_max is not None:
            if not isinstance(k_max, (int, np.integer)) or k_max < 1:
                raise ValueError(f"k_max must be a positive int or None; got {k_max!r}")
            k_max = int(k_max)
        if tau < 0:
            raise ValueError(f"tau must be >= 0; got {tau!r}")

        self._A = A
        self._y = y
        self.k_max = k_max
        self.tau = float(tau)

        if weights is not None:
            w = np.asarray(weights, dtype=float).ravel()
            if w.shape != y.shape:
                raise ValueError(
                    f"weights must have shape {y.shape}; got {w.shape}"
                )
            if np.any(w < 0):
                raise ValueError("weights must be non-negative")
            self._Aw = A * w[:, None]
            self._yw = y * w
        else:
            self._Aw = A
            self._yw = y

        #: Memo of built QUBOs keyed by (encoding, k, resolved penalties).
        self._qubo_memo = {}

    # -- task data ---------------------------------------------------------

    @property
    def A(self):
        """Raw library matrix (m, N)."""
        return self._A

    @property
    def y(self):
        """Raw target spectrum (m,)."""
        return self._y

    @property
    def n_points(self):
        """Number of spectral points ``m``."""
        return self._A.shape[0]

    @property
    def n_components(self):
        """Library size ``N``."""
        return self._A.shape[1]

    @property
    def _k_target(self):
        """QUBO cardinality target: ``k_max`` or 0 (penalty switched off)."""
        return self.k_max if self.k_max is not None else 0

    @property
    def solver_k_bound(self):
        """Cardinality bound forwarded to QUBO solvers (default 15)."""
        return self.k_max if self.k_max is not None else 15

    # -- QUBO layer (lazy + memoized) -------------------------------------

    def _get_qubo(self, encoding, lambdas, use_jit=False):
        """Build (or reuse) the QUBO for this problem.

        Returns ``(Q, q, var_map, decode)``.  Memoized on the problem so a
        second QUBO-backed solver with the same (encoding, penalties) never
        rebuilds it; alternative-backend solvers never call this at all.
        ``use_jit`` selects the optional numba assembly kernel (see
        :mod:`qubic.qubo`) only when a build actually happens -- the two
        paths produce bitwise-identical QUBOs, so the memo key does not
        depend on it.
        """
        from . import qubo as qubo_layer

        k = self._k_target
        lam = qubo_layer.resolve_lambdas(
            encoding, lambdas, self._Aw, self._yw, k)
        key = (encoding, k, tuple(sorted(lam.items())))
        if key not in self._qubo_memo:
            self._qubo_memo[key] = qubo_layer.build_qubo(
                encoding, self._Aw, self._yw, k, use_jit=use_jit, **lam)
        return self._qubo_memo[key]

    # -- dispatch ----------------------------------------------------------

    def solve(self, solver, top_k=10, use_jit=False):
        """Run ``solver`` on this problem and return a :class:`Result`.

        ``solver`` is a solver object bundling its own settings, e.g.
        ``qubic.MILP(time_limit=30.0)``, ``qubic.SA(restarts=20, T0=1.0)``
        or ``qubic.OMP()``.  ``top_k`` caps the number of refined QUBO
        candidates kept in ``Result.top_K`` (ignored by the alternative
        backend, which has a single candidate).

        ``use_jit`` (default ``False``) opts into the optional numba
        acceleration of the whole run: the QUBO assembly, the solver's own
        kernel where one exists (SA/Tabu/QAOA) and the NNLS refinement.
        The default path never imports numba ("don't pay for what you don't
        use").  The numba/JIT acceleration was NOT used in the article.
        """
        from .solver import Solver

        if not isinstance(solver, Solver):
            raise TypeError(
                "solver must be a solver object such as qubic.MILP(), "
                "qubic.SA(...) or qubic.OMP(); got "
                f"{type(solver).__name__}"
            )
        if not isinstance(top_k, (int, np.integer)) or top_k < 1:
            raise ValueError(f"top_k must be a positive int; got {top_k!r}")
        return solver._run(self, int(top_k), use_jit=bool(use_jit))
