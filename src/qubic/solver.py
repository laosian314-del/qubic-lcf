"""Public solver facade: the :class:`Solver` class hierarchy.

A solver object bundles all of its own settings and is passed whole to a
task object (see :class:`qubic.Problem`); the same interface covers the
QUBO-backed solvers and the alternative backend::

    import qubic

    task = qubic.Problem(A, y, k_max=2, tau=0.01)
    res = task.solve(qubic.MILP(time_limit=30.0))    # QUBO-backed
    res = task.solve(qubic.SA(restarts=20, T0=1.0))  # QUBO-backed
    res = task.solve(qubic.OMP())                    # alternative backend

Layered architecture (internal -> public facade):

    encodings -> QUBO builder (uses lambdas) -> low-level solver registry
        -> refinement (NNLS + tau threshold + selection)

The classes below are the elegant public facade on those layers:
:class:`QuboSolver` holds the QUBO-construction settings (``encoding``,
``lambdas``) and builds the QUBO lazily inside its solve call only;
alternative-backend classes never trigger any QUBO work.

Lazy semantics ("don't pay for what you don't use"):
running ``task.solve(qubic.OMP())`` builds no QUBO at all; solving one
problem with several QUBO solvers builds the QUBO at most once per
(encoding, penalties) combination -- it is memoized on the problem.
"""
from __future__ import annotations

import abc
import time

import numpy as np

from .pipeline import full_library_candidate, refine_candidate
from .problem import Result
from .qubo import PENALTY_KEYS


class Solver(abc.ABC):
    """Abstract base of every solver object.

    A solver bundles its own settings; the task data lives in
    :class:`qubic.Problem`.  The internal contract is
    ``Problem.solve(solver) -> Result``: ``Problem.solve`` delegates to
    ``solver._run(problem, top_k)``.  Solver objects are stateless with
    respect to problems and can be reused across problems.
    """

    #: Method registry name recorded in :attr:`qubic.Result.method`.
    name = ''

    @abc.abstractmethod
    def _run(self, problem, top_k):
        """Solve ``problem`` and return a :class:`qubic.Result`."""


class QuboSolver(Solver):
    """Base class of the QUBO-backed solvers.

    Holds the QUBO-construction settings shared by all of them:

    encoding : str
        Encoding scheme name (default ``'linked_B2'``), see
        :data:`qubic.encodings.SCHEMES`.
    lambdas : dict of floats or callable or None
        Penalty coefficients of the QUBO builder (normally hardcoded
        there).  Two variants are supported:

        * a dict of floats keyed by the scheme's penalty names
          (``lam_sparse``, ``lam_link``, ``lam_antimix`` -- see
          :data:`qubic.qubo.PENALTY_KEYS`)::

              qubic.MILP(lambdas={'lam_sparse': 1.0, 'lam_link': 5.0})

        * a callable ``(A, y) -> dict of floats`` so the penalties can
          depend on the spectrum scale, e.g. proportional to ``||y||^2``::

              qubic.MILP(lambdas=lambda A, y: {
                  'lam_sparse': 0.01 * float(y @ y),
                  'lam_link': 0.05 * float(y @ y)})

        ``None`` (default) uses the encoding builder defaults
        (``lam_sparse=1.0`` with a cardinality target else 0.0,
        ``lam_link=5.0``, ``lam_antimix=5.0``).

    The QUBO is built lazily inside the solve call only -- never at
    construction time, never by alternative-backend solvers -- and is
    memoized on the :class:`qubic.Problem`.
    """

    #: Name of the low-level solver in :data:`qubic.solvers.SOLVERS`.
    solver_name = ''

    def __init__(self, encoding='linked_B2', lambdas=None):
        if encoding not in PENALTY_KEYS:
            raise ValueError(
                f"Unknown encoding {encoding!r}; choose from "
                f"{sorted(PENALTY_KEYS)}"
            )
        self.encoding = encoding
        self.lambdas = lambdas

    def _settings(self):
        """Low-level solver-specific kwargs forwarded to qubic.solvers.solve."""
        return {}

    def _run(self, problem, top_k):
        from . import solvers as _lowlevel

        Q, q, var_map, decode = problem._get_qubo(self.encoding, self.lambdas)
        solutions, wall_time, gap = _lowlevel.solve(
            self.solver_name, Q, q, var_map,
            k_max=problem.solver_k_bound, top_k=top_k,
            time_limit=self.time_limit, **self._settings())

        # Decode -> refine (NNLS + tau cutoff) -> BIC.
        candidates = []
        for sample, energy in solutions:
            sel, _raw_conc = decode(sample)
            refined = refine_candidate(
                problem._Aw, problem._yw, sel, problem.tau)
            if refined is None:
                continue
            sel_f, conc_f, rss, bic = refined
            candidates.append({
                'selected': tuple(int(i) for i in sel_f),
                'concentrations': conc_f,
                'rss': rss,
                'bic': bic,
                'qubo_energy': float(energy),
            })
        stats = {
            'wall_time': float(wall_time),
            'solver_gap': gap,
            'encoding': self.encoding,
            'n_solutions': len(solutions),
        }
        if not candidates:
            # Fallback: NNLS on the full library (run_pipeline step 5).
            sel_f, conc_f, rss, bic = full_library_candidate(
                problem._Aw, problem._yw, problem.tau)
            candidates.append({
                'selected': tuple(int(i) for i in sel_f),
                'concentrations': conc_f,
                'rss': rss,
                'bic': bic,
                'qubo_energy': float('inf'),
            })
            stats['fallback'] = 'full_library_nnls'
        candidates.sort(key=lambda c: c['bic'])
        top_K = candidates[:top_k]
        best = top_K[0]
        return Result(self.name, best['selected'], best['concentrations'],
                      best['rss'], best['bic'], top_K=top_K, stats=stats)


# --------------------------------------------------------------------------
# Concrete QUBO-backed solvers (own settings on top of QuboSolver)
# --------------------------------------------------------------------------

class MILP(QuboSolver):
    """MILP on the McCormick linearization via PuLP (port of benchmark_v3.py).

    Settings: ``time_limit`` (wall-clock budget in seconds), ``msg`` (solver
    log), ``prune_threshold`` (skip McCormick constraints for tiny |Q_ij|),
    ``solver_backend`` (``'auto'`` | ``'cbc'`` | ``'highs'``).
    """

    name = 'milp'
    solver_name = 'milp'

    def __init__(self, time_limit=None, encoding='linked_B2', lambdas=None,
                 msg=False, prune_threshold=1e-8, solver_backend='auto'):
        super().__init__(encoding, lambdas)
        self.time_limit = time_limit
        self.msg = msg
        self.prune_threshold = prune_threshold
        self.solver_backend = solver_backend

    def _settings(self):
        return {'msg': self.msg, 'prune_threshold': self.prune_threshold,
                'solver_backend': self.solver_backend}


class HiGHS(QuboSolver):
    """MILP via the PuLP HiGHS interface with automatic CBC fallback.

    Settings: ``time_limit``, ``msg``, ``prune_threshold``, ``prefer``
    (``'highs'`` | ``'cbc'``).
    """

    name = 'highs'
    solver_name = 'highs'

    def __init__(self, time_limit=None, encoding='linked_B2', lambdas=None,
                 msg=False, prune_threshold=1e-8, prefer='highs'):
        super().__init__(encoding, lambdas)
        self.time_limit = time_limit
        self.msg = msg
        self.prune_threshold = prune_threshold
        self.prefer = prefer

    def _settings(self):
        return {'msg': self.msg, 'prune_threshold': self.prune_threshold,
                'prefer': self.prefer}


class CP_SAT(QuboSolver):
    """OR-Tools CP-SAT solver.  Settings: ``time_limit``."""

    name = 'cp_sat'
    solver_name = 'cp_sat'

    def __init__(self, time_limit=None, encoding='linked_B2', lambdas=None):
        super().__init__(encoding, lambdas)
        self.time_limit = time_limit


class SCIP(QuboSolver):
    """PySCIPOpt solver.  Settings: ``time_limit``."""

    name = 'scip'
    solver_name = 'scip'

    def __init__(self, time_limit=None, encoding='linked_B2', lambdas=None):
        super().__init__(encoding, lambdas)
        self.time_limit = time_limit


class SA(QuboSolver):
    """Simulated annealing (dimod SimulatedAnnealingSampler).

    Settings:

    restarts : int or None
        Independent annealing runs drawn per batch (``n_reads_per_call``;
        None -> size-based default).
    T0 : float or None
        Initial temperature of the geometric annealing schedule: the
        inverse-temperature range starts at ``1 / T0`` (None -> dimod's
        default schedule).
    T_final : float or None
        Final temperature of the schedule (default ``T0 / 1000`` when
        ``T0`` is given).
    num_sweeps : int or None
        Sweeps per run (None -> size-based default).
    time_limit : float or None
        Wall-clock budget in seconds.
    """

    name = 'sa'
    solver_name = 'sa'

    def __init__(self, restarts=None, T0=None, T_final=None, num_sweeps=None,
                 time_limit=None, encoding='linked_B2', lambdas=None):
        super().__init__(encoding, lambdas)
        self.restarts = restarts
        self.T0 = T0
        self.T_final = T_final
        self.num_sweeps = num_sweeps
        self.time_limit = time_limit

    def _settings(self):
        settings = {'num_sweeps': self.num_sweeps,
                    'n_reads_per_call': self.restarts}
        if self.T0 is not None:
            if self.T0 <= 0:
                raise ValueError(f"T0 must be positive; got {self.T0!r}")
            t_final = (self.T_final if self.T_final is not None
                       else self.T0 / 1000.0)
            if t_final <= 0:
                raise ValueError(f"T_final must be positive; got {t_final!r}")
            settings['beta_range'] = (1.0 / self.T0, 1.0 / t_final)
        return settings


class Tabu(QuboSolver):
    """Tabu search (dwave-samplers TabuSampler).

    Settings: ``per_call_timeout_ms`` (per batch), ``time_limit``.
    """

    name = 'tabu'
    solver_name = 'tabu'

    def __init__(self, per_call_timeout_ms=None, time_limit=None,
                 encoding='linked_B2', lambdas=None):
        super().__init__(encoding, lambdas)
        self.per_call_timeout_ms = per_call_timeout_ms
        self.time_limit = time_limit

    def _settings(self):
        if self.per_call_timeout_ms is None:
            return {}
        return {'per_call_timeout_ms': self.per_call_timeout_ms}


class GPU_SA(QuboSolver):
    """GPU simulated annealing (PyTorch annealer of QUBO_LCF_3).

    Settings: ``multi_pass`` (use the multi-pass variant with forbidden-set
    penalties), ``num_reads``, ``num_sweeps``, ``batch_size``,
    ``initial_temp``, ``final_temp``, ``time_limit``; further options are
    forwarded to :func:`qubic.solvers.gpu_sa.solve` (``diversity_scale``,
    ``max_passes``, ``forbidden_penalty_strength``).
    """

    name = 'gpu_sa'
    solver_name = 'gpu_sa'

    def __init__(self, multi_pass=False, num_reads=None, num_sweeps=None,
                 batch_size=None, initial_temp=None, final_temp=None,
                 time_limit=None, encoding='linked_B2', lambdas=None,
                 **settings):
        super().__init__(encoding, lambdas)
        self.multi_pass = multi_pass
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.batch_size = batch_size
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.time_limit = time_limit
        self.settings = settings

    def _settings(self):
        out = dict(self.settings)
        out['multi_pass'] = self.multi_pass
        for name in ('num_reads', 'num_sweeps', 'batch_size',
                     'initial_temp', 'final_temp'):
            value = getattr(self, name)
            if value is not None:
                out[name] = value
        return out


class SQAOD(QuboSolver):
    """Simulated quantum annealing via the optional ``sqaod`` package.

    Settings: ``n_trotters`` (replicas), ``G`` (transverse-field strength),
    ``beta`` (inverse temperature), ``max_dense_vars`` (above it the
    documented random-sampling fallback is used), ``seed``, ``time_limit``.
    """

    name = 'sqaod'
    solver_name = 'sqaod'

    def __init__(self, n_trotters=None, G=1.0, beta=1.0, max_dense_vars=20,
                 seed=42, time_limit=None, encoding='linked_B2', lambdas=None):
        super().__init__(encoding, lambdas)
        self.n_trotters = n_trotters
        self.G = G
        self.beta = beta
        self.max_dense_vars = max_dense_vars
        self.seed = seed
        self.time_limit = time_limit

    def _settings(self):
        return {'n_trotters': self.n_trotters, 'G': self.G,
                'beta': self.beta, 'max_dense_vars': self.max_dense_vars,
                'seed': self.seed}


class QAOA(QuboSolver):
    """QAOA with COBYLA optimization (scipy, no quantum hardware needed).

    Settings: ``p`` (circuit depth), ``max_iter`` (COBYLA iterations),
    ``n_shots``, ``seed``, ``time_limit`` (accepted; the classical loop is
    not budget-limited).
    """

    name = 'qaoa'
    solver_name = 'qaoa'

    def __init__(self, p=2, max_iter=100, n_shots=1024, seed=0,
                 time_limit=None, encoding='linked_B2', lambdas=None):
        super().__init__(encoding, lambdas)
        self.p = p
        self.max_iter = max_iter
        self.n_shots = n_shots
        self.seed = seed
        self.time_limit = time_limit

    def _settings(self):
        return {'p': self.p, 'max_iter': self.max_iter,
                'n_shots': self.n_shots, 'seed': self.seed}


# --------------------------------------------------------------------------
# Alternative backend (encoding-free methods of qubic.alternative)
# --------------------------------------------------------------------------

class AlternativeSolver(Solver):
    """Base class of the alternative-backend solver objects.

    These run encoding-free classical methods from :mod:`qubic.alternative`
    -- the greedy forward selection (MP/OMP) family and friends (called
    baselines only in the paper) -- and never build a QUBO.  Each subclass
    forwards its own settings to the underlying method.
    """

    def _method(self, A, y, problem):
        """Run the underlying alternative method; return its result dict."""
        raise NotImplementedError

    def _run(self, problem, top_k):
        t0 = time.time()
        res = self._method(problem._Aw, problem._yw, problem)
        wall_time = time.time() - t0
        selected = tuple(int(i) for i in res['selected'])
        concentrations = np.asarray(res['concentrations'], dtype=float)
        entry = {
            'selected': selected,
            'concentrations': concentrations,
            'rss': float(res['rss']),
            'bic': None if res['bic'] is None else float(res['bic']),
            'qubo_energy': None,
        }
        top_K = [entry] if selected else []
        return Result(self.name, selected, concentrations,
                      float(res['rss']),
                      None if res['bic'] is None else float(res['bic']),
                      top_K=top_K, stats={'wall_time': float(wall_time)})


class MP(AlternativeSolver):
    """Greedy forward selection (MP/OMP) family: Matching Pursuit.

    Settings: ``k_max`` (max support size; default: the problem's ``k_max``
    or 15).
    """

    name = 'mp_bic'

    def __init__(self, k_max=None):
        self.k_max = k_max

    def _method(self, A, y, problem):
        from .alternative import mp_bic
        k_max = self.k_max if self.k_max is not None else (problem.k_max or 15)
        return mp_bic(A, y, k_max=k_max)


class OMP(AlternativeSolver):
    """Greedy forward selection (MP/OMP) family: Orthogonal Matching Pursuit.

    Settings: ``k_max`` (max support size; default: the problem's ``k_max``
    or 15).
    """

    name = 'omp_bic'

    def __init__(self, k_max=None):
        self.k_max = k_max

    def _method(self, A, y, problem):
        from .alternative import omp_bic
        k_max = self.k_max if self.k_max is not None else (problem.k_max or 15)
        return omp_bic(A, y, k_max=k_max)


class Lasso(AlternativeSolver):
    """Alternative backend: non-negative LASSO over an alpha grid.

    Settings: ``alphas`` (grid of penalties; default: the frozen grid of
    the original LASSO+ method).
    """

    name = 'lasso_positive'

    def __init__(self, alphas=None):
        self.alphas = alphas

    def _method(self, A, y, problem):
        from .alternative import lasso_positive
        return lasso_positive(A, y, alphas=self.alphas)


class TopK(AlternativeSolver):
    """Alternative backend: top-k column ranking by |<a_j, y>|/||a_j||.

    Settings: ``k_max`` (max support size; default: the problem's ``k_max``
    or 15).
    """

    name = 'topk_corr_bic'

    def __init__(self, k_max=None):
        self.k_max = k_max

    def _method(self, A, y, problem):
        from .alternative import topk_corr_bic
        k_max = self.k_max if self.k_max is not None else (problem.k_max or 15)
        return topk_corr_bic(A, y, k_max=k_max)


class Oracle(AlternativeSolver):
    """Alternative backend: exact exhaustive oracle (2^N subsets).

    Settings: ``max_n`` (library size cap; the oracle is exponential).
    """

    name = 'oracle_exact'

    def __init__(self, max_n=15):
        self.max_n = max_n

    def _method(self, A, y, problem):
        from .alternative import oracle_exact
        return oracle_exact(A, y, max_n=self.max_n)
