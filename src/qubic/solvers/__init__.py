"""Unified QUBO solver layer for QUBIC-LCF.

Every solver implements the same contract::

    solve(solver, Q, q, var_map, k_max=15, top_k=20, time_limit=None, **kwargs)
        -> (solutions, wall_time, gap)

    solutions : list of (sample, energy)
        sample : np.ndarray (M,) of 0/1 decisions
        energy : float — exact QUBO energy x^T Q x + q^T x
        sorted by ascending energy, at most ``top_k`` entries
    wall_time : float — measured wall-clock seconds
    gap : float | None — optimality gap when the solver reports one

``SOLVERS`` maps solver name -> callable with the same signature as the
unified ``solve`` (minus the ``solver`` argument): ``milp`` (PuLP MILP on
the McCormick linearization), ``highs`` (PuLP HiGHS MILP with CBC
fallback), ``cp_sat`` (OR-Tools CP-SAT), ``scip`` (PySCIPOpt), ``sa``
(dimod simulated annealing), ``tabu`` (dwave-samplers TabuSampler),
``gpu_sa`` (PyTorch GPU annealer), ``sqaod`` (sqaod simulated quantum
annealing) and ``qaoa`` (QAOA via SciPy COBYLA). Optional dependencies are
resolved lazily inside each solver; a missing dependency raises ImportError
with a ``pip install ...`` hint. :func:`available_solvers` lists the solvers
whose dependencies are actually usable in the current environment.
"""
from __future__ import annotations

import importlib.util

from . import (base, cp_sat, gpu_sa, highs, milp, qaoa, sa, scip, sqaod,
               tabu)
from .base import DEFAULT_TIME_LIMIT, get_selection, qubo_energy

#: Registry: solver name -> callable(Q, q, var_map, k_max, top_k, time_limit, **kw).
SOLVERS = {
    "milp": milp.solve,
    "highs": highs.solve,
    "cp_sat": cp_sat.solve,
    "scip": scip.solve,
    "sa": sa.solve,
    "tabu": tabu.solve,
    "gpu_sa": gpu_sa.solve,
    "sqaod": sqaod.solve,
    "qaoa": qaoa.solve,
}

#: Optional dependencies per solver: (module path, PyPI package name).
_DEPS = {
    "milp": (("pulp", "pulp"),),
    "highs": (("pulp", "pulp"),),
    "cp_sat": (("ortools", "ortools"),),
    "scip": (("pyscipopt", "pyscipopt"),),
    "sa": (("dimod", "dimod"),),
    "tabu": (("dwave.samplers", "dwave-samplers"), ("dimod", "dimod")),
    "gpu_sa": (("torch", "torch"),),
    "sqaod": (("sqaod", "sqaod"),),
    "qaoa": (("scipy", "scipy"),),
}


def _module_available(module_name):
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def available_solvers():
    """Names of solvers whose dependencies are actually installed/usable.

    gpu_sa additionally requires a CUDA device (a CPU-only torch build cannot
    run it), so it is listed only when ``torch.cuda.is_available()`` is True.
    """
    names = []
    for name in SOLVERS:
        if not all(_module_available(mod) for mod, _ in _DEPS[name]):
            continue
        if name == "gpu_sa":
            try:
                import torch
                if not torch.cuda.is_available():
                    continue
            except Exception:
                continue
        names.append(name)
    return names


def solve(solver, Q, q, var_map, k_max=15, top_k=20, time_limit=None, **kwargs):
    """Run ``solver`` on QUBO ``(Q, q)`` with the unified contract.

    Parameters
    ----------
    solver : str
        One of ``sorted(SOLVERS)``.
    Q : np.ndarray (M, M)
        Symmetric QUBO matrix.
    q : np.ndarray (M,)
        Linear coefficients.
    var_map : dict
        Encoding metadata ('scheme', 'N', 'M', ...).
    k_max : int
        Cardinality bound on the component selection (where enforced).
    top_k : int
        Maximum number of returned solutions.
    time_limit : float or None
        Wall-clock budget in seconds; stochastic solvers substitute a default
        (base.DEFAULT_TIME_LIMIT) when None.
    **kwargs
        Solver-specific options (see each module's ``solve`` docstring).

    Returns
    -------
    (solutions, wall_time, gap) — see module docstring.
    """
    try:
        solver_fn = SOLVERS[solver]
    except KeyError:
        raise ValueError(
            f"unknown solver {solver!r}; expected one of {sorted(SOLVERS)}"
        ) from None
    return solver_fn(Q, q, var_map, k_max=k_max, top_k=top_k,
                     time_limit=time_limit, **kwargs)


__all__ = [
    "SOLVERS",
    "DEFAULT_TIME_LIMIT",
    "available_solvers",
    "get_selection",
    "qubo_energy",
    "solve",
]
