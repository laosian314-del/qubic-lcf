"""SCIP solver for QUBO-LCF via PySCIPOpt (McCormick linearization + integer cuts).

Ported from ``QUBO_LCF_3/scripts/benchmark_v3.py`` (``solve_scip`` /
``_solve_one_scip``). pyscipopt is an optional dependency and is imported
lazily; without it a plain ImportError with a ``pip install pyscipopt`` hint is
raised.

Public API (ported):
    solve_scip(Q, q, var_map, k_max, time_limit, top_k)
        -> (solutions, wall_time, gap)

Unified API:
    solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, **kwargs)
        -> (solutions, wall_time, gap)
"""
from __future__ import annotations

import time

import numpy as np

from . import base
from .base import get_selection


def _solve_one_scip(Q, q, var_map, k_max, time_limit, prev_solutions):
    """Single SCIP solve with McCormick linearization and integer cuts."""
    pyscipopt = base.require("pyscipopt", "pyscipopt")
    Model, quicksum = pyscipopt.Model, pyscipopt.quicksum

    M = Q.shape[0]
    N = var_map["N"]
    scheme = var_map["scheme"]
    m = Model("QUBO_LCF_SCIP")
    m.hideOutput()
    if time_limit is not None:
        m.setParam("limits/time", time_limit)

    # Binary variables
    x = [m.addVar(f"x{i}", "BINARY") for i in range(M)]
    z = [m.addVar(f"z{j}", "BINARY") for j in range(N)]

    # Selection variable constraints
    if scheme == "B1":
        for j in range(N):
            m.addCons(z[j] == x[j], f"sel_B1_{j}")
    elif scheme == "B2":
        idx_s = var_map["idx_s"]
        for j in range(N):
            m.addCons(z[j] == x[idx_s(j)], f"sel_B2_{j}")
    elif scheme in ("3bit", "4bit"):
        vpc = var_map["vars_per_component"]
        idx_fn = var_map["idx"]
        for j in range(N):
            for b in range(vpc):
                m.addCons(z[j] >= x[idx_fn(j, b)], f"sel_lower_{j}_{b}")
            m.addCons(z[j] <= quicksum([x[idx_fn(j, b)] for b in range(vpc)]),
                      f"sel_upper_{j}")
    else:
        raise ValueError(
            f"solve_scip: unsupported scheme {scheme!r}; "
            "expected 'B1', 'B2', '3bit' or '4bit'"
        )

    # Cardinality constraint
    m.addCons(quicksum(z) <= k_max, "card")

    # Objective: McCormick linearization for off-diagonal bilinears
    obj_expr = quicksum([(Q[i, i] + q[i]) * x[i] for i in range(M)])
    for i in range(M):
        for j in range(i + 1, M):
            if abs(Q[i, j]) < 1e-10:
                continue
            z_ij = m.addVar(f"z_{i}_{j}", "CONTINUOUS", lb=0.0, ub=1.0)
            m.addCons(z_ij <= x[i], f"mc_up_i_{i}_{j}")
            m.addCons(z_ij <= x[j], f"mc_up_j_{i}_{j}")
            m.addCons(z_ij >= x[i] + x[j] - 1, f"mc_lo_{i}_{j}")
            obj_expr += 2 * Q[i, j] * z_ij

    m.setObjective(obj_expr, "minimize")

    # Integer cuts: Hamming distance >= 1 from each previous solution
    for pi, (prev_sample, _) in enumerate(prev_solutions):
        zp = get_selection(prev_sample, var_map)
        cut_expr = quicksum([(1 - z[j]) if zp[j] else z[j] for j in range(N)])
        m.addCons(cut_expr >= 1, f"cut_{pi}")

    t0 = time.time()
    try:
        m.optimize()
    except Exception:
        return None, time.time() - t0
    t = time.time() - t0
    status = m.getStatus()
    n_sols = m.getNSols()
    if status == "infeasible" or n_sols == 0:
        return None, t

    # Only access solution values if model was solved successfully
    sol_statuses = ("optimal", "feasible", "bestsollimit", "gaplimit", "timelimit")
    if status not in sol_statuses:
        return None, t

    try:
        sample = np.array([m.getVal(x[i]) for i in range(M)])
        obj_val = m.getObjVal()
    except Exception:
        return None, t

    return (sample, obj_val), t


def solve_scip(Q, q, var_map, k_max=15, time_limit=None, top_k=20):
    """SCIP solver via pyscipopt with McCormick linearization.

    Returns ``(solutions, total_wall_time, gap)`` with ``solutions`` a list of
    ``(sample, objective)`` pairs collected via integer cuts (``gap`` is always
    ``None``: SCIP's global gap is not exposed through this enumeration loop).
    """
    per_call_budget = (max(1.0, time_limit / top_k)
                       if time_limit is not None else None)

    # Resolve the optional dependency up front so a missing pyscipopt raises
    # ImportError (with install hint) instead of being swallowed by the loop.
    base.require("pyscipopt", "pyscipopt")

    all_solutions = []
    prev_samples = []
    total_time = 0.0
    seen_subsets = set()
    gap = None

    for _ in range(top_k):
        t0 = time.time()
        try:
            result, t = _solve_one_scip(Q, q, var_map, k_max, per_call_budget,
                                        prev_samples)
        except Exception:
            total_time += time.time() - t0
            break

        total_time += time.time() - t0

        if result is None:
            break
        sample, obj_val = result
        sel_bits = tuple(get_selection(sample, var_map))
        if sel_bits in seen_subsets:
            prev_samples.append((sample, obj_val))
            continue
        seen_subsets.add(sel_bits)
        all_solutions.append((sample, obj_val))
        prev_samples.append((sample, obj_val))

    return all_solutions, total_time, gap


def solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, **kwargs):
    """Unified solver entry point (see :mod:`qubic.solvers.base`).

    Raises ImportError (with a ``pip install pyscipopt`` hint) when pyscipopt
    is not installed.
    """
    base.reject_unknown_kwargs("scip.solve", kwargs)
    pairs, wall_time, gap = solve_scip(Q, q, var_map, k_max=k_max,
                                       time_limit=time_limit, top_k=top_k)
    solutions = base.finalize((s for s, _ in pairs), Q, q, top_k)
    return solutions, wall_time, gap
