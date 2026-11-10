"""MILP solver for QUBO-LCF via the PuLP HiGHS interface (CBC fallback).

Ported from ``QUBO_LCF_3/scripts/benchmark_v3.py`` (``solve_milp`` with
solver name ``'HiGHS'`` and ``_solve_one_milp``): standard McCormick
linearization of the QUBO objective, a cardinality bound on the component
selection and top-K enumeration via integer cuts on the selection
variables.  The primary backend is ``pulp.HiGHS`` (HiGHS through its Python
interface); when HiGHS is unavailable or fails, the solve automatically
falls back to CBC through PuLP (``pulp.PULP_CBC_CMD``), so the solver works
with any PuLP installation.

Public API (ported):
    solve_highs(Q, q, var_map, k_max, time_limit, ...)
        -> (solutions, wall_time, gap=None)

Unified API:
    solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, **kwargs)
        -> (solutions, wall_time, gap=None)

Extra kwargs: ``msg`` (solver log), ``prune_threshold`` (skip McCormick
constraints for |Q_ij| below it), ``prefer`` ('highs' | 'cbc').
"""
from __future__ import annotations

import time

import numpy as np

from . import base


def _make_pulp_solver(msg, time_limit, prefer="highs"):
    """Create a PuLP solver: ``pulp.HiGHS`` first, CBC fallback.

    Args:
        msg: log to stdout when True.
        time_limit: per-call wall-clock limit in seconds (None -> 3600).
        prefer: 'highs' tries ``pulp.HiGHS`` then CBC; 'cbc' uses CBC only.

    Returns:
        A ready-to-use PuLP solver object.
    """
    pulp = base.require("pulp", "pulp")
    tl = time_limit if time_limit is not None else 3600
    if prefer not in ("highs", "cbc"):
        raise ValueError(f"prefer must be 'highs' or 'cbc', got {prefer!r}")
    if prefer == "highs":
        try:
            solver = pulp.HiGHS(msg=msg, timeLimit=tl, threads=1)
            if solver.available():
                return solver
        except Exception:
            pass
    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=tl, threads=1)
    if solver.available():
        return solver
    raise ImportError(
        "no PuLP solver backend available (pulp.HiGHS and CBC both missing); "
        "install it with: pip install pulp"
    )


def _solve_one(Q, q, var_map, k_max, time_limit, msg, prev_solutions,
               prune_threshold, prefer):
    """Solve the McCormick MILP for one top-K slot.

    Returns:
        ``(sample, selected_indices, solve_time)`` or ``(None, None,
        elapsed)`` if the model is infeasible or the backend failed.
    """
    pulp = base.require("pulp", "pulp")
    M = Q.shape[0]
    N = var_map["N"]
    scheme = var_map["scheme"]

    prob = pulp.LpProblem("QUBO_LCF_HIGHS", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(M)]
    z = [pulp.LpVariable(f"z{j}", cat="Binary") for j in range(N)]

    if scheme == "B1":
        for j in range(N):
            prob += z[j] == x[j]
    elif scheme == "B2" or scheme.startswith("linked_"):
        idx_s = var_map["idx_s"]
        for j in range(N):
            prob += z[j] == x[idx_s(j)]
    elif scheme in ("B3", "B4"):
        vpc = var_map["vars_per_component"]
        idx_fn = var_map["idx"]
        for j in range(N):
            for b in range(vpc):
                prob += z[j] >= x[idx_fn(j, b)]
            prob += z[j] <= pulp.lpSum([x[idx_fn(j, b)] for b in range(vpc)])
    else:
        raise ValueError(
            f"solve_highs: unsupported scheme {scheme!r}; "
            "expected 'B1', 'B2', 'B3', 'B4' or 'linked_B{2,3,4}'"
        )

    # Cardinality bound on the component selection
    prob += pulp.lpSum(z) <= k_max, "card"

    # Objective: linear + diagonal + McCormick linearized off-diagonals
    obj = pulp.lpSum(q[i] * x[i] for i in range(M))
    for i in range(M):
        obj += Q[i, i] * x[i]
        for j in range(i + 1, M):
            if abs(Q[i, j]) < prune_threshold:
                continue
            w = pulp.LpVariable(f"w_{i}_{j}", cat="Binary")
            prob += w <= x[i]
            prob += w <= x[j]
            prob += w >= x[i] + x[j] - 1
            obj += Q[i, j] * w
    prob += obj

    # Top-K enumeration: component-level integer cuts on previous selections
    # (scheme-agnostic, mirrors milp.collect_top_k semantics)
    for sel in prev_solutions:
        prob += pulp.lpSum([(1 - z[j]) if j in sel else z[j]
                            for j in range(N)]) >= 1

    solver = _make_pulp_solver(msg, time_limit, prefer=prefer)
    t0 = time.time()
    status = prob.solve(solver)
    elapsed = time.time() - t0

    if pulp.LpStatus[status] not in ("Optimal", "Feasible"):
        return None, None, elapsed

    sample = np.array([pulp.value(x[i]) or 0.0 for i in range(M)])
    selected = tuple(sorted(j for j in range(N) if pulp.value(z[j]) > 0.5))
    return sample, selected, elapsed


def solve_highs(Q, q, var_map, k_max=15, time_limit=None, top_k=20,
                msg=False, prune_threshold=1e-8, prefer="highs"):
    """Enumerate up to ``top_k`` distinct selections via HiGHS MILP solves.

    The wall-clock budget is split evenly across the top-K calls
    (at least 1 s per call) so the total stays within ``time_limit``.
    """
    base.require("pulp", "pulp")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if k_max < 0:
        raise ValueError(f"k_max must be non-negative, got {k_max}")
    if time_limit is not None and time_limit <= 0:
        raise ValueError(f"time_limit must be positive, got {time_limit}")

    per_call_budget = (max(1.0, time_limit / top_k)
                       if time_limit is not None else None)

    all_solutions, prev_samples, seen_subsets = [], [], set()
    total_time = 0.0
    current_prefer = prefer
    for _ in range(top_k):
        t0 = time.time()
        retry_cbc = False
        try:
            sample, selected, _t = _solve_one(
                Q, q, var_map, k_max, per_call_budget, msg, prev_samples,
                prune_threshold, current_prefer,
            )
        except Exception:
            # HiGHS failed mid-solve: fall back to CBC and retry the slot.
            if current_prefer == "highs":
                current_prefer = "cbc"
                retry_cbc = True
            sample, selected = None, None
        total_time += time.time() - t0
        if sample is None:
            if retry_cbc:
                continue
            break
        if selected in seen_subsets:
            # duplicate selection: record the integer cut and move past it
            prev_samples.append(selected)
            continue
        seen_subsets.add(selected)
        all_solutions.append((sample, base.qubo_energy(sample, Q, q)))
        prev_samples.append(selected)
    return all_solutions, total_time, None


def solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, **kwargs):
    """Unified solver interface: return (solutions, wall_time, gap=None)."""
    msg = kwargs.pop("msg", False)
    prune_threshold = kwargs.pop("prune_threshold", 1e-8)
    prefer = kwargs.pop("prefer", "highs")
    base.reject_unknown_kwargs("highs.solve", kwargs)
    solutions, wall, gap = solve_highs(
        Q, q, var_map, k_max, time_limit, top_k, msg, prune_threshold, prefer,
    )
    return base.finalize(solutions, Q, q, top_k), wall, gap
