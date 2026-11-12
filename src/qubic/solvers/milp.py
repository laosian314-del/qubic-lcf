"""MILP solver for QUBO-LCF using PuLP (CBC with HiGHS fallback).

Ported from ``QUBO_LCF_3/scripts/milp_solver.py``. For 3-bit/4-bit schemes an
optimized formulation with projected concentrations u_j is used instead of the
full McCormick linearization: N(N-1)/2 cross-component z-vars plus 3N
within-component w-vars plus N(N-1)/2 sparsity v-vars. For B1/B2 a standard
McCormick linearization is used. Distinct component subsets are enumerated via
integer cuts on the selection variables.

Public API (ported):
    solve_qubo_milp(Q, q, var_map, k_max, time_limit, ...)  -> (result, wall_time)
    collect_top_k(Q, q, var_map, k_max, time_limit, top_k, ...) -> (solutions, wall_time)

Unified API:
    solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, **kwargs)
        -> (solutions, wall_time, gap=None)

Backend: PuLP + CBC by default; if CBC is unavailable the solve falls back to
HiGHS through PuLP (``pulp.HiGHS_CMD``). Select explicitly with
``solver_backend='cbc'|'highs'|'auto'``.
"""
from __future__ import annotations

import time

import numpy as np

from . import base
from .base import get_selection


def _make_pulp_solver(msg, time_limit, backend="auto"):
    """Create a PuLP solver: CBC first, HiGHS fallback (both via PuLP)."""
    pulp = base.require("pulp", "pulp")
    tl = time_limit if time_limit is not None else 3600
    if backend not in ("auto", "cbc", "highs"):
        raise ValueError(f"solver_backend must be 'auto', 'cbc' or 'highs', got {backend!r}")
    if backend in ("auto", "cbc"):
        solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=tl, threads=1)
        if backend == "cbc" or solver.available():
            return solver
    solver = pulp.HiGHS_CMD(msg=msg, timeLimit=tl)
    if solver.available():
        return solver
    if backend == "highs":
        raise ImportError(
            "solver_backend='highs' requested, but the 'highs' executable "
            "is not available to PuLP (pulp.HiGHS_CMD); install HiGHS or "
            "use solver_backend='auto'/'cbc'"
        )
    raise ImportError(
        "no PuLP solver backend available (CBC and HiGHS both missing); "
        "install it with: pip install pulp"
    )


def _solve_standard(Q, q, var_map, k_max, time_limit, msg, prev_solutions,
                    prune_threshold, solver_backend="auto"):
    """Standard McCormick linearization for B1/B2 (ported unchanged)."""
    pulp = base.require("pulp", "pulp")
    if prev_solutions is None:
        prev_solutions = []
    M = Q.shape[0]
    N = var_map["N"]
    scheme = var_map["scheme"]

    prob = pulp.LpProblem("QUBO_LCF", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(M)]
    z = [pulp.LpVariable(f"z{j}", cat="Binary") for j in range(N)]

    if scheme == "B1":
        for j in range(N):
            prob += z[j] == x[j]
    elif scheme == "B2":
        idx_s = var_map["idx_s"]
        for j in range(N):
            prob += z[j] == x[idx_s(j)]
    elif scheme.startswith("linked_"):
        idx_s = var_map["idx_s"]
        for j in range(N):
            prob += z[j] == x[idx_s(j)]
    else:
        raise ValueError(
            f"solve_qubo_milp: unsupported scheme {scheme!r}; "
            "expected 'B1', 'B2', 'B3', 'B4' or 'linked_B{2,3,4}'"
        )

    prob += pulp.lpSum(z) <= k_max, "card"

    obj = pulp.lpSum([(Q[i, i] + q[i]) * x[i] for i in range(M)])
    for i in range(M):
        for j in range(i + 1, M):
            if abs(Q[i, j]) < prune_threshold:
                continue
            z_ij = pulp.LpVariable(f"z_{i}_{j}", 0, 1, "Continuous")
            prob += z_ij <= x[i]
            prob += z_ij <= x[j]
            prob += z_ij >= x[i] + x[j] - 1
            obj += 2 * Q[i, j] * z_ij
    prob += obj

    for si, (prev_sample, _) in enumerate(prev_solutions):
        zp = get_selection(prev_sample, var_map)
        prob += pulp.lpSum([(1 - z[j]) if zp[j] else z[j] for j in range(N)]) >= 1

    solver = _make_pulp_solver(msg, time_limit, solver_backend)
    t0 = time.time()
    status = prob.solve(solver)
    t = time.time() - t0

    if pulp.LpStatus[status] not in ("Optimal", "Feasible"):
        return None, t

    sample = np.array([x[i].varValue or 0.0 for i in range(M)])
    return (sample, pulp.value(prob.objective)), t


def _solve_antimix_optimized(Q, q, var_map, k_max, time_limit, msg, prev_solutions,
                             solver_backend="auto"):
    """
    Optimized MILP for the 3-bit / 4-bit anti-mixing encodings (ported unchanged).

    Decomposes QUBO into:
    1. Diagonal (linear in binary)
    2. Within-component bilinears (3 per component -> w-vars)
    3. Cross-component bilinears via u_j projection (N(N-1)/2 -> z-vars)
    4. Sparsity bilinears on x_{j,0} (N(N-1)/2 -> v-vars)
    """
    pulp = base.require("pulp", "pulp")
    if prev_solutions is None:
        prev_solutions = []

    N = var_map["N"]
    M = var_map["M"]
    alpha = var_map["alpha"]
    idx_fn = var_map["idx"]
    vpc = var_map["vars_per_component"]

    prob = pulp.LpProblem("QUBO_antimix", pulp.LpMinimize)

    # Binary variables
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(M)]

    # Projected concentrations u_j = sum_b alpha_b * x_{jb}
    u_max = sum(alpha)
    u = [pulp.LpVariable(f"u{j}", 0, u_max, "Continuous") for j in range(N)]
    for j in range(N):
        prob += u[j] == pulp.lpSum([alpha[b] * x[idx_fn(j, b)] for b in range(vpc)])

    # Selection proxy z_j for cardinality
    z = [pulp.LpVariable(f"z{j}", cat="Binary") for j in range(N)]
    for j in range(N):
        for b in range(vpc):
            prob += z[j] >= x[idx_fn(j, b)]
        prob += z[j] <= pulp.lpSum([x[idx_fn(j, b)] for b in range(vpc)])
    prob += pulp.lpSum(z) <= k_max

    # === Build objective ===
    # Start with diagonal + linear terms
    obj = pulp.lpSum([(Q[i, i] + q[i]) * x[i] for i in range(M)])

    # --- Extract parameters from Q ---
    # Cross-component G: Q[idx(j1,0), idx(j2,0)] = alpha[0]^2*G[j1,j2] + lam_sparse (j1!=j2)
    # Within-component: Q[idx(j,0), idx(j,1)] = alpha[0]*alpha[1]*G[j,j] - lam_antimix/2
    G_off = np.zeros((N, N))
    for j1 in range(N):
        for j2 in range(j1 + 1, N):
            G_off[j1, j2] = Q[idx_fn(j1, 0), idx_fn(j2, 0)] / (alpha[0] ** 2)
            G_off[j2, j1] = G_off[j1, j2]

    a01 = alpha[0] * alpha[1]
    a02 = alpha[0] * alpha[2] if vpc >= 3 else 0
    G_diag = np.zeros(N)
    lam_antimix_vals = []
    for j in range(N):
        Q01 = Q[idx_fn(j, 0), idx_fn(j, 1)]
        Q02 = Q[idx_fn(j, 0), idx_fn(j, 2)] if vpc >= 3 else Q01
        if abs(a01 - a02) > 1e-10:
            G_diag[j] = (Q01 - Q02) / (a01 - a02)
        lr = 2 * (a01 * G_diag[j] - Q01)
        lam_antimix_vals.append(lr)
    lam_antimix = np.mean(lam_antimix_vals)

    # Extract lam_sparse from cross-component sparsity
    lam_sparse = Q[idx_fn(0, 0), idx_fn(1, 0)] - alpha[0] ** 2 * G_off[0, 1]

    # --- Cross-component fit bilinears via u_j ---
    for j1 in range(N):
        for j2 in range(j1 + 1, N):
            g = G_off[j1, j2]
            if abs(g) < 1e-10:
                continue
            zvar = pulp.LpVariable(f"z_{j1}_{j2}", 0, u_max ** 2, "Continuous")
            prob += zvar <= u_max * u[j1]
            prob += zvar <= u_max * u[j2]
            prob += zvar >= 0
            prob += zvar >= u_max * (u[j1] + u[j2]) - u_max ** 2
            obj += g * zvar

    # --- Diagonal fit: G[j,j] * u_j^2 ---
    for j in range(N):
        # Linear part (x^2 = x for binary)
        for b in range(vpc):
            obj += G_diag[j] * alpha[b] ** 2 * x[idx_fn(j, b)]
        # Bilinear within-component
        for b1 in range(vpc):
            for b2 in range(b1 + 1, vpc):
                coeff = 2 * G_diag[j] * alpha[b1] * alpha[b2]
                if abs(coeff) < 1e-10:
                    continue
                wvar = pulp.LpVariable(f"w_{j}_{b1}_{b2}", 0, 1, "Binary")
                prob += wvar <= x[idx_fn(j, b1)]
                prob += wvar <= x[idx_fn(j, b2)]
                prob += wvar >= x[idx_fn(j, b1)] + x[idx_fn(j, b2)] - 1
                obj += coeff * wvar

    # --- Anti-mixing penalty bilinears ---
    for j in range(N):
        for b1 in range(vpc):
            for b2 in range(b1 + 1, vpc):
                coeff = 2 * Q[idx_fn(j, b1), idx_fn(j, b2)]
                # Subtract diagonal-fit contribution (already added above)
                coeff -= 2 * G_diag[j] * alpha[b1] * alpha[b2]
                if abs(coeff) < 1e-10:
                    continue
                wvar = pulp.LpVariable(f"wr_{j}_{b1}_{b2}", 0, 1, "Binary")
                prob += wvar <= x[idx_fn(j, b1)]
                prob += wvar <= x[idx_fn(j, b2)]
                prob += wvar >= x[idx_fn(j, b1)] + x[idx_fn(j, b2)] - 1
                obj += coeff * wvar

    # --- Sparsity bilinears on x_{j,0}: 2*lam_sparse*x_{j1,0}*x_{j2,0} ---
    if abs(lam_sparse) > 1e-10:
        for j1 in range(N):
            for j2 in range(j1 + 1, N):
                vvar = pulp.LpVariable(f"v_{j1}_{j2}", 0, 1, "Binary")
                prob += vvar <= x[idx_fn(j1, 0)]
                prob += vvar <= x[idx_fn(j2, 0)]
                prob += vvar >= x[idx_fn(j1, 0)] + x[idx_fn(j2, 0)] - 1
                obj += 2 * lam_sparse * vvar

    prob += obj

    # Integer cuts on z_j
    for si, (prev_sample, _) in enumerate(prev_solutions):
        zp = get_selection(prev_sample, var_map)
        prob += pulp.lpSum([(1 - z[j]) if zp[j] else z[j] for j in range(N)]) >= 1

    solver = _make_pulp_solver(msg, time_limit, solver_backend)
    t0 = time.time()
    status = prob.solve(solver)
    t = time.time() - t0

    if pulp.LpStatus[status] not in ("Optimal", "Feasible"):
        return None, t

    sample = np.array([x[i].varValue or 0.0 for i in range(M)])
    return (sample, pulp.value(prob.objective)), t


def solve_qubo_milp(Q, q, var_map, k_max=15, time_limit=None,
                    msg=False, prev_solutions=None, prune_threshold=1e-8,
                    solver_backend="auto"):
    """Single MILP solve. Dispatches to standard or optimized formulation.

    Returns ``(result, wall_time)`` where ``result`` is ``(sample, objective)``
    or ``None`` if no feasible solution was found.
    """
    scheme = var_map["scheme"]
    if scheme in ("B3", "B4"):
        return _solve_antimix_optimized(Q, q, var_map, k_max, time_limit,
                                        msg, prev_solutions, solver_backend)
    else:
        return _solve_standard(Q, q, var_map, k_max, time_limit,
                               msg, prev_solutions, prune_threshold,
                               solver_backend)


def collect_top_k(Q, q, var_map, k_max=15, time_limit=None, top_k=20,
                  msg=False, prune_threshold=1e-8, solver_backend="auto"):
    """Collect top-K distinct component-subsets via MILP + integer cuts."""
    all_solutions = []
    prev_samples = []
    total_time = 0.0
    seen_subsets = set()

    for _ in range(top_k):
        result, t = solve_qubo_milp(
            Q, q, var_map, k_max=k_max, time_limit=time_limit,
            msg=msg, prev_solutions=prev_samples,
            prune_threshold=prune_threshold, solver_backend=solver_backend,
        )
        total_time += t
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

    return all_solutions, total_time


def solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, **kwargs):
    """Unified solver entry point (see :mod:`qubic.solvers.base`).

    Extra kwargs: ``msg`` (PuLP solver log), ``prune_threshold``,
    ``solver_backend`` ('auto' | 'cbc' | 'highs').
    """
    msg = kwargs.pop("msg", False)
    prune_threshold = kwargs.pop("prune_threshold", 1e-8)
    solver_backend = kwargs.pop("solver_backend", "auto")
    base.reject_unknown_kwargs("milp.solve", kwargs)

    pairs, wall_time = collect_top_k(
        Q, q, var_map, k_max=k_max, time_limit=time_limit, top_k=top_k,
        msg=msg, prune_threshold=prune_threshold, solver_backend=solver_backend,
    )
    solutions = base.finalize((s for s, _ in pairs), Q, q, top_k)
    return solutions, wall_time, None
