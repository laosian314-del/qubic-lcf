"""CP-SAT solver for QUBO-LCF using OR-Tools CP-SAT.

Ported from ``QUBO_LCF_3/scripts/cp_sat_solver.py``. Collects diverse
component subsets up to ``k_max`` via CP-SAT solution enumeration with integer
cuts. Uses a linear proxy objective (diagonal of Q + q) to steer the search,
then evaluates the full QUBO energy for each distinct subset found.

Public API (ported):
    solve_cp_sat(Q, q, var_map, k_max, time_limit, top_k) -> (solutions, wall_time)

Unified API:
    solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, **kwargs)
        -> (solutions, wall_time, gap=None)
"""
from __future__ import annotations

import time

import numpy as np

from . import base
from .base import get_selection

TOP_K_CANDIDATES = 100


# ---------------------------------------------------------------------------
# Integer-cut helpers
# ---------------------------------------------------------------------------

def _add_integer_cut(model, z_vars, z_vals):
    """Add constraint that the next solution must differ in *z*.

    sum_{j : z_vals[j]=1} (1-z_j)  +  sum_{j : z_vals[j]=0} z_j  >= 1
    """
    N = len(z_vals)
    cut_terms = []
    for j in range(N):
        if z_vals[j] == 1:
            cut_terms.append(1 - z_vars[j])   # (1 - z_j)
        else:
            cut_terms.append(z_vars[j])       # z_j
    model.Add(sum(cut_terms) >= 1)


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def solve_cp_sat(Q, q, var_map, k_max, time_limit=None, top_k=TOP_K_CANDIDATES):
    """Collect diverse candidate component subsets via CP-SAT.

    Parameters
    ----------
    Q : np.ndarray (M, M)
        QUBO matrix (symmetric).
    q : np.ndarray (M,)
        Linear coefficients.
    var_map : dict
        Encoding specification (keys: 'N', 'scheme', ...).
    k_max : int
        Maximum number of active components,  sum_j z_j <= k_max.
    time_limit : float or None
        Wall-clock time budget in seconds (None -> no limit).
    top_k : int
        Maximum number of distinct solutions to return.

    Returns
    -------
    solutions : list of (sample_array, energy)
        sample_array : np.ndarray (M,) float — 0.0 / 1.0 decisions
        energy       : float — sample @ Q @ sample + q @ sample
    wall_time : float   seconds elapsed
    """
    cp_model = base.require("ortools.sat.python.cp_model", "ortools")
    M = Q.shape[0]
    N = var_map["N"]
    scheme = var_map["scheme"]

    # ----- 1. Build CP-SAT model ------------------------------------------
    model = cp_model.CpModel()

    # Binary variables for *all* QUBO bits
    x = [model.NewBoolVar(f"x{i}") for i in range(M)]

    # Selection variables z[j] in {0,1}
    z = [model.NewBoolVar(f"z{j}") for j in range(N)]

    # --- Encoding-specific linking constraints ---------------------------
    if scheme == "B1":
        for j in range(N):
            model.Add(z[j] == x[j])
    elif scheme == "B2":
        idx_s = var_map["idx_s"]
        for j in range(N):
            model.Add(z[j] == x[idx_s(j)])
        # Hard link: if a component is not selected, its q-bits are 0
        idx_q = var_map["idx_q"]
        for j in range(N):
            si = idx_s(j)
            for b in range(2):
                qi = idx_q(j, b)
                model.Add(x[qi] <= x[si])
    elif scheme.startswith("linked_"):
        idx_s = var_map["idx_s"]
        idx_q = var_map["idx_q"]
        B = var_map["B"]
        for j in range(N):
            model.Add(z[j] == x[idx_s(j)])
        # Hard link: if a component is not selected, its q-bits are 0
        for j in range(N):
            si = idx_s(j)
            for b in range(B):
                model.Add(x[idx_q(j, b)] <= x[si])
    elif scheme in ("B3", "B4"):
        vpc = var_map["vars_per_component"]
        idx_fn = var_map["idx"]
        for j in range(N):
            # z[j] = OR of its bits
            for b in range(vpc):
                model.Add(z[j] >= x[idx_fn(j, b)])
            model.Add(z[j] <= sum(x[idx_fn(j, b)] for b in range(vpc)))
    else:
        raise ValueError(
            f"solve_cp_sat: unsupported scheme {scheme!r}; "
            "expected 'B1', 'B2', 'B3' or 'B4'"
        )

    # --- Cardinality constraint ------------------------------------------
    model.Add(sum(z) <= k_max)

    # --- Linear proxy objective (diagonal + linear terms) ----------------
    # CP-SAT requires integer coefficients.  Scale d = Q[i,i]+q[i] so that
    # the largest |d| maps to ~1e6, preserving relative magnitudes.
    diag_vals = np.array([Q[i, i] + q[i] for i in range(M)], dtype=float)
    max_abs = float(np.max(np.abs(diag_vals))) if M > 0 else 1.0
    if max_abs > 1e-12:
        scale = min(1_000_000.0, (2 ** 62 - 1) / (max_abs + 1.0))
    else:
        scale = 1.0

    obj_terms = []
    for i in range(M):
        coeff = diag_vals[i]
        if abs(coeff) > 1e-12:
            int_coeff = int(round(coeff * scale))
            if int_coeff != 0:
                obj_terms.append(int_coeff * x[i])

    if obj_terms:
        model.Minimize(sum(obj_terms))
    else:
        # No usable linear terms — just enumerate any feasible subsets
        model.Minimize(cp_model.LinearExpr.Sum([]))

    # ----- 2. Iteratively solve and add integer cuts ----------------------
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = False
    solver.parameters.random_seed = 42

    seen_subsets = set()          # tuple-of-int (z-values)
    all_solutions = []            # list of (x_array, energy)
    t_start = time.time()

    for _ in range(top_k):
        elapsed = time.time() - t_start
        if time_limit is not None:
            remaining = time_limit - elapsed
            if remaining <= 0.0:
                break
            solver.parameters.max_time_in_seconds = max(0.1, remaining)

        status = solver.Solve(model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break

        # Read z-pattern
        z_vals = tuple(int(solver.Value(v)) for v in z)
        if z_vals in seen_subsets:
            # Degenerate case — shouldn't happen with cuts, but guard
            _add_integer_cut(model, z, z_vals)
            continue

        seen_subsets.add(z_vals)

        # Build full x-vector and compute true QUBO energy
        x_vals = np.array([solver.Value(v) for v in x], dtype=float)
        energy = float(x_vals @ Q @ x_vals + q @ x_vals)
        all_solutions.append((x_vals, energy))

        # Exclude this z-assignment for the next round
        _add_integer_cut(model, z, z_vals)

    wall_time = time.time() - t_start
    return all_solutions, wall_time


def solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, **kwargs):
    """Unified solver entry point (see :mod:`qubic.solvers.base`)."""
    base.reject_unknown_kwargs("cp_sat.solve", kwargs)
    pairs, wall_time = solve_cp_sat(Q, q, var_map, k_max=k_max,
                                    time_limit=time_limit, top_k=top_k)
    solutions = base.finalize((s for s, _ in pairs), Q, q, top_k)
    return solutions, wall_time, None
