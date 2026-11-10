"""Simulated Annealing solver for QUBO-LCF via dimod.

Ported from the inline ``solve_sa`` of ``QUBO_LCF_3/scripts/benchmark_v4_fast.py``.
dimod is an optional dependency and is imported lazily; without it a plain
ImportError with a ``pip install dimod`` hint is raised.

Public API (ported):
    solve_sa(Q, q, var_map, k_max, time_limit, top_k) -> (solutions, wall_time)

Unified API:
    solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, **kwargs)
        -> (solutions, wall_time, gap=None)
"""
from __future__ import annotations

import time

import numpy as np

from . import base
from .base import get_selection


def _build_bqm(dimod, Q, q):
    """Build the dimod BQM equivalent to x^T Q x + q^T x (Q symmetric)."""
    M = Q.shape[0]
    linear = {}
    quadratic = {}
    for i in range(M):
        linear[i] = Q[i, i] + q[i]
    for i in range(M):
        for j in range(i + 1, M):
            if abs(Q[i, j]) > 1e-10:
                quadratic[(i, j)] = 2 * Q[i, j]
    return dimod.BinaryQuadraticModel(linear, quadratic, 0.0, "BINARY")


def solve_sa(Q, q, var_map, k_max=15, time_limit=None, top_k=20,
             num_sweeps=None, n_reads_per_call=None):
    """Simulated Annealing via dimod (ported semantics).

    Repeatedly draws batches of reads until the time budget is exhausted or
    ``top_k`` distinct component selections have been collected. Batch sizes
    and sweep counts scale with problem size exactly as in the original
    benchmark implementation.

    Returns ``(solutions, wall_time)``.
    """
    dimod = base.require("dimod", "dimod")
    time_limit = base.resolve_time_limit(time_limit)
    M = Q.shape[0]

    bqm = _build_bqm(dimod, Q, q)
    sampler = dimod.SimulatedAnnealingSampler()

    if num_sweeps is None or n_reads_per_call is None:
        if M <= 20:
            default_sweeps, default_reads = 2000, 100
        elif M <= 50:
            default_sweeps, default_reads = 1000, 50
        elif M <= 100:
            default_sweeps, default_reads = 500, 25
        else:
            default_sweeps, default_reads = 100, 10
        num_sweeps = default_sweeps if num_sweeps is None else num_sweeps
        n_reads_per_call = default_reads if n_reads_per_call is None else n_reads_per_call

    all_solutions = []
    seen = set()
    total_time = 0.0

    while total_time < time_limit and len(all_solutions) < top_k:
        remaining_frac = max(0.1, min(1.0, (time_limit - total_time) / max(time_limit, 1)))
        n_reads = max(1, int(n_reads_per_call * remaining_frac))

        t0 = time.time()
        try:
            response = sampler.sample(bqm, num_reads=n_reads, num_sweeps=num_sweeps)
        except Exception:
            break
        elapsed = time.time() - t0
        total_time += elapsed

        for sample_dict, energy in response.data(["sample", "energy"]):
            sample = np.array([sample_dict.get(i, 0) for i in range(M)], dtype=float)
            sel = tuple(get_selection(sample, var_map))
            if sel not in seen:
                seen.add(sel)
                all_solutions.append((sample, float(energy)))
                if len(all_solutions) >= top_k:
                    break

    return all_solutions, total_time


def solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, **kwargs):
    """Unified solver entry point (see :mod:`qubic.solvers.base`).

    Extra kwargs: ``num_sweeps``, ``n_reads_per_call``. The cardinality bound
    ``k_max`` is accepted for interface uniformity (SA is unconstrained; the
    QUBO penalties encode the constraint).
    """
    num_sweeps = kwargs.pop("num_sweeps", None)
    n_reads_per_call = kwargs.pop("n_reads_per_call", None)
    base.reject_unknown_kwargs("sa.solve", kwargs)
    pairs, wall_time = solve_sa(Q, q, var_map, k_max=k_max, time_limit=time_limit,
                                top_k=top_k, num_sweeps=num_sweeps,
                                n_reads_per_call=n_reads_per_call)
    solutions = base.finalize((s for s, _ in pairs), Q, q, top_k)
    return solutions, wall_time, None
