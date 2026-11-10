"""Tabu search solver for QUBO-LCF via dwave-samplers (TabuSampler).

Ported from the inline ``solve_tabu`` of ``QUBO_LCF_3/scripts/benchmark_v4_fast.py``.
``dwave.samplers`` and ``dimod`` are optional dependencies and are imported
lazily; without them a plain ImportError with a ``pip install dwave-samplers``
hint is raised.

Public API (ported):
    solve_tabu(Q, q, var_map, k_max, time_limit, top_k) -> (solutions, wall_time)

Unified API:
    solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, **kwargs)
        -> (solutions, wall_time, gap=None)
"""
from __future__ import annotations

import time

import numpy as np

from . import base
from .base import get_selection

#: Per-call Tabu timeout in milliseconds (ported constant).
PER_CALL_TIMEOUT_MS = 5000


def solve_tabu(Q, q, var_map, k_max=15, time_limit=None, top_k=20,
               per_call_timeout_ms=PER_CALL_TIMEOUT_MS):
    """Tabu search via dwave-tabu (ported semantics).

    Repeatedly draws batches of reads until the time budget is exhausted or
    ``top_k`` distinct component selections have been collected.

    Returns ``(solutions, wall_time)``.
    """
    samplers = base.require("dwave.samplers", "dwave-samplers")
    dimod = base.require("dimod", "dimod")
    time_limit = base.resolve_time_limit(time_limit)
    M = Q.shape[0]

    linear = {i: Q[i, i] + q[i] for i in range(M)}
    quadratic = {}
    for i in range(M):
        for j in range(i + 1, M):
            if abs(Q[i, j]) > 1e-10:
                quadratic[(i, j)] = 2 * Q[i, j]

    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, "BINARY")
    sampler = samplers.TabuSampler()

    all_solutions = []
    seen = set()
    total_time = 0.0

    while total_time < time_limit and len(all_solutions) < top_k:
        remaining_frac = max(0.1, min(1.0, (time_limit - total_time) / max(time_limit, 1)))
        n_reads = max(1, min(20, int(20 * remaining_frac)))
        t0 = time.time()
        try:
            response = sampler.sample(bqm, num_reads=n_reads,
                                      timeout=per_call_timeout_ms)
        except Exception:
            break
        elapsed = time.time() - t0
        total_time += elapsed

        try:
            for datum in response.data(fields=["sample", "energy"]):
                sample_dict = datum.sample
                energy = datum.energy
                sample = np.array([sample_dict.get(i, 0) for i in range(M)], dtype=float)
                sel = tuple(get_selection(sample, var_map))
                if sel not in seen:
                    seen.add(sel)
                    all_solutions.append((sample, float(energy)))
                    if len(all_solutions) >= top_k:
                        break
        except Exception:
            pass
        if len(all_solutions) >= top_k:
            break

    return all_solutions, total_time


def solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, **kwargs):
    """Unified solver entry point (see :mod:`qubic.solvers.base`).

    Extra kwargs: ``per_call_timeout_ms``. The cardinality bound ``k_max`` is
    accepted for interface uniformity (Tabu is unconstrained; the QUBO
    penalties encode the constraint).
    """
    per_call_timeout_ms = kwargs.pop("per_call_timeout_ms", PER_CALL_TIMEOUT_MS)
    base.reject_unknown_kwargs("tabu.solve", kwargs)
    pairs, wall_time = solve_tabu(Q, q, var_map, k_max=k_max, time_limit=time_limit,
                                  top_k=top_k, per_call_timeout_ms=per_call_timeout_ms)
    solutions = base.finalize((s for s, _ in pairs), Q, q, top_k)
    return solutions, wall_time, None
