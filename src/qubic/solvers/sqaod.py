"""Simulated quantum annealing solver for QUBO-LCF via the optional sqaod package.

Ported from ``QUBO_LCF_3/scripts/benchmark_v3.py`` (``solve_sqaod``): the
QUBO is turned into a fully-connected Ising model and annealed with
``sqaod.cpu.dense_graph_annealer`` (path-integral Monte Carlo style,
``n_trotters`` replicas, one ``anneal_one_step(G, beta)`` call per step).
sqaod is an optional dependency and is imported lazily; without it a plain
ImportError with a ``pip install sqaod`` hint is raised.  sqaod's CPU
backend is too slow for large QUBO sizes, so above ``max_dense_vars``
(20 by default) the reference implementation's documented seeded random
sampling fallback is used instead of the annealer.

Public API (ported):
    solve_sqaod(Q, q, var_map, k_max, time_limit, top_k, ...) -> (solutions, wall_time)

Unified API:
    solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, **kwargs)
        -> (solutions, wall_time, gap=None)

Extra kwargs: ``n_trotters``, ``G``, ``beta``, ``max_dense_vars``, ``seed``.
"""
from __future__ import annotations

import time

import numpy as np

from . import base
from .base import get_selection


def _anneal(Q_full, var_map, time_limit, top_k, n_trotters, G, beta):
    """Anneal with sqaod and return a list of (sample, energy) pairs."""
    sq = base.require("sqaod", "sqaod")
    M = Q_full.shape[0]

    ann = sq.cpu.dense_graph_annealer(
        Q_full, n_trotters=(n_trotters or min(M, 8)))
    ann.prepare()
    ann.randomize_spin()

    solutions, seen = [], set()
    start = time.time()
    while time.time() - start < time_limit and len(solutions) < top_k:
        try:
            ann.anneal_one_step(G, beta)
        except Exception:
            break
        try:
            x_list = ann.get_x()
        except Exception:
            continue
        for xi in x_list:
            sample = np.asarray(xi, dtype=float).ravel()
            if sample.shape[0] < M:
                sample = np.pad(sample, (0, M - sample.shape[0]))
            sel = tuple(get_selection(sample, var_map))
            if sel not in seen:
                seen.add(sel)
                solutions.append((sample, None))
            if len(solutions) >= top_k:
                break
    return solutions, time.time() - start


def _random_sample(Q, q, var_map, time_limit, top_k, seed):
    """Seeded random-sampling fallback for large problems."""
    rng = np.random.RandomState(seed)
    M = Q.shape[0]
    solutions, seen = [], set()
    start = time.time()
    while time.time() - start < time_limit and len(solutions) < top_k:
        sample = rng.randint(0, 2, M).astype(float)
        sel = tuple(get_selection(sample, var_map))
        if sel not in seen:
            seen.add(sel)
            solutions.append((sample, None))
    return solutions, time.time() - start


def solve_sqaod(Q, q, var_map, k_max=15, time_limit=None, top_k=20,
                n_trotters=None, G=1.0, beta=1.0, max_dense_vars=20, seed=42):
    """Enumerate candidate selections with sqaod annealing.

    Only the QUBO bit patterns are decided here; the caller decodes and
    refines them.  ``k_max`` is accepted for interface symmetry (the
    reference implementation enforces the cardinality bound in decoding).
    """
    base.require("sqaod", "sqaod")
    base.validate_qubo_inputs(Q, q, var_map)
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if time_limit is not None and time_limit <= 0:
        raise ValueError(f"time_limit must be positive, got {time_limit}")

    time_limit = base.resolve_time_limit(time_limit)
    M = Q.shape[0]

    # Full QUBO matrix with the linear term absorbed into the diagonal.
    Q_full = np.asarray(Q, dtype=float).copy()
    q_arr = np.asarray(q, dtype=float).ravel()
    for i in range(M):
        Q_full[i, i] += q_arr[i]

    if M <= max_dense_vars:
        solutions, wall = _anneal(
            Q_full, var_map, time_limit, top_k, n_trotters, G, beta)
    else:
        solutions, wall = [], 0.0
    # Documented fallback: seeded random sampling when the annealer yields
    # nothing (or the problem is too big for sqaod's CPU dense backend).
    if not solutions:
        extra_solutions, extra = _random_sample(
            Q, q, var_map, time_limit, top_k, seed)
        solutions = solutions + extra_solutions
        wall += extra

    # Fill in the energies now that the run is over.
    solutions = [(s, base.qubo_energy(s, Q, q_arr)) for s, _ in solutions]
    return solutions, wall


def solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, **kwargs):
    """Unified solver interface: return (solutions, wall_time, gap=None)."""
    n_trotters = kwargs.pop("n_trotters", None)
    G = kwargs.pop("G", 1.0)
    beta = kwargs.pop("beta", 1.0)
    max_dense_vars = kwargs.pop("max_dense_vars", 20)
    seed = kwargs.pop("seed", 42)
    base.reject_unknown_kwargs("sqaod.solve", kwargs)
    solutions, wall = solve_sqaod(
        Q, q, var_map, k_max, time_limit, top_k, n_trotters, G, beta,
        max_dense_vars, seed,
    )
    return base.finalize(solutions, wall, None, top_k)
