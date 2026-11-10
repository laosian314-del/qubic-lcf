"""Tabu search solver for QUBO-LCF via dwave-samplers (TabuSampler).

Ported from the inline ``solve_tabu`` of ``QUBO_LCF_3/scripts/benchmark_v4_fast.py``.
``dwave.samplers`` and ``dimod`` are optional dependencies and are imported
lazily; without them a plain ImportError with a ``pip install dwave-samplers```
hint is raised.

Public API (ported):
    solve_tabu(Q, q, var_map, k_max, time_limit, top_k) -> (solutions, wall_time)

Unified API:
    solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, **kwargs)
        -> (solutions, wall_time, gap=None)

Opt-in JIT path (``use_jit=True``):
    A NATIVE tabu search kernel runs instead of the dwave TabuSampler and
    does NOT require dimod / dwave-samplers.  The native kernel is written
    once (``_tabu_kernel_impl``) and used twice: a pure-NumPy reference twin
    (``_tabu_kernel_np``, internal reference) and a numba ``@njit`` twin
    (``_get_tabu_kernel``) compiled from the exact same source.  Random
    variates (initial states and per-move tabu-tenure jitter) are
    pre-generated and shared between the twins, so the trajectories and
    results are bit-for-bit identical.  The search keeps O(1) delta-energy
    moves, best-improvement move selection (ties -> lowest bit index), a
    per-move randomized tenure and best-so-far aspiration.

The optional numba/JIT acceleration was NOT used in the article; article
results were obtained with the pure NumPy code paths.
"""
from __future__ import annotations

import time

import numpy as np

from . import base
from .base import get_selection
from .._accel import require_numba

#: Per-call Tabu timeout in milliseconds (ported constant).
PER_CALL_TIMEOUT_MS = 5000


# ---------------------------------------------------------------------------
# Native tabu kernel — single source shared by the NumPy and numba twins.
# ---------------------------------------------------------------------------

def _tabu_kernel_impl(Q, q, x0, tenure_jitter, n_steps, tenure_base):
    """Native tabu-search trajectories with O(1) delta-energy updates.

    Shared source: run as-is for the pure-NumPy reference twin
    (``_tabu_kernel_np``) and compiled with ``numba.njit`` for the JIT twin
    (``_get_tabu_kernel``); both produce bit-identical results because every
    operation is a plain IEEE double add/mul/compare executed in the same
    order (move ties are broken by lowest bit index) and the random variates
    are supplied by the caller.

    Parameters
    ----------
    Q : (M, M) float64 — QUBO matrix (need not be symmetric; the energy is
        ``x^T Q x + q^T x`` exactly as :func:`base.qubo_energy` computes it).
    q : (M,) float64 — linear coefficients.
    x0 : (n_reads, M) float64 — initial binary states.
    tenure_jitter : (n_reads, n_steps) int64 — pre-generated tenure jitter in
        ``[0, span)``; bit flipped at step t is tabu for
        ``tenure_base + tenure_jitter[r, t]`` steps.
    n_steps : int — tabu steps per trajectory.
    tenure_base : int — base tabu tenure (>= 1).

    Returns
    -------
    states : (n_reads, M) float64 — final state of each trajectory.
    bests : (n_reads, M) float64 — best-so-far state of each trajectory.
    best_Es : (n_reads,) float64 — running energies of ``bests``.
    """
    n_reads = x0.shape[0]
    M = Q.shape[0]
    states = np.empty((n_reads, M), dtype=np.float64)
    bests = np.empty((n_reads, M), dtype=np.float64)
    best_Es = np.empty(n_reads, dtype=np.float64)
    for r in range(n_reads):
        x = np.empty(M, dtype=np.float64)
        for i in range(M):
            x[i] = x0[r, i]
        # h[i] = energy delta of flipping bit i = q[i] + Q[i,i]
        #        + sum_{j!=i} (Q[i,j] + Q[j,i]) * x[j]
        h = np.empty(M, dtype=np.float64)
        for i in range(M):
            acc = q[i] + Q[i, i]
            for j in range(M):
                if j != i:
                    acc = acc + (Q[i, j] + Q[j, i]) * x[j]
            h[i] = acc
        E = 0.0
        for i in range(M):
            for j in range(M):
                E = E + x[i] * Q[i, j] * x[j]
            E = E + x[i] * q[i]
        tabu_until = np.zeros(M, dtype=np.int64)
        best_x = np.empty(M, dtype=np.float64)
        for i in range(M):
            best_x[i] = x[i]
        best_e = E
        for t in range(n_steps):
            step = t + 1
            move = -1
            move_dE = 0.0
            # best-improvement move among non-tabu bits (aspiration: a tabu
            # move is allowed when it strictly improves the best-so-far energy)
            for i in range(M):
                s = 1.0 - 2.0 * x[i]
                dE = s * h[i]
                if tabu_until[i] <= step or E + dE < best_e:
                    if move < 0 or dE < move_dE:
                        move = i
                        move_dE = dE
            if move < 0:
                break
            f = move
            s = 1.0 - 2.0 * x[f]
            x[f] = x[f] + s
            E = E + move_dE
            for j in range(M):
                if j != f:
                    h[j] = h[j] + (Q[j, f] + Q[f, j]) * s
            tabu_until[f] = step + tenure_base + tenure_jitter[r, t]
            if E < best_e:
                best_e = E
                for i in range(M):
                    best_x[i] = x[i]
        for i in range(M):
            states[r, i] = x[i]
            bests[r, i] = best_x[i]
        best_Es[r] = best_e
    return states, bests, best_Es


#: Pure-NumPy reference twin of the native kernel (internal reference).
_tabu_kernel_np = _tabu_kernel_impl

_TABU_KERNEL_JIT = None


def _get_tabu_kernel(nb):
    """Return the ``@njit`` twin of :func:`_tabu_kernel_impl` (compiled once)."""
    global _TABU_KERNEL_JIT
    if _TABU_KERNEL_JIT is None:
        _TABU_KERNEL_JIT = nb.njit(_tabu_kernel_impl)
    return _TABU_KERNEL_JIT


def _native_tabu_loop(Q, q, var_map, time_limit, top_k, num_steps, tenure_base,
                      tenure_span, seed):
    """Batched native-tabu candidate loop (use_jit=True path).

    Mirrors the ported batching semantics: draw batches of restarts until the
    time budget is exhausted or ``top_k`` distinct component selections have
    been collected; candidates are deduped by :func:`base.get_selection`.
    Returns ``(solutions, wall_time)`` where ``solutions`` is a list of
    ``(sample, energy)`` pairs with exact QUBO energies.
    """
    nb = require_numba()
    kernel = _get_tabu_kernel(nb)
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    q = np.ascontiguousarray(q, dtype=np.float64)
    M = Q.shape[0]

    rng = np.random.default_rng(seed)
    all_solutions = []
    seen = set()
    total_time = 0.0

    while total_time < time_limit and len(all_solutions) < top_k:
        remaining_frac = max(0.1, min(1.0, (time_limit - total_time) / max(time_limit, 1)))
        n_reads = max(1, min(20, int(20 * remaining_frac)))
        x0 = (rng.random((n_reads, M)) < 0.5).astype(np.float64)
        tenure_jitter = rng.integers(0, tenure_span,
                                     size=(n_reads, num_steps)).astype(np.int64)
        t0 = time.time()
        states, bests, _ = kernel(Q, q, x0, tenure_jitter, num_steps, tenure_base)
        elapsed = time.time() - t0
        total_time += elapsed

        for arr in (bests, states):
            for r in range(n_reads):
                sample = np.array(arr[r], dtype=float)
                sel = tuple(get_selection(sample, var_map))
                if sel not in seen:
                    seen.add(sel)
                    all_solutions.append((sample, base.qubo_energy(sample, Q, q)))
                    if len(all_solutions) >= top_k:
                        break
            if len(all_solutions) >= top_k:
                break

    return all_solutions, total_time


def solve_tabu(Q, q, var_map, k_max=15, time_limit=None, top_k=20,
               per_call_timeout_ms=PER_CALL_TIMEOUT_MS, use_jit=False,
               seed=None, num_steps=None, tenure_base=None, tenure_span=None):
    """Tabu search (ported semantics; dwave-tabu default, native JIT opt-in).

    Repeatedly draws batches of reads until the time budget is exhausted or
    ``top_k`` distinct component selections have been collected.

    Parameters
    ----------
    use_jit : bool, default False
        When True run the native tabu kernel (see module docstring) instead
        of the dwave TabuSampler; requires the optional ``numba`` dependency
        but NOT dimod / dwave-samplers.  The default path keeps the
        dwave-tabu behavior unchanged and does not import numba.
    seed : int or None
        Seed for the native kernel's RNG (ignored on the dwave path).
    num_steps : int or None
        Native path: tabu steps per restart (default ``max(1000, 100 * M)``).
    tenure_base, tenure_span : int or None
        Native path: tabu tenure of a flipped bit is
        ``tenure_base + jitter`` with jitter uniform in ``[0, tenure_span)``;
        both default to ``max(2, M // 10)``.

    Returns ``(solutions, wall_time)``.
    """
    if use_jit:
        time_limit = base.resolve_time_limit(time_limit)
        M = Q.shape[0]
        if num_steps is None:
            num_steps = max(1000, 100 * M)
        if tenure_base is None:
            tenure_base = max(2, M // 10)
        if tenure_span is None:
            tenure_span = max(2, M // 10)
        return _native_tabu_loop(Q, q, var_map, time_limit, top_k,
                                 int(num_steps), int(tenure_base),
                                 int(tenure_span), seed)

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

def solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, use_jit=False,
          **kwargs):
    """Unified solver entry point (see :mod:`qubic.solvers.base`).

    Extra kwargs: ``per_call_timeout_ms`` (dwave path), ``seed``,
    ``num_steps``, ``tenure_base``, ``tenure_span`` (native path). The
    cardinality bound ``k_max`` is accepted for interface uniformity (Tabu is
    unconstrained; the QUBO penalties encode the constraint).
    ``use_jit=True`` selects the opt-in native numba kernel (see module
    docstring).
    """
    per_call_timeout_ms = kwargs.pop("per_call_timeout_ms", PER_CALL_TIMEOUT_MS)
    seed = kwargs.pop("seed", None)
    num_steps = kwargs.pop("num_steps", None)
    tenure_base = kwargs.pop("tenure_base", None)
    tenure_span = kwargs.pop("tenure_span", None)
    base.reject_unknown_kwargs("tabu.solve", kwargs)
    pairs, wall_time = solve_tabu(Q, q, var_map, k_max=k_max, time_limit=time_limit,
                                  top_k=top_k,
                                  per_call_timeout_ms=per_call_timeout_ms,
                                  use_jit=use_jit, seed=seed,
                                  num_steps=num_steps, tenure_base=tenure_base,
                                  tenure_span=tenure_span)
    solutions = base.finalize((s for s, _ in pairs), Q, q, top_k)
    return solutions, wall_time, None