"""Simulated Annealing solver for QUBO-LCF via dimod.

Ported from the inline ``solve_sa`` of ``QUBO_LCF_3/scripts/benchmark_v4_fast.py``.
dimod is an optional dependency and is imported lazily; without it a plain
ImportError with a ``pip install dimod`` hint is raised.

Public API (ported):
    solve_sa(Q, q, var_map, k_max, time_limit, top_k) -> (solutions, wall_time)

Unified API:
    solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, **kwargs)
        -> (solutions, wall_time, gap=None)

Opt-in JIT path (``use_jit=True``):
    A NATIVE Metropolis single-bit-flip annealer with O(1) delta-energy
    updates runs instead of the dimod sampler and does NOT require dimod.
    The native kernel is written once (``_sa_kernel_impl``) and used twice:
    a pure-NumPy reference twin (``_sa_kernel_np``, internal reference) and a
    numba ``@njit`` twin (``_get_sa_kernel``) compiled from the exact same
    source.  Random variates (initial states, flip streams and Exp(1)
    Metropolis noise) are pre-generated and shared between the twins, so the
    trajectories and results are bit-for-bit identical; the noise is passed
    as ``-log(u)`` variates so the acceptance test is a plain comparison and
    no transcendental is evaluated inside the kernels.

The optional numba/JIT acceleration was NOT used in the article; article
results were obtained with the pure NumPy code paths.
"""
from __future__ import annotations

import time

import numpy as np

from . import base
from .base import get_selection
from .._accel import require_numba


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


def _sa_batch_defaults(M):
    """Default sweep/read batching scaled with problem size (ported)."""
    if M <= 20:
        return 2000, 100
    elif M <= 50:
        return 1000, 50
    elif M <= 100:
        return 500, 25
    return 100, 10


# ---------------------------------------------------------------------------
# Native SA kernel — single source shared by the NumPy and numba twins.
# ---------------------------------------------------------------------------

def _sa_kernel_impl(Q, q, x0, flips, exp_noise, betas):
    """Metropolis single-bit-flip SA trajectories with delta-energy updates.

    Shared source: run as-is for the pure-NumPy reference twin
    (``_sa_kernel_np``) and compiled with ``numba.njit`` for the JIT twin
    (``_get_sa_kernel``); both produce bit-identical results because every
    operation is a plain IEEE double add/mul/compare executed in the same
    order and the random variates are supplied by the caller.

    Parameters
    ----------
    Q : (M, M) float64 — QUBO matrix (need not be symmetric; the energy is
        ``x^T Q x + q^T x`` exactly as :func:`base.qubo_energy` computes it).
    q : (M,) float64 — linear coefficients.
    x0 : (n_reads, M) float64 — initial binary states.
    flips : (n_reads, n_moves) int64 — pre-generated flip-bit stream
        (``n_moves == betas.shape[0] * M``: M flip attempts per sweep).
    exp_noise : (n_reads, n_moves) float64 — pre-generated Exp(1) variates
        (``-log(u)``, u ~ U(0,1)); the Metropolis test is
        ``noise > beta * dE`` which is exactly ``u < exp(-beta * dE)``.
    betas : (n_sweeps,) float64 — inverse temperature per sweep.

    Returns
    -------
    states : (n_reads, M) float64 — final state of each trajectory.
    bests : (n_reads, M) float64 — best-so-far state of each trajectory.
    best_Es : (n_reads,) float64 — running energies of ``bests``.
    """
    n_reads = x0.shape[0]
    M = Q.shape[0]
    n_moves = flips.shape[1]
    n_sweeps = betas.shape[0]
    states = np.empty((n_reads, M), dtype=np.float64)
    bests = np.empty((n_reads, M), dtype=np.float64)
    best_Es = np.empty(n_reads, dtype=np.float64)
    for r in range(n_reads):
        x = np.empty(M, dtype=np.float64)
        for i in range(M):
            x[i] = x0[r, i]
        # h[i] = dE/d(coin) of flipping bit i = q[i] + Q[i,i]
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
        best_x = np.empty(M, dtype=np.float64)
        for i in range(M):
            best_x[i] = x[i]
        best_e = E
        for t in range(n_moves):
            b = t // M
            if b >= n_sweeps:
                b = n_sweeps - 1
            f = flips[r, t]
            s = 1.0 - 2.0 * x[f]
            dE = s * h[f]
            # accept downhill always; uphill iff u < exp(-beta*dE), i.e.
            # -log(u) > beta*dE with -log(u) pre-generated as exp_noise
            if dE <= 0.0 or exp_noise[r, t] > betas[b] * dE:
                x[f] = x[f] + s
                E = E + dE
                for j in range(M):
                    if j != f:
                        h[j] = h[j] + (Q[j, f] + Q[f, j]) * s
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
_sa_kernel_np = _sa_kernel_impl

_SA_KERNEL_JIT = None


def _get_sa_kernel(nb):
    """Return the ``@njit`` twin of :func:`_sa_kernel_impl` (compiled once)."""
    global _SA_KERNEL_JIT
    if _SA_KERNEL_JIT is None:
        _SA_KERNEL_JIT = nb.njit(_sa_kernel_impl)
    return _SA_KERNEL_JIT


def _native_sa_loop(Q, q, var_map, time_limit, top_k, num_sweeps,
                    n_reads_per_call, beta_range, seed):
    """Batched native-SA candidate loop (use_jit=True path).

    Mirrors the ported dimod batching semantics: draw batches of reads until
    the time budget is exhausted or ``top_k`` distinct component selections
    have been collected; candidates are deduped by
    :func:`base.get_selection`.  Returns ``(solutions, wall_time)`` where
    ``solutions`` is a list of ``(sample, energy)`` pairs with exact QUBO
    energies (the raw kernel energies only track best-so-far internally).
    """
    nb = require_numba()
    kernel = _get_sa_kernel(nb)
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    q = np.ascontiguousarray(q, dtype=np.float64)
    M = Q.shape[0]

    if beta_range is None:
        beta_start, beta_end = 0.1, 2.0
    else:
        beta_start, beta_end = float(beta_range[0]), float(beta_range[1])
    denom = max(1, num_sweeps - 1)
    betas = beta_start * (beta_end / beta_start) ** (
        np.arange(num_sweeps, dtype=np.float64) / denom)

    rng = np.random.default_rng(seed)
    all_solutions = []
    seen = set()
    total_time = 0.0

    while total_time < time_limit and len(all_solutions) < top_k:
        remaining_frac = max(0.1, min(1.0, (time_limit - total_time) / max(time_limit, 1)))
        n_reads = max(1, int(n_reads_per_call * remaining_frac))
        n_moves = num_sweeps * M
        x0 = (rng.random((n_reads, M)) < 0.5).astype(np.float64)
        flips = rng.integers(0, M, size=(n_reads, n_moves)).astype(np.int64)
        exp_noise = rng.exponential(1.0, size=(n_reads, n_moves))
        t0 = time.time()
        states, bests, _ = kernel(Q, q, x0, flips, exp_noise, betas)
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


def solve_sa(Q, q, var_map, k_max=15, time_limit=None, top_k=20,
             num_sweeps=None, n_reads_per_call=None, beta_range=None,
             use_jit=False, seed=None):
    """Simulated Annealing (ported semantics; dimod default, native JIT opt-in).

    Repeatedly draws batches of reads until the time budget is exhausted or
    ``top_k`` distinct component selections have been collected. Batch sizes
    and sweep counts scale with problem size exactly as in the original
    benchmark implementation.  ``beta_range`` is an optional ``(beta_start,
    beta_end)`` inverse-temperature schedule; on the dimod path it is
    forwarded to dimod (None -> dimod's default schedule), on the native path
    it defines a geometric beta ramp over the sweeps (None -> (0.1, 2.0)).

    Parameters
    ----------
    use_jit : bool, default False
        When True run the native Metropolis kernel (see module docstring)
        instead of dimod; requires the optional ``numba`` dependency but NOT
        dimod.  The default path keeps the dimod behavior unchanged and does
        not import numba.
    seed : int or None
        Seed for the native kernel's RNG (ignored on the dimod path).

    Returns ``(solutions, wall_time)``.
    """
    if use_jit:
        time_limit = base.resolve_time_limit(time_limit)
        if num_sweeps is None or n_reads_per_call is None:
            default_sweeps, default_reads = _sa_batch_defaults(Q.shape[0])
            num_sweeps = default_sweeps if num_sweeps is None else num_sweeps
            n_reads_per_call = default_reads if n_reads_per_call is None else n_reads_per_call
        return _native_sa_loop(Q, q, var_map, time_limit, top_k, num_sweeps,
                               n_reads_per_call, beta_range, seed)

    dimod = base.require("dimod", "dimod")
    time_limit = base.resolve_time_limit(time_limit)
    M = Q.shape[0]

    bqm = _build_bqm(dimod, Q, q)
    sampler = dimod.SimulatedAnnealingSampler()

    if num_sweeps is None or n_reads_per_call is None:
        default_sweeps, default_reads = _sa_batch_defaults(M)
        num_sweeps = default_sweeps if num_sweeps is None else num_sweeps
        n_reads_per_call = default_reads if n_reads_per_call is None else n_reads_per_call

    all_solutions = []
    seen = set()
    total_time = 0.0

    while total_time < time_limit and len(all_solutions) < top_k:
        remaining_frac = max(0.1, min(1.0, (time_limit - total_time) / max(time_limit, 1)))
        n_reads = max(1, int(n_reads_per_call * remaining_frac))

        sample_kwargs = {"num_reads": n_reads, "num_sweeps": num_sweeps}
        if beta_range is not None:
            sample_kwargs["beta_range"] = tuple(beta_range)
        t0 = time.time()
        try:
            response = sampler.sample(bqm, **sample_kwargs)
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

def solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, use_jit=False,
          **kwargs):
    """Unified solver entry point (see :mod:`qubic.solvers.base`).

    Extra kwargs: ``num_sweeps``, ``n_reads_per_call``, ``beta_range``,
    ``seed`` (native path only). The cardinality bound ``k_max`` is accepted
    for interface uniformity (SA is unconstrained; the QUBO penalties encode
    the constraint).  ``use_jit=True`` selects the opt-in native numba kernel
    (see module docstring).
    """
    num_sweeps = kwargs.pop("num_sweeps", None)
    n_reads_per_call = kwargs.pop("n_reads_per_call", None)
    beta_range = kwargs.pop("beta_range", None)
    seed = kwargs.pop("seed", None)
    base.reject_unknown_kwargs("sa.solve", kwargs)
    pairs, wall_time = solve_sa(Q, q, var_map, k_max=k_max, time_limit=time_limit,
                                top_k=top_k, num_sweeps=num_sweeps,
                                n_reads_per_call=n_reads_per_call,
                                beta_range=beta_range, use_jit=use_jit,
                                seed=seed)
    solutions = base.finalize((s for s, _ in pairs), Q, q, top_k)
    return solutions, wall_time, None