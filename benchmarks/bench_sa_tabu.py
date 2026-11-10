#!/usr/bin/env python
"""Benchmark: native SA / tabu kernels — pure-NumPy twin vs numba (use_jit).

The native kernels are written once and run twice: as the pure-NumPy
reference twin ("default / NumPy" below — the no-JIT baseline for the SAME
algorithm) and as the numba ``@njit`` twin ("use_jit").  Both consume the
same pre-generated random variates, so the workloads are identical.  The
dimod / dwave-tabu default solver paths are timed too when those optional
packages are installed (they are different C++ samplers, shown for context).

Usage:
    python benchmarks/bench_sa_tabu.py [M]

The optional numba/JIT acceleration was NOT used in the article; article
results were obtained with the pure NumPy code paths.
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from qubic import _accel
from qubic.solvers import sa, tabu


def make_qubo(M, seed=1234):
    rng = np.random.RandomState(seed)
    U = rng.uniform(-1.5, 2.0, size=(M, M))
    Q = np.ascontiguousarray(np.triu(U) + np.triu(U, 1).T, dtype=np.float64)
    q = np.ascontiguousarray(rng.uniform(-1.0, 1.0, size=M), dtype=np.float64)
    var_map = {"scheme": "B1", "N": M, "M": M,
               "vars_per_component": 1, "selection": "explicit"}
    return Q, q, var_map


def timeit(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return time.perf_counter() - t0, out


def bench_kernels(M):
    print(f"\n== kernel-level benchmark (identical pre-generated variates), M={M} ==")
    Q, q, var_map = make_qubo(M)

    # ---- SA kernel -------------------------------------------------------
    n_sweeps, n_reads = 100, 6
    n_moves = n_sweeps * M
    rng = np.random.default_rng(0)
    x0 = (rng.random((n_reads, M)) < 0.5).astype(np.float64)
    flips = rng.integers(0, M, size=(n_reads, n_moves)).astype(np.int64)
    exp_noise = rng.exponential(1.0, size=(n_reads, n_moves))
    betas = 0.1 * (2.0 / 0.1) ** (np.arange(n_sweeps, dtype=np.float64)
                                 / max(1, n_sweeps - 1))

    t_np, ref = timeit(sa._sa_kernel_np, Q, q, x0, flips, exp_noise, betas)
    jit_sa = sa._get_sa_kernel(_accel.require_numba())
    jit_sa(Q, q, x0[:1], flips[:1], exp_noise[:1], betas)  # warm-up compile
    t_jit, got = timeit(jit_sa, Q, q, x0, flips, exp_noise, betas)
    assert all(np.array_equal(a, b) for a, b in zip(ref, got)), "SA parity broken"
    print(f"SA   kernel  (n_reads={n_reads}, n_sweeps={n_sweeps}): "
          f"NumPy twin {t_np:.3f}s   use_jit {t_jit:.4f}s   "
          f"speedup {t_np / t_jit:.1f}x")

    # ---- tabu kernel -----------------------------------------------------
    n_steps, tenure_base, span = 500, max(2, M // 10), max(2, M // 10)
    rng = np.random.default_rng(1)
    x0 = (rng.random((n_reads, M)) < 0.5).astype(np.float64)
    tenure_jitter = rng.integers(0, span, size=(n_reads, n_steps)).astype(np.int64)

    t_np, ref = timeit(tabu._tabu_kernel_np, Q, q, x0, tenure_jitter,
                       n_steps, tenure_base)
    jit_tabu = tabu._get_tabu_kernel(_accel.require_numba())
    jit_tabu(Q, q, x0[:1], tenure_jitter[:1], n_steps, tenure_base)  # warm-up
    t_jit, got = timeit(jit_tabu, Q, q, x0, tenure_jitter, n_steps, tenure_base)
    assert all(np.array_equal(a, b) for a, b in zip(ref, got)), "tabu parity broken"
    print(f"tabu kernel  (n_reads={n_reads}, n_steps={n_steps}):    "
          f"NumPy twin {t_np:.3f}s   use_jit {t_jit:.4f}s   "
          f"speedup {t_np / t_jit:.1f}x")
    return t_np


def bench_public_api(M):
    print(f"\n== public API benchmark (time_limit=2.0s, top_k=10), M={M} ==")
    Q, q, var_map = make_qubo(M)
    for name, mod in (("sa", sa), ("tabu", tabu)):
        t0 = time.perf_counter()
        sols, wall, _gap = mod.solve(Q, q, var_map, top_k=10, time_limit=2.0,
                                     use_jit=True, seed=0)
        el = time.perf_counter() - t0
        best = min(e for _, e in sols)
        print(f"{name:4s} use_jit=True : {len(sols)} solutions, "
              f"solver wall_time={wall:.3f}s, end-to-end={el:.3f}s, "
              f"best energy={best:.4f}")
        try:
            t0 = time.perf_counter()
            sols, wall, _gap = mod.solve(Q, q, var_map, top_k=10, time_limit=2.0)
            el = time.perf_counter() - t0
            best = min(e for _, e in sols)
            print(f"{name:4s} default     : {len(sols)} solutions, "
                  f"solver wall_time={wall:.3f}s, end-to-end={el:.3f}s, "
                  f"best energy={best:.4f}")
        except ImportError as exc:
            print(f"{name:4s} default     : skipped ({exc})")


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(f"numba available: {_accel.numba_available()}")
    bench_kernels(M)
    bench_public_api(M)


if __name__ == "__main__":
    main()