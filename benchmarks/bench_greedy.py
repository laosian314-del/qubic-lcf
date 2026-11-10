#!/usr/bin/env python
"""Benchmark: greedy forward selection family — pure-NumPy twin vs numba (use_jit).

The hot loops of ``mp_bic`` / ``omp_bic`` (correlation / Gram computation,
per-step atom selection, rank-1 / incremental least-squares updates) are
written twice: the pure-NumPy reference twin ("default / NumPy", the no-JIT
baseline for the SAME algorithm) and the numba ``@njit`` twin ("use_jit").
Both are timed at kernel level and end-to-end through ``mp_bic`` /
``omp_bic`` (which also run the shared NNLS-threshold-BIC stage).  The first
``use_jit=True`` call includes the one-time numba compilation and is not
timed.

Usage:
    python benchmarks/bench_greedy.py

The optional numba/JIT acceleration was NOT used in the article; article results were obtained with the pure NumPy code paths.
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

import numpy as np

from qubic.alternative import (_omp_path_jit, _omp_path_np, _mp_path_jit,
                               _mp_path_np, mp_bic, omp_bic)


def make_case(K, N, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.random((K, N))
    true = rng.choice(N, size=2, replace=False)
    y = A[:, true] @ np.array([0.6, 0.4]) + 1e-3 * rng.standard_normal(K)
    return A, y


def best_of(reps, fn, *args, **kwargs):
    """Min wall time over ``reps`` runs; returns (best_time, last_output)."""
    out = None
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return min(times), out


def bench(K, N, k_max, reps=5):
    print(f"\n== K={K} rows, N={N} atoms, k_max={k_max} "
          f"(min of {reps} runs) ==")
    A, y = make_case(K, N)
    k = min(k_max, N)

    # ---- kernel level: MP path ------------------------------------------
    t_np, (order_np, alpha_np, _n2_np) = best_of(
        reps, _mp_path_np, A, y, k)
    _mp_path_jit(A, y, 2)  # warm-up compile
    t_jit, (order_jit, alpha_jit, _n2_jit) = best_of(
        reps, _mp_path_jit, A, y, k)
    assert np.array_equal(order_np, order_jit), 'MP parity broken'
    np.testing.assert_allclose(alpha_np, alpha_jit)
    print(f"mp  path kernel : NumPy twin {t_np * 1e3:8.3f} ms   "
          f"use_jit {t_jit * 1e3:8.3f} ms   speedup {t_np / t_jit:6.2f}x")

    # ---- kernel level: OMP path ----------------------------------------
    t_np, (order_np, coefs_np, n_np) = best_of(
        reps, _omp_path_np, A, y, k)
    _omp_path_jit(A, y, 2)  # warm-up compile
    t_jit, (order_jit, coefs_jit, n_jit) = best_of(
        reps, _omp_path_jit, A, y, k)
    assert n_np == n_jit, 'OMP parity broken'
    assert np.array_equal(order_np[:n_np], order_jit[:n_jit]), 'OMP parity broken'
    np.testing.assert_allclose(coefs_np, coefs_jit)
    print(f"omp path kernel : NumPy twin {t_np * 1e3:8.3f} ms   "
          f"use_jit {t_jit * 1e3:8.3f} ms   speedup {t_np / t_jit:6.2f}x")

    # ---- end-to-end: mp_bic / omp_bic (NNLS + BIC stage included) -------
    end_reps = max(3, reps // 2)
    for fn in (mp_bic, omp_bic):
        t_np, res_np = best_of(end_reps, fn, A, y, k_max=k_max)
        fn(A, y, k_max=k_max, use_jit=True)  # warm-up
        t_jit, res_jit = best_of(end_reps, fn, A, y, k_max=k_max, use_jit=True)
        assert res_np['selected'] == res_jit['selected'], 'parity broken'
        np.testing.assert_allclose(res_np['concentrations'],
                                   res_jit['concentrations'])
        print(f"{fn.__name__:14s}: NumPy twin {t_np * 1e3:8.3f} ms   "
              f"use_jit {t_jit * 1e3:8.3f} ms   speedup {t_np / t_jit:6.2f}x")


if __name__ == '__main__':
    print('greedy forward selection (MP/OMP) family: default vs use_jit')
    print('(first use_jit=True call compiles the @njit twins and is untimed)')
    for K, N, k_max in ((200, 50, 15), (1000, 100, 20)):
        bench(K, N, k_max)
    print('\nAll parity checks (identical supports, allclose coefficients) OK.')