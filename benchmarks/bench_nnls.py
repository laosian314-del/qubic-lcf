"""Wall-time benchmark: scipy NNLS refinement vs opt-in batched numba NNLS.

Refines 100 candidate supports of an LCF fit (m=200 points, N=50 library
columns, k=3 components per support) through the full refinement stage
(NNLS refit with the ridge row -> sum-1 normalization -> tau cutoff -> BIC)
on three backends:

* ``default``  — ``refine_candidates(..., use_jit=False)``: one
  ``scipy.optimize.nnls`` call per candidate (the original loop);
* ``pg-numpy`` — the pure-NumPy projected-gradient twin
  (``nnls_pg_batch(..., use_jit=False)``), including building the augmented
  systems;
* ``pg-jit``   — ``refine_candidates(..., use_jit=True)``: one batched numba
  ``@njit`` projected-gradient call for all candidates (first call includes
  numba compilation and is reported separately).

Usage::

    /opt/anaconda/bin/python benchmarks/bench_nnls.py [--repeats R] [--seed S]

The optional numba/JIT acceleration was NOT used in the article; article results were obtained with the pure NumPy code paths.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

from qubic.pipeline import RIDGE, nnls_pg_batch, refine_candidates  # noqa: E402


def build_problem(seed=0, m=200, N=50, k=3, n_cand=100, noise=0.01):
    rng = np.random.default_rng(seed)
    A = rng.random((m, N))  # [0, 1) library, like measured spectra columns
    true_sel = [4, 17, 33]
    target = A[:, true_sel] @ np.array([0.55, 0.30, 0.15])
    target = target + noise * rng.standard_normal(m)
    cands, seen = [], set()
    while len(cands) < n_cand:
        sel = tuple(sorted(int(i) for i in
                           rng.choice(N, size=k, replace=False)))
        if sel not in seen:
            seen.add(sel)
            cands.append(list(sel))
    return A, target, cands


def time_best(fn, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tau', type=float, default=0.01)
    args = ap.parse_args()

    A, target, cands = build_problem(seed=args.seed)
    m, N, k, B = A.shape[0], A.shape[1], len(cands[0]), len(cands)
    conds = [np.linalg.cond(np.vstack([A[:, s], np.full(k, RIDGE)]))
             for s in cands]
    print(f'problem: m={m} points, N={N} library, k={k}, '
          f'{B} candidate supports, tau={args.tau}, seed={args.seed}')
    print(f'support conditioning: cond(M) min={min(conds):.1f} '
          f'median={np.median(conds):.1f} max={max(conds):.1f}')

    def default():
        return refine_candidates(A, target, cands, tau=args.tau)

    def pg_numpy():
        mats = [np.vstack([A[:, s], np.full(k, RIDGE)]) for s in cands]
        rhs = [np.hstack([target, [RIDGE]]) for _ in cands]
        return nnls_pg_batch(mats, rhs, use_jit=False)

    def pg_jit():
        return refine_candidates(A, target, cands, tau=args.tau, use_jit=True)

    t_default = time_best(default, args.repeats)
    t_numpy = time_best(pg_numpy, args.repeats)
    t_jit_cold = time_best(pg_jit, 1)  # includes numba compilation
    t_jit = time_best(pg_jit, args.repeats)

    out_def = default()
    out_jit = pg_jit()
    pairs = []
    for a, b in zip(out_def, out_jit):
        assert a is not None and b is not None
        pairs.append((a, b))
    same_supports = sum(1 for a, b in pairs if a[0] == b[0])
    rss_rel = max(abs(a[2] - b[2]) / max(abs(a[2]), 1e-300)
                  for a, b in pairs)
    bic_rel = max(abs(a[3] - b[3]) / max(abs(a[3]), 1e-300)
                  for a, b in pairs)
    conc_max = max(float(np.max(np.abs(a[1] - b[1])))
                   if a[1].shape == b[1].shape else float('inf')
                   for a, b in pairs)

    print()
    print(f"{'path':<12} {'wall time (s)':>14} {'speedup vs default':>20}")
    print(f"{'-' * 12} {'-' * 14:>14} {'-' * 20:>20}")
    print(f"{'default':<12} {t_default:14.6f} {1.0:20.2f}x")
    print(f"{'pg-numpy':<12} {t_numpy:14.6f} {t_default / t_numpy:20.2f}x")
    print(f"{'pg-jit cold':<12} {t_jit_cold:14.6f} {t_default / t_jit_cold:20.2f}x")
    print(f"{'pg-jit warm':<12} {t_jit:14.6f} {t_default / t_jit:20.2f}x")
    print()
    print(f'best-of-{args.repeats} wall times; cold = first call in the '
          f'process (numba compilation / cache load)')
    print('parity default vs pg-jit: '
          f'{same_supports}/{B} identical trimmed supports, '
          f'max |conc| diff={conc_max:.2e}, '
          f'max rel rss diff={rss_rel:.2e}, max rel bic diff={bic_rel:.2e}')
    print(f'per-candidate cost: default={1e6 * t_default / B:.1f} us, '
          f'pg-jit={1e6 * t_jit / B:.1f} us')


if __name__ == '__main__':
    main()