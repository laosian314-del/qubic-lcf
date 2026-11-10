"""Benchmark QUBO build: pure-NumPy default vs opt-in numba JIT.

The optional numba/JIT acceleration was NOT used in the article;
article results were obtained with the pure NumPy code paths.

Times ``qubic.qubo.build_qubo`` at the real problem size (m=300 spectral
points, N=324 components) for a 2-bit scheme (``B2``) and a linked scheme
(``linked_B3``), default path (``use_jit=False``) against the numba twin
(``use_jit=True``).  The first ``use_jit=True`` call includes the one-time
numba compilation cost and is reported separately.

Usage::

    python benchmarks/bench_qubo_build.py [--m 300] [--N 324] [--repeat 5]
"""
import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

from qubic.qubo import build_qubo  # noqa: E402

#: Benchmark cases: one 2-bit scheme and one linked scheme.
CASES = {
    'B2': dict(lam_sparse=1.0, lam_link=5.0),
    'linked_B3': dict(lam_sparse=1.0, lam_link=5.0),
}


def bench_scheme(scheme, penalties, A, y, k, repeat):
    """Return timing dict for one scheme (times in seconds)."""
    # ---- default path (warmup + repeats) ----
    Q, q, var_map, _ = build_qubo(scheme, A, y, k, **penalties)
    t_default = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        Q, q, var_map, _ = build_qubo(scheme, A, y, k, **penalties)
        t_default.append(time.perf_counter() - t0)

    # ---- jit path: first call may include numba compilation ----
    t0 = time.perf_counter()
    Qj, qj, _, _ = build_qubo(scheme, A, y, k, use_jit=True, **penalties)
    t_first = time.perf_counter() - t0

    t_jit = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        Qj, qj, _, _ = build_qubo(scheme, A, y, k, use_jit=True, **penalties)
        t_jit.append(time.perf_counter() - t0)

    # sanity: the twins must agree (they are asserted in the test suite too)
    np.testing.assert_allclose(Q, Qj, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(q, qj, rtol=1e-12, atol=1e-12)

    return {
        'scheme': scheme,
        'M': var_map['M'],
        'default_best': min(t_default),
        'default_mean': float(np.mean(t_default)),
        'jit_first': t_first,
        'jit_best': min(t_jit),
        'jit_mean': float(np.mean(t_jit)),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--m', type=int, default=300, help='spectral points')
    ap.add_argument('--N', type=int, default=324, help='library components')
    ap.add_argument('--k', type=int, default=10, help='cardinality target')
    ap.add_argument('--repeat', type=int, default=5, help='timed repeats')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    A = rng.normal(size=(args.m, args.N))
    y = rng.normal(size=args.m)

    print(f'QUBO build benchmark: m={args.m} points, N={args.N} components, '
          f'k={args.k}, repeat={args.repeat} (best / mean of {args.repeat})')
    print(f'numpy {np.__version__}')
    header = (f'{"scheme":10s} {"M":>6s} {"default best":>13s} '
              f'{"jit best":>10s} {"speedup":>8s} {"jit 1st call":>13s}')
    print(header)
    print('-' * len(header))
    for scheme, penalties in CASES.items():
        r = bench_scheme(scheme, penalties, A, y, args.k, args.repeat)
        speedup = r['default_best'] / r['jit_best'] if r['jit_best'] else float('inf')
        print(f'{r["scheme"]:10s} {r["M"]:6d} {r["default_best"]:11.4f} s '
              f'{r["jit_best"]:8.4f} s {speedup:7.2f}x '
              f'{r["jit_first"]:11.4f} s')
    print()
    print('("jit 1st call" includes numba compilation when the kernel is not')
    print(' yet compiled; use_jit=True steady-state is the "jit best" column.)')


if __name__ == '__main__':
    main()
