#!/usr/bin/env python
"""Benchmark: QAOA state evolution — scipy.expm vs analytic NumPy vs analytic numba.

Times one optimizer iteration (one ``<psi|H_C|psi>`` evaluation, the dominant
cost inside the COBYLA loop) at p layers on n = 10..12 qubits, for three
implementations of the state evolution:

  * expm           — legacy path: ``scipy.linalg.expm`` single-qubit mixer
                     unitaries (``qaoa.final_state_expm``),
  * analytic-numpy — analytic two-level rotations, pure NumPy twin
                     (``qaoa.qaoa_expectation(..., use_jit=False)``),
  * analytic-numba — numba ``@njit`` twin of the state-evolution inner loop
                     (``qaoa.qaoa_expectation(..., use_jit=True)``; numba
                     JIT compilation time is reported separately and excluded
                     from per-iteration timings).

Usage:
    /opt/anaconda/bin/python benchmarks/bench_qaoa.py [--p 2] [--n 10,11,12]
        [--repeats 7] [--seed 0]

The optional numba/JIT acceleration was NOT used in the article; article
results were obtained with the pure NumPy code paths.
"""
import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

from qubic.solvers.qaoa import (  # noqa: E402
    cost_diagonal,
    final_state_expm,
    qaoa_expectation,
    qubo_to_ising_fixed,
)


def make_diag(n, seed):
    """Random n-qubit QUBO -> cost-Hamiltonian diagonal."""
    rng = np.random.RandomState(seed)
    U = rng.uniform(0.5, 2.0, size=(n, n))
    Q = np.triu(U) + np.triu(U, 1).T
    q = rng.uniform(-1.0, 1.0, size=n)
    h, J, _ = qubo_to_ising_fixed(Q, q)
    return cost_diagonal(h, J)


def expm_expectation(x, diag, p):
    """Reference expectation via the scipy.expm state evolution."""
    psi = final_state_expm(x[:p], x[p:], diag)
    return float((np.abs(psi) ** 2) @ diag)


def time_per_iter(fn, p, repeats, seed):
    """Return (median_ms, min_ms) of one expectation call per repeat."""
    rng = np.random.RandomState(seed)
    times = []
    for _ in range(repeats):
        x = rng.uniform(0, np.pi, 2 * p)
        t0 = time.perf_counter()
        fn(x)
        times.append(time.perf_counter() - t0)
    times = np.asarray(times) * 1e3
    return float(np.median(times)), float(times.min())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p", type=int, default=2, help="QAOA depth (layers)")
    ap.add_argument("--n", type=str, default="10,11,12",
                    help="comma-separated qubit counts")
    ap.add_argument("--repeats", type=int, default=7,
                    help="timed repetitions per implementation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n_list = [int(s) for s in args.n.split(",") if s.strip()]
    p = args.p

    print(f"QAOA state-evolution benchmark: p={p}, repeats={args.repeats}, "
          f"one optimizer iteration = one <psi|H_C|psi> evaluation")
    print(f"{'n':>3} {'dim':>7} {'expm (ms)':>12} {'analytic-np (ms)':>18} "
          f"{'analytic-jit (ms)':>18} {'np/expm':>8} {'jit/np':>8}")
    for n in n_list:
        diag = make_diag(n, seed=args.seed + n)
        dim = diag.size

        # numba JIT compile time (excluded from per-iteration timings)
        x0 = np.random.RandomState(args.seed).uniform(0, np.pi, 2 * p)
        t0 = time.perf_counter()
        qaoa_expectation(x0, diag, p, use_jit=True)
        compile_s = time.perf_counter() - t0

        f_expm = lambda x: expm_expectation(x, diag, p)          # noqa: E731
        f_np = lambda x: qaoa_expectation(x, diag, p)            # noqa: E731
        f_jit = lambda x: qaoa_expectation(x, diag, p, use_jit=True)  # noqa: E731

        med_expm, min_expm = time_per_iter(f_expm, p, args.repeats, args.seed + 1)
        med_np, min_np = time_per_iter(f_np, p, args.repeats, args.seed + 2)
        med_jit, min_jit = time_per_iter(f_jit, p, args.repeats, args.seed + 3)

        # parity sanity check at a fixed parameter vector
        e_expm = f_expm(x0)
        e_np = f_np(x0)
        e_jit = f_jit(x0)
        assert abs(e_np - e_expm) < 1e-9 * max(1.0, abs(e_expm)), (e_np, e_expm)
        assert e_np == e_jit, (e_np, e_jit)

        print(f"{n:>3} {dim:>7} {med_expm:>12.3f} {med_np:>18.3f} "
              f"{med_jit:>18.3f} {med_expm / med_np:>8.2f} "
              f"{med_np / med_jit:>8.2f}")
        print(f"    (min: expm {min_expm:.3f} ms, np {min_np:.3f} ms, "
              f"jit {min_jit:.3f} ms; numba compile {compile_s:.2f} s; "
              f"max |E_np - E_expm| = {abs(e_np - e_expm):.2e})")


if __name__ == "__main__":
    main()
