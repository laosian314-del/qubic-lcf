"""Quickstart: QUBIC-LCF on a synthetic LCF mixture.

Mirrors the quickstart of the original QUBO_LCF research repository
(``QUBO_LCF_3/scripts/quickstart.py``: N=5 pseudo-Voigt components, a
2-component mixture), but uses only the high-level API of :mod:`qubic`: a
:class:`qubic.Problem` task object and solver objects passed whole to
:meth:`qubic.Problem.solve`.  Both backends share the same
:class:`qubic.Result`: QUBO-backed solvers (:class:`qubic.MILP`,
:class:`qubic.SA`, ...) and the alternative backend (:class:`qubic.OMP`,
...).  No PyFitIt required.

Usage::

    python examples/quickstart.py [--backend qubo|alternative|both]
"""
import argparse

import numpy as np

import qubic


def pseudo_voigt(x, x0, gamma, eta):
    """Pseudo-Voigt component: eta*Lorentzian + (1-eta)*Gaussian."""
    lor = gamma**2 / ((x - x0)**2 + gamma**2)
    gauss = np.exp(-((x - x0)**2) / (2 * gamma**2))
    return eta * lor + (1 - eta) * gauss


def make_synthetic_mixture(noise=0.0):
    """Synthetic 5-component library and 2-component mixture (fixed seed)."""
    rng = np.random.RandomState(42)
    x = np.linspace(0, 10, 200)
    N = 5
    A = np.zeros((len(x), N))
    for j in range(N):
        x0 = rng.uniform(1, 9)
        gamma = rng.uniform(0.3, 0.8)
        eta = rng.uniform(0.3, 0.7)
        A[:, j] = pseudo_voigt(x, x0, gamma, eta)
    true_idx = (1, 3)
    y = A[:, list(true_idx)] @ np.array([0.7, 0.3])
    if noise > 0:
        y = y + rng.normal(0, noise, size=y.shape)
    return A, y, true_idx


def build_problem(A, y, k_max=2, tau=0.01):
    """Wrap the task data in a qubic.Problem (the only task object needed)."""
    return qubic.Problem(A, y, k_max=k_max, tau=tau)


def run_qubo(task, true_idx, top_k=10):
    """Identify the mixture with one QUBO-backed solver object."""
    print("--- QUBO backend ---")
    res = task.solve(qubic.MILP(time_limit=10.0), top_k=top_k)
    _print_result(res, true_idx)
    return res


def run_alternative(task, true_idx):
    """Identify the mixture with the alternative-backend solver objects."""
    print("--- alternative backend (greedy forward selection (MP/OMP) family etc.) ---")
    for solver in (qubic.MP(), qubic.OMP(), qubic.Lasso(),
                   qubic.TopK(), qubic.Oracle()):
        res = task.solve(solver)
        _print_result(res, true_idx)
    return True


def _print_result(res, true_idx):
    f1 = qubic.compute_f1(res.selected, true_idx) if true_idx else None
    print(f"method={res.method}  selected={res.selected}  f1={f1}")
    print(f"  concentrations={np.round(res.concentrations, 3)}  "
          f"bic={res.bic}  rss={res.rss:.3e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--backend', choices=['qubo', 'alternative', 'both'],
                        default='both', help='which backend to run')
    parser.add_argument('--top-k', type=int, default=10,
                        help='candidate pool size for the QUBO backend')
    args = parser.parse_args()

    A, y, true_idx = make_synthetic_mixture(noise=1e-3)
    task = build_problem(A, y, k_max=2, tau=0.01)
    print(f"Library: A {A.shape}, target y {y.shape}, "
          f"true components {list(true_idx)}")
    print(f"task = qubic.Problem(A, y, k_max={task.k_max}, tau={task.tau})")

    if args.backend in ('qubo', 'both'):
        run_qubo(task, true_idx, top_k=args.top_k)
    if args.backend in ('alternative', 'both'):
        run_alternative(task, true_idx)


if __name__ == '__main__':
    main()
