"""QAOA solver for QUBO-LCF spectral decomposition.

Ported from ``QUBO_LCF_3/scripts/qaoa_solver.py`` (statevector simulation +
COBYLA, no Qiskit dependency), with one deliberate correction: the QUBO->Ising
mapping is the FIXED ``qubo_to_ising_fixed`` from
``deliverables/2026-10-29_qaoa_review/repo/qaoa-lcf/experiments/run_pilot.py``
(the prototype ``qubo_to_ising`` in qaoa_solver.py violates the identity
E_Q(x) == H(1-2x) by ~1e4 on real QUBOs and is therefore not used here).

Algorithm:
  1. QUBO matrix Q -> Ising Hamiltonian H_C = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j + C
  2. QAOA ansatz: |psi(gamma,beta)> = prod_{p=1..P} e^{-i beta_p H_M} e^{-i gamma_p H_C} |+>^{xn}
     with H_M = sum_i X_i (mixer Hamiltonian)
  3. Optimize (gamma, beta) via COBYLA to minimize <psi|H_C|psi>
  4. Sample the final state -> bitstrings -> top_k candidate solutions

The cost unitary is applied as an exact phase on the statevector diagonal and
the mixer unitary is built with ``scipy.linalg.expm`` (single-qubit X
rotations). Exact statevector simulation is limited to MAX_QUBITS = 15 qubits.

Public API:
    qubo_to_ising_fixed(Q, q) -> (h, J, const)   (corrected mapping)
    ising_energy(s, h, J, const) -> float
    solve_qaoa(Q, q, var_map, ...) -> dict

Unified API:
    solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, **kwargs)
        -> (solutions, wall_time, gap=None)
"""
from __future__ import annotations

import time

import numpy as np

from . import base
from .base import get_selection

#: Hard limit for exact statevector simulation (2**15 complex amplitudes).
MAX_QUBITS = 15


# ---------------------------------------------------------------- Ising mapping
def qubo_to_ising_fixed(Q, q):
    """Correct QUBO->Ising mapping for E = x^T Q x + q^T x, x=(1-s)/2, Qs=(Q+Q.T)/2:
    J_ij = Qs_ij / 2 (i<j),  h_i = -0.5*(q_i + sum_j Qs_ij),
    C = 0.25*sum_ij Qs_ij + 0.25*trace(Qs) + 0.5*sum_i q_i.

    (Ported from experiments/run_pilot.py — replaces the buggy prototype
    mapping whose deviation from the E_Q == H identity is ~1e4.)
    """
    Qs = (Q + Q.T) / 2.0
    n = Q.shape[0]
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = Qs[i, j] / 2.0
    h = -0.5 * (q + Qs.sum(axis=1))
    C = 0.25 * float(Qs.sum()) + 0.25 * float(np.trace(Qs)) + 0.5 * float(q.sum())
    return h, J, C


def ising_energy(s, h, J, const):
    """H = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j + C  (convention of qaoa_solver)."""
    n = len(s)
    val = float(h @ s) + const
    for i in range(n):
        for j in range(i + 1, n):
            val += J[i, j] * s[i] * s[j]
    return val


# ---------------------------------------------------------------- QAOA core
def cost_diagonal(h, J):
    """Diagonal of H_C in the computational basis (spin s_i = 1 - 2*x_i)."""
    n = len(h)
    idx = np.arange(2 ** n)
    bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
    spins = 1.0 - 2.0 * bits
    diag = spins @ h
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0.0:
                diag = diag + J[i, j] * spins[:, i] * spins[:, j]
    return diag


def final_state(gammas, betas, diag):
    """QAOA statevector after p = len(gammas) layers (ported from run_pilot.py).

    Cost evolution e^{-i gamma H_C} is an exact phase on the diagonal; the
    mixer e^{-i beta H_M} = prod_i e^{-i beta X_i} uses the scipy.linalg.expm
    single-qubit unitary.
    """
    scipy_linalg = base.require("scipy.linalg", "scipy")
    expm = scipy_linalg.expm

    n = int(np.log2(len(diag)))
    psi = np.ones(2 ** n, dtype=np.complex128) / np.sqrt(2 ** n)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    idx_all = np.arange(2 ** n)
    for layer in range(len(gammas)):
        psi = psi * np.exp(-1j * gammas[layer] * diag)
        U = expm(-1j * betas[layer] * X)   # [[c, -i s], [-i s, c]]
        for i in range(n):
            flipped = idx_all ^ (1 << i)
            psi = U[0, 0] * psi + U[0, 1] * psi[flipped]
    return psi


def qaoa_expectation(gamma_beta, diag, p):
    """<psi(gamma,beta)|H_C|psi(gamma,beta)> for QAOA parameters (2p,)."""
    gammas = gamma_beta[:p]
    betas = gamma_beta[p:]
    psi = final_state(gammas, betas, diag)
    probs = np.abs(psi) ** 2
    return float(probs @ diag)


def sample_qaoa_state(psi, n_shots=1024, rng=None):
    """Sample bitstrings from a QAOA statevector (ported; deterministic rng)."""
    if rng is None:
        rng = np.random
    n = int(np.log2(len(psi)))
    probs = np.abs(psi) ** 2
    probs = probs / probs.sum()
    indices = rng.choice(2 ** n, size=n_shots, p=probs)
    return [np.array([(int(idx) >> k) & 1 for k in range(n)], dtype=int)
            for idx in indices]


def solve_qaoa(Q, q, var_map, p=2, max_iter=100, n_shots=1024, seed=0):
    """Solve a QUBO problem using QAOA (ported; corrected Ising mapping).

    Parameters
    ----------
    Q, q : np.ndarray
        QUBO matrix (symmetric) and linear term.
    var_map : dict
        Variable layout metadata ('N', 'scheme', ...).
    p : int
        QAOA depth (number of layers).
    max_iter : int
        Max COBYLA iterations.
    n_shots : int
        Number of measurement shots.
    seed : int or None
        Seed for parameter init and sampling (reproducibility).

    Returns
    -------
    dict with keys: solution_indices, best_sample, best_energy, energy,
    candidates, energy_ising, wall_time_s, n_qubits, p, optimizer_iters,
    optimizer_success, gammas, betas.

    Raises
    ------
    ValueError
        If the problem has more than MAX_QUBITS (15) variables.
    """
    optimize = base.require("scipy.optimize", "scipy").minimize

    n_qubits = int(Q.shape[0])
    if n_qubits > MAX_QUBITS:
        raise ValueError(
            f"QAOA statevector simulation is limited to {MAX_QUBITS} qubits; "
            f"got {n_qubits} variables. Reduce the problem size or use a "
            "classical solver (milp, cp_sat, scip, sa, tabu)."
        )

    rng = np.random.RandomState(seed if seed is not None else 0)
    t0 = time.time()

    # Convert to Ising with the CORRECTED mapping
    h_vec, J, const = qubo_to_ising_fixed(np.asarray(Q, dtype=float),
                                          np.asarray(q, dtype=float))
    diag = cost_diagonal(h_vec, J)

    # Optimize QAOA parameters
    result = optimize(
        lambda x: qaoa_expectation(x, diag, p),
        x0=rng.uniform(0, np.pi, 2 * p),
        method="COBYLA",
        options={"maxiter": max_iter, "disp": False},
    )

    gammas = result.x[:p]
    betas = result.x[p:]

    # Reconstruct final state with optimal parameters and sample candidates
    psi = final_state(gammas, betas, diag)
    samples = sample_qaoa_state(psi, n_shots=n_shots, rng=rng)

    best_energy = float("inf")
    best_sample = None
    candidates = []
    seen = set()
    for bits in samples:
        x = bits.astype(float)
        key = tuple(x.tolist())
        if key in seen:
            continue
        seen.add(key)
        energy = float(x @ np.asarray(Q, dtype=float) @ x
                       + np.asarray(q, dtype=float) @ x)
        candidates.append((x, energy))
        if energy < best_energy:
            best_energy = energy
            best_sample = x.copy()

    candidates.sort(key=lambda pair: pair[1])

    solution_indices = [j for j, flag in enumerate(get_selection(best_sample, var_map))
                        if flag]
    wall_time = time.time() - t0

    return {
        "solution_indices": solution_indices,
        "best_sample": best_sample,
        "best_energy": float(best_energy),
        "energy": float(best_energy),
        "candidates": candidates,
        "energy_ising": float(result.fun + const),
        "wall_time_s": wall_time,
        "n_qubits": n_qubits,
        "p": p,
        "optimizer_iters": getattr(result, "nit", getattr(result, "nfev", 0)),
        "optimizer_success": bool(result.success),
        "gammas": gammas.tolist(),
        "betas": betas.tolist(),
    }


def solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, **kwargs):
    """Unified solver entry point (see :mod:`qubic.solvers.base`).

    Extra kwargs: ``p`` (QAOA depth), ``max_iter`` (COBYLA iterations),
    ``n_shots``, ``seed``. ``time_limit`` and ``k_max`` are accepted for
    interface uniformity (QAOA is budgeted by ``max_iter``/``n_shots`` and is
    unconstrained — cardinality is encoded in the QUBO penalties).

    Raises ValueError for problems with more than 15 variables.
    """
    p = kwargs.pop("p", 2)
    max_iter = kwargs.pop("max_iter", 100)
    n_shots = kwargs.pop("n_shots", 1024)
    seed = kwargs.pop("seed", 0)
    base.reject_unknown_kwargs("qaoa.solve", kwargs)

    result = solve_qaoa(Q, q, var_map, p=p, max_iter=max_iter,
                        n_shots=n_shots, seed=seed)
    solutions = base.finalize((s for s, _ in result["candidates"]), Q, q, top_k)
    return solutions, result["wall_time_s"], None
