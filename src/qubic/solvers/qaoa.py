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

State evolution uses analytic parameterized gate application — no matrix
exponentials are built or factorized. The cost unitary is an exact
per-amplitude phase and the mixer unitary is a product of single-qubit X
rotations ``e^{-i beta X_i} = cos(beta) I - i sin(beta) X_i`` applied as
two-level rotations directly on the statevector (linear in the state
dimension per gate, O(P n 2^n) in total). The previous
``scipy.linalg.expm``-based evolution is kept as :func:`final_state_expm`
for validation (parity tests) and benchmarking.

The evolution has two twins selected by ``use_jit``: a pure-NumPy reference
(:func:`_evolve_numpy`, the default path) and a numba ``@njit`` twin of the
state-evolution inner loop (:func:`_build_jit_evolve`, opt-in via
``use_jit=True`` on :func:`final_state`, :func:`qaoa_expectation`,
:func:`solve_qaoa` and :func:`solve`). The twins consume bitwise-identical
precomputed gate factors and perform the same scalar arithmetic, so they
produce bitwise-identical statevectors.

The optional numba/JIT acceleration was NOT used in the article; article
results were obtained with the pure NumPy code paths.

Public API:
    qubo_to_ising_fixed(Q, q) -> (h, J, const)   (corrected mapping)
    ising_energy(s, h, J, const) -> float
    solve_qaoa(Q, q, var_map, ..., use_jit=False) -> dict

Unified API:
    solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, use_jit=False, **kwargs)
        -> (solutions, wall_time, gap=None)
"""
from __future__ import annotations

import time

import numpy as np

from .._accel import require_numba
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


def _evolve_numpy(re, im, ph_re, ph_im, cos_betas, sin_betas, n_qubits):
    """Pure-NumPy analytic gate application (default twin), in place.

    Applies p = ph_re.shape[0] QAOA layers to the real/imaginary parts of the
    statevector ``re + i*im`` (both length 2**n, contiguous float64):

    * cost layer:   psi[j] *= exp(-i gamma diag[j]) (per-amplitude phase
      given as precomputed real factors ph_re, ph_im),
    * mixer layer:  for each qubit i, the two-level rotation
      ``cos(beta) I - i sin(beta) X_i`` applied to every (j, j^(1<<i)) pair:
      ``new_a = c*a + s*(i*b),  new_b = c*b + s*(i*a)`` in explicit real
      arithmetic (gather of the flipped amplitudes, in-place update).

    The numba twin :func:`_build_jit_evolve` performs the same scalar
    operations on the same precomputed factors, so both twins agree bitwise.
    """
    dim = re.size
    idx_all = np.arange(dim)
    flips = [idx_all ^ (1 << i) for i in range(n_qubits)]
    for layer in range(ph_re.shape[0]):
        pr = ph_re[layer]
        pi = ph_im[layer]
        nr = re * pr - im * pi
        ni = re * pi + im * pr
        re[:] = nr
        im[:] = ni
        c = cos_betas[layer]
        s = sin_betas[layer]
        for i in range(n_qubits):
            flipped = flips[i]
            re_f = re[flipped]   # snapshot of the old flipped amplitudes
            im_f = im[flipped]
            re *= c
            re += s * im_f
            im *= c
            im -= s * re_f
    return re, im


def _build_jit_evolve(nb):
    """Build the numba ``@njit`` twin of :func:`_evolve_numpy` (lazy, cached)."""

    @nb.njit(cache=True)
    def evolve(re, im, ph_re, ph_im, cos_betas, sin_betas, n_qubits):
        p = ph_re.shape[0]
        dim = re.shape[0]
        for layer in range(p):
            pr = ph_re[layer]
            pi = ph_im[layer]
            for j in range(dim):
                x = re[j]
                y = im[j]
                re[j] = x * pr[j] - y * pi[j]
                im[j] = x * pi[j] + y * pr[j]
            c = cos_betas[layer]
            s = sin_betas[layer]
            for i in range(n_qubits):
                bit = 1 << i
                stride = bit << 1
                for base in range(0, dim, stride):
                    for r in range(bit):
                        j = base + r
                        k = j + bit
                        ar = re[j]
                        ai = im[j]
                        br = re[k]
                        bi = im[k]
                        re[j] = c * ar + s * bi
                        im[j] = c * ai - s * br
                        re[k] = c * br + s * ai
                        im[k] = c * bi - s * ar
        return re, im

    return evolve


_JIT_EVOLVE = None


def _get_jit_evolve():
    """Build (once) and return the numba twin of :func:`_evolve_numpy`.

    Imports numba lazily through :func:`qubic._accel.require_numba`, which
    raises a helpful ImportError when the optional dependency is missing.
    """
    global _JIT_EVOLVE
    if _JIT_EVOLVE is None:
        _JIT_EVOLVE = _build_jit_evolve(require_numba())
    return _JIT_EVOLVE


def _layer_factors(gammas, betas, diag):
    """Per-layer analytic gate factors (computed with NumPy for BOTH twins).

    Cost layer l: exp(-i gamma_l diag) = (ph_re[l], ph_im[l]) elementwise,
    computed via ``np.exp(-1j * gamma_l * diag)`` (bitwise the same phase the
    ``scipy.expm`` reference path applies).
    Mixer layer l: cos(beta_l) I - i sin(beta_l) X_i given as (cos_b[l], sin_b[l]).
    """
    p = gammas.size
    dim = diag.size
    ph_re = np.empty((p, dim), dtype=np.float64)
    ph_im = np.empty((p, dim), dtype=np.float64)
    for layer in range(p):
        z = np.exp(-1j * gammas[layer] * diag)
        ph_re[layer] = z.real
        ph_im[layer] = z.imag
    cos_b = np.cos(betas)
    sin_b = np.sin(betas)
    return ph_re, ph_im, cos_b, sin_b


def final_state(gammas, betas, diag, use_jit=False):
    """QAOA statevector after p = len(gammas) layers (analytic gate application).

    Cost evolution e^{-i gamma H_C} is an exact per-amplitude phase; the mixer
    e^{-i beta H_M} = prod_i e^{-i beta X_i} is applied as analytic two-level
    rotations ``cos(beta) I - i sin(beta) X_i`` directly on the statevector.

    Parameters
    ----------
    gammas, betas : sequence of float
        QAOA cost/mixer angles, one pair per layer.
    diag : np.ndarray
        Diagonal of the cost Hamiltonian (length 2**n).
    use_jit : bool, optional
        When True, run the numba ``@njit`` twin of the state-evolution inner
        loop (requires the optional ``numba`` dependency). The default runs
        the pure-NumPy twin and never imports numba.

    Returns
    -------
    np.ndarray
        Complex statevector of length 2**n.
    """
    diag = np.asarray(diag, dtype=np.float64)
    gammas = np.asarray(gammas, dtype=np.float64)
    betas = np.asarray(betas, dtype=np.float64)
    n = int(np.log2(diag.size))
    dim = diag.size

    ph_re, ph_im, cos_b, sin_b = _layer_factors(gammas, betas, diag)

    re = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.float64)   # |+>^{xn}
    im = np.zeros(dim, dtype=np.float64)

    if use_jit:
        _get_jit_evolve()(re, im, ph_re, ph_im, cos_b, sin_b, n)
    else:
        _evolve_numpy(re, im, ph_re, ph_im, cos_b, sin_b, n)
    return re + 1j * im


def final_state_expm(gammas, betas, diag):
    """Reference QAOA statevector using ``scipy.linalg.expm`` (legacy path).

    Kept for validation (parity tests) and benchmarking; the production path
    is :func:`final_state` (analytic gate application). Behavior is identical
    to the pre-analytic implementation.
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


def qaoa_expectation(gamma_beta, diag, p, use_jit=False):
    """<psi(gamma,beta)|H_C|psi(gamma,beta)> for QAOA parameters (2p,).

    ``use_jit`` selects the numba twin of the state-evolution inner loop
    (see :func:`final_state`).
    """
    gammas = gamma_beta[:p]
    betas = gamma_beta[p:]
    psi = final_state(gammas, betas, diag, use_jit=use_jit)
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


def solve_qaoa(Q, q, var_map, p=2, max_iter=100, n_shots=1024, seed=0,
               use_jit=False):
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
    use_jit : bool, optional
        When True, accelerate the state-evolution inner loop with the
        optional numba ``@njit`` twin. The default (False) uses the pure
        NumPy analytic path and never imports numba.

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
        lambda x: qaoa_expectation(x, diag, p, use_jit=use_jit),
        x0=rng.uniform(0, np.pi, 2 * p),
        method="COBYLA",
        options={"maxiter": max_iter, "disp": False},
    )

    gammas = result.x[:p]
    betas = result.x[p:]

    # Reconstruct final state with optimal parameters and sample candidates
    psi = final_state(gammas, betas, diag, use_jit=use_jit)
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


def solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, use_jit=False,
          **kwargs):
    """Unified solver entry point (see :mod:`qubic.solvers.base`).

    Extra kwargs: ``p`` (QAOA depth), ``max_iter`` (COBYLA iterations),
    ``n_shots``, ``seed``, ``use_jit`` (numba twin for the state-evolution
    inner loop). ``time_limit`` and ``k_max`` are accepted for
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
                        n_shots=n_shots, seed=seed, use_jit=use_jit)
    solutions = base.finalize((s for s, _ in result["candidates"]), Q, q, top_k)
    return solutions, result["wall_time_s"], None
