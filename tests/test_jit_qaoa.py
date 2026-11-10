"""Parity and behavior tests for the analytic QAOA state evolution twins.

Covers:
  * analytic gate application vs the legacy ``scipy.linalg.expm`` path
    (expectation values and final energies, tight tolerance),
  * pure-NumPy twin vs numba ``@njit`` twin (bitwise-identical states),
  * default path (``use_jit=False``) never imports numba (subprocess test),
  * QUBO->Ising mapping helpers and the MAX_QUBITS guard are intact.

The optional numba/JIT acceleration was NOT used in the article; article
results were obtained with the pure NumPy code paths.
"""
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))

import qubic.solvers.qaoa as qaoa_mod
from qubic.solvers.qaoa import (
    MAX_QUBITS,
    cost_diagonal,
    final_state,
    final_state_expm,
    ising_energy,
    qaoa_expectation,
    qubo_to_ising_fixed,
    solve,
    solve_qaoa,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ helpers
def make_qubo(seed=1234, n=4):
    """Small synthetic symmetric QUBO (same construction as test_solvers)."""
    rng = np.random.RandomState(seed)
    U = rng.uniform(0.5, 2.0, size=(n, n))
    Q = np.triu(U) + np.triu(U, 1).T
    q = rng.uniform(-1.0, 1.0, size=n)
    var_map = {
        "scheme": "B1",
        "N": n,
        "M": n,
        "vars_per_component": 1,
        "selection": "explicit",
    }
    return Q, q, var_map


def random_diag(n, seed):
    """Random Ising cost diagonal on n qubits."""
    rng = np.random.RandomState(seed)
    h = rng.uniform(-1.0, 1.0, size=n)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.rand() < 0.7:
                J[i, j] = rng.uniform(-1.0, 1.0)
    return cost_diagonal(h, J)


def random_params(p, seed):
    rng = np.random.RandomState(seed)
    return rng.uniform(0, np.pi, 2 * p)


def expm_expectation(x, diag, p):
    """Reference expectation using the scipy.expm state evolution."""
    psi = final_state_expm(x[:p], x[p:], diag)
    probs = np.abs(psi) ** 2
    return float(probs @ diag)


# --------------------------------------------------- analytic vs expm parity
@pytest.mark.parametrize("n", [3, 5, 8])
@pytest.mark.parametrize("p", [1, 2, 3])
def test_final_state_matches_expm(n, p):
    """Analytic statevector matches scipy.expm evolution to tight tolerance."""
    diag = random_diag(n, seed=10 * n + p)
    x = random_params(p, seed=100 + 10 * n + p)
    psi_analytic = final_state(x[:p], x[p:], diag)
    psi_expm = final_state_expm(x[:p], x[p:], diag)
    np.testing.assert_allclose(psi_analytic, psi_expm, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("n", [3, 5, 8])
@pytest.mark.parametrize("p", [1, 2, 3])
def test_expectation_matches_expm(n, p):
    """<psi|H_C|psi> matches the scipy.expm path to tight tolerance."""
    diag = random_diag(n, seed=20 * n + p)
    for k in range(3):
        x = random_params(p, seed=200 + 10 * n + p + k)
        e_analytic = qaoa_expectation(x, diag, p)
        e_expm = expm_expectation(x, diag, p)
        np.testing.assert_allclose(e_analytic, e_expm, rtol=1e-11, atol=1e-11)


def test_final_energies_match_expm(monkeypatch):
    """Full solve_qaoa final energies match the scipy.expm path (same seed)."""
    Q, q, var_map = make_qubo(seed=11, n=4)
    res_analytic = solve_qaoa(Q, q, var_map, p=2, max_iter=25,
                              n_shots=256, seed=3)

    monkeypatch.setattr(
        qaoa_mod, "final_state",
        lambda g, b, d, use_jit=False: qaoa_mod.final_state_expm(g, b, d),
    )
    res_expm = solve_qaoa(Q, q, var_map, p=2, max_iter=25,
                          n_shots=256, seed=3)

    np.testing.assert_allclose(res_analytic["energy_ising"],
                               res_expm["energy_ising"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(res_analytic["best_energy"],
                               res_expm["best_energy"], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(res_analytic["gammas"], res_expm["gammas"],
                               rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(res_analytic["betas"], res_expm["betas"],
                               rtol=1e-7, atol=1e-9)


# --------------------------------------------------- numpy vs numba twins
@pytest.mark.parametrize("n", [3, 5, 8])
@pytest.mark.parametrize("p", [1, 2, 3])
def test_numba_twin_matches_numpy_exactly(n, p):
    """The numba twin reproduces the NumPy twin bitwise ('match exactly')."""
    pytest.importorskip("numba")
    diag = random_diag(n, seed=30 * n + p)
    x = random_params(p, seed=300 + 10 * n + p)
    psi_np = final_state(x[:p], x[p:], diag, use_jit=False)
    psi_jit = final_state(x[:p], x[p:], diag, use_jit=True)
    np.testing.assert_array_equal(psi_np, psi_jit)
    assert (qaoa_expectation(x, diag, p, use_jit=True)
            == qaoa_expectation(x, diag, p, use_jit=False))


def test_solve_qaoa_use_jit_matches_default():
    """solve_qaoa(use_jit=True) reproduces the default path exactly."""
    pytest.importorskip("numba")
    Q, q, var_map = make_qubo(seed=21, n=5)
    res_np = solve_qaoa(Q, q, var_map, p=2, max_iter=15, n_shots=256, seed=7)
    res_jit = solve_qaoa(Q, q, var_map, p=2, max_iter=15, n_shots=256,
                         seed=7, use_jit=True)
    assert res_jit["energy_ising"] == res_np["energy_ising"]
    assert res_jit["best_energy"] == res_np["best_energy"]
    assert res_jit["solution_indices"] == res_np["solution_indices"]
    np.testing.assert_array_equal(res_jit["best_sample"], res_np["best_sample"])


def test_solve_unified_accepts_use_jit():
    """The unified entry points (solve dispatch + qaoa.solve) forward use_jit."""
    pytest.importorskip("numba")
    from qubic.solvers import solve as dispatch_solve

    Q, q, var_map = make_qubo(seed=31, n=4)
    solutions, wall_time, gap = dispatch_solve("qaoa", Q, q, var_map, top_k=3,
                                              p=1, max_iter=5, n_shots=64,
                                              seed=0, use_jit=True)
    assert gap is None
    assert len(solutions) >= 1
    assert wall_time > 0

    solutions2, _, _ = solve(Q, q, var_map, top_k=3, p=1, max_iter=5,
                             n_shots=64, seed=0, use_jit=True)
    assert len(solutions2) >= 1


# --------------------------------------------------- default path sanity
def test_default_path_runs_without_jit():
    """Default (use_jit=False) solve_qaoa returns a well-formed result."""
    Q, q, var_map = make_qubo(seed=41, n=4)
    res = solve_qaoa(Q, q, var_map, p=1, max_iter=10, n_shots=128, seed=0)
    for key in ("solution_indices", "best_sample", "best_energy", "energy",
                "candidates", "energy_ising", "wall_time_s", "n_qubits", "p",
                "optimizer_iters", "optimizer_success", "gammas", "betas"):
        assert key in res
    assert res["n_qubits"] == 4
    assert res["p"] == 1
    assert res["best_energy"] <= min(e for _, e in res["candidates"]) + 1e-12
    # candidates are sorted by ascending energy
    energies = [e for _, e in res["candidates"]]
    assert energies == sorted(energies)


def test_default_path_does_not_import_numba():
    """A use_jit=False run must not import numba (fresh interpreter)."""
    script = (
        "import sys\n"
        "assert 'numba' not in sys.modules\n"
        "import numpy as np\n"
        "from qubic.solvers import qaoa\n"
        "rng = np.random.RandomState(0)\n"
        "n = 4\n"
        "U = rng.uniform(0.5, 2.0, size=(n, n))\n"
        "Q = np.triu(U) + np.triu(U, 1).T\n"
        "q = rng.uniform(-1.0, 1.0, size=n)\n"
        "var_map = {'scheme': 'B1', 'N': n, 'M': n,\n"
        "           'vars_per_component': 1, 'selection': 'explicit'}\n"
        "res = qaoa.solve_qaoa(Q, q, var_map, p=1, max_iter=3,\n"
        "                      n_shots=32, seed=0)\n"
        "psi = qaoa.final_state([0.3], [0.7],\n"
        "                       qaoa.cost_diagonal(*qaoa.qubo_to_ising_fixed(\n"
        "                           Q, q)[:2]))\n"
        "assert psi.shape == (2 ** n,)\n"
        "assert 'numba' not in sys.modules, sorted(\n"
        "    m for m in sys.modules if 'numba' in m)\n"
        "print('NO_NUMBA_OK')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    proc = subprocess.run([sys.executable, "-c", script], env=env,
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr
    assert "NO_NUMBA_OK" in proc.stdout


# --------------------------------------------------- unchanged contracts
def test_mapping_helpers_unchanged():
    """qubo_to_ising_fixed/ising_energy keep the E_Q == H(1-2x) identity."""
    Q, q, _ = make_qubo(seed=51, n=5)
    h, J, const = qubo_to_ising_fixed(Q, q)
    rng = np.random.RandomState(0)
    for _ in range(5):
        x = (rng.rand(5) < 0.5).astype(float)
        s = 1.0 - 2.0 * x
        e_q = float(x @ Q @ x + q @ x)
        assert e_q == pytest.approx(ising_energy(s, h, J, const), abs=1e-10)


def test_max_qubits_guard_unchanged():
    """Problems with more than MAX_QUBITS (15) variables still raise."""
    n = MAX_QUBITS + 1
    Q, q, var_map = make_qubo(seed=61, n=n)
    with pytest.raises(ValueError, match="15 qubits"):
        solve_qaoa(Q, q, var_map, p=1, max_iter=1, n_shots=8, seed=0)
    with pytest.raises(ValueError, match="15 qubits"):
        solve_qaoa(Q, q, var_map, p=1, max_iter=1, n_shots=8, seed=0,
                   use_jit=True)
