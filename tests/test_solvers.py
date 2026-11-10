"""Tests for qubic.solvers — the unified QUBO solver layer.

All QUBO instances are built by hand with seeded numpy RNG (no import of
qubic.encodings — that module is developed in parallel). Optional solvers are
guarded with pytest.mark.skipif: sa/tabu/scip skip when their dependency is
missing, gpu_sa skips without CUDA. The QAOA identity test verifies the
corrected QUBO->Ising mapping (E_Q(x) == H(1-2x)) and the qubit limit.
"""
import importlib.util

import numpy as np
import pytest

from qubic.solvers import SOLVERS, available_solvers, solve
from qubic.solvers.qaoa import MAX_QUBITS, ising_energy, qubo_to_ising_fixed

SOLVER_NAMES = {"milp", "highs", "cp_sat", "scip", "sa", "tabu",
                "gpu_sa", "sqaod", "qaoa"}


def _has(module_name):
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _cuda_available():
    if not _has("torch"):
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


HAS_PULP = _has("pulp")
HAS_ORTOOLS = _has("ortools")
HAS_PYSCIPOPT = _has("pyscipopt")
HAS_DIMOD = _has("dimod")
HAS_DWAVE = _has("dwave.samplers")
HAS_SQAOD = _has("sqaod")
HAS_CUDA = _cuda_available()

NEED_PULP = pytest.mark.skipif(not HAS_PULP, reason="pulp not installed")
NEED_ORTOOLS = pytest.mark.skipif(not HAS_ORTOOLS, reason="ortools not installed")
NEED_PYSCIPOPT = pytest.mark.skipif(not HAS_PYSCIPOPT, reason="pyscipopt not installed")
NEED_DIMOD = pytest.mark.skipif(not HAS_DIMOD, reason="dimod not installed")
NEED_DWAVE = pytest.mark.skipif(not HAS_DWAVE, reason="dwave-samplers not installed")
NEED_SQAOD = pytest.mark.skipif(not HAS_SQAOD, reason="sqaod not installed")
NEED_CUDA = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")


# ---------------------------------------------------------------------------
# Synthetic QUBO construction (hand-built, seeded — no qubic.encodings)
# ---------------------------------------------------------------------------

def make_qubo(seed=1234, n=4):
    """Small synthetic symmetric QUBO with both triangles of Q filled.

    Returns (Q, q, var_map) with var_map = {'scheme': 'B1', 'N': 4, 'M': 4,
    'vars_per_component': 1, 'selection': 'explicit'} for n = 4.
    """
    rng = np.random.RandomState(seed)
    U = rng.uniform(0.5, 2.0, size=(n, n))
    # Symmetric with BOTH triangles filled (no zero off-diagonal entries).
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


def check_solutions(solutions, Q, q, var_map, k_max, top_k, enforce_kmax=True):
    """Shared validity checks for the unified solver contract.

    ``enforce_kmax`` is True for solvers that enforce the cardinality bound
    (milp/cp_sat/scip); heuristic samplers only encode it in the QUBO
    penalties, so for them the bound is not asserted.
    """
    assert isinstance(solutions, list)
    assert 0 < len(solutions) <= top_k
    M = Q.shape[0]
    energies = []
    for sample, energy in solutions:
        assert isinstance(sample, np.ndarray)
        assert sample.shape == (M,)
        assert np.all(sample >= -1e-6) and np.all(sample <= 1.0 + 1e-6)
        assert np.allclose(sample, np.round(sample)), "sample must be 0/1"
        # energy must be the exact QUBO energy of the sample
        expected = float(sample @ Q @ sample + q @ sample)
        assert energy == pytest.approx(expected, rel=1e-9, abs=1e-9)
        # cardinality bound on the selection (only where enforced)
        if enforce_kmax:
            sel = sum(1 for j in range(var_map["N"]) if sample[j] > 0.5)
            assert sel <= k_max
        energies.append(energy)
    # energies must be sorted ascending (monotone)
    assert energies == sorted(energies)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_contains_all_solvers():
    assert set(SOLVERS) == SOLVER_NAMES


def test_available_solvers_lists_installed():
    avail = available_solvers()
    assert isinstance(avail, list)
    assert set(avail) <= SOLVER_NAMES
    assert "milp" in avail and "cp_sat" in avail and "qaoa" in avail


def test_unknown_solver_raises():
    Q, q, var_map = make_qubo()
    with pytest.raises(ValueError, match="unknown solver"):
        solve("nope", Q, q, var_map)


# ---------------------------------------------------------------------------
# MILP (pulp — installed)
# ---------------------------------------------------------------------------

@NEED_PULP
def test_milp_solutions_valid():
    Q, q, var_map = make_qubo()
    k_max, top_k = 3, 6
    solutions, wall_time, gap = solve("milp", Q, q, var_map,
                                      k_max=k_max, top_k=top_k, time_limit=30)
    check_solutions(solutions, Q, q, var_map, k_max, top_k)
    assert wall_time > 0.0
    assert gap is None


@NEED_PULP
def test_milp_top_k_respected():
    Q, q, var_map = make_qubo(seed=7)
    solutions, _, _ = solve("milp", Q, q, var_map, k_max=3, top_k=3, time_limit=30)
    assert len(solutions) == 3


@NEED_PULP
def test_milp_highs_backend():
    Q, q, var_map = make_qubo(seed=11)
    solutions, _, _ = solve("milp", Q, q, var_map, k_max=2, top_k=4,
                            time_limit=30, solver_backend="highs")
    check_solutions(solutions, Q, q, var_map, 2, 4)


# ---------------------------------------------------------------------------
# HiGHS (pulp.HiGHS interface — installed; CBC fallback)
# ---------------------------------------------------------------------------

@NEED_PULP
def test_highs_solutions_valid():
    Q, q, var_map = make_qubo()
    k_max, top_k = 3, 6
    solutions, wall_time, gap = solve("highs", Q, q, var_map,
                                      k_max=k_max, top_k=top_k, time_limit=30)
    check_solutions(solutions, Q, q, var_map, k_max, top_k)
    assert wall_time > 0.0
    assert gap is None


@NEED_PULP
def test_highs_top_k_respected():
    Q, q, var_map = make_qubo(seed=7)
    solutions, _, _ = solve("highs", Q, q, var_map, k_max=3, top_k=3,
                            time_limit=30)
    assert len(solutions) == 3


# ---------------------------------------------------------------------------
# CP-SAT (ortools — installed)
# ---------------------------------------------------------------------------

@NEED_ORTOOLS
def test_cp_sat_solutions_valid():
    Q, q, var_map = make_qubo()
    k_max, top_k = 3, 6
    solutions, wall_time, gap = solve("cp_sat", Q, q, var_map,
                                      k_max=k_max, top_k=top_k, time_limit=30)
    check_solutions(solutions, Q, q, var_map, k_max, top_k)
    assert wall_time > 0.0
    assert gap is None


@NEED_ORTOOLS
def test_cp_sat_top_k_respected():
    Q, q, var_map = make_qubo(seed=23)
    solutions, _, _ = solve("cp_sat", Q, q, var_map, k_max=3, top_k=3, time_limit=30)
    assert len(solutions) == 3


# ---------------------------------------------------------------------------
# SQAOD — simulated quantum annealing (optional package; skip when missing)
# ---------------------------------------------------------------------------

@NEED_SQAOD
def test_sqaod_solutions_valid():
    Q, q, var_map = make_qubo()
    k_max, top_k = 3, 6
    solutions, wall_time, gap = solve("sqaod", Q, q, var_map,
                                      k_max=k_max, top_k=top_k, time_limit=30)
    check_solutions(solutions, Q, q, var_map, k_max, top_k, enforce_kmax=False)
    assert gap is None


# ---------------------------------------------------------------------------
# SA / Tabu / SCIP — skip when the dependency is missing
# ---------------------------------------------------------------------------

@NEED_DIMOD
def test_sa_solutions_valid():
    Q, q, var_map = make_qubo()
    k_max, top_k = 3, 6
    solutions, wall_time, gap = solve("sa", Q, q, var_map,
                                      k_max=k_max, top_k=top_k, time_limit=5)
    check_solutions(solutions, Q, q, var_map, k_max, top_k, enforce_kmax=False)
    assert gap is None


@NEED_DWAVE
def test_tabu_solutions_valid():
    Q, q, var_map = make_qubo()
    k_max, top_k = 3, 6
    solutions, wall_time, gap = solve("tabu", Q, q, var_map,
                                      k_max=k_max, top_k=top_k, time_limit=5)
    check_solutions(solutions, Q, q, var_map, k_max, top_k, enforce_kmax=False)
    assert gap is None


@NEED_PYSCIPOPT
def test_scip_solutions_valid():
    Q, q, var_map = make_qubo()
    k_max, top_k = 3, 6
    solutions, wall_time, gap = solve("scip", Q, q, var_map,
                                      k_max=k_max, top_k=top_k, time_limit=30)
    check_solutions(solutions, Q, q, var_map, k_max, top_k)
    assert gap is None


@pytest.mark.skipif(HAS_DIMOD, reason="dimod is installed")
def test_sa_missing_dependency_raises_import_error():
    Q, q, var_map = make_qubo()
    with pytest.raises(ImportError, match="pip install"):
        solve("sa", Q, q, var_map, time_limit=1)


@pytest.mark.skipif(HAS_DWAVE, reason="dwave-samplers is installed")
def test_tabu_missing_dependency_raises_import_error():
    Q, q, var_map = make_qubo()
    with pytest.raises(ImportError, match="pip install"):
        solve("tabu", Q, q, var_map, time_limit=1)


@pytest.mark.skipif(HAS_PYSCIPOPT, reason="pyscipopt is installed")
def test_scip_missing_dependency_raises_import_error():
    Q, q, var_map = make_qubo()
    with pytest.raises(ImportError, match="pip install"):
        solve("scip", Q, q, var_map, top_k=1)


# ---------------------------------------------------------------------------
# GPU SA — skip without CUDA; clear RuntimeError when CUDA is missing
# ---------------------------------------------------------------------------

@NEED_CUDA
def test_gpu_sa_solutions_valid():
    Q, q, var_map = make_qubo()
    k_max, top_k = 3, 6
    solutions, wall_time, gap = solve("gpu_sa", Q, q, var_map,
                                      k_max=k_max, top_k=top_k, time_limit=5,
                                      num_reads=64, num_sweeps=64, batch_size=32)
    check_solutions(solutions, Q, q, var_map, k_max, top_k, enforce_kmax=False)
    assert gap is None


@pytest.mark.skipif(HAS_CUDA, reason="CUDA available")
def test_gpu_sa_requires_cuda():
    Q, q, var_map = make_qubo()
    if not _has("torch"):
        pytest.skip("torch not installed")
    with pytest.raises(RuntimeError, match="CUDA"):
        solve("gpu_sa", Q, q, var_map, time_limit=1)


# ---------------------------------------------------------------------------
# QAOA — identity test for the corrected QUBO->Ising mapping + qubit limit
# ---------------------------------------------------------------------------

def test_qaoa_identity_mapping():
    """E_Q(x) == H(1-2x) for the corrected mapping (tol 1e-10).

    Ported from qaoa-lcf/tests/test_identity_mapping.py (synthetic part).
    """
    rng = np.random.RandomState(7)
    for n in (3, 5, 8):
        M = rng.randn(n, n)
        Q = (M + M.T) / 2
        q = rng.randn(n)
        h, J, const = qubo_to_ising_fixed(Q, q)
        err = 0.0
        for _ in range(100):
            x = rng.randint(0, 2, n).astype(float)
            s = 1 - 2 * x
            e_q = float(x @ Q @ x + q @ x)
            err = max(err, abs(e_q - ising_energy(s, h, J, const)))
        assert err < 1e-10, f"fixed mapping failed identity for n={n}: {err:.2e}"


def test_qaoa_identity_mapping_on_synthetic_qubo():
    """Identity also holds on the hand-built solver QUBO."""
    Q, q, var_map = make_qubo()
    h, J, const = qubo_to_ising_fixed(Q, q)
    rng = np.random.RandomState(0)
    for _ in range(50):
        x = rng.randint(0, 2, Q.shape[0]).astype(float)
        s = 1 - 2 * x
        e_q = float(x @ Q @ x + q @ x)
        assert e_q == pytest.approx(ising_energy(s, h, J, const), abs=1e-10)


def test_qaoa_qubit_limit():
    """Problems with more than MAX_QUBITS (15) variables raise a clear error."""
    n = MAX_QUBITS + 1
    Q, q, var_map = make_qubo(seed=5, n=n)
    with pytest.raises(ValueError, match="15 qubits"):
        solve("qaoa", Q, q, var_map, top_k=1)


def test_qaoa_smoke():
    """Tiny QAOA run returns valid, energy-sorted solutions."""
    Q, q, var_map = make_qubo()
    k_max, top_k = 4, 4
    solutions, wall_time, gap = solve("qaoa", Q, q, var_map,
                                      k_max=k_max, top_k=top_k,
                                      p=1, max_iter=20, n_shots=256, seed=0)
    check_solutions(solutions, Q, q, var_map, k_max, top_k, enforce_kmax=False)
    assert gap is None
