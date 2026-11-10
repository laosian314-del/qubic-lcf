"""Tests for the opt-in numba/JIT SA and tabu kernels (qubic.solvers.sa/tabu).

Covers:
- default path (``use_jit=False``) keeps the dimod / dwave-tabu behavior and
  NEVER imports numba (subprocess assertion);
- the native kernels (``use_jit=True``) work WITHOUT dimod and satisfy the
  unified solver contract (return shape, exact energies, selection dedupe);
- exact parity between the pure-NumPy reference twin and the numba ``@njit``
  twin of each native kernel, with pre-generated random variates shared
  between both twins;
- kernel energy bookkeeping matches ``base.qubo_energy`` (also on
  non-symmetric Q);
- quality sanity on small seeded synthetic QUBOs (native vs brute force, and
  native vs the dimod/dwave default where installed).

The optional numba/JIT acceleration was NOT used in the article; article
results were obtained with the pure NumPy code paths.
"""
import importlib.util
import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from qubic import _accel
from qubic.solvers import base, sa, tabu


def _has(module_name):
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


HAS_DIMOD = _has("dimod")
HAS_DWAVE = _has("dwave.samplers")

_SRC = str(pathlib.Path(__file__).resolve().parents[1] / "src")


# ---------------------------------------------------------------------------
# Synthetic QUBO construction (hand-built, seeded — like tests/test_solvers.py)
# ---------------------------------------------------------------------------

def make_qubo(seed=1234, n=6, symmetric=True):
    """Seeded synthetic QUBO; var_map is the B1 scheme ('selection': 'explicit')."""
    rng = np.random.RandomState(seed)
    U = rng.uniform(-1.5, 2.0, size=(n, n))
    if symmetric:
        Q = np.triu(U) + np.triu(U, 1).T
    else:
        Q = U  # deliberately non-symmetric: energy is still x^T Q x + q^T x
    q = rng.uniform(-1.0, 1.0, size=n)
    var_map = {
        "scheme": "B1",
        "N": n,
        "M": n,
        "vars_per_component": 1,
        "selection": "explicit",
    }
    return Q, q, var_map


def check_contract(solutions, wall_time, gap, Q, q, var_map, top_k):
    """Unified solver contract for the (sample, energy) pairs output."""
    assert isinstance(solutions, list)
    assert gap is None
    assert wall_time > 0.0
    assert 0 < len(solutions) <= top_k
    M = Q.shape[0]
    energies = []
    selections = set()
    for item in solutions:
        assert isinstance(item, tuple) and len(item) == 2
        sample, energy = item
        assert isinstance(sample, np.ndarray)
        assert sample.shape == (M,)
        assert np.all(sample >= -1e-6) and np.all(sample <= 1.0 + 1e-6)
        assert np.allclose(sample, np.round(sample)), "sample must be 0/1"
        expected = float(sample @ Q @ sample + q @ sample)
        assert energy == pytest.approx(expected, rel=1e-9, abs=1e-9)
        selections.add(tuple(base.get_selection(sample, var_map)))
        energies.append(energy)
    assert energies == sorted(energies), "energies must be sorted ascending"
    # candidates are deduped by component selection
    assert len(selections) == len(solutions)


def brute_force(Q, q):
    """Exact minimum of x^T Q x + q^T x by enumeration (small M only)."""
    M = Q.shape[0]
    best = np.inf
    for mask in range(2 ** M):
        x = np.array([(mask >> i) & 1 for i in range(M)], dtype=float)
        e = float(x @ Q @ x + q @ x)
        if e < best:
            best = e
    return best


# ---------------------------------------------------------------------------
# Default path: no numba import, dimod/dwave behavior preserved
# ---------------------------------------------------------------------------

def test_default_path_does_not_import_numba():
    """use_jit=False must never import numba (checked in a clean subprocess)."""
    code = textwrap.dedent("""
        import sys
        sys.path.insert(0, @SRC@)
        import numpy as np
        from qubic.solvers import sa, tabu
        assert "numba" not in sys.modules, "numba imported at module import time"
        Q = np.array([[1.0, -0.5], [-0.5, 1.0]])
        q = np.array([-0.3, 0.2])
        var_map = {"scheme": "B1", "N": 2, "M": 2,
                   "vars_per_component": 1, "selection": "explicit"}
        for fn in (sa.solve_sa, tabu.solve_tabu):
            try:
                fn(Q, q, var_map, time_limit=0.5, top_k=1)  # use_jit=False default
            except ImportError:
                pass  # dimod / dwave-tabu missing is fine here
        assert "numba" not in sys.modules, "use_jit=False run imported numba"
        print("OK")
    """).replace("@SRC@", repr(_SRC))
    res = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr
    assert "OK" in res.stdout


@pytest.mark.skipif(not HAS_DIMOD, reason="dimod not installed")
def test_sa_default_path_dimod_contract():
    Q, q, var_map = make_qubo()
    pairs, wall_time = sa.solve_sa(Q, q, var_map, time_limit=2, top_k=4)
    assert wall_time > 0.0
    assert 0 < len(pairs) <= 4
    for sample, energy in pairs:
        assert sample.shape == (Q.shape[0],)
        assert energy == pytest.approx(float(sample @ Q @ sample + q @ sample),
                                       rel=1e-9, abs=1e-9)


@pytest.mark.skipif(not (HAS_DWAVE and HAS_DIMOD),
                    reason="dwave-samplers/dimod not installed")
def test_tabu_default_path_dwave_contract():
    Q, q, var_map = make_qubo()
    pairs, wall_time = tabu.solve_tabu(Q, q, var_map, time_limit=2, top_k=4)
    assert wall_time > 0.0
    assert 0 < len(pairs) <= 4
    for sample, energy in pairs:
        assert energy == pytest.approx(float(sample @ Q @ sample + q @ sample),
                                       rel=1e-9, abs=1e-9)


def test_default_path_still_raises_import_error_without_dimod():
    """Default path keeps its lazy dimod/dwave ImportError contract."""
    if HAS_DIMOD:
        pytest.skip("dimod installed")
    Q, q, var_map = make_qubo()
    with pytest.raises(ImportError, match="pip install"):
        sa.solve_sa(Q, q, var_map, time_limit=0.5, top_k=1)


# ---------------------------------------------------------------------------
# Kernel parity: pure-NumPy twin vs numba twin on shared random variates
# ---------------------------------------------------------------------------

def _sa_variates(rng, n_reads, M, n_sweeps):
    n_moves = n_sweeps * M
    x0 = (rng.random((n_reads, M)) < 0.5).astype(np.float64)
    flips = rng.integers(0, M, size=(n_reads, n_moves)).astype(np.int64)
    exp_noise = rng.exponential(1.0, size=(n_reads, n_moves))
    betas = 0.1 * (2.0 / 0.1) ** (np.arange(n_sweeps, dtype=np.float64)
                                 / max(1, n_sweeps - 1))
    return x0, flips, exp_noise, betas


def _tabu_variates(rng, n_reads, M, n_steps, span):
    x0 = (rng.random((n_reads, M)) < 0.5).astype(np.float64)
    tenure_jitter = rng.integers(0, span, size=(n_reads, n_steps)).astype(np.int64)
    return x0, tenure_jitter


@pytest.mark.parametrize("symmetric", [True, False])
def test_sa_kernel_np_jit_exact_parity(symmetric):
    nb = pytest.importorskip("numba")
    Q, q, var_map = make_qubo(seed=3, n=6, symmetric=symmetric)
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    q = np.ascontiguousarray(q, dtype=np.float64)
    rng = np.random.default_rng(42)
    x0, flips, exp_noise, betas = _sa_variates(rng, n_reads=5, M=6, n_sweeps=40)

    ref_states, ref_bests, ref_Es = sa._sa_kernel_np(Q, q, x0, flips,
                                                     exp_noise, betas)
    jit = sa._get_sa_kernel(nb)
    jit_states, jit_bests, jit_Es = jit(Q, q, x0, flips, exp_noise, betas)

    # trajectories must be identical bit-for-bit
    np.testing.assert_array_equal(ref_states, jit_states)
    np.testing.assert_array_equal(ref_bests, jit_bests)
    np.testing.assert_allclose(ref_Es, jit_Es, rtol=0, atol=0)

    # kernel energy bookkeeping matches the exact QUBO energy
    for r in range(x0.shape[0]):
        exact = base.qubo_energy(ref_bests[r], Q, q)
        np.testing.assert_allclose(ref_Es[r], exact, rtol=1e-12, atol=1e-8)


@pytest.mark.parametrize("symmetric", [True, False])
def test_tabu_kernel_np_jit_exact_parity(symmetric):
    nb = pytest.importorskip("numba")
    Q, q, var_map = make_qubo(seed=5, n=6, symmetric=symmetric)
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    q = np.ascontiguousarray(q, dtype=np.float64)
    rng = np.random.default_rng(43)
    n_steps, tenure_base, span = 60, 3, 3
    x0, tenure_jitter = _tabu_variates(rng, n_reads=5, M=6, n_steps=n_steps,
                                       span=span)

    ref_states, ref_bests, ref_Es = tabu._tabu_kernel_np(
        Q, q, x0, tenure_jitter, n_steps, tenure_base)
    jit = tabu._get_tabu_kernel(nb)
    jit_states, jit_bests, jit_Es = jit(Q, q, x0, tenure_jitter, n_steps,
                                        tenure_base)

    np.testing.assert_array_equal(ref_states, jit_states)
    np.testing.assert_array_equal(ref_bests, jit_bests)
    np.testing.assert_allclose(ref_Es, jit_Es, rtol=0, atol=0)

    for r in range(x0.shape[0]):
        exact = base.qubo_energy(ref_bests[r], Q, q)
        np.testing.assert_allclose(ref_Es[r], exact, rtol=1e-12, atol=1e-8)


# ---------------------------------------------------------------------------
# Native public path (use_jit=True): contract, no dimod required
# ---------------------------------------------------------------------------

def test_sa_native_solve_contract():
    pytest.importorskip("numba")
    Q, q, var_map = make_qubo(seed=7, n=8)
    solutions, wall_time, gap = sa.solve(Q, q, var_map, k_max=3, top_k=5,
                                         time_limit=10, use_jit=True, seed=0)
    check_contract(solutions, wall_time, gap, Q, q, var_map, top_k=5)


def test_tabu_native_solve_contract():
    pytest.importorskip("numba")
    Q, q, var_map = make_qubo(seed=9, n=8)
    solutions, wall_time, gap = tabu.solve(Q, q, var_map, k_max=3, top_k=5,
                                           time_limit=10, use_jit=True, seed=0)
    check_contract(solutions, wall_time, gap, Q, q, var_map, top_k=5)


def test_native_solve_via_registry():
    pytest.importorskip("numba")
    from qubic.solvers import solve as solve_any
    Q, q, var_map = make_qubo(seed=11, n=8)
    for name in ("sa", "tabu"):
        solutions, wall_time, gap = solve_any(name, Q, q, var_map, k_max=3,
                                              top_k=4, time_limit=10,
                                              use_jit=True, seed=1)
        check_contract(solutions, wall_time, gap, Q, q, var_map, top_k=4)


@pytest.mark.skipif(HAS_DIMOD, reason="dimod installed; no-dimod case covered")
def test_sa_native_works_without_dimod():
    pytest.importorskip("numba")
    Q, q, var_map = make_qubo(seed=13, n=6)
    solutions, wall_time = sa.solve_sa(Q, q, var_map, time_limit=5,
                                       top_k=3, use_jit=True, seed=2)
    assert 0 < len(solutions) <= 3
    assert wall_time > 0.0


@pytest.mark.skipif(HAS_DIMOD and HAS_DWAVE,
                    reason="dwave-samplers installed; no-dimod case covered")
def test_tabu_native_works_without_dwave():
    pytest.importorskip("numba")
    Q, q, var_map = make_qubo(seed=15, n=6)
    solutions, wall_time = tabu.solve_tabu(Q, q, var_map, time_limit=5,
                                           top_k=3, use_jit=True, seed=2)
    assert 0 < len(solutions) <= 3
    assert wall_time > 0.0


def test_use_jit_true_requires_numba(monkeypatch):
    """use_jit=True raises a helpful ImportError when numba is missing."""
    monkeypatch.setattr(_accel, "_checked", True)
    monkeypatch.setattr(_accel, "_numba", None)
    Q, q, var_map = make_qubo(seed=17, n=4)
    with pytest.raises(ImportError, match="numba"):
        sa.solve_sa(Q, q, var_map, time_limit=1, top_k=1, use_jit=True)
    with pytest.raises(ImportError, match="numba"):
        tabu.solve_tabu(Q, q, var_map, time_limit=1, top_k=1, use_jit=True)


def test_native_seed_reproducible():
    """Same seed -> identical native solution lists (deterministic variates)."""
    pytest.importorskip("numba")
    Q, q, var_map = make_qubo(seed=19, n=8)
    for fn in (sa.solve, tabu.solve):
        a, _, _ = fn(Q, q, var_map, k_max=3, top_k=3, time_limit=30,
                     use_jit=True, seed=7)
        b, _, _ = fn(Q, q, var_map, k_max=3, top_k=3, time_limit=30,
                     use_jit=True, seed=7)
        assert len(a) == len(b) == 3
        for (s1, e1), (s2, e2) in zip(a, b):
            np.testing.assert_array_equal(s1, s2)
            assert e1 == e2


# ---------------------------------------------------------------------------
# Quality sanity on small seeded synthetic QUBOs
# ---------------------------------------------------------------------------

def test_sa_native_hits_bruteforce_optimum():
    pytest.importorskip("numba")
    Q, q, var_map = make_qubo(seed=23, n=8)
    opt = brute_force(Q, q)
    solutions, _, _ = sa.solve(Q, q, var_map, top_k=8, time_limit=20,
                               use_jit=True, seed=3)
    native_min = min(e for _, e in solutions)
    assert native_min <= opt + 1e-6, (native_min, opt)


def test_tabu_native_hits_bruteforce_optimum():
    pytest.importorskip("numba")
    Q, q, var_map = make_qubo(seed=29, n=8)
    opt = brute_force(Q, q)
    solutions, _, _ = tabu.solve(Q, q, var_map, top_k=8, time_limit=20,
                                 use_jit=True, seed=3)
    native_min = min(e for _, e in solutions)
    assert native_min <= opt + 1e-6, (native_min, opt)


@pytest.mark.skipif(not HAS_DIMOD, reason="dimod not installed")
def test_sa_native_quality_vs_dimod_default():
    pytest.importorskip("numba")
    Q, q, var_map = make_qubo(seed=31, n=10)
    dimod_pairs, _ = sa.solve_sa(Q, q, var_map, time_limit=3, top_k=4)
    native_pairs, _ = sa.solve_sa(Q, q, var_map, time_limit=3, top_k=4,
                                  use_jit=True, seed=0)
    dimod_min = min(e for _, e in dimod_pairs)
    native_min = min(e for _, e in native_pairs)
    assert native_min <= dimod_min + max(1e-9, 0.05 * abs(dimod_min))


@pytest.mark.skipif(not (HAS_DWAVE and HAS_DIMOD),
                    reason="dwave-samplers/dimod not installed")
def test_tabu_native_quality_vs_dwave_default():
    pytest.importorskip("numba")
    Q, q, var_map = make_qubo(seed=37, n=10)
    dwave_pairs, _ = tabu.solve_tabu(Q, q, var_map, time_limit=3, top_k=4)
    native_pairs, _ = tabu.solve_tabu(Q, q, var_map, time_limit=3, top_k=4,
                                      use_jit=True, seed=0)
    dwave_min = min(e for _, e in dwave_pairs)
    native_min = min(e for _, e in native_pairs)
    assert native_min <= dwave_min + max(1e-9, 0.05 * abs(dwave_min))