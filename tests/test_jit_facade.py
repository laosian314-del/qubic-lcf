"""Facade-level tests of the opt-in ``use_jit`` run flag.

The optional numba/JIT acceleration was NOT used in the article; article
results were obtained with the pure NumPy code paths.

Here ``task.solve(solver, use_jit=True)`` is checked end-to-end through the
public :class:`qubic.Problem` / :class:`qubic.Solver` facade: results must be
identical to the default path, and the default path must never import numba.
"""
import subprocess
import sys

import numpy as np
import pytest

import qubic


def _problem(seed=0, m=60, n=10, k=2, noise=0.01):
    rng = np.random.default_rng(seed)
    A = rng.uniform(0.1, 1.0, size=(m, n))
    true_c = np.zeros(n)
    idx = sorted({n // 3, 2 * n // 3})
    true_c[idx] = [0.6, 0.4][:len(idx)]
    y = A @ true_c + noise * rng.standard_normal(m)
    return qubic.Problem(A, y, k_max=k, tau=0.01), true_c


def test_omp_use_jit_parity():
    task, _ = _problem()
    r0 = task.solve(qubic.OMP(), use_jit=False)
    r1 = task.solve(qubic.OMP(), use_jit=True)
    assert r1.selected == r0.selected
    np.testing.assert_allclose(r1.concentrations, r0.concentrations)
    np.testing.assert_allclose(r1.rss, r0.rss)


def test_mp_use_jit_parity():
    task, _ = _problem(seed=1)
    r0 = task.solve(qubic.MP(), use_jit=False)
    r1 = task.solve(qubic.MP(), use_jit=True)
    assert r1.selected == r0.selected


def test_sa_native_kernel_without_dimod():
    # use_jit=True switches SA to the native numba kernel, which must work
    # without the optional dimod package.
    task, _ = _problem(seed=2)
    res = task.solve(qubic.SA(restarts=4, num_sweeps=200), use_jit=True)
    assert isinstance(res.selected, tuple) and len(res.selected) >= 1
    assert res.rss >= 0.0


def test_tabu_native_kernel_without_dimod():
    task, _ = _problem(seed=3)
    res = task.solve(qubic.Tabu(), use_jit=True)
    assert isinstance(res.selected, tuple) and len(res.selected) >= 1


def test_milp_use_jit_parity():
    pytest.importorskip("pulp")
    task, _ = _problem(seed=4)
    r0 = task.solve(qubic.MILP(time_limit=10.0), use_jit=False)
    r1 = task.solve(qubic.MILP(time_limit=10.0), use_jit=True)
    assert r1.selected == r0.selected


def test_use_jit_qaoa_parity():
    pytest.importorskip("scipy")
    # linked_B2 uses 3 binary variables per component and QAOA is capped at
    # MAX_QUBITS=15 -> at most 5 components here.
    task, _ = _problem(seed=5, n=5)
    r0 = task.solve(qubic.QAOA(p=1, max_iter=5, seed=0), use_jit=False)
    r1 = task.solve(qubic.QAOA(p=1, max_iter=5, seed=0), use_jit=True)
    assert r1.selected == r0.selected
    np.testing.assert_allclose(r1.rss, r0.rss)


def _subprocess_env():
    import os
    from pathlib import Path

    env = dict(os.environ)
    src = str(Path(qubic.__file__).resolve().parents[1])
    env['PYTHONPATH'] = src + os.pathsep + env.get('PYTHONPATH', '')
    return env


def test_default_run_never_imports_numba():
    code = (
        "import sys, numpy as np\n"
        "import qubic\n"
        "rng = np.random.default_rng(0)\n"
        "A = rng.uniform(0.1, 1.0, size=(60, 10))\n"
        "y = A @ np.array([0., 0., 0.6, 0., 0., 0., 0., 0.4, 0., 0.])\n"
        "task = qubic.Problem(A, y, k_max=2, tau=0.01)\n"
        "task.solve(qubic.OMP(), use_jit=False)\n"
        "task.solve(qubic.MP(), use_jit=False)\n"
        "assert not any(m.split('.')[0] == 'numba' for m in sys.modules), "
        "'numba was imported on the default path'\n"
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=300,
                          env=_subprocess_env())
    assert proc.returncode == 0, proc.stderr


def test_use_jit_missing_numba_helps(monkeypatch):
    from qubic import _accel
    monkeypatch.setattr(_accel, "get_numba", lambda: None)
    task, _ = _problem()
    with pytest.raises(ImportError, match="use_jit=True"):
        task.solve(qubic.OMP(), use_jit=True)