"""Tests for the opt-in numba acceleration of the greedy forward selection family.

- the default path (``use_jit=False``) preserves the pre-JIT behavior of
  ``mp_bic`` (verbatim original MP loop) and ``omp_bic`` (same greedy OMP
  selections as sklearn ``OrthogonalMatchingPursuit`` per k);
- exact parity between the pure-NumPy reference kernels and their numba
  ``@njit`` twins: identical selected supports, allclose coefficients;
- the default path never imports numba (checked in a clean subprocess);
  ``use_jit=True`` raises a helpful ImportError when numba is missing.

The optional numba/JIT acceleration was NOT used in the article; article results were obtained with the pure NumPy code paths.
"""
import os
import pathlib
import subprocess
import sys
import warnings

import numpy as np
import pytest

from qubic import _accel
from qubic.alternative import (_omp_greedy_path, _omp_path_jit, _omp_path_np,
                               _mp_greedy_path, _mp_path_jit, _mp_path_np,
                               mp_bic, nnls_thresh_refit, omp_bic)
from qubic.pipeline import EPS_RSS, compute_bic

REQUIRED_KEYS = {'method', 'selected', 'concentrations', 'k_pred', 'bic', 'rss'}


def make_case(K=12, N=6, noise=1e-3, seed=0, true=(1, 3), conc=(0.6, 0.4)):
    """Synthetic LCF mixture: random non-negative dictionary + sparse y."""
    rng = np.random.default_rng(seed)
    A = rng.random((K, N))
    y = A[:, list(true)] @ np.asarray(conc, dtype=float)
    if noise:
        y = y + noise * rng.standard_normal(K)
    return A, y


# ------------------------------------------------------------------
# (a) default path preserves the pre-JIT behavior
# ------------------------------------------------------------------

def select_bic(method, A, y, cands):
    """The shared NNLS-threshold-BIC candidate selection (unchanged code)."""
    K = A.shape[0]
    best = None
    for cols in cands:
        support, rss, conc = nnls_thresh_refit(A, y, cols)
        if not support:
            continue
        bic = float(compute_bic(max(rss, EPS_RSS), len(support), K))
        if best is None or bic < best[3]:
            best = (support, conc, rss, bic)
    if best is None:
        return {'method': method, 'selected': [], 'concentrations': [],
                'k_pred': 0, 'bic': None, 'rss': float(np.sum(y ** 2))}
    support, conc, rss, bic = best
    return {'method': method, 'selected': [int(i) for i in support],
            'concentrations': [float(v) for v in np.asarray(conc).ravel()],
            'k_pred': len(support), 'bic': float(bic), 'rss': float(rss)}


def reference_mp_bic(A, y, k_max=15):
    """Verbatim pre-JIT mp_bic algorithm (MP loop + NNLS/BIC post-processing)."""
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    k_max_eff = min(k_max, A.shape[1])
    r = y.astype(float).copy()
    norms2 = np.sum(A ** 2, axis=0)
    if not (norms2 > 0).all():
        raise ValueError('zero column in A')
    norms = np.sqrt(norms2)
    chosen = []
    cands = []
    for _ in range(k_max_eff):
        ip = np.abs(A.T @ r) / norms
        if chosen:
            ip[np.asarray(chosen, dtype=int)] = -np.inf
        j = int(np.argmax(ip))
        chosen.append(j)
        r = r - (float(A[:, j] @ r) / norms2[j]) * A[:, j]
        cands.append(list(chosen))
    return select_bic('mp_bic', A, y, cands)


def reference_omp_bic(A, y, k_max=15):
    """Verbatim pre-JIT omp_bic algorithm (sklearn OMP per k + NNLS/BIC)."""
    from sklearn.linear_model import OrthogonalMatchingPursuit
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    cands = []
    for k in range(1, min(k_max, A.shape[1]) + 1):
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k, fit_intercept=False)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            omp.fit(A, y)
        cols = [int(i) for i in np.flatnonzero(omp.coef_)]
        if cols:
            cands.append(cols)
    return select_bic('omp_bic', A, y, cands)


def assert_results_equal(res0, res1):
    """Identical supports and allclose coefficients/metrics."""
    assert res0['method'] == res1['method']
    assert res0['selected'] == res1['selected'], 'selected supports differ'
    assert res0['k_pred'] == res1['k_pred']
    np.testing.assert_allclose(res0['concentrations'], res1['concentrations'])
    np.testing.assert_allclose([res0['rss']], [res1['rss']])
    if res0['bic'] is None:
        assert res1['bic'] is None
    else:
        np.testing.assert_allclose([res0['bic']], [res1['bic']])


@pytest.mark.parametrize('seed', [0, 1, 2, 3])
@pytest.mark.parametrize('k_max', [3, 15])
def test_mp_bic_default_matches_reference_loop(seed, k_max):
    A, y = make_case(seed=seed)
    assert_results_equal(mp_bic(A, y, k_max=k_max),
                         reference_mp_bic(A, y, k_max=k_max))


@pytest.mark.parametrize('seed', [0, 1, 2, 3])
@pytest.mark.parametrize('k_max', [3, 15])
def test_omp_bic_default_matches_sklearn(seed, k_max):
    A, y = make_case(seed=seed)
    assert_results_equal(omp_bic(A, y, k_max=k_max),
                         reference_omp_bic(A, y, k_max=k_max))


def test_result_contract_default():
    A, y = make_case()
    for fn, name in ((mp_bic, 'mp_bic'), (omp_bic, 'omp_bic')):
        res = fn(A, y)  # use_jit defaults to False
        assert set(res) == REQUIRED_KEYS
        assert res['method'] == name
        assert res['k_pred'] == len(res['selected']) == len(res['concentrations'])
        assert res['k_pred'] >= 1
        assert sorted(res['selected']) == res['selected']
        assert np.isfinite(res['rss'])
        if res['bic'] is not None:
            assert np.isfinite(res['bic'])


def test_mp_bic_rejects_zero_column():
    A, y = make_case()
    A[:, 2] = 0.0
    with pytest.raises(ValueError, match='zero column'):
        mp_bic(A, y)
    with pytest.raises(ValueError, match='zero column'):
        mp_bic(A, y, use_jit=False)


# ------------------------------------------------------------------
# (b) exact parity between the pure-NumPy and numba twins
# ------------------------------------------------------------------

@pytest.mark.parametrize('K,N,k_max,seed', [(12, 6, 6, 0), (40, 15, 15, 1),
                                            (64, 32, 10, 2)])
def test_mp_kernel_twin_parity(K, N, k_max, seed):
    pytest.importorskip('numba')
    A, y = make_case(K, N, noise=1e-2, seed=seed)
    k = min(k_max, N)
    order_np, alpha_np, norms2_np = _mp_path_np(A, y, k)
    order_jit, alpha_jit, norms2_jit = _mp_path_jit(A, y, k)
    np.testing.assert_array_equal(order_np, order_jit)  # identical selection
    np.testing.assert_allclose(alpha_np, alpha_jit, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(norms2_np, norms2_jit, rtol=1e-10, atol=0.0)


@pytest.mark.parametrize('K,N,k_max,seed', [(12, 6, 6, 0), (40, 15, 15, 1),
                                            (64, 32, 10, 2)])
def test_omp_kernel_twin_parity(K, N, k_max, seed):
    pytest.importorskip('numba')
    A, y = make_case(K, N, noise=1e-2, seed=seed)
    k = min(k_max, N)
    order_np, coefs_np, n_np = _omp_path_np(A, y, k)
    order_jit, coefs_jit, n_jit = _omp_path_jit(A, y, k)
    assert n_np == n_jit, 'twins disagree on the number of selected atoms'
    np.testing.assert_array_equal(order_np[:n_np], order_jit[:n_jit])
    np.testing.assert_allclose(coefs_np, coefs_jit, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize('K,N,k_max,seed', [(12, 6, 6, 0), (40, 15, 15, 1),
                                            (64, 32, 10, 2)])
def test_mp_bic_result_parity(K, N, k_max, seed):
    pytest.importorskip('numba')
    A, y = make_case(K, N, noise=1e-2, seed=seed)
    assert_results_equal(mp_bic(A, y, k_max=k_max),
                         mp_bic(A, y, k_max=k_max, use_jit=True))


@pytest.mark.parametrize('K,N,k_max,seed', [(12, 6, 6, 0), (40, 15, 15, 1),
                                            (64, 32, 10, 2)])
def test_omp_bic_result_parity(K, N, k_max, seed):
    pytest.importorskip('numba')
    A, y = make_case(K, N, noise=1e-2, seed=seed)
    assert_results_equal(omp_bic(A, y, k_max=k_max),
                         omp_bic(A, y, k_max=k_max, use_jit=True))


def test_use_jit_result_contract():
    pytest.importorskip('numba')
    A, y = make_case()
    for fn, name in ((mp_bic, 'mp_bic'), (omp_bic, 'omp_bic')):
        res = fn(A, y, k_max=4, use_jit=True)
        assert set(res) == REQUIRED_KEYS
        assert res['method'] == name
        assert sorted(res['selected']) == res['selected']
        assert res['k_pred'] == len(res['selected']) == len(res['concentrations'])
        assert np.isfinite(res['rss'])


def test_greedy_path_dispatch_matches_kernels():
    """The shared helpers select the twin named by ``use_jit``."""
    pytest.importorskip('numba')
    A, y = make_case(seed=4)
    for path_np, path_jit, dispatch in (
            (_mp_path_np, _mp_path_jit, _mp_greedy_path),
            (_omp_path_np, _omp_path_jit, _omp_greedy_path)):
        out_np = dispatch(A, y, 5)
        out_jit = dispatch(A, y, 5, use_jit=True)
        ref_np = path_np(A, y, 5)
        ref_jit = path_jit(A, y, 5)
        for got, ref in ((out_np, ref_np), (out_jit, ref_jit)):
            for a, b in zip(got, ref):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# ------------------------------------------------------------------
# (c) the default path must not import numba
# ------------------------------------------------------------------

def test_default_path_does_not_import_numba():
    src = str(pathlib.Path(__file__).resolve().parents[1] / 'src')
    code = (
        "import sys\n"
        "import numpy as np\n"
        "from qubic.alternative import mp_bic, omp_bic, run_all_alternative\n"
        "rng = np.random.default_rng(0)\n"
        "A = rng.normal(size=(8, 4))\n"
        "y = rng.normal(size=8)\n"
        "mp_bic(A, y, k_max=3)\n"
        "mp_bic(A, y, k_max=3, use_jit=False)\n"
        "omp_bic(A, y, k_max=3)\n"
        "omp_bic(A, y, k_max=3, use_jit=False)\n"
        "run_all_alternative(A, y, k_max=3)\n"
        "assert 'numba' not in sys.modules, "
        "'numba was imported on the default path'\n"
        "print('NO_NUMBA_OK')\n"
    )
    env = dict(os.environ)
    env['PYTHONPATH'] = src + os.pathsep + env.get('PYTHONPATH', '')
    proc = subprocess.run([sys.executable, '-c', code], capture_output=True,
                          text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    assert 'NO_NUMBA_OK' in proc.stdout


# ------------------------------------------------------------------
# (d) helpful error when numba is missing
# ------------------------------------------------------------------

def test_use_jit_true_requires_numba(monkeypatch):
    monkeypatch.setattr(_accel, 'get_numba', lambda: None)
    A, y = make_case()
    with pytest.raises(ImportError, match='use_jit=True'):
        mp_bic(A, y, use_jit=True)
    with pytest.raises(ImportError, match='use_jit=True'):
        omp_bic(A, y, use_jit=True)
    with pytest.raises(ImportError, match='use_jit=True'):
        _mp_path_jit(A, y, 2)
    with pytest.raises(ImportError, match='use_jit=True'):
        _omp_path_jit(A, y, 2)


# ------------------------------------------------------------------
# (e) greedy selection semantics (stopping rules) unchanged
# ------------------------------------------------------------------

def test_omp_premature_stop_cuts_path_short():
    """Duplicate atoms trigger the linear-dependence premature stop."""
    A, y = make_case(K=10, N=4, noise=0.0, seed=1, true=(0, 2))
    A[:, 3] = A[:, 1]  # exact duplicate column
    order, _coefs, n_active = _omp_path_np(A, y, 4)
    assert n_active < 4, 'duplicate columns must stop the greedy path early'
    assert len(set(int(i) for i in order[:n_active])) == n_active
    res = omp_bic(A, y, k_max=4)
    assert set(res) == REQUIRED_KEYS
    assert res['k_pred'] <= n_active
    pytest.importorskip('numba')
    order_j, _coefs_j, n_j = _omp_path_jit(A, y, 4)
    assert n_j == n_active
    np.testing.assert_array_equal(order[:n_active], order_j[:n_j])


def test_k_max_larger_than_n_and_zero_edge_cases():
    A, y = make_case(K=10, N=4, seed=2)
    for fn in (mp_bic, omp_bic):
        res_full = fn(A, y, k_max=50)  # clamped to N
        res_n = fn(A, y, k_max=4)
        assert_results_equal(res_full, res_n)
        res_empty = fn(A, y, k_max=0)
        assert set(res_empty) == REQUIRED_KEYS
        assert res_empty['bic'] is None and res_empty['k_pred'] == 0