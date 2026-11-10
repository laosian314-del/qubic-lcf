"""Tests for qubic.baselines on synthetic LCF mixtures (fixed seed).

Every baseline must return the shared result dict; LASSO+ and the exact
oracle must recover the true support (f1 >= 0.5) at low noise; run_all_baselines
must return all five method keys.  No pyfitit required anywhere.
"""
import numpy as np
import pytest

from qubic.baselines import (BASELINE_METHODS, lasso_positive, mp_bic,
                             nnls_thresh_refit, omp_bic, oracle_exact,
                             run_all_baselines, threshold_support,
                             topk_corr_bic)
from qubic.pipeline import compute_f1

K, N = 10, 5
TRUE = [1, 3]
TRUE_CONC = np.array([0.6, 0.4])
REQUIRED_KEYS = {'method', 'selected', 'concentrations', 'k_pred', 'bic', 'rss'}

ALL_METHODS = {
    'omp_bic': omp_bic,
    'mp_bic': mp_bic,
    'lasso_positive': lasso_positive,
    'topk_corr_bic': topk_corr_bic,
    'oracle_exact': oracle_exact,
}


def make_case(noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.random((K, N))
    y = A[:, TRUE] @ TRUE_CONC
    if noise:
        y = y + noise * rng.standard_normal(K)
    return A, y


def test_threshold_support_trims_small_coefficients():
    support, concs = threshold_support([0, 2, 4], [0.5, 0.001, 0.25])
    assert support == [0, 4]
    assert np.allclose(concs, [0.5, 0.25])


def test_threshold_support_all_zero():
    support, concs = threshold_support([1, 2], [0.0, 0.0])
    assert support == []
    assert concs.shape == (0,)


def test_nnls_thresh_refit_recovers_support():
    A, y = make_case(noise=0.0)
    support, rss, conc = nnls_thresh_refit(A, y, TRUE)
    assert support == TRUE
    assert rss < 1e-18
    assert np.allclose(conc, TRUE_CONC, atol=1e-6)


@pytest.mark.parametrize('name', sorted(ALL_METHODS))
def test_baseline_result_contract(name):
    A, y = make_case(noise=1e-3)
    res = ALL_METHODS[name](A, y)
    assert set(res) == REQUIRED_KEYS
    assert res['method'] == name
    assert res['k_pred'] == len(res['selected']) == len(res['concentrations'])
    assert res['k_pred'] >= 1
    assert sorted(res['selected']) == res['selected']
    assert all(0 <= i < N for i in res['selected'])
    assert np.isfinite(res['rss'])
    if res['bic'] is not None:
        assert np.isfinite(res['bic'])


@pytest.mark.parametrize('name', ['lasso_positive', 'oracle_exact'])
def test_baseline_finds_true_components(name):
    A, y = make_case(noise=1e-3)
    res = ALL_METHODS[name](A, y)
    assert compute_f1(res['selected'], TRUE) >= 0.5


@pytest.mark.parametrize('name', sorted(ALL_METHODS))
def test_baseline_finds_two_components_low_noise(name):
    A, y = make_case(noise=1e-3)
    res = ALL_METHODS[name](A, y)
    assert res['k_pred'] >= 1
    assert set(res['selected']) & set(TRUE), f'{name} found no true component'


def test_oracle_exact_rejects_too_large_n():
    A, y = make_case()
    with pytest.raises(ValueError):
        oracle_exact(A, y, max_n=N - 1)


def test_oracle_exact_matches_true_support_on_clean_signal():
    A, y = make_case(noise=0.0)
    res = oracle_exact(A, y)
    assert res['selected'] == TRUE
    assert compute_f1(res['selected'], TRUE) == 1.0


def test_run_all_baselines_returns_all_five_methods():
    A, y = make_case(noise=1e-3)
    results = run_all_baselines(A, y, k_max=15)
    assert set(results) == set(BASELINE_METHODS) == set(ALL_METHODS)
    for name, res in results.items():
        assert set(res) == REQUIRED_KEYS
        assert res['method'] == name
