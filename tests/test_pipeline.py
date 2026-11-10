"""Tests for qubic.pipeline on synthetic LCF mixtures (fixed seed).

Covers nnls_refine / compute_bic / compute_f1 unit behaviour and the
run_pipeline contract for a clean and a noisy 2-component mixture with
solvers 'milp' and 'cp_sat', plus a pyfitit_api smoke test that must work
without pyfitit installed.
"""
import numpy as np
import pytest

from qubic.pipeline import compute_bic, compute_f1, nnls_refine, run_pipeline
from qubic.solvers import available_solvers

K, N = 10, 5
TRUE = [1, 3]
TRUE_CONC = np.array([0.6, 0.4])


def make_case(noise=0.0, seed=0):
    """A (10x5) reference library and a 2-component mixture target."""
    rng = np.random.default_rng(seed)
    A = rng.random((K, N))
    y = A[:, TRUE] @ TRUE_CONC
    if noise:
        y = y + noise * rng.standard_normal(K)
    return A, y


def test_compute_bic_formula():
    rss, k, K_ = 2.5, 3, 10
    expected = K_ * np.log(rss / K_) + k * np.log(K_)
    assert compute_bic(rss, k, K_) == pytest.approx(expected)


def test_compute_bic_degenerate():
    assert compute_bic(0.0, 2, 10) == 1e10
    assert compute_bic(-1.0, 2, 10) == 1e10
    assert compute_bic(1.0, 0, 10) == 1e10


def test_compute_f1_sets():
    assert compute_f1([1, 3], [3, 1]) == 1.0
    assert compute_f1([1], [1, 2]) == pytest.approx(2 / 3)
    assert compute_f1([0], [1]) == 0.0
    assert compute_f1([], [1]) == 0.0
    assert compute_f1([1], []) == 0.0


def test_nnls_refine_recovers_clean_mixture():
    A, y = make_case(noise=0.0)
    conc, rss = nnls_refine(A, y, TRUE)
    assert conc.shape == (2,)
    assert np.allclose(conc, TRUE_CONC, atol=1e-6)
    assert conc.sum() == pytest.approx(1.0)
    assert rss < 1e-18


def test_nnls_refine_empty_support():
    A, y = make_case()
    conc, rss = nnls_refine(A, y, [])
    assert conc.shape == (0,)
    assert rss == 1e10


def test_run_pipeline_clean_milp():
    """Clean 2-component signal: f1=1.0 / k_pred=2 for the pools that contain
    the true support.

    Pool semantics are the original ones: candidates sorted by QUBO energy,
    min-BIC inside each top-tk pool.  With the default ``lam_sparse`` the
    B1 energy favours singletons of this synthetic target, so the top-1 pool
    alone may miss the pair (see the lam_sparse variant test below).
    """
    A, y = make_case(noise=0.0)
    top_k_values = (1, 10, 20)
    out = run_pipeline(A, y, k_true=2, gt_indices=TRUE,
                       scheme='B1', solver='milp',
                       top_k_values=top_k_values)
    assert set(out) == {'results', 'wall_time', 'n_solutions'}
    assert out['n_solutions'] >= 1
    assert out['wall_time'] > 0
    assert set(out['results']) == set(top_k_values)
    for tk, res in out['results'].items():
        assert set(res) == {'f1', 'k_pred', 'k_true', 'bic', 'rss',
                            'selected', 'concentrations', 'n_candidates'}
        assert res['k_true'] == 2
        assert len(res['concentrations']) == res['k_pred']
        assert np.isfinite(res['bic'])
        assert np.isfinite(res['rss'])
        assert res['n_candidates'] >= 1
        assert 0.0 <= res['f1'] <= 1.0
    for tk in (10, 20):
        res = out['results'][tk]
        assert res['f1'] == 1.0
        assert res['k_pred'] == 2
        assert sorted(res['selected']) == TRUE


def test_run_pipeline_clean_milp_calibrated_prior():
    """With a dominant sparsity prior even the top-1 pool recovers the pair."""
    A, y = make_case(noise=0.0)
    out = run_pipeline(A, y, k_true=2, gt_indices=TRUE,
                       scheme='B1', solver='milp', lam_sparse=5.0)
    for res in out['results'].values():
        assert res['f1'] == 1.0
        assert res['k_pred'] == 2


@pytest.mark.skipif('cp_sat' not in available_solvers(),
                    reason='ortools (cp_sat) not available')
def test_run_pipeline_clean_cp_sat():
    A, y = make_case(noise=0.0)
    out = run_pipeline(A, y, k_true=2, gt_indices=TRUE,
                       scheme='B1', solver='cp_sat')
    res = out['results'][20]
    assert res['f1'] == 1.0
    assert res['k_pred'] == 2
    assert sorted(res['selected']) == TRUE


def test_run_pipeline_noisy():
    A, y = make_case(noise=1e-3)
    out = run_pipeline(A, y, k_true=2, gt_indices=TRUE,
                       scheme='B1', solver='milp')
    res = out['results'][20]
    assert res['f1'] >= 0.5
    assert res['k_pred'] >= 1
    assert np.isfinite(res['bic'])


def test_run_pipeline_without_ground_truth():
    A, y = make_case(noise=0.0)
    out = run_pipeline(A, y, scheme='B1', solver='milp')
    for res in out['results'].values():
        assert res['f1'] is None
        assert res['k_true'] is None


def test_tryallmixtures_smoke_qubo_and_enumerate():
    # Must work without pyfitit installed (no pyfitit import anywhere).
    from qubic.pyfitit_api import tryAllMixtures, tryallmixtures

    A, y = make_case(noise=0.0)
    names = ['a', 'b', 'c', 'd', 'e']

    out = tryallmixtures(A, y, componentCount=2, componentNames=names,
                         scheme='B1', solver='milp')
    assert out['selected'] == TRUE
    assert out['component_names'] == ['b', 'd']
    assert out['k_pred'] == 2
    assert np.allclose(out['concentrations'], TRUE_CONC, atol=1e-6)
    assert np.isfinite(out['bic']) and np.isfinite(out['rss'])
    assert 0.0 <= out['r_factor'] <= 1.0
    assert out['candidates']

    out2 = tryAllMixtures(A, y, componentCount=2,
                          mixtureTrysCount='all combinations of singles',
                          method='enumerate')
    assert out2['selected'] == TRUE
    assert out2['method'] == 'enumerate'
