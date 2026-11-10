"""Parity and contract tests for the opt-in batched numba NNLS refinement.

Covers the batched projected-gradient NNLS kernel (``nnls_pg_batch``) and its
pure-NumPy twin against ``scipy.optimize.nnls`` on well-conditioned seeded
problems, the ``use_jit`` contract of the public refinement entry points
(``nnls_refine`` / ``refine_candidate`` / ``refine_candidates`` /
``full_library_candidate`` / ``run_pipeline``), the unchanged ``tau`` cutoff
and BIC semantics, and the guarantee that the default paths never import
numba.

The optional numba/JIT acceleration was NOT used in the article; article results were obtained with the pure NumPy code paths.
"""
import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from scipy.optimize import nnls as scipy_nnls

from qubic.pipeline import (EPS_RSS, RIDGE, compute_bic, full_library_candidate,
                            nnls_pg_batch, nnls_refine, refine_candidate,
                            refine_candidates)

M_PTS, N_LIB, K_SUP = 200, 50, 3
TAU = 0.01


def make_problem(seed=7, n_cand=25, k=K_SUP, max_cond=25.0):
    """Well-conditioned seeded LCF problem (full-rank candidate supports).

    The library is standard-normal, so random ``k``-column submatrices are
    comfortably conditioned; supports whose augmented design matrix
    (library rows + the ridge row) is worse than ``max_cond`` are
    resampled.  Keeping the problems well-conditioned is what makes the
    projected-gradient agreement tolerance below meaningful.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M_PTS, N_LIB))
    true_sel = [2, 11, 37]
    true_conc = np.array([0.5, 0.3, 0.2])
    target = A[:, true_sel] @ true_conc + 1e-2 * rng.standard_normal(M_PTS)
    cands = []
    while len(cands) < n_cand:
        sel = tuple(sorted(int(i) for i in
                           rng.choice(N_LIB, size=k, replace=False)))
        if sel in [tuple(c) for c in cands]:
            continue
        M = np.vstack([A[:, list(sel)], np.full(k, RIDGE)])
        if np.linalg.cond(M) <= max_cond:
            cands.append(list(sel))
    return A, target, cands


def aug_mats(A, target, cands):
    mats, rhs = [], []
    for sel in cands:
        A_sel = A[:, sel]
        mats.append(np.vstack([A_sel, np.full(len(sel), RIDGE)]))
        rhs.append(np.hstack([target, [RIDGE]]))
    return mats, rhs


def test_default_paths_never_import_numba():
    """A use_jit=False run must not import numba (fresh subprocess)."""
    src = str(pathlib.Path(__file__).resolve().parents[1] / 'src')
    code = textwrap.dedent(f'''
        import sys
        sys.path.insert(0, {src!r})
        import numpy as np
        import qubic.pipeline as p

        rng = np.random.default_rng(0)
        A = rng.standard_normal((30, 10))
        y = A[:, [1, 4]] @ np.array([0.6, 0.4])
        y = y + 1e-2 * rng.standard_normal(30)

        p.nnls_refine(A, y, [1, 4])
        p.refine_candidate(A, y, [1, 4], 0.01)
        p.refine_candidates(A, y, [[1, 4], [2, 5], [3], []], 0.01)
        p.full_library_candidate(A, y, 0.01)
        p.nnls_pg_batch(
            [np.vstack([A[:, [1, 4]], [0.1, 0.1]]),
             np.vstack([A[:, [2, 5]], [0.1, 0.1]])],
            [np.hstack([y, [0.1]]), np.hstack([y, [0.1]])],
        )
        assert "numba" not in sys.modules, "numba imported on the default path"
        print("NO_NUMBA_OK")
    ''')
    proc = subprocess.run([sys.executable, '-c', code],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert 'NO_NUMBA_OK' in proc.stdout


def test_projected_gradient_numpy_matches_scipy_nnls():
    """The pure-NumPy projected-gradient twin must match scipy.optimize.nnls.

    Tolerances: the twin stops at a KKT residual <= ``PG_TOL * scale`` on
    problems with cond(M) <= 25, which bounds the concentration error far
    below 1e-8; rtol=1e-6 leaves ~2 orders of headroom while staying tight.
    RSS is stationary at the (interior) optimum, so its agreement is second
    order in the concentration error; rtol=1e-8 is safe on these noisy
    targets (every RSS is >= ~1e-2).  BIC inherits the RSS agreement, and
    must match to ~1e-8 relative as well.
    """
    A, target, cands = make_problem(seed=11)
    mats, rhs = aug_mats(A, target, cands)
    sols = nnls_pg_batch(mats, rhs)  # default path: pure-NumPy twin
    assert len(sols) == len(cands)
    for sel, M, d, x in zip(cands, mats, rhs, sols):
        c_ref, _ = scipy_nnls(M, d)
        np.testing.assert_allclose(x, c_ref, rtol=1e-6, atol=1e-8,
                                   err_msg=f'support {sel}')
        # Same normalization / RSS / BIC pipeline as nnls_refine.
        conc_s, rss_s = nnls_refine(A, target, sel)
        conc_p = x / x.sum() if x.sum() > 0 else x
        rss_p = float(np.sum((target - A[:, sel] @ conc_p) ** 2))
        np.testing.assert_allclose(conc_p, conc_s, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(rss_p, rss_s, rtol=1e-8)
        bic_p = compute_bic(max(rss_p, EPS_RSS), len(sel), len(target))
        bic_s = compute_bic(max(rss_s, EPS_RSS), len(sel), len(target))
        np.testing.assert_allclose(bic_p, bic_s, rtol=1e-8)


def test_projected_gradient_jit_matches_numpy_twin():
    pytest.importorskip('numba')
    A, target, cands = make_problem(seed=13)
    mats, rhs = aug_mats(A, target, cands)
    sols_np = nnls_pg_batch(mats, rhs, use_jit=False)
    sols_jit = nnls_pg_batch(mats, rhs, use_jit=True)  # one batched njit call
    for x_np, x_jit in zip(sols_np, sols_jit):
        np.testing.assert_allclose(x_jit, x_np, rtol=1e-6, atol=1e-8)


def test_projected_gradient_jit_matches_scipy_nnls():
    pytest.importorskip('numba')
    A, target, cands = make_problem(seed=15)
    mats, rhs = aug_mats(A, target, cands)
    sols = nnls_pg_batch(mats, rhs, use_jit=True)
    for sel, M, d, x in zip(cands, mats, rhs, sols):
        c_ref, _ = scipy_nnls(M, d)
        np.testing.assert_allclose(x, c_ref, rtol=1e-6, atol=1e-8,
                                   err_msg=f'support {sel}')


def test_refine_candidate_use_jit_matches_default():
    pytest.importorskip('numba')
    A, target, cands = make_problem(seed=17, n_cand=10)
    for sel in cands:
        ref = refine_candidate(A, target, sel, TAU)
        got = refine_candidate(A, target, sel, TAU, use_jit=True)
        assert got[0] == ref[0]  # trimmed support identical
        np.testing.assert_allclose(got[1], ref[1], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(got[2], ref[2], rtol=1e-8)  # rss
        np.testing.assert_allclose(got[3], ref[3], rtol=1e-8)  # bic


def test_refine_candidates_default_is_the_per_candidate_loop():
    """The batched default path is bitwise the refine_candidate loop."""
    A, target, cands = make_problem(seed=19, n_cand=10)
    out = refine_candidates(A, target, cands, tau=TAU)
    assert len(out) == len(cands)
    for sel, got in zip(cands, out):
        ref = refine_candidate(A, target, sel, TAU)
        assert got[0] == ref[0]
        np.testing.assert_array_equal(got[1], ref[1])
        assert got[2] == ref[2]
        assert got[3] == ref[3]


def test_refine_candidates_use_jit_matches_default():
    pytest.importorskip('numba')
    A, target, cands = make_problem(seed=21)
    out_def = refine_candidates(A, target, cands, tau=TAU)
    out_jit = refine_candidates(A, target, cands, tau=TAU, use_jit=True)
    assert len(out_jit) == len(out_def)
    for ref, got in zip(out_def, out_jit):
        assert got[0] == ref[0]  # trimmed support identical
        np.testing.assert_allclose(got[1], ref[1], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(got[2], ref[2], rtol=1e-8)  # rss
        np.testing.assert_allclose(got[3], ref[3], rtol=1e-8)  # bic


def test_tau_cutoff_and_bic_semantics_unchanged():
    """tau trim is strict '>', BIC uses the trimmed size and the EPS_RSS floor."""
    A, target, cands = make_problem(seed=23, n_cand=8)
    for use_jit in (False, True):
        if use_jit:
            pytest.importorskip('numba')
        out = refine_candidates(A, target, cands, tau=TAU, use_jit=use_jit)
        for sel, got in zip(cands, out):
            conc, rss = nnls_refine(A, target, sel, use_jit=use_jit)
            exp_sel = [s for i, s in enumerate(sel) if conc[i] > TAU]
            exp_conc = conc[conc > TAU]
            exp_bic = compute_bic(max(rss, EPS_RSS), len(exp_sel),
                                  len(target))
            assert got[0] == exp_sel
            np.testing.assert_array_equal(got[1], exp_conc)
            assert got[2] == rss
            assert got[3] == exp_bic
    # Empty supports stay None in both backends; the eps floor keeps exact
    # fits finite instead of the compute_bic degenerate 1e10.
    mixed = refine_candidates(A, target, [[], cands[0]], tau=TAU)
    assert mixed[0] is None
    assert mixed[1] is not None
    clean_sel = [2, 11, 37]
    clean = A[:, clean_sel] @ np.array([0.5, 0.3, 0.2])
    res = refine_candidate(A, clean, clean_sel, TAU)
    assert res[2] < 1e-18 and np.isfinite(res[3]) and res[3] < 1e9


def test_full_library_candidate_use_jit_matches_default():
    pytest.importorskip('numba')
    A, target, _ = make_problem(seed=25, n_cand=1)
    ref = full_library_candidate(A, target, TAU)
    got = full_library_candidate(A, target, TAU, use_jit=True)
    assert got[0] == ref[0]
    np.testing.assert_allclose(got[1], ref[1], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(got[2], ref[2], rtol=1e-8)
    np.testing.assert_allclose(got[3], ref[3], rtol=1e-8)


def test_use_jit_without_numba_raises_helpful_error(monkeypatch):
    import qubic._accel as accel
    monkeypatch.setattr(accel, 'get_numba', lambda: None)
    A, target, cands = make_problem(seed=27, n_cand=3)
    with pytest.raises(ImportError, match='numba'):
        nnls_pg_batch([np.eye(2)], [np.ones(2)], use_jit=True)
    with pytest.raises(ImportError, match='numba'):
        refine_candidates(A, target, cands, use_jit=True)
    with pytest.raises(ImportError, match='numba'):
        refine_candidate(A, target, cands[0], TAU, use_jit=True)


def test_nnls_pg_batch_edge_cases():
    assert nnls_pg_batch([], []) == []
    out = nnls_pg_batch([np.zeros((5, 0))], [np.ones(5)])
    assert out[0].shape == (0,)
    with pytest.raises(ValueError):
        nnls_pg_batch([np.eye(2)], [np.ones(2), np.ones(2)])
    with pytest.raises(ValueError):
        nnls_pg_batch([np.eye(2)], [np.ones(3)])


def test_run_pipeline_use_jit_smoke():
    """run_pipeline(use_jit=True) runs end-to-end and recovers a clean pair.

    Only the structural guarantees (f1/k_pred on the top-20 pool) are
    asserted: the tiny numerical differences between scipy.optimize.nnls
    and the projected-gradient kernel may reorder candidates with
    near-identical BIC, but cannot change which support wins on a clean
    exact fit.
    """
    pytest.importorskip('numba')
    from qubic.pipeline import run_pipeline

    rng = np.random.default_rng(0)
    A = rng.random((10, 5))
    y = A[:, [1, 3]] @ np.array([0.6, 0.4])
    out = run_pipeline(A, y, k_true=2, gt_indices=[1, 3], scheme='B1',
                       solver='milp', use_jit=True)
    assert out['n_solutions'] >= 1
    res = out['results'][20]
    assert res['f1'] == 1.0
    assert res['k_pred'] == 2
    assert sorted(res['selected']) == [1, 3]