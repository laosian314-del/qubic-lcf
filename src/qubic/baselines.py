"""Encoding-free classical baselines for LCF component identification.

Ported from the QUBO_LCF research repository
(``scripts/baseline_classics.py`` and ``scripts/oracle_exact.py``) as a clean
library module: no ``main()``, no absolute paths, no CLI hardcoding.  All
numerical semantics (candidate generation, post-processing, BIC) are kept.

Methods (each returns the same result dict)
-------------------------------------------
- ``omp_bic``        -- sklearn ``OrthogonalMatchingPursuit`` with
   ``n_nonzero_coefs=k`` for k = 1..k_max -> min-BIC.
- ``mp_bic``         -- Matching Pursuit without orthogonalization: at step t
   pick the column maximizing |<r, a_j>|/||a_j||, update
   r <- r - (<r,a_j>/||a_j||^2) a_j; support after each step -> min-BIC.
- ``lasso_positive`` -- sklearn ``Lasso(positive=True, fit_intercept=False)``
   over an alpha grid (1e-5..1e-1 by default) -> min-BIC.
- ``topk_corr_bic``  -- one-shot column ranking by |<a_j, y>|/||a_j||,
   top-k prefixes for k = 1..k_max -> min-BIC.
- ``oracle_exact``   -- exhaustive enumeration of all 2^N subsets
   (N <= ``max_n``), NNLS per subset, min-BIC (F1-based selection of the
   original script needs ground truth, which is not available at inference).

Shared candidate post-processing (``nnls_thresh_refit``)
------------------------------------------------------
1. NNLS refit on the candidate support S (``scipy.optimize.nnls``, no ridge);
2. tau = 0.05 * max(c) trim -> support S';
3. if S' non-empty: NNLS refit on S' -> RSS = ||y - A[:,S'] @ c'||^2;
4. BIC = K*ln(RSS/K) + |S'|*ln(K) (:func:`qubic.pipeline.compute_bic`).

Empty supports are skipped.  ``oracle_exact`` keeps the metric semantics of
``scripts/oracle_exact.py``: plain NNLS, 0.05*max(c) trim, BIC with a
``+1e-15`` guard inside the logarithm.

Every method returns ``{'method', 'selected', 'concentrations', 'k_pred',
'bic', 'rss'}`` with a sorted support; ``bic`` is None when no candidate
produced a non-empty support (then ``rss = ||y||^2``).
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit

from .pipeline import EPS_RSS, compute_bic

#: tau = TAU_FACTOR * max(c) support trim (original nmr_common value).
TAU_FACTOR = 0.05
#: Frozen alpha grid of the original LASSO+ baseline.
LASSO_ALPHA_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
#: Frozen max_iter of the original LASSO+ baseline.
LASSO_MAX_ITER = 50000

#: Method name -> callable, as used by :func:`run_all_baselines`.
BASELINE_METHODS = ('omp_bic', 'mp_bic', 'lasso_positive', 'topk_corr_bic',
                    'oracle_exact')

# Count of suppressed sklearn OMP "ended prematurely due to linear
# dependence" warnings (kept from the original script for honest logging).
_OMP_WARN_COUNT = 0


# --------------------------------------------------------------------------
# Shared post-processing (semantics of run_baselines.py)
# --------------------------------------------------------------------------
def threshold_support(cols, concs, tau_factor=TAU_FACTOR):
    """Trim NNLS coefficients with tau = ``tau_factor`` * max(c).

    Returns ``(support_sorted, concs_kept_aligned)``; an empty support when
    all coefficients are zero/negligible.
    """
    cols = list(cols)
    concs = np.asarray(concs, dtype=float)
    if len(concs) == 0:
        return [], np.zeros(0)
    if concs.max() <= 0:
        return [], np.zeros(0)
    keep = concs >= tau_factor * concs.max()
    kept = [(cols[i], float(concs[i])) for i in range(len(cols)) if keep[i]]
    kept.sort(key=lambda t: t[0])
    return [c for c, _ in kept], np.array([v for _, v in kept], dtype=float)


def nnls_thresh_refit(A, y, cols):
    """NNLS refit on ``cols`` -> tau trim -> NNLS refit on the trimmed support.

    Returns ``(support_sorted, rss, conc_trimmed)``.
    """
    cols = list(cols)
    if not cols:
        return [], float(np.sum(y ** 2)), np.zeros(0)
    c, _ = nnls(A[:, cols], y)
    support, _ = threshold_support(cols, c)
    if not support:
        return [], float(np.sum(y ** 2)), np.zeros(0)
    if support == cols:
        rss = float(np.sum((y - A[:, support] @ c) ** 2))
        return support, rss, c
    c2, _ = nnls(A[:, support], y)
    rss = float(np.sum((y - A[:, support] @ c2) ** 2))
    return support, rss, c2


def _select_by_bic(A, y, candidate_cols):
    """Candidates -> NNLS refit + tau + BIC; return the min-BIC one.

    Returns ``(support, concentrations, rss, bic)`` or None when no candidate
    yields a non-empty support.
    """
    K = A.shape[0]
    best = None
    for cols in candidate_cols:
        support, rss, conc = nnls_thresh_refit(A, y, cols)
        if not support:
            continue
        bic = float(compute_bic(max(rss, EPS_RSS), len(support), K))
        if best is None or bic < best[3]:
            best = (support, conc, rss, bic)
    return best


def _result(method, best, y):
    """Build the common result dict from ``_select_by_bic`` output."""
    if best is None:
        return {
            'method': method,
            'selected': [],
            'concentrations': [],
            'k_pred': 0,
            'bic': None,
            'rss': float(np.sum(y ** 2)),
        }
    support, conc, rss, bic = best
    return {
        'method': method,
        'selected': [int(i) for i in support],
        'concentrations': [float(v) for v in np.asarray(conc).ravel()],
        'k_pred': len(support),
        'bic': float(bic),
        'rss': float(rss),
    }


# --------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------
def omp_bic(A, y, k_max=15):
    """Orthogonal Matching Pursuit (sklearn), k = 1..k_max -> min-BIC."""
    global _OMP_WARN_COUNT
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    cands = []
    for k in range(1, min(k_max, A.shape[1]) + 1):
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k, fit_intercept=False)
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter('always')
            omp.fit(A, y)
        _OMP_WARN_COUNT += sum(
            1 for w in wlist if 'prematurely' in str(w.message).lower())
        cols = [int(i) for i in np.flatnonzero(omp.coef_)]
        if cols:
            cands.append(cols)
    return _result('omp_bic', _select_by_bic(A, y, cands), y)


def mp_bic(A, y, k_max=15):
    """Matching Pursuit without orthogonalization, k = 1..k_max -> min-BIC."""
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
    return _result('mp_bic', _select_by_bic(A, y, cands), y)


def lasso_positive(A, y, alphas=None):
    """Lasso(positive=True) over an alpha grid -> min-BIC (the 'LASSO+' baseline).

    ``alphas`` defaults to the frozen grid 1e-5..1e-1.
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if alphas is None:
        alphas = LASSO_ALPHA_GRID
    cands = []
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, positive=True, fit_intercept=False,
                      max_iter=LASSO_MAX_ITER)
        lasso.fit(A, y)
        cols = [int(i) for i in np.flatnonzero(lasso.coef_)]
        if cols:
            cands.append(cols)
    return _result('lasso_positive', _select_by_bic(A, y, cands), y)


def topk_corr_bic(A, y, k_max=15):
    """One-shot ranking by |<a_j, y>|/||a_j||, top-k prefixes -> min-BIC."""
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    k_max_eff = min(k_max, A.shape[1])
    scores = np.abs(A.T @ y) / np.sqrt(np.sum(A ** 2, axis=0))
    order = list(np.argsort(-scores, kind='stable'))
    cands = [order[:k] for k in range(1, k_max_eff + 1)]
    return _result('topk_corr_bic', _select_by_bic(A, y, cands), y)


def _oracle_metrics(K, A, y, cols, concs, conc_thresh=TAU_FACTOR):
    """Metrics of one subset (semantics of oracle_exact.compute_metrics_from_concs).

    Returns ``(rss, bic, trimmed_cols, trimmed_concs)``.
    """
    concs = np.asarray(concs, dtype=float)
    if len(concs) > 0 and concs.max() > 1e-12:
        keep = concs >= conc_thresh * concs.max()
        if not keep.all():
            trimmed_cols = [cols[i] for i in range(len(concs)) if keep[i]]
            trimmed_concs = concs[keep]
            if len(trimmed_cols) == 0:
                rss = float(np.sum(y ** 2))
                bic = float(K * np.log(rss / K + 1e-15))
                return rss, bic, [], np.array([])
            A_sel = A[:, trimmed_cols]
            try:
                concs2, _ = nnls(A_sel, y)
            except Exception:
                concs2 = np.ones(len(trimmed_cols))
            rss = float(np.sum((y - A_sel @ concs2) ** 2))
            bic = float(K * np.log(rss / K + 1e-15)
                        + len(trimmed_cols) * np.log(K))
            return rss, bic, trimmed_cols, concs2
        rss = float(np.sum((y - A[:, cols] @ concs) ** 2))
        bic = float(K * np.log(rss / K + 1e-15) + len(cols) * np.log(K))
        return rss, bic, list(cols), concs
    if len(cols) == 0:
        rss = float(np.sum(y ** 2))
        return rss, float(K * np.log(rss / K + 1e-15)), [], np.array([])
    rss = float(np.sum((y - A[:, cols] @ concs) ** 2))
    bic = float(K * np.log(rss / K + 1e-15) + len(cols) * np.log(K))
    return rss, bic, list(cols), concs


def oracle_exact(A, y, max_n=15):
    """Exhaustive best-subset oracle: enumerate all 2^N subsets, min-BIC.

    For every non-empty subset: NNLS, tau = 0.05*max(c) trim, refit on the
    trimmed support, BIC (metric semantics of ``scripts/oracle_exact.py``).
    The original script ranked subsets by F1 against ground truth; here the
    winner is the min-BIC subset because no ground truth is available at
    inference time.

    Raises
    ------
    ValueError
        If ``N > max_n`` (the enumeration is exponential).
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    N = A.shape[1]
    if N > max_n:
        raise ValueError(
            f'oracle_exact is exponential: N={N} > max_n={max_n}')
    K = A.shape[0]
    best = None  # (bic, rss, cols, concs)
    for mask in range(1, 1 << N):
        cols = [j for j in range(N) if (mask >> j) & 1]
        try:
            concs, _ = nnls(A[:, cols], y)
        except Exception:
            concs = np.zeros(len(cols))
        rss, bic, trimmed_cols, trimmed_concs = _oracle_metrics(
            K, A, y, cols, concs)
        if best is None or bic < best[0]:
            best = (bic, rss, trimmed_cols, trimmed_concs)
    if best is None or not best[2]:  # every subset trimmed to empty
        rss = float(np.sum(y ** 2))
        return {
            'method': 'oracle_exact',
            'selected': [],
            'concentrations': [],
            'k_pred': 0,
            'bic': float(K * np.log(rss / K + 1e-15)),
            'rss': rss,
        }
    bic, rss, cols, concs = best
    return {
        'method': 'oracle_exact',
        'selected': [int(i) for i in cols],
        'concentrations': [float(v) for v in np.asarray(concs).ravel()],
        'k_pred': len(cols),
        'bic': float(bic),
        'rss': float(rss),
    }


def run_all_baselines(A, y, k_max=15):
    """Run every baseline on ``(A, y)``.

    Returns ``{name: result_dict}`` with the five keys of
    :data:`BASELINE_METHODS`.  ``oracle_exact`` is exponential and raises
    ValueError when ``A.shape[1] > 15``.
    """
    return {
        'omp_bic': omp_bic(A, y, k_max=k_max),
        'mp_bic': mp_bic(A, y, k_max=k_max),
        'lasso_positive': lasso_positive(A, y),
        'topk_corr_bic': topk_corr_bic(A, y, k_max=k_max),
        'oracle_exact': oracle_exact(A, y),
    }
