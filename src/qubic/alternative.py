"""Alternative backend for LCF component identification (encoding-free).

QUBIC-LCF offers two backends for identifying mixture components: the QUBO
pipeline (see :mod:`qubic.pipeline` and :class:`qubic.Problem`) and this
**alternative backend** — encoding-free classical methods used for
comparison (they are called baselines only in the paper). Ported from the
QUBO_LCF research repository (``scripts/baseline_classics.py`` and
``scripts/oracle_exact.py``) as a clean library module: no ``main()``, no
absolute paths, no CLI hardcoding.  All numerical semantics (candidate
generation, post-processing, BIC) are kept.

Methods (each returns the same :class:`AlternativeResult` dict)
--------------------------------------------------------------
Greedy forward selection (MP/OMP) family (JIT-accelerable, see below):
- ``omp_bic``        -- Orthogonal Matching Pursuit: greedy forward selection
   with incremental least-squares refit (rank-1 Cholesky updates of the
   ``A^T A`` Gram matrix); support prefixes of the greedy path for
   k = 1..k_max -> min-BIC.
- ``mp_bic``         -- Matching Pursuit without orthogonalization: at step t
   pick the column maximizing |<r, a_j>|/||a_j||, update
   r <- r - (<r,a_j>/||a_j||^2) a_j; support after each step -> min-BIC.

Other alternative-backend methods:
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

Every method returns :class:`AlternativeResult` -- ``{'method', 'selected',
'concentrations', 'k_pred', 'bic', 'rss'}`` with a sorted support; ``bic``
is None when no candidate produced a non-empty support (then
``rss = ||y||^2``).

Opt-in numba acceleration (greedy forward selection family)
-----------------------------------------------------------
:func:`mp_bic` / :func:`omp_bic` (and their shared greedy-path helpers) take
``use_jit=False``.  The default runs the pure-NumPy reference kernels and
never imports numba; ``use_jit=True`` runs the numerically equivalent
``@njit`` twins (optional ``numba`` dependency: ``pip install
"qubic-lcf[jit]"``).  Both twins implement the same deterministic greedy
selection: identical selected supports, allclose coefficients.

The optional numba/JIT acceleration was NOT used in the article; article results were obtained with the pure NumPy code paths.
"""
from __future__ import annotations

import itertools
from typing import TypedDict

import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import Lasso

from ._accel import require_numba
from .pipeline import EPS_RSS, compute_bic


class AlternativeResult(TypedDict):
    """Result record of one alternative-backend method.

    ``method``: method name; ``selected``: sorted support (list of int);
    ``concentrations``: NNLS concentrations aligned with ``selected``;
    ``k_pred``: support size; ``bic``: BIC score (None when no candidate
    produced a non-empty support); ``rss``: residual sum of squares.
    """
    method: str
    selected: list
    concentrations: list
    k_pred: int
    bic: float | None
    rss: float


#: tau = TAU_FACTOR * max(c) support trim (original nmr_common value).
TAU_FACTOR = 0.05
#: Frozen alpha grid of the original LASSO+ method.
LASSO_ALPHA_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
#: Frozen max_iter of the original LASSO+ method.
LASSO_MAX_ITER = 50000

#: Method name -> callable, as used by :func:`run_all_alternative`.
ALTERNATIVE_METHODS = ('omp_bic', 'mp_bic', 'lasso_positive', 'topk_corr_bic',
                       'oracle_exact')

# Count of suppressed "ended prematurely due to linear dependence" conditions
# of the greedy OMP path (kept from the original script for honest logging):
# one per candidate k that a premature greedy-path stop would have skipped.
_OMP_WARN_COUNT = 0


# --------------------------------------------------------------------------
# Shared post-processing (semantics of the original research scripts)
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
# Greedy forward selection kernels (twin implementations)
# --------------------------------------------------------------------------
# The hot loops of the greedy forward selection family are written twice and
# selected with ``use_jit`` on :func:`mp_bic` / :func:`omp_bic` (and their
# shared greedy-path helpers):
#
#   _mp_path_np  / _mp_path_jit   -- MP path (no orthogonalization): the
#                                    correlation computation |A^T r|/||a_j||,
#                                    the per-step atom selection and the
#                                    rank-1 residual update.
#   _omp_path_np / _omp_path_jit  -- OMP path: the Gram/correlation
#                                    computation (A^T A, A^T y), the per-step
#                                    atom selection over the correlations and
#                                    the rank-1 Cholesky / incremental
#                                    least-squares updates across steps.
#
# The pure-NumPy reference twin is the default path (``use_jit=False``, no
# numba import ever); the ``@njit`` twin runs only on ``use_jit=True``.  Both
# twins implement the same deterministic algorithm: identical selected
# supports, allclose coefficients.
#
# The optional numba/JIT acceleration was NOT used in the article; article results were obtained with the pure NumPy code paths.

#: ``np.finfo(float).eps`` guard of the OMP premature-stop checks (the
#: sklearn OMP ``min_float`` value, kept for identical stopping rules).
MIN_FLOAT = float(np.finfo(np.float64).eps)


def _mp_path_np(A, y, k_max_eff):
    """Pure-NumPy reference kernel: greedy MP path (default, ``use_jit=False``).

    At step t pick ``j = argmax_j |<r, a_j>| / ||a_j||`` over the not-yet
    chosen columns (first maximum wins, as ``np.argmax``) and apply the
    rank-1 residual update ``r <- r - (<r, a_j>/||a_j||^2) a_j``.

    Returns ``(order, step_alpha, norms2)``: the selected column indices in
    selection order (length ``k_max_eff``), the rank-1 update coefficient of
    each step, and the squared column norms ``diag(A^T A)``.
    """
    norms2 = np.sum(A ** 2, axis=0)
    norms = np.sqrt(norms2)
    r = y.astype(float).copy()
    order = np.empty(k_max_eff, dtype=np.int64)
    step_alpha = np.empty(k_max_eff, dtype=float)
    chosen = []
    for t in range(k_max_eff):
        ip = np.abs(A.T @ r) / norms
        if chosen:
            ip[np.asarray(chosen, dtype=int)] = -np.inf
        j = int(np.argmax(ip))
        chosen.append(j)
        order[t] = j
        alpha = float(A[:, j] @ r) / norms2[j]
        step_alpha[t] = alpha
        r = r - alpha * A[:, j]
    return order, step_alpha, norms2


def _omp_path_np(A, y, k_max_eff):
    """Pure-NumPy reference kernel: greedy OMP path (default, ``use_jit=False``).

    Greedy forward selection with incremental least-squares refit: compute
    the correlations ``c = A^T y`` and, per selected atom, its Gram row of
    ``A^T A`` (deferred-Gram OMP -- the same values as the full Gram, work
    O(k_max*K*N) instead of O(K*N^2)); at step t pick
    ``lam = argmax_j |cov_j|`` over ``cov = c - G[:, :t] @ gamma_t``
    (``gamma_t`` = least-squares coefficients on the selected atoms, from
    rank-1 Cholesky updates of ``G``), and stop early on a vanishing
    correlation or linearly dependent atoms (sklearn OMP stopping rules).

    Returns ``(order, coefs_path, n_active)``: the selected column indices in
    selection order (length ``n_active``), the per-step full-length
    least-squares coefficient vectors (row ``t`` holds the solution on the
    first ``t + 1`` atoms), and the number of atoms selected before any
    premature stop.
    """
    N = A.shape[1]
    c = A.T @ y
    order = np.empty(k_max_eff, dtype=np.int64)
    coefs_path = np.zeros((k_max_eff, N), dtype=float)
    gram_rows = np.zeros((k_max_eff, N), dtype=float)
    selected = np.zeros(N, dtype=bool)
    L = np.zeros((k_max_eff, k_max_eff), dtype=float)
    cov = c.copy()
    n_active = 0
    for t in range(k_max_eff):
        lam = int(np.argmax(np.abs(cov)))
        if selected[lam] or cov[lam] ** 2 < MIN_FLOAT:
            break  # premature stop: vanishing correlation / re-selection
        grow = A.T @ A[:, lam]  # Gram row of the newly selected atom
        if t == 0:
            L[0, 0] = np.sqrt(grow[lam])
        else:
            w = grow[order[:t]]
            z = np.empty(t, dtype=float)
            for j in range(t):
                z[j] = (w[j] - L[j, :j] @ z[:j]) / L[j, j]
            Lkk = float(grow[lam]) - float(z @ z)
            if Lkk <= MIN_FLOAT:
                break  # premature stop: linearly dependent atoms
            L[t, :t] = z
            L[t, t] = np.sqrt(Lkk)
        gram_rows[n_active] = grow
        selected[lam] = True
        order[n_active] = lam
        n_active += 1
        # incremental least squares: solve L L^T gamma = c[order[:n_active]]
        u = np.empty(n_active, dtype=float)
        for j in range(n_active):
            u[j] = (c[order[j]] - L[j, :j] @ u[:j]) / L[j, j]
        gamma = np.empty(n_active, dtype=float)
        for j in range(n_active - 1, -1, -1):
            gamma[j] = (u[j] - L[j + 1:n_active, j] @ gamma[j + 1:]) / L[j, j]
        cov = c - gram_rows[:n_active].T @ gamma
        coefs_path[n_active - 1, order[:n_active]] = gamma
    return order, coefs_path, n_active


_mp_kernel = None
_omp_kernel = None


def _get_mp_kernel(nb):
    """Build (once) and return the ``@njit`` twin of :func:`_mp_path_np`."""
    global _mp_kernel
    if _mp_kernel is None:
        @nb.njit(cache=True)
        def mp_path(A, y, k_max_eff):
            K, N = A.shape
            norms2 = np.zeros(N, dtype=np.float64)
            for i in range(K):
                for j in range(N):
                    aij = A[i, j]
                    norms2[j] += aij * aij
            norms = np.empty(N, dtype=np.float64)
            for j in range(N):
                norms[j] = np.sqrt(norms2[j])
            r = y.copy()
            order = np.empty(k_max_eff, dtype=np.int64)
            step_alpha = np.empty(k_max_eff, dtype=np.float64)
            chosen = np.zeros(N, dtype=np.uint8)
            dots = np.empty(N, dtype=np.float64)
            for t in range(k_max_eff):
                # hot loop: correlations |<r, a_j>| = |A^T r| + atom selection
                for j in range(N):
                    dots[j] = 0.0
                for i in range(K):
                    ri = r[i]
                    for j in range(N):
                        dots[j] += A[i, j] * ri
                best_j = -1
                best_v = -np.inf
                for j in range(N):
                    if chosen[j] != 0:
                        continue
                    v = abs(dots[j]) / norms[j]
                    if v > best_v:
                        best_v = v
                        best_j = j
                j = best_j
                chosen[j] = 1
                order[t] = j
                alpha = dots[j] / norms2[j]
                step_alpha[t] = alpha
                for i in range(K):
                    r[i] -= alpha * A[i, j]
            return order, step_alpha, norms2
        _mp_kernel = mp_path
    return _mp_kernel


def _get_omp_kernel(nb):
    """Build (once) and return the ``@njit`` twin of :func:`_omp_path_np`."""
    global _omp_kernel
    if _omp_kernel is None:
        @nb.njit(cache=True)
        def omp_path(A, y, k_max_eff):
            K, N = A.shape
            # hot loop 1: correlations A^T y and, per selected atom, its Gram
            # row of A^T A (deferred-Gram OMP: same values as the full Gram)
            c = np.zeros(N, dtype=np.float64)
            for i in range(K):
                yi = y[i]
                for j in range(N):
                    c[j] += A[i, j] * yi
            order = np.empty(k_max_eff, dtype=np.int64)
            coefs_path = np.zeros((k_max_eff, N), dtype=np.float64)
            gram_rows = np.zeros((k_max_eff, N), dtype=np.float64)
            selected = np.zeros(N, dtype=np.uint8)
            L = np.zeros((k_max_eff, k_max_eff), dtype=np.float64)
            cov = c.copy()
            n_active = 0
            for t in range(k_max_eff):
                # hot loop 2: per-step atom selection over the correlations
                lam = -1
                best = -np.inf
                for j in range(N):
                    v = abs(cov[j])
                    if v > best:
                        best = v
                        lam = j
                if selected[lam] != 0 or cov[lam] * cov[lam] < MIN_FLOAT:
                    break  # premature stop: vanishing correlation / re-selection
                grow = np.zeros(N, dtype=np.float64)  # Gram row of a_lam
                for i in range(K):
                    ali = A[i, lam]
                    for j in range(N):
                        grow[j] += ali * A[i, j]
                # hot loop 3: rank-1 Cholesky + incremental least squares
                if t == 0:
                    L[0, 0] = np.sqrt(grow[lam])
                else:
                    z = np.empty(t, dtype=np.float64)
                    for j in range(t):
                        s = grow[order[j]]
                        for m in range(j):
                            s -= L[j, m] * z[m]
                        z[j] = s / L[j, j]
                    v = 0.0
                    for j in range(t):
                        v += z[j] * z[j]
                    Lkk = grow[lam] - v
                    if Lkk <= MIN_FLOAT:
                        break  # premature stop: linearly dependent atoms
                    for j in range(t):
                        L[t, j] = z[j]
                    L[t, t] = np.sqrt(Lkk)
                for j in range(N):
                    gram_rows[n_active, j] = grow[j]
                selected[lam] = 1
                order[n_active] = lam
                n_active += 1
                u = np.empty(n_active, dtype=np.float64)
                for j in range(n_active):
                    s = c[order[j]]
                    for m in range(j):
                        s -= L[j, m] * u[m]
                    u[j] = s / L[j, j]
                gamma = np.empty(n_active, dtype=np.float64)
                for j in range(n_active - 1, -1, -1):
                    s = u[j]
                    for m in range(j + 1, n_active):
                        s -= L[m, j] * gamma[m]
                    gamma[j] = s / L[j, j]
                for l in range(N):
                    s = 0.0
                    for j in range(n_active):
                        s += gram_rows[j, l] * gamma[j]
                    cov[l] = c[l] - s
                for j in range(n_active):
                    coefs_path[n_active - 1, order[j]] = gamma[j]
            return order, coefs_path, n_active
        _omp_kernel = omp_path
    return _omp_kernel


def _mp_path_jit(A, y, k_max_eff):
    """numba ``@njit`` twin of :func:`_mp_path_np` (opt-in ``use_jit=True``)."""
    nb = require_numba()
    kernel = _get_mp_kernel(nb)
    return kernel(np.ascontiguousarray(A, dtype=np.float64),
                  np.ascontiguousarray(y, dtype=np.float64), k_max_eff)


def _omp_path_jit(A, y, k_max_eff):
    """numba ``@njit`` twin of :func:`_omp_path_np` (opt-in ``use_jit=True``)."""
    nb = require_numba()
    kernel = _get_omp_kernel(nb)
    return kernel(np.ascontiguousarray(A, dtype=np.float64),
                  np.ascontiguousarray(y, dtype=np.float64), k_max_eff)


def _mp_greedy_path(A, y, k_max_eff, use_jit=False):
    """Greedy MP path via the twin selected by ``use_jit``.

    Shared helper of :func:`mp_bic`; returns ``(order, step_alpha, norms2)``
    exactly as :func:`_mp_path_np`.
    """
    if use_jit:
        return _mp_path_jit(A, y, k_max_eff)
    return _mp_path_np(A, y, k_max_eff)


def _omp_greedy_path(A, y, k_max_eff, use_jit=False):
    """Greedy OMP path via the twin selected by ``use_jit``.

    Shared helper of :func:`omp_bic`; returns ``(order, coefs_path,
    n_active)`` exactly as :func:`_omp_path_np`.
    """
    if use_jit:
        return _omp_path_jit(A, y, k_max_eff)
    return _omp_path_np(A, y, k_max_eff)


# --------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------
def omp_bic(A, y, k_max=15, use_jit=False):
    """Orthogonal Matching Pursuit (greedy), k = 1..k_max -> min-BIC.

    The candidate supports are the prefixes of one greedy forward-selection
    path with incremental least-squares refit (equivalent to the per-k OMP
    fits of the original code; premature stops cut the path short exactly as
    before).  ``use_jit`` selects the numba ``@njit`` twin of the greedy-path
    kernel (the default pure-NumPy twin preserves current behavior).
    """
    global _OMP_WARN_COUNT
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    k_max_eff = min(k_max, A.shape[1])
    if k_max_eff <= 0:
        return _result('omp_bic', None, y)
    order, _coefs, n_active = _omp_greedy_path(A, y, k_max_eff,
                                               use_jit=use_jit)
    if n_active < k_max_eff:
        # premature stop: the original per-k fits warned once per k beyond
        # the stop -- keep that accounting honest
        _OMP_WARN_COUNT += k_max_eff - n_active
    cands = [[int(i) for i in sorted(order[:s])]
             for s in range(1, n_active + 1)]
    return _result('omp_bic', _select_by_bic(A, y, cands), y)


def mp_bic(A, y, k_max=15, use_jit=False):
    """Matching Pursuit without orthogonalization, k = 1..k_max -> min-BIC.

    ``use_jit`` selects the numba ``@njit`` twin of the greedy-path kernel
    (the default pure-NumPy twin preserves current behavior).
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    k_max_eff = min(k_max, A.shape[1])
    norms2 = np.sum(A ** 2, axis=0)
    if not (norms2 > 0).all():
        raise ValueError('zero column in A')
    if k_max_eff <= 0:
        return _result('mp_bic', None, y)
    order, _alphas, _norms2 = _mp_greedy_path(A, y, k_max_eff,
                                              use_jit=use_jit)
    cands = [[int(i) for i in order[:s]] for s in range(1, k_max_eff + 1)]
    return _result('mp_bic', _select_by_bic(A, y, cands), y)


def lasso_positive(A, y, alphas=None):
    """Lasso(positive=True) over an alpha grid -> min-BIC (the 'LASSO+' method).

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


def run_all_alternative(A, y, k_max=15, use_jit=False):
    """Run every alternative-backend method on ``(A, y)``.

    Returns ``{name: result_dict}`` with the five keys of
    :data:`ALTERNATIVE_METHODS`.  ``oracle_exact`` is exponential and raises
    ValueError when ``A.shape[1] > 15``.  ``use_jit`` is forwarded to the
    greedy forward selection family (:func:`omp_bic`, :func:`mp_bic`).
    """
    return {
        'omp_bic': omp_bic(A, y, k_max=k_max, use_jit=use_jit),
        'mp_bic': mp_bic(A, y, k_max=k_max, use_jit=use_jit),
        'lasso_positive': lasso_positive(A, y),
        'topk_corr_bic': topk_corr_bic(A, y, k_max=k_max),
        'oracle_exact': oracle_exact(A, y),
    }
