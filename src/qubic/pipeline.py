"""Full QUBIC-LCF pipeline: QUBO construction -> solving -> NNLS refinement -> BIC.

Ported from the QUBO_LCF research repository (``scripts/pipeline.py``) with
the numerical semantics preserved; absolute paths and CLI hardcoding are gone,
and the QUBO solver is now a parameter (``solver='milp'`` by default) instead
of a hardwired MILP call.

Pipeline steps
--------------
1. ``build_qubo(scheme, A, target, k, **qubo_kwargs)`` encodes the LCF problem
   (see :mod:`qubic.encodings`).
2. ``qubic.solvers.solve(solver, ...)`` collects up to ``max_solutions``
   candidate supports (bit samples) sorted by ascending QUBO energy.
3. Every decoded candidate is refined with NNLS on its support (ridge row
   ``0.1``, concentrations normalized to sum 1) and thresholded at
   ``threshold_frac`` (0.01 by default, as in the original).
4. BIC selects the best candidate inside each top-``top_k`` pool.
5. Fallback when no candidate decodes to a non-empty support: NNLS on the
   whole library with the same ridge row 0.1 and the same threshold.

Cardinality prior
-----------------
``run_pipeline`` passes ``k_true`` to ``build_qubo`` as the cardinality target
``k`` (original semantics).  ``k`` may also be given explicitly through
``**qubo_kwargs`` and takes precedence.  When neither is given the cardinality
penalty is switched off (``lam_sparse=0``) and model selection is left to BIC.

Optional batched numba NNLS
---------------------------
The public refinement entry points (:func:`nnls_refine`,
:func:`refine_candidate`, :func:`refine_candidates`,
:func:`full_library_candidate`, :func:`run_pipeline`) accept an opt-in
``use_jit`` keyword.  ``use_jit=False`` (the default) keeps the original
``scipy.optimize.nnls`` behaviour exactly and never imports numba;
``use_jit=True`` refits the candidate supports with one batched
projected-gradient NNLS kernel (:func:`nnls_pg_batch`, numba ``@njit``),
whose pure-NumPy twin is the default path of :func:`nnls_pg_batch` itself.

The optional numba/JIT acceleration was NOT used in the article; article results were obtained with the pure NumPy code paths.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from ._accel import require_numba
from .encodings import build_qubo
from .solvers import solve as _solve

#: Ridge row weight used by every NNLS refinement (original value).
RIDGE = 0.1
#: RSS floor for BIC ranking of exact fits (rss == 0): ``compute_bic``'s
#: degenerate guard (original ``rss <= 0 -> 1e10``) would otherwise rank a
#: perfect fit as the worst model; the floor keeps BIC monotone in RSS.
EPS_RSS = 1e-300
#: Projected-gradient NNLS: KKT stopping tolerance, scaled by
#: ``max(1, ||M^T d||_inf)`` at every candidate problem.
PG_TOL = 1e-12
#: Projected-gradient NNLS: iteration cap (well-conditioned problems
#: converge far earlier; the cap only bounds ill-conditioned tails).
PG_MAX_ITER = 20000


def _pg_nnls_numpy(M, d, tol=PG_TOL, max_iter=PG_MAX_ITER):
    """Pure-NumPy projected-gradient NNLS: min ``||M x - d||^2`` s.t. ``x >= 0``.

    Exact twin of the numba kernel :func:`_pg_nnls_batch_kernel` (same
    FISTA iteration, adaptive restart and KKT stopping rule).  The step
    size ``1/L`` uses ``L = ||M^T M||_F``, a guaranteed upper bound on the
    spectral radius of the PSD Gram matrix.  Iteration stops when the
    KKT residual ``max_i |v_i|`` (``v_i = grad_i`` for ``x_i > 0``,
    ``v_i = min(grad_i, 0)`` for ``x_i == 0``) falls below
    ``tol * max(1, ||M^T d||_inf)``.
    """
    M = np.asarray(M, dtype=float)
    d = np.asarray(d, dtype=float).ravel()
    n = M.shape[1]
    if n == 0:
        return np.zeros(0)
    G = M.T @ M
    g0 = M.T @ d
    L = float(np.sqrt(np.sum(G * G)))
    if L <= 0.0:
        return np.zeros(n)
    scale = max(1.0, float(np.max(np.abs(g0))))
    thr = tol * scale
    c = np.zeros(n)
    y = np.zeros(n)
    t = 1.0
    for _ in range(max_iter):
        c_new = np.maximum(y - (G @ y - g0) / L, 0.0)
        if float((y - c_new) @ (c_new - c)) > 0.0:  # adaptive restart
            t = 1.0
            c = c_new
            y = c_new.copy()
        else:
            t_new = 0.5 * (1.0 + float(np.sqrt(1.0 + 4.0 * t * t)))
            y = c_new + ((t - 1.0) / t_new) * (c_new - c)
            c = c_new
            t = t_new
        grad = G @ c - g0
        v = np.where(c > 0.0, np.abs(grad), np.maximum(-grad, 0.0))
        if float(v.max()) <= thr:
            break
    return c


def _pg_nnls_batch_kernel(As, ds, lens, tol, max_iter, outs):
    """Numba-compiled batched projected-gradient NNLS (see task notes).

    ``As[b]`` is the (m, kmax) design matrix of candidate ``b`` (only its
    first ``lens[b]`` columns are meaningful), ``ds[b]`` its right-hand
    side; the solution is written to ``outs[b, :lens[b]]``.  Algorithm and
    stopping rule are identical to :func:`_pg_nnls_numpy`.
    """
    B = lens.shape[0]
    m = As.shape[1]
    kmax = As.shape[2]
    G = np.zeros((kmax, kmax))
    g0 = np.zeros(kmax)
    c = np.zeros(kmax)
    y = np.zeros(kmax)
    c_new = np.zeros(kmax)
    for b in range(B):
        n = lens[b]
        if n == 0:
            continue
        # Gram matrix and gradient of the quadratic at 0.
        for i in range(n):
            for j in range(i, n):
                s = 0.0
                for r in range(m):
                    s += As[b, r, i] * As[b, r, j]
                G[i, j] = s
                G[j, i] = s
            s2 = 0.0
            for r in range(m):
                s2 += As[b, r, i] * ds[b, r]
            g0[i] = s2
        acc = 0.0
        for i in range(n):
            for j in range(n):
                acc += G[i, j] * G[i, j]
        L = np.sqrt(acc)
        scale = 1.0
        for i in range(n):
            v = abs(g0[i])
            if v > scale:
                scale = v
        thr = tol * scale
        for i in range(n):
            c[i] = 0.0
            y[i] = 0.0
        t = 1.0
        if L <= 0.0:
            for i in range(n):
                outs[b, i] = 0.0
            continue
        for _ in range(max_iter):
            for i in range(n):
                s = -g0[i]
                for j in range(n):
                    s += G[i, j] * y[j]
                v = y[i] - s / L
                if v < 0.0:
                    v = 0.0
                c_new[i] = v
            dot = 0.0
            for i in range(n):
                dot += (y[i] - c_new[i]) * (c_new[i] - c[i])
            if dot > 0.0:  # adaptive restart
                t = 1.0
                for i in range(n):
                    c[i] = c_new[i]
                    y[i] = c_new[i]
            else:
                t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
                coef = (t - 1.0) / t_new
                for i in range(n):
                    y[i] = c_new[i] + coef * (c_new[i] - c[i])
                    c[i] = c_new[i]
                t = t_new
            worst = 0.0
            for i in range(n):
                s = -g0[i]
                for j in range(n):
                    s += G[i, j] * c[j]
                if c[i] > 0.0:
                    v = s if s >= 0.0 else -s
                else:
                    v = -s if s < 0.0 else 0.0
                if v > worst:
                    worst = v
            if worst <= thr:
                break
        for i in range(n):
            outs[b, i] = c[i]


_PG_KERNEL = None


def _get_pg_kernel():
    """Compile (once, lazily) and return the numba batched NNLS kernel."""
    global _PG_KERNEL
    if _PG_KERNEL is None:
        nb = require_numba()
        _PG_KERNEL = nb.njit(cache=True)(_pg_nnls_batch_kernel)
    return _PG_KERNEL


def nnls_pg_batch(matrices, rhs, tol=PG_TOL, max_iter=PG_MAX_ITER,
                  use_jit=False):
    """Batched projected-gradient NNLS for many candidate supports at once.

    Solves ``min ||M_b x - d_b||^2`` s.t. ``x >= 0`` for every problem
    ``b``.  This is the raw refit engine used by :func:`refine_candidates`
    (``use_jit=True``); it does not apply the ridge row, the sum-1
    normalization, the ``tau`` cutoff or BIC — see :func:`refine_candidates`
    for the full refinement stage.

    Parameters
    ----------
    matrices : sequence of np.ndarray
        One 2-D design matrix ``(m_b, k_b)`` per candidate.
    rhs : sequence of np.ndarray
        One 1-D right-hand side ``(m_b,)`` per candidate.
    tol, max_iter : float, int
        KKT tolerance (scaled per problem) and iteration cap.
    use_jit : bool, optional
        False (default): the pure-NumPy twin of the kernel.  True: one
        batched numba ``@njit`` call (requires the optional ``numba``
        dependency; raises a helpful ImportError otherwise).

    Returns
    -------
    list of np.ndarray
        One non-negative solution ``(k_b,)`` per input problem.
    """
    mats = [np.ascontiguousarray(np.asarray(M, dtype=float)) for M in matrices]
    ds = [np.ascontiguousarray(np.asarray(d, dtype=float).ravel()) for d in rhs]
    if len(mats) != len(ds):
        raise ValueError('matrices and rhs must have the same number of problems')
    for M, d in zip(mats, ds):
        if M.ndim != 2:
            raise ValueError('each design matrix must be 2-D')
        if d.shape[0] != M.shape[0]:
            raise ValueError('design matrix and rhs row counts must match')
    if not mats:
        return []
    if not use_jit:
        return [_pg_nnls_numpy(M, d, tol, max_iter) for M, d in zip(mats, ds)]
    require_numba()
    kernel = _get_pg_kernel()
    B = len(mats)
    mmax = max(M.shape[0] for M in mats)
    kmax = max(M.shape[1] for M in mats)
    if kmax == 0:
        return [np.zeros(0) for _ in range(B)]
    As = np.zeros((B, mmax, kmax))
    ds_arr = np.zeros((B, mmax))
    lens = np.zeros(B, dtype=np.int64)
    outs = np.zeros((B, kmax))
    for b, (M, d) in enumerate(zip(mats, ds)):
        n = M.shape[1]
        m = M.shape[0]
        As[b, :m, :n] = M
        ds_arr[b, :m] = d
        lens[b] = n
    kernel(As, ds_arr, lens, float(tol), int(max_iter), outs)
    return [outs[b, :lens[b]].copy() for b in range(B)]


def nnls_refine(A, target, selected, use_jit=False):
    """NNLS refinement of the concentrations on ``selected`` components.

    A ridge row of weight 0.1 softly enforces sum(c) = 1 and the resulting
    coefficients are normalized to sum 1 (original ``nnls_refine`` semantics,
    minus the residual norm which is not part of the public contract).

    Parameters
    ----------
    A : np.ndarray (K, N)
        Reference library; column ``j`` is component ``j``.
    target : np.ndarray (K,)
        Experimental spectrum.
    selected : sequence of int
        Column indices of the candidate support.
    use_jit : bool, optional
        False (default): the original ``scipy.optimize.nnls`` call.  True:
        solve the same augmented NNLS problem with the opt-in batched
        projected-gradient kernel (:func:`nnls_pg_batch`; requires numba).

    Returns
    -------
    (concentrations, rss) : (np.ndarray (len(selected),), float)
        Non-negative concentrations (sum 1 when non-degenerate) and the
        residual sum of squares ``||target - A[:, selected] @ c||^2``.
        For an empty support: ``(np.zeros(0), 1e10)``.
    """
    if len(selected) == 0:
        return np.zeros(0), 1e10
    sel = [int(i) for i in selected]
    A_sel = A[:, sel]
    A_aug = np.vstack([A_sel, np.ones(A_sel.shape[1]) * RIDGE])
    b_aug = np.hstack([target, [RIDGE]])
    if use_jit:
        c = nnls_pg_batch([A_aug], [b_aug], use_jit=True)[0]
    else:
        c, _residual = nnls(A_aug, b_aug)
    if c.sum() > 0:
        c = c / c.sum()
    rss = float(np.sum((target - A_sel @ c) ** 2))
    return c, rss


def compute_bic(rss, k, K):
    """BIC = K * ln(RSS/K) + k * ln(K) (original formula).

    Returns ``1e10`` for degenerate inputs (``rss <= 0`` or ``k == 0``).

    Note: an exact fit (``rss == 0``) hits the degenerate guard; pipeline
    candidate ranking therefore feeds this function ``max(rss, EPS_RSS)``.
    """
    if rss <= 0 or k == 0:
        return 1e10
    return float(K * np.log(rss / K) + k * np.log(K))


def compute_f1(pred, gt):
    """F1 score between the predicted and ground-truth index sets."""
    ps, gs = set(pred), set(gt)
    tp = len(ps & gs)
    p = tp / len(ps) if ps else 0.0
    r = tp / len(gs) if gs else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def refine_candidate(A, target, selected, tau, use_jit=False):
    """Refinement stage for one candidate support: NNLS -> tau cutoff -> BIC.

    Mirrors the candidate loop of :func:`run_pipeline`:
    1. NNLS refit on the candidate support (ridge row ``RIDGE``,
       concentrations normalized to sum 1);
    2. concentrations <= ``tau`` are trimmed from the reported support;
    3. BIC is computed on the trimmed support (RSS floored at ``EPS_RSS``).

    Parameters
    ----------
    use_jit : bool, optional
        Forwarded to :func:`nnls_refine`: False (default) keeps
        ``scipy.optimize.nnls`` exactly; True uses the opt-in batched
        projected-gradient kernel (requires numba).

    Returns ``(support, concentrations, rss, bic)`` or ``None`` when
    ``selected`` is empty.
    """
    sel = [int(i) for i in selected]
    if not sel:
        return None
    conc, rss = nnls_refine(A, target, sel, use_jit=use_jit)
    sel_filtered = [sel[i] for i in range(len(sel)) if conc[i] > tau]
    conc_filtered = conc[conc > tau]
    bic = compute_bic(max(rss, EPS_RSS), len(sel_filtered), len(target))
    return sel_filtered, conc_filtered, rss, bic


def refine_candidates(A, target, candidates, tau=0.01, use_jit=False):
    """Batched refinement of many candidate supports (``run_pipeline`` 3-4).

    Applies exactly the :func:`refine_candidate` stage (NNLS refit with ridge
    row ``RIDGE`` and sum-1 normalization -> ``tau`` concentration cutoff ->
    BIC with the ``EPS_RSS`` floor) to every candidate support, in one call.

    Parameters
    ----------
    A : np.ndarray (K, N)
        Reference library; column ``j`` is component ``j``.
    target : np.ndarray (K,)
        Experimental spectrum.
    candidates : sequence of sequence of int
        Candidate supports (column index sequences).
    tau : float, optional
        Concentration cutoff (0.01 by default, as in the original).
    use_jit : bool, optional
        False (default): every candidate is refit with
        ``scipy.optimize.nnls``, exactly as the original per-candidate loop.
        True: all candidate supports are refit in a single batched numba
        projected-gradient NNLS call (:func:`nnls_pg_batch`); requires the
        optional ``numba`` dependency.

    Returns
    -------
    list
        One ``(support, concentrations, rss, bic)`` tuple per input
        candidate, in input order (``None`` for empty supports) — the
        :func:`refine_candidate` contract.
    """
    A = np.asarray(A, dtype=float)
    target = np.asarray(target, dtype=float).ravel()
    sels = [[int(i) for i in sel] for sel in candidates]
    out = [None] * len(sels)
    kept = [(pos, sel) for pos, sel in enumerate(sels) if sel]
    if not kept:
        return out
    if not use_jit:
        for pos, sel in kept:
            out[pos] = refine_candidate(A, target, sel, tau, use_jit=False)
        return out
    require_numba()
    K = A.shape[0]
    b_aug = np.empty(K + 1)
    b_aug[:K] = target
    b_aug[K] = RIDGE
    mats, rhs, blocks = [], [], []
    for _pos, sel in kept:
        n = len(sel)
        A_sel = A[:, sel]  # fancy indexing: fresh C-contiguous (K, n) block
        M = np.empty((K + 1, n))
        M[:K, :] = A_sel
        M[K, :] = RIDGE
        mats.append(M)
        rhs.append(b_aug)
        blocks.append(A_sel)
    sols = nnls_pg_batch(mats, rhs, use_jit=True)
    for (pos, sel), A_sel, c in zip(kept, blocks, sols):
        if c.sum() > 0:
            c = c / c.sum()
        rss = float(np.sum((target - A_sel @ c) ** 2))
        sel_filtered = [sel[i] for i in range(len(sel)) if c[i] > tau]
        conc_filtered = c[c > tau]
        bic = compute_bic(max(rss, EPS_RSS), len(sel_filtered), len(target))
        out[pos] = (sel_filtered, conc_filtered, rss, bic)
    return out


def full_library_candidate(A, target, tau, use_jit=False):
    """Fallback candidate: NNLS on the whole library (``run_pipeline`` step 5).

    Same ridge row, ``tau`` cutoff and BIC as :func:`refine_candidate`.
    ``use_jit`` selects the refit backend as in :func:`nnls_refine`
    (False keeps ``scipy.optimize.nnls`` exactly).
    Returns ``(support, concentrations, rss, bic)``.
    """
    N = A.shape[1]
    A_aug = np.vstack([A, np.ones(N) * RIDGE])
    b_aug = np.hstack([target, [RIDGE]])
    if use_jit:
        c_full = nnls_pg_batch([A_aug], [b_aug], use_jit=True)[0]
    else:
        c_full, _ = nnls(A_aug, b_aug)
    if c_full.sum() > 0:
        c_full = c_full / c_full.sum()
    sel_full = [i for i in range(N) if c_full[i] > tau]
    conc_full = c_full[c_full > tau]
    rss_full = float(np.sum((target - A[:, sel_full] @ conc_full) ** 2))
    bic_full = compute_bic(max(rss_full, EPS_RSS), len(sel_full), len(target))
    return sel_full, conc_full, rss_full, bic_full


def run_pipeline(A, target, k_true=None, gt_indices=None, scheme='B1',
                 solver='milp', top_k_values=(1, 10, 20), k_max=15,
                 time_limit=None, max_solutions=20, threshold_frac=0.01,
                 use_jit=False, **qubo_kwargs):
    """Run the full QUBO-LCF pipeline on one mixture.

    Parameters
    ----------
    A : np.ndarray (K, N)
        Reference library (columns = components).
    target : np.ndarray (K,)
        Experimental spectrum.
    k_true : int or None
        Ground-truth number of components.  Also used as the QUBO cardinality
        target ``k`` unless ``k`` is passed through ``**qubo_kwargs`` (original
        semantics).  When both are None the cardinality penalty is disabled.
    gt_indices : sequence of int or None
        Ground-truth component indices; ``f1`` is None when not given.
    scheme : str
        Encoding scheme, see :data:`qubic.encodings.SCHEMES`.
    solver : str
        QUBO solver name passed to :func:`qubic.solvers.solve`
        (default ``'milp'``).
    top_k_values : sequence of int
        Pool sizes; for each value the min-BIC candidate of the best ``tk``
        candidates (by QUBO energy) is reported.
    k_max : int
        Cardinality bound forwarded to the solver.
    time_limit : float or None
        Per-solve wall-clock cap in seconds. None = solver default.
    max_solutions : int
        Max distinct candidate supports to collect from the solver.
    threshold_frac : float
        Concentrations <= this value (after sum-1 normalization) are trimmed
        from the reported support (0.01 by default, as in the original).
    use_jit : bool, optional
        Refit backend for the candidate loop (step 3) and the fallback
        (step 5), forwarded to :func:`refine_candidates` /
        :func:`full_library_candidate`.  False (default) keeps
        ``scipy.optimize.nnls`` per candidate exactly as before; True refits
        all candidates in one batched numba projected-gradient NNLS call.
    **qubo_kwargs
        Extra encoding options (``lam_sparse``, ``lam_link``, ``lam_antimix``, or
        an explicit cardinality ``k``).

    Returns
    -------
    dict
        ``{'results': {top_k: {'f1', 'k_pred', 'k_true', 'bic', 'rss',
        'selected', 'concentrations', 'n_candidates'}}, 'wall_time',
        'n_solutions'}``.
    """
    A = np.asarray(A, dtype=float)
    target = np.asarray(target, dtype=float).ravel()
    N = A.shape[1]
    K = A.shape[0]

    # Cardinality target: explicit k wins over k_true; neither -> no penalty.
    k = qubo_kwargs.pop('k', k_true)
    if k is None:
        k = 0  # dummy value, unused with lam_sparse = 0
        qubo_kwargs.setdefault('lam_sparse', 0.0)

    Q, q, var_map, decode_fn = build_qubo(scheme, A, target, k, **qubo_kwargs)

    solutions, wall_time, _gap = _solve(
        solver, Q, q, var_map, k_max=k_max,
        top_k=max_solutions, time_limit=time_limit,
    )

    # Decode -> NNLS refine -> threshold -> BIC (one batched refit when
    # use_jit=True; per-candidate scipy.optimize.nnls otherwise).
    kept = []
    for sample, obj_val in solutions:
        sel, _raw_conc = decode_fn(sample)
        sel = [int(i) for i in sel]
        if not sel:
            continue
        kept.append((sel, float(obj_val)))
    candidates = []
    refined = refine_candidates(A, target, [s for s, _ in kept],
                                tau=threshold_frac, use_jit=use_jit)
    for (sel, obj_val), ref in zip(kept, refined):
        sel_filtered, conc_filtered, rss, bic = ref
        candidates.append({
            'sel': sel_filtered,
            'conc': conc_filtered,
            'rss': rss,
            'bic': bic,
            'qubo_energy': obj_val,
        })

    # Fallback: NNLS on the full library (ridge row, threshold, BIC).
    if not candidates:
        sel_full, conc_full, rss_full, bic_full = full_library_candidate(
            A, target, threshold_frac, use_jit=use_jit)
        candidates.append({
            'sel': sel_full,
            'conc': conc_full,
            'rss': rss_full,
            'bic': bic_full,
            'qubo_energy': float('inf'),
        })

    candidates.sort(key=lambda c: c['qubo_energy'])

    results = {}
    for tk in top_k_values:
        pool = candidates[:tk]
        if not pool:
            continue
        best = min(pool, key=lambda c: c['bic'])
        results[tk] = {
            'f1': compute_f1(best['sel'], gt_indices) if gt_indices is not None else None,
            'k_pred': len(best['sel']),
            'k_true': k_true,
            'bic': best['bic'],
            'rss': best['rss'],
            'selected': [int(i) for i in best['sel']],
            'concentrations': [float(v) for v in np.asarray(best['conc']).ravel()],
            'n_candidates': len(pool),
        }

    return {
        'results': results,
        'wall_time': float(wall_time),
        'n_solutions': len(solutions),
    }
