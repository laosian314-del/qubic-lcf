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
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from .encodings import build_qubo
from .solvers import solve as _solve

#: Ridge row weight used by every NNLS refinement (original value).
RIDGE = 0.1
#: RSS floor for BIC ranking of exact fits (rss == 0): ``compute_bic``'s
#: degenerate guard (original ``rss <= 0 -> 1e10``) would otherwise rank a
#: perfect fit as the worst model; the floor keeps BIC monotone in RSS.
EPS_RSS = 1e-300


def nnls_refine(A, target, selected):
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


def run_pipeline(A, target, k_true=None, gt_indices=None, scheme='B1',
                 solver='milp', top_k_values=(1, 10, 20), k_max=15,
                 time_limit=None, max_solutions=20, threshold_frac=0.01,
                 **qubo_kwargs):
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
    **qubo_kwargs
        Extra encoding options (``lam_sparse``, ``lam_link``, ``lam_rub``, or
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

    # Decode -> NNLS refine -> threshold -> BIC
    candidates = []
    for sample, obj_val in solutions:
        sel, _raw_conc = decode_fn(sample)
        sel = [int(i) for i in sel]
        if not sel:
            continue
        conc, rss = nnls_refine(A, target, sel)
        sel_filtered = [sel[i] for i in range(len(sel)) if conc[i] > threshold_frac]
        conc_filtered = conc[conc > threshold_frac]
        bic = compute_bic(max(rss, EPS_RSS), len(sel_filtered), K)
        candidates.append({
            'sel': sel_filtered,
            'conc': conc_filtered,
            'rss': rss,
            'bic': bic,
            'qubo_energy': float(obj_val),
        })

    # Fallback: NNLS on the full library (ridge row, threshold, BIC).
    if not candidates:
        A_aug = np.vstack([A, np.ones(N) * RIDGE])
        b_aug = np.hstack([target, [RIDGE]])
        c_full, _ = nnls(A_aug, b_aug)
        if c_full.sum() > 0:
            c_full = c_full / c_full.sum()
        sel_full = [i for i in range(N) if c_full[i] > threshold_frac]
        conc_full = c_full[c_full > threshold_frac]
        rss_full = float(np.sum((target - A[:, sel_full] @ conc_full) ** 2))
        bic_full = compute_bic(max(rss_full, EPS_RSS), len(sel_full), K)
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
