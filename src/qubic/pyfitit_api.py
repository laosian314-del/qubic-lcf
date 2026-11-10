"""pyfitit-compatible LCF backend on top of QUBIC-LCF.

Drop-in-style replacement for ``pyfitit.mixture.tryAllMixtures`` ("похожий
бекенд на который легко перейти"): it accepts a library of reference spectra
plus an experimental target spectrum, tries linear combinations of reference
components and returns the fitted concentrations and quality metrics.

Signature and naming follow ``pyfitit.mixture.tryAllMixtures``
(the stock pyfitit mixture module) where they overlap (``componentCount``,
``mixtureTrysCount``, ``componentNames``, ``optimizeConcentrations``,
``randomSeed``, ``unknownCharacterization``, ``componentNameColumn``,
``spType``), and add the QUBO knobs ``scheme=`` and ``solver=``.",

pyfitit is NOT required: the module imports without it and the core fitting
uses :func:`qubic.pipeline.run_pipeline` (QUBO backend) or exhaustive mixture
enumeration (enumerate backend).  pyfitit ``ML.Sample`` objects are supported
duck-typed (via ``getSpectra``/``getDefaultSpType``), so no pyfitit import is
needed even then.

ASSUMPTIONS (documented deviations from pyfitit, since pyfitit's plotting /
label-map machinery is out of scope here):
- ``r_factor`` is the plain R-factor ``sum|y - y_mix| / sum|y|``; pyfitit's
  ``utils.rFactorSp`` with its interval/weight parameters is not reproduced.
- Concentrations are non-negative and normalized to sum 1 (mixture-of-spectra
  convention), as in the QUBIC-LCF pipeline's NNLS refinement.
- ``mixtureTrysCount``: 'all combinations of singles' enumerates
  itertools.combinations (distinct components; pyfitit allows repetitions for
  label sampling); an int N draws N random subsets of size ``componentCount``
  with ``randomSeed``; 'all combinations for each label' needs pyfitit labels
  and raises here.
- ``optimizeConcentrations=False`` draws one random Dirichlet split seeded by
  ``randomSeed`` per mixture (pyfitit draws random concentrations as well);
  ``True`` fits concentrations by NNLS (pipeline ``nnls_refine``).
- No plots are written; ``folder`` is accepted for signature compatibility
  only and ignored.
"""
from __future__ import annotations

import itertools
import time

import numpy as np

from .pipeline import EPS_RSS, compute_bic, nnls_refine, run_pipeline

#: Cap on the number of enumerated mixtures in the 'enumerate' backend.
_MAX_ENUMERATED = 200000


# --------------------------------------------------------------------------
# Input adapters (no pyfitit import needed)
# --------------------------------------------------------------------------
def _as_matrix(reference_spectra, spType=None, componentNameColumn=None):
    """Normalize the reference library to an (K, N) array + component names.

    Accepts: 2D array-like (columns = components), 1D array-like (single
    component), sequence of 1D spectra, dict {name: spectrum}, or a pyfitit
    ``ML.Sample``-like object exposing ``getSpectra``/``getDefaultSpType``.
    """
    names = None
    if hasattr(reference_spectra, 'getSpectra'):  # pyfitit ML.Sample (duck-typed)
        if spType is None:
            spType = reference_spectra.getDefaultSpType()
        spectra = np.asarray(reference_spectra.getSpectra(spType), dtype=float)
        A = spectra.T  # Sample rows are individual spectra; QUBIC needs columns
        if componentNameColumn is not None:
            params = reference_spectra.params
            names = [str(v) for v in params[componentNameColumn].tolist()]
        return A, names
    if isinstance(reference_spectra, dict):
        names = [str(k) for k in reference_spectra]
        A = np.column_stack([np.asarray(reference_spectra[k], dtype=float).ravel()
                             for k in reference_spectra])
        return A, names
    if isinstance(reference_spectra, (list, tuple)):
        cols = [np.asarray(s, dtype=float).ravel() for s in reference_spectra]
        return np.column_stack(cols), names
    A = np.asarray(reference_spectra, dtype=float)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if A.ndim != 2:
        raise ValueError(f'reference_spectra must be 2D, got shape {A.shape}')
    return A, names


def _as_target(target, unknownCharacterization=None):
    """Extract the experimental spectrum as a 1D float array."""
    if unknownCharacterization is not None:
        if not isinstance(unknownCharacterization, dict):
            raise TypeError('unknownCharacterization must be a dict')
        if unknownCharacterization.get('type', 'spectrum') != 'spectrum':
            raise ValueError("only unknownCharacterization['type'] == "
                             "'spectrum' is supported")
        if 'spectrum' in unknownCharacterization:
            target = unknownCharacterization['spectrum']
    if isinstance(target, dict):
        if 'spectrum' not in target:
            raise ValueError("target dict must contain a 'spectrum' entry")
        target = target['spectrum']
    if hasattr(target, 'y'):  # pyfitit utils.Spectrum
        target = target.y
    y = np.asarray(target, dtype=float).ravel()
    return y


def _r_factor(y, y_mix):
    """Plain R-factor sum|y - y_mix| / sum|y| (see module docstring)."""
    denom = float(np.sum(np.abs(y)))
    if denom <= 0:
        return float('nan')
    return float(np.sum(np.abs(y - y_mix))) / denom


def _candidate_record(A, y, selected, concentrations, bic):
    """Assemble one mixture record with metrics."""
    selected = [int(i) for i in selected]
    concentrations = [float(c) for c in np.asarray(concentrations, dtype=float).ravel()]
    y_mix = A[:, selected] @ np.asarray(concentrations) if selected \
        else np.zeros_like(y)
    return {
        'selected': selected,
        'concentrations': concentrations,
        'k_pred': len(selected),
        'rss': float(np.sum((y - y_mix) ** 2)),
        'bic': float(bic),
        'r_factor': _r_factor(y, y_mix),
        'mixture_spectrum': y_mix,
    }


# --------------------------------------------------------------------------
# Main API
# --------------------------------------------------------------------------
def tryallmixtures(reference_spectra, target, componentCount=None,
                   mixtureTrysCount='all combinations of singles',
                   componentNames=None, unknownCharacterization=None,
                   optimizeConcentrations=True, componentNameColumn=None,
                   spType=None, method='qubo', scheme='B1', solver='milp',
                   top_k_values=(1, 10, 20), k_max=15, time_limit=None,
                   max_solutions=20, threshold_frac=0.01, randomSeed=0,
                   folder=None, **qubo_kwargs):
    """Fit ``target`` by linear combinations of reference spectra.

    A pyfitit-``tryAllMixtures``-style interface over QUBIC-LCF.  Two
    backends (``method=``):

    - ``'qubo'`` (default): :func:`qubic.pipeline.run_pipeline` with the
      ``scheme=``/``solver=`` knobs (QUBIC buttons); ``componentCount`` is
      used as the QUBO cardinality target when given.
    - ``'enumerate'``: try mixtures of ``componentCount`` reference spectra
      (``mixtureTrysCount`` subsets), fit concentrations and keep the min-BIC
      mixture — the 'перебор смесей' mode of the original.

    Parameters
    ----------
    reference_spectra : array-like (K, N), sequence of spectra, dict
        {name: spectrum} or pyfitit ``ML.Sample``-like object.
        Columns/entries are reference components.
    target : 1D array-like, pyfitit Spectrum-like (``.y``) or dict
        {'spectrum': ...}; alternatively pass ``unknownCharacterization``
        (pyfitit-style dict with 'type': 'spectrum' and 'spectrum').
    componentCount : int or None
        Number of mixture components.  None: no cardinality target (QUBO
        cardinality penalty off) / sizes 1..min(k_max, N) in 'enumerate'.
    mixtureTrysCount : str or int
        'all combinations of singles' (default) or an int number of randomly
        drawn subsets (see module docstring).  Only used by 'enumerate'.
    componentNames : list of str or None
        Names of the reference components (default: indices as strings).
    optimizeConcentrations : bool
        True -> NNLS concentration fit; False -> random Dirichlet split.

    Other Parameters
    ----------------
    scheme, solver : QUBIC buttons forwarded to the QUBO pipeline.
    top_k_values, k_max, time_limit, max_solutions, threshold_frac :
        forwarded to :func:`qubic.pipeline.run_pipeline`.
    randomSeed : int (pyfitit spelling), folder : ignored (signature compat).
    **qubo_kwargs : extra encoding options (lam_sparse, lam_link, lam_antimix, k).

    Returns
    -------
    dict
        ``{'method', 'scheme', 'solver', 'componentCount', 'selected',
        'component_names', 'concentrations', 'k_pred', 'rss', 'bic',
        'r_factor', 'mixture_spectrum', 'wall_time', 'candidates'}`` where
        ``candidates`` holds every tried mixture (enumerate) or the per-top-k
        winners (qubo), each with the same metric keys.
    """
    A, sample_names = _as_matrix(reference_spectra, spType=spType,
                                 componentNameColumn=componentNameColumn)
    y = _as_target(target, unknownCharacterization=unknownCharacterization)
    if A.shape[0] != y.shape[0]:
        raise ValueError(f'shape mismatch: A is {A.shape}, target is {y.shape}')
    K, N = A.shape
    if componentNames is None:
        componentNames = sample_names if sample_names is not None \
            else [str(i) for i in range(N)]
    if len(componentNames) != N:
        raise ValueError('componentNames length must match the number of '
                         'reference spectra')
    if componentCount is not None and not 1 <= componentCount <= N:
        raise ValueError(f'componentCount must be in 1..{N}')

    t0 = time.time()
    if method == 'qubo':
        out = run_pipeline(
            A, y, k_true=None, gt_indices=None, scheme=scheme, solver=solver,
            top_k_values=top_k_values, k_max=k_max, time_limit=time_limit,
            max_solutions=max_solutions, threshold_frac=threshold_frac,
            k=componentCount, **qubo_kwargs)
        candidates = []
        for res in out['results'].values():
            rec = _candidate_record(A, y, res['selected'], res['concentrations'],
                                    res['bic'])
            rec['n_candidates'] = res['n_candidates']
            candidates.append(rec)
        candidates.sort(key=lambda c: c['bic'])
    elif method == 'enumerate':
        candidates = _enumerate_mixtures(
            A, y, componentCount, mixtureTrysCount, optimizeConcentrations,
            randomSeed, k_max)
    else:
        raise ValueError("method must be 'qubo' or 'enumerate'")
    wall_time = time.time() - t0

    if not candidates:
        raise RuntimeError('no mixture candidates were produced')
    best = candidates[0]
    return {
        'method': method,
        'scheme': scheme,
        'solver': solver if method == 'qubo' else None,
        'componentCount': componentCount,
        'selected': best['selected'],
        'component_names': [componentNames[i] for i in best['selected']],
        'concentrations': best['concentrations'],
        'k_pred': best['k_pred'],
        'rss': best['rss'],
        'bic': best['bic'],
        'r_factor': best['r_factor'],
        'mixture_spectrum': best['mixture_spectrum'],
        'wall_time': float(wall_time),
        'candidates': candidates,
    }


def _enumerate_mixtures(A, y, componentCount, mixtureTrysCount,
                        optimizeConcentrations, randomSeed, k_max):
    """Try mixtures of reference spectra and rank them by BIC.

    'all combinations of singles' -> itertools.combinations of size
    ``componentCount`` (or 1..min(k_max, N) when it is None); an int -> that
    many random subsets drawn with ``randomSeed``.
    """
    K, N = A.shape
    rng = np.random.default_rng(randomSeed)

    if mixtureTrysCount == 'all combinations for each label':
        raise ValueError("'all combinations for each label' requires pyfitit "
                         "label data and is not supported; pass a number or "
                         "'all combinations of singles'")

    if mixtureTrysCount == 'all combinations of singles':
        sizes = [componentCount] if componentCount is not None \
            else list(range(1, min(k_max, N) + 1))
        subsets = []
        for size in sizes:
            subsets.extend(itertools.combinations(range(N), size))
        if len(subsets) > _MAX_ENUMERATED:
            raise ValueError(
                f'{len(subsets)} mixtures exceed _MAX_ENUMERATED; '
                "use method='qubo' or a smaller componentCount")
    elif isinstance(mixtureTrysCount, (int, np.integer)):
        if componentCount is None:
            raise ValueError('a numeric mixtureTrysCount requires componentCount')
        subsets = []
        for _ in range(int(mixtureTrysCount)):
            subsets.append(tuple(sorted(rng.choice(N, componentCount,
                                                   replace=False).tolist())))
    else:
        raise ValueError(f'unknown mixtureTrysCount: {mixtureTrysCount!r}')

    candidates = []
    for cols in subsets:
        cols = [int(i) for i in cols]
        if optimizeConcentrations:
            conc, rss = nnls_refine(A, y, cols)
        else:
            conc = rng.dirichlet(np.ones(len(cols)))
            rss = float(np.sum((y - A[:, cols] @ conc) ** 2))
        bic = compute_bic(max(rss, EPS_RSS), len(cols), K)
        candidates.append(_candidate_record(A, y, cols, conc, bic))
    candidates.sort(key=lambda c: c['bic'])
    return candidates


#: Camel-case alias matching the pyfitit function name.
tryAllMixtures = tryallmixtures