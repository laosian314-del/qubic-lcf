"""QUBIC-LCF: QUBO-based combinatorial linear-combination fitting for XAS spectra.

Public API. Submodules are imported lazily (PEP 562), so importing this
package requires only numpy/scipy for whatever you actually use; optional
solver dependencies (pulp, ortools, dimod, torch) are resolved only when the
corresponding solver is requested.
"""

from __future__ import annotations

import importlib

__version__ = "0.1.0"

_EXPORTS = {
    # encodings
    "SCHEMES": (".encodings", "SCHEMES"),
    "build_qubo": (".encodings", "build_qubo"),
    "build_qubo_B1": (".encodings", "build_qubo_B1"),
    "build_qubo_B2": (".encodings", "build_qubo_B2"),
    "build_qubo_3bit": (".encodings", "build_qubo_3bit"),
    "build_qubo_4bit": (".encodings", "build_qubo_4bit"),
    "build_qubo_linked": (".encodings", "build_qubo_linked"),
    "energy_terms": (".encodings", "energy_terms"),
    "scale_s_A": (".encodings", "scale_s_A"),
    "gammas_from_absolute": (".encodings", "gammas_from_absolute"),
    "lambdas_for_A": (".encodings", "lambdas_for_A"),
    # solvers
    "solve": (".solvers", "solve"),
    "SOLVERS": (".solvers", "SOLVERS"),
    "available_solvers": (".solvers", "available_solvers"),
    # pipeline
    "run_pipeline": (".pipeline", "run_pipeline"),
    "nnls_refine": (".pipeline", "nnls_refine"),
    "compute_bic": (".pipeline", "compute_bic"),
    "compute_f1": (".pipeline", "compute_f1"),
    # baselines
    "omp_bic": (".baselines", "omp_bic"),
    "mp_bic": (".baselines", "mp_bic"),
    "lasso_positive": (".baselines", "lasso_positive"),
    "topk_corr_bic": (".baselines", "topk_corr_bic"),
    "oracle_exact": (".baselines", "oracle_exact"),
    "run_all_baselines": (".baselines", "run_all_baselines"),
}

__all__ = sorted(_EXPORTS) + ["__version__"]


def __getattr__(name):
    if name in _EXPORTS:
        module_name, attr = _EXPORTS[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__
