"""QUBIC: QUBO-based combinatorial search for sparse LCF mixtures.

Scientific Python package for identifying the components of a linear
combination of reference spectra (LCF):

    y ~= sum_j c_j * A[:, j],  c_j >= 0,  sum_j c_j = 1,  |supp(c)| small

using binary encodings of the concentrations (B1/B2/B3/B4 and linked
variants) and QUBO combinatorial solvers, followed by NNLS refinement and
BIC model selection.

High-level usage (the public facade)::

    import qubic

    task = qubic.Problem(A, y, k_max=2, tau=0.01)   # A: (m, N) library, y: (m,) target
    res = task.solve(qubic.MILP(time_limit=30.0))    # QUBO-backed solver object
    res = task.solve(qubic.HiGHS(time_limit=30.0))   # QUBO-backed solver object
    res = task.solve(qubic.SA(restarts=20, T0=1.0))  # QUBO-backed solver object
    res = task.solve(qubic.OMP())                    # alternative backend
    print(res.selected, res.concentrations, res.bic)

Solver objects bundle their own settings and are passed whole to
:meth:`qubic.Problem.solve`; :class:`qubic.Result` is shared by both
backends.  The low-level tuple interface (``build_qubo`` -> ``solvers.solve``
-> decode) lives in :mod:`qubic.qubo` and :mod:`qubic.solvers` as an
internal/advanced layer and is deliberately not re-exported here.

Layered architecture:

1. :mod:`qubic.encodings` -- binary encodings of the concentration vector:
   ``B1`` (binary selection), ``B2`` (selection + 2-bit quantization),
   ``B3``/``B4`` (3-bit / 4-bit anti-mixing encodings), and the linked
   variants ``linked_B2``, ``linked_B3``, ``linked_B4``.
2. :mod:`qubic.qubo` -- internal QUBO builder; penalty coefficients are
   configurable through each QUBO solver's ``lambdas`` setting.
3. :mod:`qubic.solvers` -- low-level solver registry: ``milp``, ``highs``
   (PuLP HiGHS with CBC fallback), ``cp_sat``, ``scip``, ``sa``, ``tabu``,
   ``gpu_sa``, ``sqaod``, ``qaoa``.
4. :mod:`qubic.pipeline` -- refinement: NNLS + tau concentration cutoff +
   BIC/size selection.
5. :mod:`qubic.solver` -- the public :class:`Solver` class hierarchy
   (:class:`QuboSolver` subclasses and the alternative-backend classes).
6. :mod:`qubic.alternative` -- alternative backend: encoding-free methods,
   the greedy forward selection (MP/OMP) family and friends (called
   baselines only in the paper).
7. :mod:`qubic.pyfitit_api` -- ready-made PyFitIt method.

Lazy imports throughout: heavy optional dependencies (pulp, ortools,
pyscipopt, dimod, torch, sqaod) are only imported when their solver runs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

#: Package version (kept in sync with pyproject.toml).
__version__ = "0.1.0"

# name -> (module, attribute); resolved lazily on first access.
_EXPORTS = {
    # --- high-level facade ---
    "Problem": (".problem", "Problem"),
    "Result": (".problem", "Result"),
    "Solver": (".solver", "Solver"),
    "QuboSolver": (".solver", "QuboSolver"),
    "AlternativeSolver": (".solver", "AlternativeSolver"),
    # QUBO-backed solver classes
    "MILP": (".solver", "MILP"),
    "HiGHS": (".solver", "HiGHS"),
    "CP_SAT": (".solver", "CP_SAT"),
    "SCIP": (".solver", "SCIP"),
    "SA": (".solver", "SA"),
    "Tabu": (".solver", "Tabu"),
    "GPU_SA": (".solver", "GPU_SA"),
    "SQAOD": (".solver", "SQAOD"),
    "QAOA": (".solver", "QAOA"),
    # alternative-backend solver classes
    "MP": (".solver", "MP"),
    "OMP": (".solver", "OMP"),
    "Lasso": (".solver", "Lasso"),
    "TopK": (".solver", "TopK"),
    "Oracle": (".solver", "Oracle"),
    # --- encoding metadata ---
    "SCHEMES": (".encodings", "SCHEMES"),
    "SCHEME_NAMES": (".encodings", "SCHEME_NAMES"),
    "VARS_PER_COMPONENT": (".encodings", "VARS_PER_COMPONENT"),
    "RUN_SIZES": (".encodings", "RUN_SIZES"),
    # --- pipeline (internal/advanced) ---
    "run_pipeline": (".pipeline", "run_pipeline"),
    "refine_candidate": (".pipeline", "refine_candidate"),
    "compute_bic": (".pipeline", "compute_bic"),
    "compute_f1": (".pipeline", "compute_f1"),
    # --- alternative backend (called baselines only in the paper) ---
    "omp_bic": (".alternative", "omp_bic"),
    "mp_bic": (".alternative", "mp_bic"),
    "lasso_positive": (".alternative", "lasso_positive"),
    "topk_corr_bic": (".alternative", "topk_corr_bic"),
    "oracle_exact": (".alternative", "oracle_exact"),
    "run_all_alternative": (".alternative", "run_all_alternative"),
    "ALTERNATIVE_METHODS": (".alternative", "ALTERNATIVE_METHODS"),
    # --- low-level solver registry (advanced) ---
    "SOLVERS": (".solvers", "SOLVERS"),
    "available_solvers": (".solvers", "available_solvers"),
    # --- pyfitit adapter ---
    "tryallmixtures": (".pyfitit_api", "tryallmixtures"),
}

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from .alternative import (ALTERNATIVE_METHODS, lasso_positive, mp_bic,
                              omp_bic, oracle_exact, run_all_alternative,
                              topk_corr_bic)
    from .encodings import (RUN_SIZES, SCHEMES, SCHEME_NAMES,
                            VARS_PER_COMPONENT)
    from .pipeline import (compute_bic, compute_f1, refine_candidate,
                           run_pipeline)
    from .problem import Problem, Result
    from .pyfitit_api import tryallmixtures
    from .solver import (AlternativeSolver, CP_SAT, GPU_SA, HiGHS, Lasso,
                         MILP, MP, OMP, Oracle, QAOA, SA, SQAOD, SCIP,
                         Solver, Tabu, TopK, QuboSolver)
    from .solvers import SOLVERS, available_solvers


def __getattr__(name):
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'qubic' has no attribute {name!r}") from None
    import importlib
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = sorted(_EXPORTS)
