"""Smoke tests: the package imports and exposes the documented public API.

These tests intentionally avoid running any solver: they must pass in a
minimal numpy/scipy environment such as the CI setup.  The public facade is
the Problem/Solver object API; the low-level tuple interface (build_qubo,
solvers.solve) is internal/advanced and is not re-exported at the top level.
"""
import qubic


def test_version_is_set_and_string():
    assert qubic.__version__
    assert isinstance(qubic.__version__, str)


def test_public_api_names_present():
    names = dir(qubic)
    required = (
        # high-level facade
        "Problem", "Result", "Solver", "QuboSolver", "AlternativeSolver",
        # QUBO-backed solver classes
        "MILP", "HiGHS", "CP_SAT", "SCIP", "SA", "Tabu", "GPU_SA", "SQAOD",
        "QAOA",
        # alternative-backend solver classes
        "MP", "OMP", "Lasso", "TopK", "Oracle",
        # encoding metadata
        "SCHEMES", "SCHEME_NAMES",
        # alternative backend + pyfitit adapter
        "run_all_alternative", "ALTERNATIVE_METHODS", "tryallmixtures",
    )
    for name in required:
        assert name in names, f"missing public name: {name}"


def test_internal_qubo_layer_not_exported():
    # build_qubo/compute_qubo live in qubic.qubo (internal/advanced layer)
    # and are deliberately not part of the top-level facade.
    assert "build_qubo" not in dir(qubic)
    assert "compute_qubo" not in dir(qubic)


def test_scheme_registry_has_documented_names():
    assert set(qubic.SCHEMES) == {
        "B1", "B2", "B3", "B4", "linked_B2", "linked_B3", "linked_B4",
    }
