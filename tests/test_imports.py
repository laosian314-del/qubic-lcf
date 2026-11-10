"""Smoke tests: the package imports and exposes the documented public API.

These tests intentionally avoid any optional dependency (pulp, ortools,
dimod, torch): they must pass in a minimal numpy/scipy/scikit-learn
environment such as the CI setup.
"""

import qubic


def test_version_is_set_and_string():
    assert qubic.__version__
    assert isinstance(qubic.__version__, str)


def test_public_api_names_present():
    names = dir(qubic)
    for required in ("build_qubo", "solve", "run_pipeline"):
        assert required in names, f"missing public name: {required}"
