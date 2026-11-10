"""Tests for the high-level facade: Problem + Solver objects + Result.

Covers the uniform solve() contract (QUBO-backed and alternative backend),
the lazy QUBO semantics ("don't pay for what you don't use"), configurable
penalties (lambdas as dict or callable) and the tau concentration cutoff.
"""
import numpy as np
import pytest

import qubic


def make_lcf_problem(seed=0, m=40, n=6, k=2, tau=0.01):
    """Small synthetic LCF task with a known sparse non-negative truth."""
    rng = np.random.RandomState(seed)
    A = rng.uniform(0.0, 1.0, size=(m, n))
    c_true = np.zeros(n)
    support = rng.choice(n, size=k, replace=False)
    c_true[support] = rng.uniform(0.3, 0.8, size=k)
    y = A @ c_true
    return qubic.Problem(A, y, k_max=k, tau=tau), support


def check_result(res, k_max):
    assert isinstance(res, qubic.Result)
    assert isinstance(res.selected, tuple)
    assert 1 <= len(res.selected) <= k_max
    assert len(res.concentrations) == len(res.selected)
    assert np.all(res.concentrations >= 0.0)
    assert np.isfinite(res.rss)
    assert res.bic is not None
    assert isinstance(res.top_K, list)
    assert isinstance(res.stats, dict)
    # aliases agree with the canonical fields
    assert res.best_support == res.selected
    assert res.best_score == res.bic


def test_qubo_backend_solve():
    task, _ = make_lcf_problem()
    res = task.solve(qubic.MILP(time_limit=10.0), top_k=5)
    check_result(res, task.k_max)
    assert len(res.top_K) <= 5
    assert res.stats  # run statistics recorded


def test_alternative_backend_solve():
    task, _ = make_lcf_problem()
    res = task.solve(qubic.OMP())
    check_result(res, task.k_max)
    # alternative backend: no QUBO energy attached to candidates
    assert all(c.get("qubo_energy") is None for c in res.top_K)


def test_lazy_qubo_not_built_for_alternative():
    task, _ = make_lcf_problem()
    task.solve(qubic.OMP())
    assert task._qubo_memo == {}


def test_qubo_memoized_across_solvers():
    task, _ = make_lcf_problem()
    task.solve(qubic.MILP(time_limit=10.0))
    assert len(task._qubo_memo) == 1
    task.solve(qubic.HiGHS(time_limit=10.0))  # same encoding + penalties
    assert len(task._qubo_memo) == 1  # built at most once


def test_lambdas_dict_accepted():
    task, _ = make_lcf_problem()
    res = task.solve(qubic.MILP(time_limit=10.0,
                                lambdas={"lam_sparse": 0.5, "lam_link": 2.0}))
    check_result(res, task.k_max)


def test_lambdas_callable_spectrum_dependent():
    task, _ = make_lcf_problem()

    def penalties(A, y):
        scale = float(y @ y)
        return {"lam_sparse": 0.01 * scale, "lam_link": 0.05 * scale}

    res = task.solve(qubic.MILP(time_limit=10.0, lambdas=penalties))
    check_result(res, task.k_max)


def test_tau_trims_small_concentrations():
    task, _ = make_lcf_problem(tau=0.05)
    res = task.solve(qubic.MILP(time_limit=10.0))
    # refinement reports only concentrations above the tau cutoff
    assert all(c > task.tau for c in res.concentrations)


def test_solve_rejects_non_solver_object():
    task, _ = make_lcf_problem()
    with pytest.raises(TypeError, match="solver object"):
        task.solve("milp")


def test_problem_validates_shapes():
    with pytest.raises(ValueError):
        qubic.Problem(np.ones((3, 2)), np.ones(4))
    with pytest.raises(ValueError):
        qubic.Problem(np.ones((3, 2)), np.ones(3), k_max=0)
    with pytest.raises(ValueError):
        qubic.Problem(np.ones((3, 2)), np.ones(3), tau=-1.0)
