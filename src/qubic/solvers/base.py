"""Shared signature and utilities for the QUBIC-LCF solver layer.

Every solver module exposes a unified entry point::

    solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, **kwargs)
        -> (solutions, wall_time, gap)

where ``solutions`` is a list of ``(sample, energy)`` pairs with ``sample`` a
``np.ndarray`` of shape ``(M,)`` holding 0/1 decisions and ``energy`` the exact
QUBO energy ``x^T Q x + q^T x``, sorted by ascending energy and truncated to
``top_k``; ``wall_time`` is the measured wall-clock seconds and ``gap`` is an
optimality gap (float) when the solver can report one, else ``None``.

Only numpy may be imported at module level; every optional solver dependency
(pulp, ortools, pyscipopt, dimod, dwave-samplers, torch, scipy) is imported
lazily inside the functions that need it, via :func:`require`.
"""
from __future__ import annotations

import importlib
import time

import numpy as np

#: Default wall-clock budget (seconds) for stochastic solvers when the caller
#: passes ``time_limit=None``.
DEFAULT_TIME_LIMIT = 30.0


def require(module_name, pip_name=None):
    """Import ``module_name`` or raise ImportError with an install hint.

    Parameters
    ----------
    module_name : str
        Dotted module path, e.g. ``'dwave.samplers'``.
    pip_name : str, optional
        PyPI package name to suggest if it differs from the module path
        (e.g. module ``'dwave.samplers'`` comes from package ``dwave-samplers``).
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        pkg = pip_name or module_name
        raise ImportError(
            f"solver dependency {module_name!r} is not installed; "
            f"install it with: pip install {pkg}"
        ) from exc


def get_selection(sample, var_map):
    """Extract the boolean component-selection vector z_j from a raw sample.

    Ported from the QUBO_LCF_3 scripts (``_get_selection`` / ``_get_selection_from_x``).
    """
    N = var_map["N"]
    scheme = var_map["scheme"]
    z = [False] * N
    if scheme == "B1":
        for j in range(N):
            z[j] = sample[j] > 0.5
    elif scheme == "B2":
        idx_s = var_map["idx_s"]
        for j in range(N):
            z[j] = sample[idx_s(j)] > 0.5
    elif scheme in ("B3", "B4"):
        vpc = var_map["vars_per_component"]
        idx_fn = var_map["idx"]
        for j in range(N):
            z[j] = any(sample[idx_fn(j, b)] > 0.5 for b in range(vpc))
    elif scheme.startswith("linked_"):
        idx_s = var_map["idx_s"]
        for j in range(N):
            z[j] = sample[idx_s(j)] > 0.5
    else:
        raise ValueError(
            f"get_selection: unsupported scheme {scheme!r}; "
            "expected 'B1', 'B2', 'B3', 'B4' or 'linked_B{2,3,4}'"
        )
    return z


def qubo_energy(sample, Q, q):
    """Exact QUBO energy ``x^T Q x + q^T x`` of a binary sample."""
    x = np.asarray(sample, dtype=float).reshape(-1)
    return float(x @ np.asarray(Q, dtype=float) @ x + np.asarray(q, dtype=float) @ x)


def resolve_time_limit(time_limit, default=DEFAULT_TIME_LIMIT):
    """Map ``time_limit=None`` to ``default`` (used by stochastic solvers)."""
    return default if time_limit is None else float(time_limit)


def finalize(solutions, Q, q, top_k):
    """Normalize raw solver output to the unified ``(solutions, ...)`` shape.

    Accepts an iterable of samples or ``(sample, energy)`` pairs; dedupes by
    sample, recomputes the exact QUBO energy (so all solvers report energies on
    the same scale), sorts by ascending energy and truncates to ``top_k``.

    Returns
    -------
    list of (np.ndarray (M,) 0/1, float)
    """
    Q = np.asarray(Q, dtype=float)
    q = np.asarray(q, dtype=float)
    seen = {}
    for item in solutions:
        sample = item[0] if isinstance(item, tuple) else item
        arr = np.asarray(sample, dtype=float).reshape(-1)
        if arr.shape[0] != Q.shape[0]:
            raise ValueError(
                f"finalize: sample length {arr.shape[0]} != Q dimension {Q.shape[0]}"
            )
        key = tuple(arr.tolist())
        if key not in seen:
            seen[key] = arr
    result = [(arr, qubo_energy(arr, Q, q)) for arr in seen.values()]
    result.sort(key=lambda pair: pair[1])
    if top_k is not None:
        result = result[: int(top_k)]
    return result


def reject_unknown_kwargs(fn_name, kwargs):
    """Raise TypeError for unexpected keyword arguments (fail fast, loudly)."""
    if kwargs:
        raise TypeError(
            f"{fn_name}() got unexpected keyword argument(s): {sorted(kwargs)}"
        )


__all__ = [
    "DEFAULT_TIME_LIMIT",
    "require",
    "get_selection",
    "qubo_energy",
    "resolve_time_limit",
    "finalize",
    "reject_unknown_kwargs",
    "time",
]
