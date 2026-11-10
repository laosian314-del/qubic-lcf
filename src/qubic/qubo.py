"""Internal QUBO construction layer (advanced entry points).

Wraps :mod:`qubic.encodings` behind one stable, penalty-aware API used by the
solver facade (:mod:`qubic.solver`):

- :func:`build_qubo` -- build ``(Q, q, var_map, decode)`` for a scheme with
  explicit penalty coefficients;
- :func:`resolve_lambdas` -- resolve the ``lambdas`` setting (dict of floats
  or callable ``(A, y) -> dict of floats``) to concrete floats;
- :func:`compute_qubo` -- convenience combination of the two.

This is an internal/advanced layer of the stack

    encodings -> QUBO builder (uses lambdas) -> low-level solver registry
        -> refinement (NNLS + tau threshold + selection)

and is NOT re-exported from the :mod:`qubic` top-level facade; the public
surface is :class:`qubic.Problem` and the solver classes.
"""
from __future__ import annotations

from .encodings import SCHEMES

#: Penalty keyword names accepted by each scheme's builder.  These are the
#: allowed keys of the ``lambdas`` mapping: ``lam_sparse`` (cardinality
#: penalty), ``lam_link`` (selection/quantization link penalty), ``lam_antimix``
#: (3-bit / 4-bit anti-mixing penalty).
PENALTY_KEYS = {
    'B1': ('lam_sparse',),
    'B2': ('lam_sparse', 'lam_link'),
    'B3': ('lam_sparse', 'lam_antimix'),
    'B4': ('lam_sparse', 'lam_antimix'),
    'linked_B2': ('lam_sparse', 'lam_link'),
    'linked_B3': ('lam_sparse', 'lam_link'),
    'linked_B4': ('lam_sparse', 'lam_link'),
}


def _check_lambdas(encoding, lambdas):
    """Validate that ``lambdas`` only names penalties of ``encoding``."""
    allowed = set(PENALTY_KEYS[encoding])
    unknown = set(lambdas) - allowed
    if unknown:
        raise ValueError(
            f"lambdas keys {sorted(unknown)} are not penalty names of "
            f"encoding {encoding!r} (allowed: {sorted(allowed)})"
        )


def build_qubo(scheme, A, target, k, **penalties):
    """Build ``(Q, q, var_map, decode)`` for one scheme (advanced entry).

    ``penalties`` must only name the scheme's penalty keywords (see
    :data:`PENALTY_KEYS`); missing keys fall back to the encoding builder
    defaults (``lam_sparse=1.0``, ``lam_link=5.0``, ``lam_antimix=5.0``).
    """
    if scheme not in SCHEMES:
        raise ValueError(
            f"Unknown scheme: {scheme}. Choose from {sorted(SCHEMES)}"
        )
    _check_lambdas(scheme, penalties)
    return SCHEMES[scheme](A, target, k, **penalties)


def resolve_lambdas(encoding, lambdas, A, y, k):
    """Resolve the ``lambdas`` setting to a concrete dict of floats.

    ``lambdas`` may be:

    - ``None``: use the encoding builder defaults;
    - a dict of floats keyed by the scheme's penalty names
      (e.g. ``{'lam_sparse': 1.0, 'lam_link': 5.0}``);
    - a callable ``(A, y) -> dict of floats`` so the penalties can depend on
      the spectrum scale, e.g. proportional to ``||y||^2``::

          lambda A, y: {'lam_sparse': 0.01 * float(y @ y),
                        'lam_link': 0.05 * float(y @ y)}

    Missing ``lam_sparse`` defaults to 1.0 when a cardinality target ``k``
    is given and to 0.0 (penalty switched off) when ``k`` is 0.
    """
    if lambdas is None:
        lam = {}
    elif callable(lambdas):
        lam = lambdas(A, y)
        if not isinstance(lam, dict):
            raise TypeError("lambdas callable must return a dict of floats")
        lam = dict(lam)
    elif isinstance(lambdas, dict):
        lam = dict(lambdas)
    else:
        raise TypeError(
            "lambdas must be a dict of floats, a callable (A, y) -> dict, "
            "or None"
        )
    _check_lambdas(encoding, lam)
    lam.setdefault('lam_sparse', 1.0 if k else 0.0)
    return {name: float(value) for name, value in lam.items()}


def compute_qubo(A, y, encoding, k, lambdas=None):
    """Build the QUBO ``(Q, q, var_map, decode)`` with resolved ``lambdas``.

    Combination of :func:`resolve_lambdas` and :func:`build_qubo`.
    """
    lam = resolve_lambdas(encoding, lambdas, A, y, k)
    return build_qubo(encoding, A, y, k, **lam)
