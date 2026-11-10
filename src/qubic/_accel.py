"""Shared lazy-acceleration infrastructure ("don't pay for what you don't use").

The optional numba/JIT acceleration paths throughout the package are
**opt-in** (``use_jit=True`` on solver/method entry points) and require the
optional ``numba`` dependency::

    pip install "qubic-lcf[jit]"

Nothing here imports numba eagerly: :func:`get_numba` performs the import on
first use, so pure-NumPy users never pay for it.

.. note::
   The optional JIT acceleration paths were **NOT used in the article**;
   all article results were obtained with the pure NumPy code paths.
"""
from __future__ import annotations

_numba = None
_checked = False


def get_numba():
    """Return the ``numba`` module, or None when it is not installed.

    The import happens on first call only (lazy); the result is cached.
    """
    global _numba, _checked
    if not _checked:
        _checked = True
        try:
            import numba
            _numba = numba
        except ImportError:
            _numba = None
    return _numba


def numba_available():
    """True when the optional ``numba`` dependency can be imported."""
    return get_numba() is not None


def require_numba():
    """Return the ``numba`` module or raise a helpful ImportError.

    To be called from opt-in (``use_jit=True``) code paths only.
    """
    nb = get_numba()
    if nb is None:
        raise ImportError(
            "use_jit=True requires the optional 'numba' dependency; "
            "install it with: pip install 'qubic-lcf[jit]'"
        )
    return nb
