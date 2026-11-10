"""Tests for the QUBO-build acceleration twins (default NumPy vs numba JIT).

The optional numba/JIT acceleration was NOT used in the article;
article results were obtained with the pure NumPy code paths.

Covers:
  (a) the default (pure-NumPy) path reproduces the pre-refactor
      implementation output exactly (reference values captured before
      the refactor and embedded below, seeded synthetic case);
  (b) default vs ``use_jit=True`` parity on every scheme in SCHEMES
      (Q, q, var_map, decode) on seeded synthetic data;
  (c) the default path never imports numba (subprocess test);
  (d) ``use_jit=True`` raises a helpful ImportError when numba is absent;
  (e) lambdas semantics (dict / callable / None via resolve_lambdas) and
      the var_map contract are unchanged.
"""
import base64
import io
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from qubic import _accel
from qubic import encodings
from qubic.qubo import (
    PENALTY_KEYS,
    build_qubo,
    compute_qubo,
    resolve_lambdas,
)

ALL_SCHEMES = sorted(encodings.SCHEMES)

# Reference Q / q values captured from the PRE-REFACTOR implementation
# (git HEAD of encodings.py before vectorization) on the exact seeded
# case of _reference_case(): np.savez_compressed -> base64.
REF_B64 = (
    "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"
    "FER/zN+99+NOjhf255YveuoH+b27zdHxkbHJJkPAsCo7nP4ZnzwgyYOuftkkyS6vb8ln99i8Pofz6/1h+4LfWX3f77YvOquve81W"
    "k2S29pmkkNTganXVVucp9Enha6mtV2IQrJ/uq05kP7JeziQINE2b075n7A45CvGEAk3T5hR1v8O23+No21agKdqsti9bYnsd0VkF"
    "GqQNK6yPLrTI+ehCi5yPLvSC89GFxjgfXegt56MLveV8dKHXnI8uNKk7379+PvTmqzpdhhX39EaeQEu0Fe13y8aJt8xxWIHGaJPa"
    "/98JZncvPUkCPaLN686X9dGFPnM+utAjzkcXWtKdL+ujC20ZYn10oXnORxdaIb9zfht/PtSld35F81Wd7uaZs+jHCYEWaG8097F9"
    "KhX3LivQNdrot8+vaL6sjy70gfPRhR58+/yK5sv66ELfOR9d6OE/3e/i50P9nOh+Vvf1zq9ovqoTmLtLbc1nCPSc9oqwfv39zPri"
    "8yuaL+ujC73n/Pr7mfXrz6/efFkfXegL+QRQSwMELQAAAAgAAAAhAHiZRaD//////////xAAFABsaW5rZWRfQjNfX3EubnB5AQAQ"
    "AAABAAAAAAAAsgAAAAAAAACb7BfqGxDJyFDGUK2eklqcXKRupaBuk2ahrqOgnpZfVFKUmBefX5SSChJ3S8wpTgWKF2ckFqQC+RqG"
    "ZjqaOgq1CuQDLgYoqHE3fvla8L+99qGZrNzhP+1F8s/WTDr02h4mv/uNc9f36YIOzstCHu7+I+mwime3u1KyqgNMXueo1ZW1/NwO"
    "vyayL6y6I+hw4MNtvkOtUnD5L4t/yF6ey+VwoODoq/lZgg5+68IX8SySdAAAUEsDBC0AAAAIAAAAIQBalPQW//////////8TABQA"
    "bGlua2VkX0IzX192bV9OLm5weQEAEACIAAAAAAAAAEgAAAAAAAAAm+wX6hsQychQxlCtnpJanFykbqWgbpNpoa6joJ6WX1RSlJgX"
    "n1+UkgoSd0vMKU4FihdnJBakAvkahjqaOgq1ChQALhYGCAAAUEsDBC0AAAAIAAAAIQCLqwXu//////////8TABQAbGlua2VkX0Iz"
    "X192bV9NLm5weQEAEACIAAAAAAAAAEgAAAAAAAAAm+wX6hsQychQxlCtnpJanFykbqWgbpNpoa6joJ6WX1RSlJgXn1+UkgoSd0vM"
    "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"
)

# One non-default penalty set per scheme so every penalty term is active.
PEN = {
    'B1': dict(lam_sparse=1.7),
    'B2': dict(lam_sparse=1.7, lam_link=2.3),
    'B3': dict(lam_sparse=1.7, lam_antimix=3.1),
    'B4': dict(lam_sparse=1.7, lam_antimix=3.1),
    'linked_B2': dict(lam_sparse=1.7, lam_link=2.3),
    'linked_B3': dict(lam_sparse=1.7, lam_link=2.3),
    'linked_B4': dict(lam_sparse=1.7, lam_link=2.3),
}

EXPECTED_VAR_MAP = {
    'B1': dict(scheme='B1', vars_per_component=1, selection='direct'),
    'B2': dict(scheme='B2', vars_per_component=3, selection='explicit'),
    'B3': dict(scheme='B3', vars_per_component=3, selection='any_bit'),
    'B4': dict(scheme='B4', vars_per_component=4, selection='any_bit'),
    'linked_B2': dict(scheme='linked_B2', vars_per_component=3,
                      selection='explicit'),
    'linked_B3': dict(scheme='linked_B3', vars_per_component=4,
                      selection='explicit'),
    'linked_B4': dict(scheme='linked_B4', vars_per_component=5,
                      selection='explicit'),
}


def _reference_case():
    """The exact seeded case the embedded reference was captured on."""
    rng = np.random.default_rng(20261110)
    A = rng.normal(size=(7, 4))
    y = rng.normal(size=7)
    return A, y, 2


def _load_reference():
    return np.load(io.BytesIO(base64.b64decode(REF_B64)))


# ------------------------------------------------------------------
# (a) behavior preservation vs the pre-refactor implementation
# ------------------------------------------------------------------

@pytest.mark.parametrize('scheme', ALL_SCHEMES)
def test_default_matches_pre_refactor_reference(scheme):
    ref = _load_reference()
    A, y, k = _reference_case()
    Q, q, var_map, decode = build_qubo(scheme, A, y, k, **PEN[scheme])
    np.testing.assert_allclose(Q, ref[f'{scheme}__Q'], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(q, ref[f'{scheme}__q'], rtol=1e-12, atol=1e-12)
    assert var_map['N'] == int(ref[f'{scheme}__vm_N'][0])
    assert var_map['M'] == int(ref[f'{scheme}__vm_M'][0])
    assert (var_map['vars_per_component']
            == int(ref[f'{scheme}__vm_vars_per_component'][0]))


# ------------------------------------------------------------------
# (b) default vs jit parity
# ------------------------------------------------------------------

@pytest.mark.parametrize('scheme', ALL_SCHEMES)
def test_jit_matches_default(scheme):
    pytest.importorskip('numba')
    rng = np.random.default_rng(20261110 + 1)
    A = rng.normal(size=(23, 9))
    y = rng.normal(size=23)
    k = 3
    Q0, q0, vm0, dec0 = build_qubo(scheme, A, y, k,
                                   use_jit=False, **PEN[scheme])
    Q1, q1, vm1, dec1 = build_qubo(scheme, A, y, k,
                                   use_jit=True, **PEN[scheme])
    np.testing.assert_allclose(Q0, Q1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(q0, q1, rtol=1e-12, atol=1e-12)
    # var_map contract identical between the twins
    assert vm0.keys() == vm1.keys()
    for key in vm0:
        v0, v1 = vm0[key], vm1[key]
        if callable(v0):
            assert callable(v1)   # closures: compared via decode below
        elif isinstance(v0, np.ndarray):
            np.testing.assert_allclose(v0, v1)
        else:
            assert v0 == v1
    # decode identical on a fixed random bit state
    x = (rng.random(vm0['M']) < 0.4).astype(float)
    s0, c0 = dec0(x)
    s1, c1 = dec1(x)
    np.testing.assert_array_equal(s0, s1)
    np.testing.assert_allclose(c0, c1)


def test_compute_qubo_forwards_use_jit():
    pytest.importorskip('numba')
    A, y, k = _reference_case()
    lam = PEN['linked_B4']
    Q0, q0, _, _ = compute_qubo(A, y, 'linked_B4', k, lambdas=lam)
    Q1, q1, _, _ = compute_qubo(A, y, 'linked_B4', k, lambdas=lam,
                                use_jit=True)
    np.testing.assert_allclose(Q0, Q1, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(q0, q1, rtol=1e-12, atol=1e-12)


def test_energy_terms_forwards_use_jit():
    pytest.importorskip('numba')
    A, y, k = _reference_case()
    rng = np.random.default_rng(5)
    x = (rng.random(3 * A.shape[1]) < 0.5).astype(float)
    t0 = encodings.energy_terms(x, A, y, k, 'B3',
                                lam_sparse=1.7, lam_antimix=3.1)
    t1 = encodings.energy_terms(x, A, y, k, 'B3',
                                lam_sparse=1.7, lam_antimix=3.1,
                                use_jit=True)
    for key in ('residual', 'sparsity', 'link', 'pattern', 'base',
                'xQx_plus_qx', 'constant_excluded', 'match_error'):
        np.testing.assert_allclose(t0[key], t1[key], rtol=1e-12, atol=1e-12)


# ------------------------------------------------------------------
# (c) default path must not import numba
# ------------------------------------------------------------------

def test_default_path_does_not_import_numba():
    src = str(pathlib.Path(__file__).resolve().parents[1] / 'src')
    code = (
        "import sys\n"
        "import numpy as np\n"
        "from qubic.qubo import build_qubo, compute_qubo\n"
        "from qubic.encodings import energy_terms\n"
        "rng = np.random.default_rng(0)\n"
        "A = rng.normal(size=(5, 3))\n"
        "y = rng.normal(size=5)\n"
        "for scheme in ('B1', 'B2', 'B3', 'B4', "
        "'linked_B2', 'linked_B3', 'linked_B4'):\n"
        "    build_qubo(scheme, A, y, 1)\n"
        "    build_qubo(scheme, A, y, 1, use_jit=False)\n"
        "compute_qubo(A, y, 'B2', 1, "
        "lambdas={'lam_sparse': 2.0, 'lam_link': 3.0})\n"
        "energy_terms((rng.random(9) < 0.5).astype(float), A, y, 1, 'B3')\n"
        "assert 'numba' not in sys.modules, "
        "'numba was imported on the default path'\n"
        "print('NO_NUMBA_OK')\n"
    )
    env = dict(os.environ)
    env['PYTHONPATH'] = src + os.pathsep + env.get('PYTHONPATH', '')
    proc = subprocess.run([sys.executable, '-c', code], capture_output=True,
                          text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    assert 'NO_NUMBA_OK' in proc.stdout


# ------------------------------------------------------------------
# (d) helpful error when numba is missing
# ------------------------------------------------------------------

def test_use_jit_raises_helpful_error_without_numba(monkeypatch):
    monkeypatch.setattr(_accel, 'get_numba', lambda: None)
    A, y, k = _reference_case()
    with pytest.raises(ImportError, match='use_jit=True'):
        build_qubo('B2', A, y, k, use_jit=True)
    with pytest.raises(ImportError, match='use_jit=True'):
        compute_qubo(A, y, 'linked_B3', k, use_jit=True)


# ------------------------------------------------------------------
# (e) lambdas semantics and var_map contract unchanged
# ------------------------------------------------------------------

def test_resolve_lambdas_semantics_unchanged():
    A, y, k = _reference_case()
    assert resolve_lambdas('B2', None, A, y, k) == {'lam_sparse': 1.0}
    assert resolve_lambdas('B2', None, A, y, 0) == {'lam_sparse': 0.0}
    lam = resolve_lambdas('B2', {'lam_sparse': 2.0, 'lam_link': 3.0},
                          A, y, k)
    assert lam == {'lam_sparse': 2.0, 'lam_link': 3.0}
    lam = resolve_lambdas(
        'B2', lambda A, y: {'lam_sparse': float(y @ y)}, A, y, k)
    assert lam == {'lam_sparse': float(y @ y)}
    with pytest.raises(ValueError):
        resolve_lambdas('B2', {'lam_antimix': 1.0}, A, y, k)
    with pytest.raises(TypeError):
        resolve_lambdas('B2', 'not-a-lambda', A, y, k)
    with pytest.raises(TypeError):
        resolve_lambdas('B2', lambda A, y: 3.0, A, y, k)


def test_build_qubo_rejects_unknown_penalties():
    A, y, k = _reference_case()
    with pytest.raises(ValueError):
        build_qubo('B2', A, y, k, lam_antimix=1.0)
    with pytest.raises(ValueError):
        build_qubo('B2', A, y, k, use_jit=True, lam_antimix=1.0)
    with pytest.raises(ValueError):
        build_qubo('nope', A, y, k)


@pytest.mark.parametrize('scheme', ALL_SCHEMES)
@pytest.mark.parametrize('use_jit', [False, True])
def test_var_map_contract(scheme, use_jit):
    if use_jit:
        pytest.importorskip('numba')
    A, y, k = _reference_case()
    Q, q, var_map, decode = build_qubo(scheme, A, y, k,
                                       use_jit=use_jit, **PEN[scheme])
    exp = EXPECTED_VAR_MAP[scheme]
    N = A.shape[1]
    assert var_map['scheme'] == exp['scheme']
    assert var_map['N'] == N
    assert var_map['M'] == Q.shape[0] == exp['vars_per_component'] * N
    assert var_map['vars_per_component'] == exp['vars_per_component']
    assert var_map['selection'] == exp['selection']
    if 'idx_s' in var_map:
        assert var_map['idx_s'](0) == 0
        assert var_map['idx_q'](0, 0) == 1
    if 'idx' in var_map:
        assert var_map['idx'](0, 0) == 0
    if 'alpha' in var_map:
        alpha = np.asarray(var_map['alpha'], dtype=float)
        assert alpha.shape == (exp['vars_per_component']
                               if exp['selection'] == 'any_bit'
                               else exp['vars_per_component'] - 1,)
    x = np.zeros(var_map['M'])
    x[::2] = 1.0
    sel, conc = decode(x)
    assert isinstance(sel, np.ndarray) and sel.dtype.kind == 'i'
    assert isinstance(conc, np.ndarray) and conc.shape == (N,)
