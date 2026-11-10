"""Unit tests for qubic.encodings.

Adapted from the research repository's scripts/test_package3_lib.py to
pytest style. Covers:
  (a) linked_B2 == original B2 (Q, q allclose 1e-12);
  (b) variable counts M for all 7 registered schemes;
  (c) energy_terms decomposition + link / anti-mixing penalties;
  (d) decode_fn contract (types, selection rule, concentration values);
  (e) Q generation on a small synthetic A (5x2) for every scheme and the
      E(x) identity against a direct objective recomputation.
"""
import numpy as np
import pytest

from qubic.encodings import (
    SCHEMES,
    build_qubo,
    build_qubo_B1,
    build_qubo_B2,
    build_qubo_3bit,
    build_qubo_4bit,
    build_qubo_linked,
    energy_terms,
    gammas_from_absolute,
    lambdas_for_A,
    scale_s_A,
)

RNG = np.random.default_rng(20261019)

ALL_SCHEMES = ['B1', 'B2', '3bit', '4bit',
               'linked_B2', 'linked_B3', 'linked_B4']

EXPECTED_VARS_PER_COMPONENT = {
    'B1': 1, 'B2': 3, '3bit': 3, '4bit': 4,
    'linked_B2': 3, 'linked_B3': 4, 'linked_B4': 5,
}


def random_problem():
    K = int(RNG.integers(3, 8))
    N = int(RNG.integers(2, 6))
    A = RNG.normal(size=(K, N))
    y = RNG.normal(size=K)
    k = int(RNG.integers(1, N + 1))
    return A, y, k


def _direct_objective(scheme, A, y, k, x, lam_sparse, lam_link, lam_rub):
    """Independent recomputation of the full encoding objective H(x).

    Includes the state-independent constants ||y||^2 + lam_sparse * k^2 that
    the builders omit from Q, q. Mirrors the documented physics of each
    encoding, not the builder code.
    """
    N = A.shape[1]
    x = np.asarray(x, dtype=float).ravel()
    c = np.zeros(N)
    sel_bits = np.zeros(N)
    link = 0.0
    pattern = 0.0

    if scheme == 'B1':
        c = x.copy()
        sel_bits = x.copy()
    elif scheme in ('B2', 'linked_B2', 'linked_B3', 'linked_B4'):
        B = 2 if scheme == 'B2' else int(scheme.split('_B')[1])
        vpc = 3 if scheme == 'B2' else B + 1
        alpha = [2 ** b / float(2 ** B - 1) for b in range(B)]
        for i in range(N):
            s = x[vpc * i]
            sel_bits[i] = s
            for b in range(B):
                qb = x[vpc * i + 1 + b]
                c[i] += alpha[b] * qb
                link += qb * (1 - s)
        link *= lam_link
    elif scheme == '3bit':
        alpha = np.array([4.0 / 7, 2.0 / 7, 1.0 / 7])
        for j in range(N):
            b = x[3 * j:3 * j + 3]
            c[j] = alpha @ b
            sel_bits[j] = b[0]
            f = b[0] + b[1] + b[2] - b[0] * b[1] - b[1] * b[2] - b[0] * b[2]
            pattern += f
        pattern *= lam_rub
    elif scheme == '4bit':
        alpha = np.array([8.0 / 15, 4.0 / 15, 2.0 / 15, 1.0 / 15])
        for j in range(N):
            b = x[4 * j:4 * j + 4]
            c[j] = alpha @ b
            sel_bits[j] = b[0]
            b1, b2, b3 = b[1], b[2], b[3]    # anti-mixing on bits 1..3 only
            f = b1 + b2 + b3 - b1 * b2 - b2 * b3 - b1 * b3
            pattern += f
        pattern *= lam_rub
    else:
        raise ValueError(scheme)

    residual = float(np.sum((y - A @ c) ** 2))
    sparsity = float(lam_sparse * (sel_bits.sum() - k) ** 2)
    return residual + sparsity + link + pattern


# --------------------------------------------------------------------------
# (a) linked_B2 == original B2 exactly
# --------------------------------------------------------------------------

def test_linked_B2_matches_original_B2():
    for trial in range(5):
        A, y, k = random_problem()
        lam_s = float(RNG.choice([1.0, 2.0]))
        lam_l = float(RNG.choice([5.0, 3.5]))
        Q1, q1, vm1, dec1 = build_qubo('B2', A, y, k,
                                       lam_sparse=lam_s, lam_link=lam_l)
        Q2, q2, vm2, dec2 = build_qubo_linked(A, y, k, 2,
                                              lam_sparse=lam_s, lam_link=lam_l)
        assert Q1.shape == Q2.shape, (Q1.shape, Q2.shape)
        assert np.allclose(Q1, Q2, rtol=1e-12, atol=1e-12), \
            f'trial {trial}: Q mismatch'
        assert np.allclose(q1, q2, rtol=1e-12, atol=1e-12), \
            f'trial {trial}: q mismatch'
        # decoders agree on random states
        for _ in range(20):
            x = RNG.integers(0, 2, size=Q1.shape[0]).astype(float)
            s1, c1 = dec1(x)
            s2, c2 = dec2(x)
            assert np.array_equal(s1, s2)
            assert np.allclose(c1, c2)


# --------------------------------------------------------------------------
# (b) registry + variable counts for every scheme
# --------------------------------------------------------------------------

def test_schemes_registry_keys():
    assert set(SCHEMES.keys()) == set(ALL_SCHEMES)
    # dispatcher rejects unknown schemes
    A, y, k = random_problem()
    with pytest.raises(ValueError):
        build_qubo('nope', A, y, k)


@pytest.mark.parametrize('scheme', ALL_SCHEMES)
def test_variable_counts(scheme):
    A, y, k = random_problem()
    N = A.shape[1]
    vpc = EXPECTED_VARS_PER_COMPONENT[scheme]
    M = vpc * N
    Q, q, vm, dec = build_qubo(scheme, A, y, k)
    assert Q.shape == (M, M), (scheme, Q.shape)
    assert q.shape == (M,)
    assert vm['scheme'] == scheme
    assert vm['N'] == N
    assert vm['M'] == M
    assert vm['vars_per_component'] == vpc
    assert vm['selection'] in ('direct', 'explicit', 'any_bit')


def test_variable_counts_linked_direct():
    for B in (2, 3, 4):
        A, y, k = random_problem()
        N = A.shape[1]
        Q, q, vm, dec = build_qubo_linked(A, y, k, B)
        M = (B + 1) * N
        assert Q.shape == (M, M), (B, Q.shape)
        assert q.shape == (M,)
        assert vm['M'] == M
        assert vm['N'] == N
        assert vm['B'] == B
        assert vm['idx_s'](0) == 0
        assert vm['idx_q'](0, 0) == 1
        assert vm['idx_q'](0, B - 1) == B
        assert vm['idx_s'](1) == B + 1
        assert len(vm['alpha']) == B


# --------------------------------------------------------------------------
# (c) energy decomposition consistency on every scheme
# --------------------------------------------------------------------------

def test_energy_terms_all_schemes():
    for scheme in ALL_SCHEMES:
        for trial in range(3):
            A, y, k = random_problem()
            lam_s = float(RNG.choice([1.0, 2.0]))
            lam_l = 5.0
            lam_r = 5.0
            M = EXPECTED_VARS_PER_COMPONENT[scheme] * A.shape[1]
            for _ in range(30):
                x = RNG.integers(0, 2, size=M).astype(float)
                t = energy_terms(x, A, y, k, scheme,
                                 lam_sparse=lam_s, lam_link=lam_l,
                                 lam_rub=lam_r)
                # required keys
                for key in ('residual', 'sparsity', 'link', 'pattern', 'base',
                            'xQx_plus_qx', 'constant_excluded', 'match_error'):
                    assert key in t, f'{scheme}: missing key {key}'
                # base == sum of parts
                assert abs(t['base'] - (t['residual'] + t['sparsity'] +
                                        t['link'] + t['pattern'])) < 1e-9
                # excluded constants
                assert abs(t['constant_excluded'] -
                           (float(y @ y) + lam_s * k * k)) < 1e-9
                assert t['match_error'] <= 1e-9, \
                    f'{scheme} trial {trial}: match_error={t["match_error"]}'


def test_linker_states():
    """One-sided link penalty lam_link * sum q_ib * (1 - s_i) on crafted states."""
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([0.3, -0.2])
    k = 1
    lam_s, lam_l = 1.0, 5.0

    for B in (2, 3, 4):
        M = (B + 1) * 2

        def idx_s(i, B=B):
            return (B + 1) * i

        def idx_q(i, b, B=B):
            return (B + 1) * i + 1 + b

        # s=0, single q-bit=1 -> link penalty lam_link
        x = np.zeros(M)
        x[idx_q(0, 0)] = 1.0
        t = energy_terms(x, A, y, k, f'linked_B{B}',
                         lam_sparse=lam_s, lam_link=lam_l)
        assert abs(t['link'] - lam_l) < 1e-12, (B, t['link'])
        assert t['match_error'] <= 1e-9

        # s=1, all q-bits=0 -> link penalty 0
        x = np.zeros(M)
        x[idx_s(1)] = 1.0
        t = energy_terms(x, A, y, k, f'linked_B{B}',
                         lam_sparse=lam_s, lam_link=lam_l)
        assert abs(t['link']) < 1e-12, (B, t['link'])
        assert t['match_error'] <= 1e-9

        # s=0, all B q-bits=1 -> link penalty B*lam_link
        x = np.zeros(M)
        for b in range(B):
            x[idx_q(1, b)] = 1.0
        t = energy_terms(x, A, y, k, f'linked_B{B}',
                         lam_sparse=lam_s, lam_link=lam_l)
        assert abs(t['link'] - B * lam_l) < 1e-12, (B, t['link'])
        assert t['match_error'] <= 1e-9

        # s=1, all q-bits=1 -> link 0; pattern must be 0 for linked schemes
        x = np.ones(M)
        t = energy_terms(x, A, y, k, f'linked_B{B}',
                         lam_sparse=lam_s, lam_link=lam_l)
        assert abs(t['link']) < 1e-12
        assert abs(t['pattern']) < 1e-12
        assert t['match_error'] <= 1e-9

        # one-sided linker: s=1 with q=0 is NOT penalized even though
        # s*(1-q)=1 (reverse term absent) -- covered by 'link == 0' above.


def test_anti_mixing_penalties():
    """Rubtsov anti-mixing penalty on crafted bit states."""
    A = np.eye(2)
    y = np.array([0.3, -0.2])
    k = 1
    lam_s, lam_r = 1.0, 5.0

    # ---- 3bit: f_j = b0 + b1 + b2 - b0*b1 - b1*b2 - b0*b2 ----
    M = 6
    # all-0 and all-1 states -> pattern 0
    for x in (np.zeros(M), np.ones(M)):
        t = energy_terms(x, A, y, k, '3bit',
                         lam_sparse=lam_s, lam_rub=lam_r)
        assert abs(t['pattern']) < 1e-12, t['pattern']
        assert t['match_error'] <= 1e-9
    # mixed states -> pattern = lam_rub * sum f_j
    x = np.zeros(M)
    x[0] = 1.0                       # (1,0,0): f = 1
    x[3:6] = [1.0, 1.0, 0.0]         # (1,1,0): f = 2 - 1 = 1
    t = energy_terms(x, A, y, k, '3bit',
                     lam_sparse=lam_s, lam_rub=lam_r)
    assert abs(t['pattern'] - lam_r * 2.0) < 1e-12, t['pattern']
    assert t['match_error'] <= 1e-9

    # ---- 4bit: anti-mixing on bits 1..3 only ----
    M = 8
    # bit pattern (1,1,1,1): lower bits all-1 -> pattern 0
    x = np.ones(M)
    t = energy_terms(x, A, y, k, '4bit',
                     lam_sparse=lam_s, lam_rub=lam_r)
    assert abs(t['pattern']) < 1e-12
    assert t['match_error'] <= 1e-9
    # component 0 bits (1,1,0,0): f on bits 1..3 = (1,0,0) -> f = 1
    x = np.zeros(M)
    x[0:4] = [1.0, 1.0, 0.0, 0.0]
    t = energy_terms(x, A, y, k, '4bit',
                     lam_sparse=lam_s, lam_rub=lam_r)
    assert abs(t['pattern'] - lam_r * 1.0) < 1e-12, t['pattern']
    assert t['match_error'] <= 1e-9


# --------------------------------------------------------------------------
# (d) decode_fn contract
# --------------------------------------------------------------------------

@pytest.mark.parametrize('scheme', ALL_SCHEMES)
def test_decode_fn_contract(scheme):
    A, y, k = random_problem()
    N = A.shape[1]
    Q, q, vm, dec = build_qubo(scheme, A, y, k)
    x = RNG.integers(0, 2, size=Q.shape[0]).astype(float)
    sel, conc = dec(x)
    # types: selected_indices int ndarray, concentrations float ndarray
    assert isinstance(sel, np.ndarray)
    assert sel.dtype.kind in ('i', 'u')
    assert isinstance(conc, np.ndarray)
    assert conc.dtype.kind == 'f'
    assert conc.shape == (N,)
    # selection indices are valid, sorted and unique
    assert sel.ndim == 1
    assert np.all((sel >= 0) & (sel < N))
    assert list(sel) == sorted(set(sel.tolist()))
    # concentrations within the encoding's range [0, 1]
    assert np.all(conc >= -1e-15) and np.all(conc <= 1.0 + 1e-15)


def test_decode_fn_values_linked():
    B = 3
    A = np.array([[1.0, 0.5, 0.2], [0.1, 1.0, 0.3]])
    y = np.array([0.7, 0.4])
    k = 2
    Q, q, vm, dec = build_qubo_linked(A, y, k, B, lam_sparse=1.5, lam_link=4.0)
    N = 3
    M = (B + 1) * N
    x = np.zeros(M)
    # select components 0 and 2
    x[vm['idx_s'](0)] = 1.0
    x[vm['idx_s'](2)] = 1.0
    x[vm['idx_q'](0, 0)] = 1.0          # c0 = 1/7
    x[vm['idx_q'](2, 1)] = 1.0          # c2 = 2/7
    x[vm['idx_q'](1, 2)] = 1.0          # unselected -> conc must be zeroed
    sel, conc = dec(x)
    assert np.array_equal(sel, np.array([0, 2]))
    alpha = [2 ** b / (2 ** B - 1) for b in range(B)]
    assert abs(conc[0] - alpha[0]) < 1e-12
    assert abs(conc[1]) < 1e-12          # zeroed for unselected
    assert abs(conc[2] - alpha[1]) < 1e-12


def test_invalid_B_rejected():
    A = np.ones((2, 2))
    y = np.ones(2)
    for B in (1, 5, -1):
        with pytest.raises(ValueError):
            build_qubo_linked(A, y, 1, B)


# --------------------------------------------------------------------------
# (e) Q generation on small synthetic A (5x2) + E(x) identity
# --------------------------------------------------------------------------

def test_q_generation_and_energy_identity():
    A = np.array([
        [1.0, 0.2],
        [0.3, 1.1],
        [0.8, 0.5],
        [0.1, 0.7],
        [1.2, 0.4],
    ])
    y = np.array([0.9, 0.6, 0.5, 0.3, 0.8])
    k = 1
    lam_s, lam_l, lam_r = 1.3, 5.0, 4.0
    N = A.shape[1]

    for scheme in ALL_SCHEMES:
        kwargs = {'lam_sparse': lam_s}
        if scheme in ('B2', 'linked_B2', 'linked_B3', 'linked_B4'):
            kwargs['lam_link'] = lam_l
        if scheme in ('3bit', '4bit'):
            kwargs['lam_rub'] = lam_r
        Q, q, vm, dec = build_qubo(scheme, A, y, k, **kwargs)
        M = EXPECTED_VARS_PER_COMPONENT[scheme] * N

        # shapes and Q convention: symmetric, both triangles filled
        assert Q.shape == (M, M), (scheme, Q.shape)
        assert q.shape == (M,)
        assert np.allclose(Q, Q.T, rtol=0, atol=1e-12), scheme

        # E(x) identity against direct objective for random binary states:
        # x^T Q x + q . x + (||y||^2 + lam_sparse k^2) == H_direct(x)
        const = float(y @ y) + lam_s * k * k
        for _ in range(20):
            x = RNG.integers(0, 2, size=M).astype(float)
            e_quad = float(x @ Q @ x + q @ x)
            h_direct = _direct_objective(
                scheme, A, y, k, x, lam_s, lam_l, lam_r)
            assert abs(e_quad + const - h_direct) < 1e-9, \
                f'{scheme}: {e_quad + const} vs {h_direct}'


# --------------------------------------------------------------------------
# scale helpers
# --------------------------------------------------------------------------

def test_scale_helpers():
    A = RNG.normal(size=(6, 4))
    s_A = scale_s_A(A)
    assert isinstance(s_A, float)
    assert abs(s_A - float(A.ravel() @ A.ravel() / A.shape[1])) < 1e-12

    lam_s0, lam_l0, s_ref = 1.0, 5.0, 2.5
    g_s, g_l = gammas_from_absolute(lam_s0, lam_l0, s_ref)
    assert abs(g_s - 1.0 / 2.5) < 1e-15
    assert abs(g_l - 5.0 / 2.5) < 1e-15

    ls, ll = lambdas_for_A(g_s, g_l, s_A)
    assert abs(ls - g_s * s_A) < 1e-15
    assert abs(ll - g_l * s_A) < 1e-15

    # round-trip: absolute -> gamma at s_ref -> absolute at s_ref
    ls2, ll2 = lambdas_for_A(g_s, g_l, s_ref)
    assert abs(ls2 - lam_s0) < 1e-12
    assert abs(ll2 - lam_l0) < 1e-12