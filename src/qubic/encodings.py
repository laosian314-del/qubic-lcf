"""QUBO matrix construction for LCF with multiple encoding schemes.

Ported from the QUBO_LCF research repository (scripts/qubo_lib.py and
scripts/package3_lib.py) with the numerical conventions preserved exactly.

Schemes (keys of :data:`SCHEMES`):
  B1        -- Binary selection only (N vars). Concentrations via NNLS post-hoc.
  B2        -- Selection s_i + 2-bit quantization q_i0,q_i1 (3N vars).
               c_i = s_i * (1/3 * q_i0 + 2/3 * q_i1)
  3bit      -- Rubtsov 3-bit encoding (3N vars, no explicit selection).
               c_j = (1/7) * (4*s_j0 + 2*s_j1 + s_j2)
               Anti-mixing penalty pushes bits toward 000 or near-111.
  4bit      -- Rubtsov 4-bit encoding (4N vars).
               c_j = (1/15) * (8*s_j0 + 4*s_j1 + 2*s_j2 + s_j3)
  linked_B{2,3,4} -- Linked-selection encoding: per component i one selector
               s_i plus B quantization bits q_i0..q_i{B-1}, M = (B+1)*N.
               c_i = sum_b alpha_b * q_ib, alpha_b = 2^b / (2^B - 1)
               (LSB-first binary weights; for B=2 identical to B2).
               Objective: ||y - A c||^2
                 + lam_sparse * (sum_i s_i - k)^2
                 + lam_link   * sum_i sum_b q_ib * (1 - s_i)
               (one-sided linker, fit term on q-bits only, no anti-mixing).

Design decision: NO normalization penalty inside the QUBO.
NNLS refinement handles normalization post-hoc. This keeps Q sparse
and makes MILP tractable for N up to ~60.

Q convention (all builders): symmetric, BOTH triangles filled; a quadratic
coefficient c on x_u*x_v (u != v) is stored as Q[u,v] = Q[v,u] = c/2, so
E(x) = x^T Q x + q . x carries it exactly once.

Each builder returns:
  Q       -- symmetric matrix (M x M)
  q       -- linear vector (M,)
  var_map -- dict describing variable layout
  decode  -- function(sample) -> (selected_indices, concentrations)
             with selected_indices an int np.ndarray and concentrations a
             float np.ndarray of length N.
"""
import numpy as np


# --------------------------------------------------------------------------
# Non-linked schemes (original qubo_lib builders)
# --------------------------------------------------------------------------

def build_qubo_B1(A, target, k, lam_sparse=1.0):
    """
    B=1: Binary selection. x_i in {0,1}.
    H = ||y - A @ x||^2 + lam_sparse * (sum(x) - k)^2
    Variables: N
    """
    N = A.shape[1]
    G = A.T @ A       # N x N
    p = A.T @ target  # N

    Q = G.copy().astype(float)
    q = -2.0 * p

    # Sparsity: lam * (sum(x) - k)^2
    for i in range(N):
        Q[i, i] += lam_sparse * (1 - 2 * k)
    for i in range(N):
        for j in range(i + 1, N):
            Q[i, j] += lam_sparse
            Q[j, i] += lam_sparse

    var_map = {
        'scheme': 'B1', 'N': N, 'M': N,
        'vars_per_component': 1, 'selection': 'direct',
    }

    def decode(sample):
        x = np.asarray(sample, dtype=float).ravel()
        sel = np.array([i for i in range(N) if x[i] > 0.5], dtype=int)
        conc = np.array([float(x[i]) for i in range(N)], dtype=float)
        return sel, conc

    return Q, q, var_map, decode


def build_qubo_B2(A, target, k, lam_sparse=1.0, lam_link=5.0):
    """
    B=2: Selection s_i + 2-bit quantization.
    c_i = s_i * (1/3 * q_i0 + 2/3 * q_i1)
    Layout: [s_0, q_00, q_01, s_1, q_10, q_11, ...]
    idx(i,'s')=3*i, idx(i,0)=3*i+1, idx(i,1)=3*i+2

    NO normalization penalty — NNLS handles it.
    Link penalty: lam_link * sum q_ib * (1-s_i).
    """
    N = A.shape[1]
    M = 3 * N
    alpha = np.array([1.0 / 3, 2.0 / 3])

    Q = np.zeros((M, M))
    q = np.zeros(M)

    def idx_s(i):
        return 3 * i

    def idx_q(i, b):
        return 3 * i + 1 + b

    G = A.T @ A
    p = A.T @ target

    # Fit term (only on q variables)
    for i in range(N):
        for b1 in range(2):
            ii = idx_q(i, b1)
            q[ii] += -2.0 * alpha[b1] * p[i]
            Q[ii, ii] += alpha[b1] ** 2 * G[i, i]
            for b2 in range(b1 + 1, 2):
                jj = idx_q(i, b2)
                Q[ii, jj] += alpha[b1] * alpha[b2] * G[i, i]
                Q[jj, ii] += alpha[b1] * alpha[b2] * G[i, i]
            for j in range(i + 1, N):
                for b2 in range(2):
                    jj = idx_q(j, b2)
                    Q[ii, jj] += alpha[b1] * alpha[b2] * G[i, j]
                    Q[jj, ii] += alpha[b1] * alpha[b2] * G[i, j]

    # Sparsity: lam * (sum s_i - k)^2
    for i in range(N):
        ii = idx_s(i)
        Q[ii, ii] += lam_sparse * (1 - 2 * k)
        for j in range(i + 1, N):
            jj = idx_s(j)
            Q[ii, jj] += lam_sparse
            Q[jj, ii] += lam_sparse

    # Link: lam_link * sum_i sum_b q_ib * (1 - s_i)
    # = lam_link * [sum q_ib - sum q_ib * s_i]
    for i in range(N):
        si = idx_s(i)
        for b in range(2):
            qi = idx_q(i, b)
            q[qi] += lam_link
            Q[qi, si] += -lam_link / 2
            Q[si, qi] += -lam_link / 2

    var_map = {
        'scheme': 'B2', 'N': N, 'M': M,
        'vars_per_component': 3, 'selection': 'explicit',
        'idx_s': idx_s, 'idx_q': idx_q, 'alpha': alpha,
    }

    def decode(sample):
        x = np.asarray(sample, dtype=float).ravel()
        sel = np.array([i for i in range(N) if x[idx_s(i)] > 0.5], dtype=int)
        conc = np.zeros(N)
        for i in sel:
            conc[i] = sum(alpha[b] * x[idx_q(i, b)] for b in range(2))
        return sel, conc

    return Q, q, var_map, decode


def build_qubo_3bit(A, target, k, lam_sparse=1.0, lam_rub=5.0):
    """
    3-bit Rubtsov encoding (3N vars).
    c_j = (1/7) * (4*s_j0 + 2*s_j1 + s_j2)
    alpha = [4/7, 2/7, 1/7]

    Anti-mixing penalty per component j:
      f_j = s_j0 + s_j1 + s_j2 - s_j0*s_j1 - s_j1*s_j2 - s_j0*s_j2
    = 0 iff bits are all-0 or all-1. Positive for mixed states.

    Sparsity: lam_sparse * (sum_j s_j0 - k)^2
    Uses s_j0 as selection proxy (with Rubtsov penalty, selected
    components tend to have all bits ~ 1).

    NO normalization penalty — NNLS handles it.
    This keeps Q much sparser (only within-component + cross-component
    fit terms + sparsity on s_j0 column).
    """
    N = A.shape[1]
    M = 3 * N
    alpha = np.array([4.0 / 7, 2.0 / 7, 1.0 / 7])

    Q = np.zeros((M, M))
    q_vec = np.zeros(M)

    def idx(j, b):
        return 3 * j + b

    G = A.T @ A
    p = A.T @ target

    # ---- Fit term ----
    for j1 in range(N):
        for b1 in range(3):
            ii = idx(j1, b1)
            q_vec[ii] += -2.0 * alpha[b1] * p[j1]
            # Same component
            for b2 in range(b1, 3):
                jj = idx(j1, b2)
                coeff = alpha[b1] * alpha[b2] * G[j1, j1]
                if b1 == b2:
                    Q[ii, ii] += coeff
                else:
                    Q[ii, jj] += coeff
                    Q[jj, ii] += coeff
            # Cross-component
            for j2 in range(j1 + 1, N):
                for b2 in range(3):
                    jj = idx(j2, b2)
                    coeff = alpha[b1] * alpha[b2] * G[j1, j2]
                    Q[ii, jj] += coeff
                    Q[jj, ii] += coeff

    # ---- Rubtsov anti-mixing penalty ----
    for j in range(N):
        for b in range(3):
            Q[idx(j, b), idx(j, b)] += lam_rub
        for b1 in range(3):
            for b2 in range(b1 + 1, 3):
                Q[idx(j, b1), idx(j, b2)] -= lam_rub / 2
                Q[idx(j, b2), idx(j, b1)] -= lam_rub / 2

    # ---- Sparsity on s_j0 ----
    for j1 in range(N):
        ii = idx(j1, 0)
        Q[ii, ii] += lam_sparse * (1 - 2 * k)
        for j2 in range(j1 + 1, N):
            jj = idx(j2, 0)
            Q[ii, jj] += lam_sparse
            Q[jj, ii] += lam_sparse

    var_map = {
        'scheme': '3bit', 'N': N, 'M': M,
        'vars_per_component': 3, 'selection': 'any_bit',
        'alpha': alpha, 'idx': idx,
    }

    def decode(sample):
        """Selected if ANY bit > 0.5 (not just s_j0)."""
        x = np.asarray(sample, dtype=float).ravel()
        sel = np.array(
            [j for j in range(N) if any(x[idx(j, b)] > 0.5 for b in range(3))],
            dtype=int)
        conc = np.zeros(N)
        for j in range(N):
            conc[j] = sum(alpha[b] * x[idx(j, b)] for b in range(3))
        return sel, conc

    return Q, q_vec, var_map, decode


def build_qubo_4bit(A, target, k, lam_sparse=1.0, lam_rub=5.0):
    """
    4-bit Rubtsov encoding (4N vars).
    c_j = (1/15) * (8*s_j0 + 4*s_j1 + 2*s_j2 + s_j3)

    Anti-mixing on lower 3 bits (b=1,2,3).
    Sparsity on s_j0.
    NO normalization penalty.
    """
    N = A.shape[1]
    M = 4 * N
    alpha = np.array([8.0 / 15, 4.0 / 15, 2.0 / 15, 1.0 / 15])

    Q = np.zeros((M, M))
    q_vec = np.zeros(M)

    def idx(j, b):
        return 4 * j + b

    G = A.T @ A
    p = A.T @ target

    # Fit term
    for j1 in range(N):
        for b1 in range(4):
            ii = idx(j1, b1)
            q_vec[ii] += -2.0 * alpha[b1] * p[j1]
            for b2 in range(b1, 4):
                jj = idx(j1, b2)
                coeff = alpha[b1] * alpha[b2] * G[j1, j1]
                if b1 == b2:
                    Q[ii, ii] += coeff
                else:
                    Q[ii, jj] += coeff
                    Q[jj, ii] += coeff
            for j2 in range(j1 + 1, N):
                for b2 in range(4):
                    jj = idx(j2, b2)
                    coeff = alpha[b1] * alpha[b2] * G[j1, j2]
                    Q[ii, jj] += coeff
                    Q[jj, ii] += coeff

    # Anti-mixing on lower 3 bits
    for j in range(N):
        for b in range(1, 4):
            Q[idx(j, b), idx(j, b)] += lam_rub
        for b1 in range(1, 4):
            for b2 in range(b1 + 1, 4):
                Q[idx(j, b1), idx(j, b2)] -= lam_rub / 2
                Q[idx(j, b2), idx(j, b1)] -= lam_rub / 2

    # Sparsity on s_j0
    for j1 in range(N):
        ii = idx(j1, 0)
        Q[ii, ii] += lam_sparse * (1 - 2 * k)
        for j2 in range(j1 + 1, N):
            jj = idx(j2, 0)
            Q[ii, jj] += lam_sparse
            Q[jj, ii] += lam_sparse

    var_map = {
        'scheme': '4bit', 'N': N, 'M': M,
        'vars_per_component': 4, 'selection': 'any_bit',
        'alpha': alpha, 'idx': idx,
    }

    def decode(sample):
        x = np.asarray(sample, dtype=float).ravel()
        sel = np.array(
            [j for j in range(N) if any(x[idx(j, b)] > 0.5 for b in range(4))],
            dtype=int)
        conc = np.zeros(N)
        for j in range(N):
            conc[j] = sum(alpha[b] * x[idx(j, b)] for b in range(4))
        return sel, conc

    return Q, q_vec, var_map, decode


# --------------------------------------------------------------------------
# Linked-selection builder (package3)
# --------------------------------------------------------------------------

def build_qubo_linked(A, y, k, B, lam_sparse=1.0, lam_link=5.0):
    """Build the linked-selection QUBO for B in {2,3,4}.

    Returns (Q, q, var_map, decode_fn) with
      Q       -- symmetric M x M matrix (both triangles filled)
      q       -- linear vector (M,)
      var_map -- dict with 'N', 'alpha', 'idx_s', 'idx_q', ...
      decode_fn(sample) -> (sel, conc)
        sel  -- int np.ndarray of selected component indices (s-bit rule)
        conc -- float np.ndarray of length N, zeroed for unselected

    Variables per component i:  s_i, q_i0, ..., q_i{B-1}
    Layout (matches build_qubo_B2 convention for B=2):
        idx_s(i)   = (B+1)*i
        idx_q(i,b) = (B+1)*i + 1 + b
        M = (B+1)*N

    Concentrations (LSB-first binary weights):
        c_i = sum_b alpha_b * q_ib,   alpha_b = 2^b / (2^B - 1)

    Objective (fit term on q-bits ONLY, no s gate; one-sided linker; no
    anti-mixing term):

        H = ||y - A c||^2
          + lam_sparse * (sum_i s_i - k)^2
          + lam_link   * sum_i sum_b q_ib * (1 - s_i)

    build_qubo_linked(A, y, k, 2, ...) reproduces build_qubo_B2(...)
    exactly (numerically equal Q and q).
    """
    B = int(B)
    if B not in (2, 3, 4):
        raise ValueError(f"B must be 2, 3 or 4, got {B!r}")

    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    N = A.shape[1]
    vpc = B + 1                      # vars per component: s + B q-bits
    M = vpc * N

    # LSB-first weights: alpha_b = 2^b / (2^B - 1)
    denom = float(2 ** B - 1)
    alpha = [2 ** b / denom for b in range(B)]

    Q = np.zeros((M, M))
    q = np.zeros(M)

    def idx_s(i):
        return vpc * i

    def idx_q(i, b):
        return vpc * i + 1 + b

    G = A.T @ A
    p = A.T @ y

    # ---- Fit term: ||y - A c||^2 - ||y||^2 with c_i = sum_b alpha_b q_ib ----
    # (q-bits only, no s gate -- exactly the build_qubo_B2 convention)
    for i in range(N):
        for b1 in range(B):
            ii = idx_q(i, b1)
            q[ii] += -2.0 * alpha[b1] * p[i]
            Q[ii, ii] += alpha[b1] ** 2 * G[i, i]
            # within-component cross terms b1 < b2
            for b2 in range(b1 + 1, B):
                jj = idx_q(i, b2)
                Q[ii, jj] += alpha[b1] * alpha[b2] * G[i, i]
                Q[jj, ii] += alpha[b1] * alpha[b2] * G[i, i]
            # cross-component terms j > i
            for j in range(i + 1, N):
                for b2 in range(B):
                    jj = idx_q(j, b2)
                    Q[ii, jj] += alpha[b1] * alpha[b2] * G[i, j]
                    Q[jj, ii] += alpha[b1] * alpha[b2] * G[i, j]

    # ---- Sparsity: lam_sparse * (sum_i s_i - k)^2 ----
    for i in range(N):
        ii = idx_s(i)
        Q[ii, ii] += lam_sparse * (1 - 2 * k)
        for j in range(i + 1, N):
            jj = idx_s(j)
            Q[ii, jj] += lam_sparse
            Q[jj, ii] += lam_sparse

    # ---- One-sided linker: lam_link * sum_i sum_b q_ib * (1 - s_i) ----
    # = lam_link * [sum q_ib - sum q_ib * s_i]
    for i in range(N):
        si = idx_s(i)
        for b in range(B):
            qi = idx_q(i, b)
            q[qi] += lam_link
            Q[qi, si] += -lam_link / 2
            Q[si, qi] += -lam_link / 2

    var_map = {
        'scheme': f'linked_B{B}',
        'N': N,
        'M': M,
        'B': B,
        'vars_per_component': vpc,
        'selection': 'explicit',
        'alpha': list(alpha),
        'idx_s': idx_s,
        'idx_q': idx_q,
    }

    def decode_fn(sample):
        """(sel, conc): selected by s-bit rule; conc zeroed for unselected."""
        x = np.asarray(sample, dtype=float).ravel()
        sel = np.array([i for i in range(N) if x[idx_s(i)] > 0.5], dtype=int)
        conc = np.zeros(N)
        for i in sel:
            conc[i] = sum(alpha[b] * x[idx_q(i, b)] for b in range(B))
        return sel, conc

    return Q, q, var_map, decode_fn


def _build_qubo_linked_B2(A, target, k, lam_sparse=1.0, lam_link=5.0):
    """linked_B2 scheme entry: fixed B=2 wrapper around build_qubo_linked."""
    return build_qubo_linked(A, target, k, 2,
                             lam_sparse=lam_sparse, lam_link=lam_link)


def _build_qubo_linked_B3(A, target, k, lam_sparse=1.0, lam_link=5.0):
    """linked_B3 scheme entry: fixed B=3 wrapper around build_qubo_linked."""
    return build_qubo_linked(A, target, k, 3,
                             lam_sparse=lam_sparse, lam_link=lam_link)


def _build_qubo_linked_B4(A, target, k, lam_sparse=1.0, lam_link=5.0):
    """linked_B4 scheme entry: fixed B=4 wrapper around build_qubo_linked."""
    return build_qubo_linked(A, target, k, 4,
                             lam_sparse=lam_sparse, lam_link=lam_link)


# --------------------------------------------------------------------------
# Registry / dispatcher
# --------------------------------------------------------------------------

SCHEMES = {
    'B1': build_qubo_B1,
    'B2': build_qubo_B2,
    '3bit': build_qubo_3bit,
    '4bit': build_qubo_4bit,
    'linked_B2': _build_qubo_linked_B2,
    'linked_B3': _build_qubo_linked_B3,
    'linked_B4': _build_qubo_linked_B4,
}


def build_qubo(scheme, A, target, k, **kwargs):
    """Dispatch to the right builder."""
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown scheme: {scheme}. Choose from {list(SCHEMES.keys())}")
    return SCHEMES[scheme](A, target, k, **kwargs)


# --------------------------------------------------------------------------
# Direct energy decomposition
# --------------------------------------------------------------------------

def _direct_terms_original(scheme, A, y, k, x, lam_sparse, lam_link, lam_rub):
    """Direct H decomposition for the 4 original non-linked schemes.

    Returns (residual, sparsity, link, pattern).
    """
    N = A.shape[1]
    if scheme == 'B1':
        c = x.astype(float)
        sel_bits = x.astype(float)
        link = 0.0
        pattern = 0.0
    elif scheme == 'B2':
        alpha = np.array([1.0 / 3, 2.0 / 3])
        c = np.zeros(N)
        sel_bits = np.zeros(N)
        link = 0.0
        for i in range(N):
            s = x[3 * i]
            sel_bits[i] = s
            q0, q1 = x[3 * i + 1], x[3 * i + 2]
            c[i] = alpha[0] * q0 + alpha[1] * q1
            link += q0 * (1 - s) + q1 * (1 - s)
        link *= lam_link
        pattern = 0.0
    elif scheme == '3bit':
        alpha = np.array([4.0 / 7, 2.0 / 7, 1.0 / 7])
        c = np.zeros(N)
        sel_bits = np.zeros(N)
        pattern = 0.0
        for j in range(N):
            b = x[3 * j:3 * j + 3]
            c[j] = alpha @ b
            sel_bits[j] = b[0]
            f = b[0] + b[1] + b[2] - b[0] * b[1] - b[1] * b[2] - b[0] * b[2]
            pattern += f
        pattern *= lam_rub
        link = 0.0
    elif scheme == '4bit':
        alpha = np.array([8.0 / 15, 4.0 / 15, 2.0 / 15, 1.0 / 15])
        c = np.zeros(N)
        sel_bits = np.zeros(N)
        pattern = 0.0
        for j in range(N):
            b = x[4 * j:4 * j + 4]
            c[j] = alpha @ b
            sel_bits[j] = b[0]
            b1, b2, b3 = b[1], b[2], b[3]   # anti-mixing on bits 1..3 only
            f = b1 + b2 + b3 - b1 * b2 - b2 * b3 - b1 * b3
            pattern += f
        pattern *= lam_rub
        link = 0.0
    else:
        raise ValueError(scheme)

    residual = float(np.sum((y - A @ c) ** 2))
    sparsity = float(lam_sparse * (sel_bits.sum() - k) ** 2)
    return residual, sparsity, float(link), float(pattern)


def _direct_terms_linked(B, A, y, k, x, lam_sparse, lam_link):
    """Direct H decomposition for linked_B{B} schemes.

    Returns (residual, sparsity, link, pattern) with pattern = 0.
    """
    N = A.shape[1]
    vpc = B + 1
    denom = float(2 ** B - 1)
    alpha = [2 ** b / denom for b in range(B)]

    c = np.zeros(N)
    sel_bits = np.zeros(N)
    link = 0.0
    for i in range(N):
        s = x[vpc * i]
        sel_bits[i] = s
        for b in range(B):
            qb = x[vpc * i + 1 + b]
            c[i] += alpha[b] * qb
            link += qb * (1 - s)
    link *= lam_link
    pattern = 0.0

    residual = float(np.sum((y - A @ c) ** 2))
    sparsity = float(lam_sparse * (sel_bits.sum() - k) ** 2)
    return residual, sparsity, float(link), float(pattern)


def energy_terms(sample, A, y, k, scheme,
                 lam_sparse=1.0, lam_link=5.0, lam_rub=5.0):
    """Direct energy decomposition of H for one bit state.

    scheme in {'B1','B2','3bit','4bit','linked_B2','linked_B3','linked_B4'}

    Returns dict with keys:
      'residual', 'sparsity', 'link', 'pattern', 'base',
      'xQx_plus_qx', 'constant_excluded', 'match_error'

    base              = residual + sparsity + link + pattern
    constant_excluded = ||y||^2 + lam_sparse * k^2   (state-independent terms
                        omitted from Q, q)
    xQx_plus_qx       = x^T Q x + q^T x from the matching builder
    match_error       = |base - xQx_plus_qx - constant_excluded|
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(sample, dtype=float).ravel()

    # ---- matching builder (also gives Q, q for the quadratic form) ----
    if scheme in ('B1', 'B2', '3bit', '4bit'):
        kwargs = {'lam_sparse': lam_sparse}
        if scheme == 'B2':
            kwargs['lam_link'] = lam_link
        if scheme in ('3bit', '4bit'):
            kwargs['lam_rub'] = lam_rub
        Q, q, var_map, _ = build_qubo(scheme, A, y, k, **kwargs)
    elif scheme in ('linked_B2', 'linked_B3', 'linked_B4'):
        B = int(scheme.split('_B')[1])
        Q, q, var_map, _ = build_qubo_linked(
            A, y, k, B, lam_sparse=lam_sparse, lam_link=lam_link)
    else:
        raise ValueError(f"Unknown scheme: {scheme!r}")

    # ---- direct decomposition ----
    if scheme.startswith('linked_'):
        B = int(scheme.split('_B')[1])
        residual, sparsity, link, pattern = _direct_terms_linked(
            B, A, y, k, x, lam_sparse, lam_link)
    else:
        residual, sparsity, link, pattern = _direct_terms_original(
            scheme, A, y, k, x, lam_sparse, lam_link, lam_rub)

    base = residual + sparsity + link + pattern
    xQx_plus_qx = float(x @ Q @ x + q @ x)
    constant_excluded = float(y @ y + lam_sparse * k * k)
    match_error = abs(base - xQx_plus_qx - constant_excluded)

    return {
        'residual': residual,
        'sparsity': sparsity,
        'link': link,
        'pattern': pattern,
        'base': base,
        'xQx_plus_qx': xQx_plus_qx,
        'constant_excluded': constant_excluded,
        'match_error': match_error,
    }


# --------------------------------------------------------------------------
# Dimensionless-penalty scale helpers
# --------------------------------------------------------------------------

def scale_s_A(A):
    """Scale factor s_A = ||A||_F^2 / N (N = number of columns)."""
    A = np.asarray(A, dtype=float)
    return float(A.ravel() @ A.ravel() / A.shape[1])


def gammas_from_absolute(lam_sparse0, lam_link0, s_ref):
    """Dimensionless penalties (gammas) from absolute lambdas at scale s_ref."""
    return (lam_sparse0 / s_ref, lam_link0 / s_ref)


def lambdas_for_A(gamma_sparse, gamma_link, s_A):
    """Absolute lambdas for a library with scale s_A from dimensionless gammas."""
    return (gamma_sparse * s_A, gamma_link * s_A)