"""QUBO matrix construction for LCF with multiple encoding schemes.

Ported from the QUBO_LCF research repository (scripts/qubo_lib.py and
scripts/package3_lib.py) with the numerical conventions preserved exactly.

Schemes (keys of :data:`SCHEMES`):
  B1        -- Binary selection only (N vars). Concentrations via NNLS post-hoc.
  B2        -- Selection s_i + 2-bit quantization q_i0,q_i1 (3N vars).
               c_i = s_i * (1/3 * q_i0 + 2/3 * q_i1)
  B3        -- 3-bit anti-mixing encoding (3N vars, no explicit selection).
               c_j = (1/7) * (4*s_j0 + 2*s_j1 + s_j2)
               Anti-mixing penalty pushes bits toward 000 or near-111.
  B4        -- 4-bit anti-mixing encoding (4N vars).
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

.. note::
   The optional numba/JIT acceleration was NOT used in the article;
   article results were obtained with the pure NumPy code paths.
"""
import numpy as np

from ._accel import require_numba

# --------------------------------------------------------------------------
# Residual quadratic-form assembly (shared by all builders)
# --------------------------------------------------------------------------
#
# The residual term ||y - A c||^2 (up to the state-independent ||y||^2) is
#   c^T G c - 2 p^T c,   G = A^T A,  p = A^T y,
# with c_i = sum_b alpha_b * x[fit_idx[i, b]] on the fit variables.  Two
# numerically identical twins assemble it into the preallocated (zeroed)
# ``Q`` / ``q``:
#
#   _fit_assembly_np   -- pure NumPy (default path; Python loops only over
#                         the <= 4 bit offsets, everything else vectorized);
#   _fit_assembly_jit  -- numba @njit scalar-loop twin (use_jit=True only).
#
# Penalty terms are vectorized NumPy helpers shared by both paths.

def _component_indices(N, vpc, offsets):
    """Global variable indices of per-component slots.

    Returns an ``(N, len(offsets))`` int array whose ``(i, b)`` entry is the
    global index of slot ``offsets[b]`` of component ``i`` on a layout with
    ``vpc`` variables per component.
    """
    base = np.arange(N, dtype=np.intp)[:, None] * vpc
    return base + np.asarray(offsets, dtype=np.intp)[None, :]


def _fit_assembly_np(G, p, alpha, fit_idx, Q, q):
    """Pure-NumPy twin of :func:`_fit_assembly_jit` (default path).

    Adds the state-dependent residual form ``c.T @ G @ c - 2 * p.T @ c``
    (with ``c_i = sum_b alpha_b * x[fit_idx[i, b]]``) into the zeroed ``Q``
    and ``q``.  Python loops only run over the (<= 4) bit offsets; the
    per-component work is fully vectorized.
    """
    for b1 in range(fit_idx.shape[1]):
        rows = fit_idx[:, b1]
        q[rows] += (-2.0 * alpha[b1]) * p
        for b2 in range(fit_idx.shape[1]):
            Q[np.ix_(rows, fit_idx[:, b2])] += (alpha[b1] * alpha[b2]) * G


_fit_kernel = None


def _get_fit_kernel(nb):
    """Build (once) and return the ``@njit`` residual-assembly kernel."""
    global _fit_kernel
    if _fit_kernel is None:
        @nb.njit(cache=True)
        def kernel(G, p, alpha, fit_idx, Q, q):
            N, B = fit_idx.shape
            for i in range(N):
                for b1 in range(B):
                    ii = fit_idx[i, b1]
                    q[ii] += (-2.0 * alpha[b1]) * p[i]
                    for j in range(N):
                        g = G[i, j]
                        for b2 in range(B):
                            Q[ii, fit_idx[j, b2]] += (alpha[b1] * alpha[b2]) * g
        _fit_kernel = kernel
    return _fit_kernel


def _fit_assembly_jit(G, p, alpha, fit_idx, Q, q):
    """numba @njit twin of :func:`_fit_assembly_np` (opt-in ``use_jit=True``)."""
    nb = require_numba()
    kernel = _get_fit_kernel(nb)
    kernel(np.ascontiguousarray(G, dtype=np.float64),
           np.ascontiguousarray(p, dtype=np.float64),
           np.ascontiguousarray(alpha, dtype=np.float64),
           np.ascontiguousarray(fit_idx, dtype=np.intp),
           Q, q)


def _assemble_residual(Q, q, G, p, alpha, fit_idx, use_jit):
    """Assemble the residual quadratic form via the selected twin."""
    alpha = np.asarray(alpha, dtype=np.float64)
    fit_idx = np.asarray(fit_idx, dtype=np.intp)
    if use_jit:
        _fit_assembly_jit(G, p, alpha, fit_idx, Q, q)
    else:
        _fit_assembly_np(G, p, alpha, fit_idx, Q, q)


def _add_sparsity_penalty(Q, sel_idx, lam_sparse, k):
    """lam_sparse * (sum_i s_i - k)^2 on the selection variables ``sel_idx``.

    Diagonal gets ``lam_sparse * (1 - 2k)`` (the ``-k^2`` constant is
    state-independent and excluded), every cross term ``s_i s_j`` (i != j)
    gets ``2 * lam_sparse`` in ``x^T Q x`` i.e. ``lam_sparse`` per triangle.
    """
    N = sel_idx.shape[0]
    block = np.full((N, N), lam_sparse)
    np.fill_diagonal(block, lam_sparse * (1 - 2 * k))
    Q[np.ix_(sel_idx, sel_idx)] += block


def _add_link_penalty(Q, q, sel_idx, q_idx, lam_link):
    """lam_link * sum_i sum_b q_ib * (1 - s_i) (one-sided linker).

    = lam_link * [sum q_ib - sum q_ib * s_i]: a linear term on every q-bit
    and one symmetric cross term per (component, bit) pair.
    """
    flat_q = q_idx.ravel()
    flat_s = np.repeat(sel_idx, q_idx.shape[1])
    q[flat_q] += lam_link
    Q[flat_q, flat_s] += -lam_link / 2
    Q[flat_s, flat_q] += -lam_link / 2


def _add_antimixing_penalty(Q, bit_idx, lam_antimix):
    """Anti-mixing penalty on each component's bit slots ``bit_idx``.

    f_j = sum_b x_jb - sum_{b1<b2} x_jb1 * x_jb2  (= 0 on all-0 / all-1).
    """
    ns = bit_idx.shape[1]
    block = np.full((ns, ns), -lam_antimix / 2)
    np.fill_diagonal(block, lam_antimix)
    for a in range(ns):
        for b in range(ns):
            Q[bit_idx[:, a], bit_idx[:, b]] += block[a, b]


# --------------------------------------------------------------------------
# Non-linked schemes (original qubo_lib builders)
# --------------------------------------------------------------------------

def build_qubo_B1(A, target, k, lam_sparse=1.0, use_jit=False):
    """
    B=1: Binary selection. x_i in {0,1}.
    H = ||y - A @ x||^2 + lam_sparse * (sum(x) - k)^2
    Variables: N
    """
    N = A.shape[1]
    G = A.T @ A       # N x N
    p = A.T @ target  # N

    Q = np.zeros((N, N))
    q = np.zeros(N)

    fit_idx = _component_indices(N, 1, [0])
    # Residual quadratic form ||y - A x||^2 (state-dependent part)
    _assemble_residual(Q, q, G, p, (1.0,), fit_idx, use_jit)

    # Sparsity: lam * (sum(x) - k)^2
    _add_sparsity_penalty(Q, fit_idx[:, 0], lam_sparse, k)

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


def build_qubo_B2(A, target, k, lam_sparse=1.0, lam_link=5.0, use_jit=False):
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

    fit_idx = _component_indices(N, 3, [1, 2])   # q-bit variables
    sel_idx = _component_indices(N, 3, [0])[:, 0]

    # Fit term (only on q variables)
    _assemble_residual(Q, q, G, p, alpha, fit_idx, use_jit)

    # Sparsity: lam * (sum s_i - k)^2
    _add_sparsity_penalty(Q, sel_idx, lam_sparse, k)

    # Link: lam_link * sum_i sum_b q_ib * (1 - s_i)
    # = lam_link * [sum q_ib - sum q_ib * s_i]
    _add_link_penalty(Q, q, sel_idx, fit_idx, lam_link)

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


def build_qubo_B3(A, target, k, lam_sparse=1.0, lam_antimix=5.0, use_jit=False):
    """
    3-bit anti-mixing encoding (3N vars).
    c_j = (1/7) * (4*s_j0 + 2*s_j1 + s_j2)
    alpha = [4/7, 2/7, 1/7]

    Anti-mixing penalty per component j:
      f_j = s_j0 + s_j1 + s_j2 - s_j0*s_j1 - s_j1*s_j2 - s_j0*s_j2
    = 0 iff bits are all-0 or all-1. Positive for mixed states.

    Sparsity: lam_sparse * (sum_j s_j0 - k)^2
    Uses s_j0 as selection proxy (with the anti-mixing penalty, selected
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

    fit_idx = _component_indices(N, 3, [0, 1, 2])

    # ---- Fit term ----
    _assemble_residual(Q, q_vec, G, p, alpha, fit_idx, use_jit)

    # ---- Anti-mixing penalty ----
    _add_antimixing_penalty(Q, fit_idx, lam_antimix)

    # ---- Sparsity on s_j0 ----
    _add_sparsity_penalty(Q, fit_idx[:, 0], lam_sparse, k)

    var_map = {
        'scheme': 'B3', 'N': N, 'M': M,
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


def build_qubo_B4(A, target, k, lam_sparse=1.0, lam_antimix=5.0, use_jit=False):
    """
    4-bit anti-mixing encoding (4N vars).
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

    fit_idx = _component_indices(N, 4, [0, 1, 2, 3])

    # Fit term
    _assemble_residual(Q, q_vec, G, p, alpha, fit_idx, use_jit)

    # Anti-mixing on lower 3 bits
    _add_antimixing_penalty(Q, fit_idx[:, 1:4], lam_antimix)

    # Sparsity on s_j0
    _add_sparsity_penalty(Q, fit_idx[:, 0], lam_sparse, k)

    var_map = {
        'scheme': 'B4', 'N': N, 'M': M,
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

def build_qubo_linked(A, y, k, B, lam_sparse=1.0, lam_link=5.0, use_jit=False):
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

    fit_idx = _component_indices(N, vpc, range(1, B + 1))  # q-bit variables
    sel_idx = _component_indices(N, vpc, [0])[:, 0]

    # ---- Fit term: ||y - A c||^2 - ||y||^2 with c_i = sum_b alpha_b q_ib ----
    # (q-bits only, no s gate -- exactly the build_qubo_B2 convention)
    _assemble_residual(Q, q, G, p, alpha, fit_idx, use_jit)

    # ---- Sparsity: lam_sparse * (sum_i s_i - k)^2 ----
    _add_sparsity_penalty(Q, sel_idx, lam_sparse, k)

    # ---- One-sided linker: lam_link * sum_i sum_b q_ib * (1 - s_i) ----
    # = lam_link * [sum q_ib - sum q_ib * s_i]
    _add_link_penalty(Q, q, sel_idx, fit_idx, lam_link)

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


def _build_qubo_linked_B2(A, target, k, lam_sparse=1.0, lam_link=5.0,
                          use_jit=False):
    """linked_B2 scheme entry: fixed B=2 wrapper around build_qubo_linked."""
    return build_qubo_linked(A, target, k, 2,
                             lam_sparse=lam_sparse, lam_link=lam_link,
                             use_jit=use_jit)


def _build_qubo_linked_B3(A, target, k, lam_sparse=1.0, lam_link=5.0,
                          use_jit=False):
    """linked_B3 scheme entry: fixed B=3 wrapper around build_qubo_linked."""
    return build_qubo_linked(A, target, k, 3,
                             lam_sparse=lam_sparse, lam_link=lam_link,
                             use_jit=use_jit)


def _build_qubo_linked_B4(A, target, k, lam_sparse=1.0, lam_link=5.0,
                          use_jit=False):
    """linked_B4 scheme entry: fixed B=4 wrapper around build_qubo_linked."""
    return build_qubo_linked(A, target, k, 4,
                             lam_sparse=lam_sparse, lam_link=lam_link,
                             use_jit=use_jit)


# --------------------------------------------------------------------------
# Registry / dispatcher
# --------------------------------------------------------------------------

SCHEMES = {
    'B1': build_qubo_B1,
    'B2': build_qubo_B2,
    'B3': build_qubo_B3,
    'B4': build_qubo_B4,
    'linked_B2': _build_qubo_linked_B2,
    'linked_B3': _build_qubo_linked_B3,
    'linked_B4': _build_qubo_linked_B4,
}


def build_qubo(scheme, A, target, k, use_jit=False, **kwargs):
    """Dispatch to the right builder (``use_jit`` selects the numba twin)."""
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown scheme: {scheme}. Choose from {list(SCHEMES.keys())}")
    return SCHEMES[scheme](A, target, k, use_jit=use_jit, **kwargs)


# --------------------------------------------------------------------------
# Direct energy decomposition
# --------------------------------------------------------------------------

def _direct_terms_original(scheme, A, y, k, x, lam_sparse, lam_link, lam_antimix):
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
    elif scheme == 'B3':
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
        pattern *= lam_antimix
        link = 0.0
    elif scheme == 'B4':
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
        pattern *= lam_antimix
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
                 lam_sparse=1.0, lam_link=5.0, lam_antimix=5.0, use_jit=False):
    """Direct energy decomposition of H for one bit state.

    scheme in {'B1','B2','B3','B4','linked_B2','linked_B3','linked_B4'}

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
    if scheme in ('B1', 'B2', 'B3', 'B4'):
        kwargs = {'lam_sparse': lam_sparse}
        if scheme == 'B2':
            kwargs['lam_link'] = lam_link
        if scheme in ('B3', 'B4'):
            kwargs['lam_antimix'] = lam_antimix
        Q, q, var_map, _ = build_qubo(scheme, A, y, k, use_jit=use_jit, **kwargs)
    elif scheme in ('linked_B2', 'linked_B3', 'linked_B4'):
        B = int(scheme.split('_B')[1])
        Q, q, var_map, _ = build_qubo_linked(
            A, y, k, B, lam_sparse=lam_sparse, lam_link=lam_link,
            use_jit=use_jit)
    else:
        raise ValueError(f"Unknown scheme: {scheme!r}")

    # ---- direct decomposition ----
    if scheme.startswith('linked_'):
        B = int(scheme.split('_B')[1])
        residual, sparsity, link, pattern = _direct_terms_linked(
            B, A, y, k, x, lam_sparse, lam_link)
    else:
        residual, sparsity, link, pattern = _direct_terms_original(
            scheme, A, y, k, x, lam_sparse, lam_link, lam_antimix)

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