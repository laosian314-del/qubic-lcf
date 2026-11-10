"""GPU-accelerated Simulated Annealing for QUBO via PyTorch CUDA.

Ported from ``QUBO_LCF_3/scripts/gpu_sa_solver.py``. Fully vectorized
Metropolis: (bsize x M) state matrix, single-bit flips per sweep.

Key behaviours (preserved from the original):
- k-balanced initialization: states are initialized with k ones randomly
  placed, where k sweeps uniformly from 1..k_max across the batch.
- Component diversity penalty: after each batch, a small repulsive bias is
  added to frequently-visited component indices.
- Adaptive sweeps: continues until wall time exceeds time_limit.
- Stall detection with early break.
- v2 integer-cut emulation via multi-pass SA with component penalty
  (``solve_gpu_sa_multi_pass``): singleton-trap components are hard-forbidden
  across passes.

Adaptations for the unified layer: the component selection is derived from
``var_map`` via :func:`qubic.solvers.base.get_selection` (the original took a
``decode_fn``).

Requires a CUDA-capable GPU: without one a RuntimeError with an explicit
message is raised.

Public API (ported):
    solve_gpu_sa(Q, q, var_map, k_max, time_limit, ...) -> (solutions, wall_time, gap)
    solve_gpu_sa_multi_pass(Q, q, var_map, k_max, time_limit, ...) -> (solutions, wall_time, gap)

Unified API:
    solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, **kwargs)
        -> (solutions, wall_time, gap=None)
"""
from __future__ import annotations

import time

import numpy as np

from . import base
from .base import get_selection


def _require_cuda_torch():
    """Import torch and require a CUDA device; raise RuntimeError otherwise."""
    torch = base.require("torch", "torch")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "gpu_sa solver requires a CUDA-capable GPU, but "
            "torch.cuda.is_available() is False (this PyTorch build is CPU-only). "
            "Install CUDA-enabled PyTorch (pip install torch --index-url "
            "https://download.pytorch.org/whl/cu124) or use a CPU solver "
            "(milp, cp_sat, scip, sa, tabu)."
        )
    return torch


def _make_q_forbidden_penalty(Q, q_orig, forbidden_components,
                              N, encoding, var_map,
                              penalty_strength=80.0):
    """Modify q-vector to penalize forbidden components (ported unchanged).

    forbidden_components: {component_index: penalty_multiplier}
    Penalty is added to q-vector entries corresponding to given component
    indices. For B1: directly to q[i]. For B2/3bit/4bit: penalty spread across
    all bits of that component.
    """
    q_mod = q_orig.copy()
    if not forbidden_components:
        return q_mod

    for comp_idx, multiplier in forbidden_components.items():
        penalty = penalty_strength * multiplier
        if encoding == "B1":
            if comp_idx < len(q_mod):
                q_mod[comp_idx] += penalty
        elif encoding == "B2":
            for b in range(2):
                qi = var_map["idx_q"](comp_idx, b)
                if qi < len(q_mod):
                    q_mod[qi] += penalty * 0.3
        elif encoding in ("3bit", "4bit"):
            idx_fn = var_map["idx"]
            vpc = var_map["vars_per_component"]
            for b in range(vpc):
                qi = idx_fn(comp_idx, b)
                if qi < len(q_mod):
                    q_mod[qi] += penalty * 0.3

    return q_mod


def solve_gpu_sa(Q, q_input, var_map, k_max=15, time_limit=None, top_k=20,
                 num_reads=5000, num_sweeps=5000, initial_temp=100.0,
                 final_temp=0.001, batch_size=4096, diversity_scale=0.05,
                 forbidden_components=None, forbidden_penalty_strength=80.0):
    """GPU SA solver with k-balanced init, component diversity, and optional
    component penalty for integer-cut emulation (ported core loop).

    If ``forbidden_components`` is provided, the q-vector is modified before SA
    to penalize those components (emulating MILP integer cuts).

    Returns ``(solutions, wall_time, gap=None)``.
    """
    torch = _require_cuda_torch()
    time_limit = base.resolve_time_limit(time_limit)

    device = torch.device("cuda")
    M = Q.shape[0]
    N = var_map["N"]
    encoding = var_map["scheme"]

    # Apply forbidden-component penalty if requested
    if forbidden_components:
        q_orig = _make_q_forbidden_penalty(
            Q, np.asarray(q_input, dtype=float), forbidden_components, N,
            encoding, var_map, penalty_strength=forbidden_penalty_strength)
    else:
        q_orig = np.asarray(q_input, dtype=float).copy()

    Q_gpu = torch.tensor(Q, dtype=torch.float32, device=device)
    q_orig_gpu = torch.tensor(q_orig, dtype=torch.float32, device=device)
    q_mean_abs = q_orig_gpu.abs().mean().item()
    penalty_strength = diversity_scale * max(q_mean_abs, 1e-6)

    t_start = time.time()
    all_solutions = []          # list of (sample, energy)
    seen_hashes = set()
    component_hits = np.zeros(M, dtype=np.float32)

    stall_batches = 0
    stall_limit = 3
    sweeps_per_check = 100

    reads_done = 0

    while True:
        wall_elapsed = time.time() - t_start
        remaining_budget = time_limit - wall_elapsed
        if remaining_budget <= 0.1:
            break

        bsize = min(batch_size, max(1, num_reads - reads_done))

        # Diversity penalty on q
        if component_hits.sum() > 0:
            q_penalty = torch.tensor(
                penalty_strength * component_hits / max(component_hits.max(), 1),
                dtype=torch.float32, device=device)
            q_gpu = q_orig_gpu + q_penalty
        else:
            q_gpu = q_orig_gpu

        # --- k-balanced initialization ---
        states = torch.zeros((bsize, M), dtype=torch.float32, device=device)
        for b in range(bsize):
            k_init = (b % k_max) + 1
            ones = torch.randperm(M, device=device)[:k_init]
            states[b, ones] = 1.0

        # Temperature schedule
        temps = np.logspace(np.log10(initial_temp), np.log10(final_temp), num_sweeps)

        prev_best_energy = float("inf")
        stall_sweeps = 0

        for sweep in range(num_sweeps):
            temp = float(temps[sweep])

            # --- Pick one random bit to flip per state ---
            flip_idx = torch.randint(0, M, (bsize,), device=device)
            x_i = states[torch.arange(bsize, device=device), flip_idx]

            # --- Compute dE with current q_gpu ---
            d_i = 1.0 - 2.0 * x_i
            Q_cols = Q_gpu[:, flip_idx].T
            Q_rows = Q_gpu[flip_idx, :]
            contrib = (Q_cols + Q_rows) * states
            contrib_sum = contrib.sum(dim=1)

            q_i = q_gpu[flip_idx]
            Q_ii = Q_gpu.diag()[flip_idx]

            delta_E = d_i * (contrib_sum + q_i) + Q_ii

            # --- Metropolis ---
            neg_dE = delta_E < 0
            pos_dE = ~neg_dE
            prob_accept = torch.zeros(bsize, device=device)
            if pos_dE.any():
                prob_accept[pos_dE] = torch.exp(-delta_E[pos_dE] / max(temp, 1e-10))
            accept = neg_dE | (torch.rand(bsize, device=device) < prob_accept)

            if accept.any():
                idx_b = torch.arange(bsize, device=device)[accept]
                idx_i = flip_idx[accept]
                states[idx_b, idx_i] = 1.0 - states[idx_b, idx_i]

            # Periodic checks
            if sweep % sweeps_per_check == 0 and sweep > 0:
                if time.time() - t_start >= time_limit:
                    break
                energies_check = (states @ Q_gpu * states).sum(dim=1) + states @ q_gpu
                batch_best = energies_check.min().item()
                if batch_best >= prev_best_energy - 1e-8:
                    stall_sweeps += sweeps_per_check
                else:
                    stall_sweeps = 0
                    prev_best_energy = batch_best
                if stall_sweeps >= num_sweeps // 2:
                    break

        # --- Final energies with original q ---
        energies = (states @ Q_gpu * states).sum(dim=1) + states @ q_orig_gpu

        # --- Collect unique solutions ---
        states_np = states.cpu().numpy()
        n_new = 0
        batch_hits = np.zeros(M, dtype=np.float32)
        for b in range(bsize):
            sample = states_np[b].astype(float)
            sel = [j for j, flag in enumerate(get_selection(sample, var_map)) if flag]
            sel_hash = hash(tuple(sorted(sel)))
            if sel_hash not in seen_hashes:
                seen_hashes.add(sel_hash)
                all_solutions.append((sample.copy(), float(energies[b])))
                n_new += 1
                for idx in sel:
                    if idx < M:
                        batch_hits[idx] = 1

        component_hits += batch_hits

        # Reset component_hits periodically to prevent monotonic penalty growth
        # (was accumulating across ALL batches — bug found 2026-06-15)
        if reads_done >= 5 * bsize:
            component_hits = np.zeros(M, dtype=np.float32)
            reads_done = 0
        reads_done += bsize

        if n_new == 0:
            stall_batches += 1
        else:
            stall_batches = 0

        if time.time() - t_start >= time_limit:
            break

    all_solutions.sort(key=lambda x: x[1])
    all_solutions = all_solutions[:top_k]

    wall_time = time.time() - t_start
    return all_solutions, wall_time, None


def solve_gpu_sa_multi_pass(Q, q, var_map, k_max=15, time_limit=None, top_k=20,
                            num_reads=5000, num_sweeps=5000, max_passes=5,
                            batch_size=4096, forbidden_penalty_strength=80.0,
                            **kwargs):
    """Multi-pass GPU SA with integer-cut emulation (v3 fix: hard-forbid).

    Runs GPU SA repeatedly. After each pass, the top-2 most frequent
    singleton (k=1) components are HARD-FORBIDDEN (penalty x100) to force
    exploration of higher cardinalities. Unlike v2 which penalized ALL
    singleton components uniformly (leading to k=1 trap regression), v3 only
    forbids the worst offenders, leaving the rest of the component pool
    available for genuine k>=2 solutions.

    Returns ``(solutions, total_wall_time, gap=None)``.
    """
    time_limit = base.resolve_time_limit(time_limit)
    all_solutions = []
    seen_subsets = set()
    forbidden_components = {}  # {comp_idx: cumulative_penalty_multiplier}
    total_wall_time = 0.0
    per_pass_time = max(1.0, time_limit / max_passes)

    for pass_idx in range(max_passes):
        wall_remaining = time_limit - total_wall_time
        if wall_remaining <= 0.05:
            break

        pass_limit = min(per_pass_time, wall_remaining)

        # Increase penalty in later passes for hard forbid
        pass_penalty = forbidden_penalty_strength
        if pass_idx >= 2 and forbidden_components:
            pass_penalty = pass_penalty * 10.0  # 10x for hard forbid

        solutions, wall_time, _gap = solve_gpu_sa(
            Q, q, var_map, k_max=k_max, time_limit=pass_limit, top_k=top_k,
            num_reads=num_reads, num_sweeps=num_sweeps, batch_size=batch_size,
            forbidden_components=(forbidden_components if pass_idx > 0 else None),
            forbidden_penalty_strength=pass_penalty,
            **kwargs)

        total_wall_time += wall_time

        # Analyze solutions: identify singleton traps
        singleton_counts = {}
        for sample, energy in solutions:
            sel = [j for j, flag in enumerate(get_selection(sample, var_map)) if flag]
            sel_tuple = tuple(sorted(sel))
            if sel_tuple not in seen_subsets:
                seen_subsets.add(sel_tuple)
                all_solutions.append((sample.copy(), float(energy)))
                if len(sel) == 1:
                    comp_idx = int(sel[0])
                    singleton_counts[comp_idx] = singleton_counts.get(comp_idx, 0) + 1
            if len(all_solutions) >= top_k:
                break

        # v3 FIX: hard-forbid only top-2 singleton components.
        # Penalizing ALL singletons creates uniform penalty -> k=1 trap persists
        if len(singleton_counts) > 2:
            top_items = sorted(singleton_counts.items(), key=lambda x: -x[1])[:2]
            singleton_counts = dict(top_items)

        # Hard-forbid: multiplier x100 for top offenders
        for comp_idx, count in singleton_counts.items():
            forbidden_components[comp_idx] = forbidden_components.get(comp_idx, 0) + count * 100

        # Safety: max 4 forbidden components total
        if len(forbidden_components) > 4:
            # Keep only the 4 with highest penalty
            top_forbidden = sorted(forbidden_components.items(), key=lambda x: -x[1])[:4]
            forbidden_components = dict(top_forbidden)

        if len(all_solutions) >= top_k:
            break

    # Sort all solutions by energy
    all_solutions.sort(key=lambda x: x[1])
    all_solutions = all_solutions[:top_k]

    return all_solutions, total_wall_time, None


def solve(Q, q, var_map, k_max=15, top_k=20, time_limit=None, **kwargs):
    """Unified solver entry point (see :mod:`qubic.solvers.base`).

    Extra kwargs: ``multi_pass`` (bool, default False — use
    :func:`solve_gpu_sa_multi_pass`), ``num_reads``, ``num_sweeps``,
    ``batch_size``, ``initial_temp``, ``final_temp``, ``diversity_scale``,
    ``max_passes``, ``forbidden_penalty_strength``.

    Raises RuntimeError when no CUDA device is available.
    """
    multi_pass = kwargs.pop("multi_pass", False)
    if multi_pass:
        max_passes = kwargs.pop("max_passes", 5)
        batch_size = kwargs.pop("batch_size", 4096)
        num_reads = kwargs.pop("num_reads", 5000)
        num_sweeps = kwargs.pop("num_sweeps", 5000)
        forbidden_penalty_strength = kwargs.pop("forbidden_penalty_strength", 80.0)
        base.reject_unknown_kwargs("gpu_sa.solve", kwargs)
        pairs, wall_time, gap = solve_gpu_sa_multi_pass(
            Q, q, var_map, k_max=k_max, time_limit=time_limit, top_k=top_k,
            num_reads=num_reads, num_sweeps=num_sweeps, max_passes=max_passes,
            batch_size=batch_size,
            forbidden_penalty_strength=forbidden_penalty_strength)
    else:
        num_reads = kwargs.pop("num_reads", 5000)
        num_sweeps = kwargs.pop("num_sweeps", 5000)
        batch_size = kwargs.pop("batch_size", 4096)
        initial_temp = kwargs.pop("initial_temp", 100.0)
        final_temp = kwargs.pop("final_temp", 0.001)
        diversity_scale = kwargs.pop("diversity_scale", 0.05)
        base.reject_unknown_kwargs("gpu_sa.solve", kwargs)
        pairs, wall_time, gap = solve_gpu_sa(
            Q, q, var_map, k_max=k_max, time_limit=time_limit, top_k=top_k,
            num_reads=num_reads, num_sweeps=num_sweeps,
            initial_temp=initial_temp, final_temp=final_temp,
            batch_size=batch_size, diversity_scale=diversity_scale)

    solutions = base.finalize((s for s, _ in pairs), Q, q, top_k)
    return solutions, wall_time, gap
