"""
merging.py — Iterative attention merging via KL divergence.

This is the core of DiffSeg.  Given the aggregated 4D attention tensor Af,
we want to group the 64×64 spatial positions into "object proposals" where
each proposal is a probability map over all positions.

Algorithm (paper Section 3.2):
    1. Sample an M×M grid of anchor positions.
    2. For each anchor, extract its attention map: Af[i, j, :, :].
    3. Iteratively merge anchors whose attention maps are similar
       (measured by symmetric KL divergence < threshold).
    4. After N_iter iterations, output a list of merged probability maps
       (one per surviving group).

The KL threshold is the main user-facing hyperparameter:
    - Low  (~0.3): many fine-grained segments
    - High (~2.0): few coarse segments
    The live slider controls this directly.
"""

from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn.functional as F


# ── KL divergence between two probability distributions ──────────────────────

def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Symmetric KL divergence: 0.5 * (KL(p||q) + KL(q||p)).
    p, q: 1D probability vectors, already normalized.
    """
    p = p + eps
    q = q + eps
    kl_pq = (p * (p / q).log()).sum()
    kl_qp = (q * (q / p).log()).sum()
    return float(0.5 * (kl_pq + kl_qp))


# ── iterative merge ───────────────────────────────────────────────────────────

def iterative_merge(
    Af: torch.Tensor,
    kl_threshold: float = 0.5,
    M: int = 8,
    n_iter: int = 10,
) -> List[torch.Tensor]:
    """
    Run iterative KL-divergence merging on the aggregated attention tensor.

    Args:
        Af:            Aggregated attention tensor (T, T, T, T), T=64.
        kl_threshold:  Merge anchors whose symmetric KL < this value.
                       This is the "K slider" in the live demo.
        M:             Grid size for anchor sampling.  M×M anchors are sampled.
        n_iter:        Number of merging iterations.

    Returns:
        proposals: List of probability maps, each shape (T*T,).
                   Each map represents one object/stuff category proposal.
    """
    T = Af.shape[0]
    N = T * T
    device = Af.device

    # Flatten spatial dims: (T, T, T, T) → (N, N)
    Af_flat = Af.reshape(N, N)  # Af_flat[idx] is the attention map of position idx

    # ── 1. Sample M×M anchor grid ─────────────────────────────────────────────
    step_h = T // M
    step_w = T // M
    anchor_coords: List[Tuple[int, int]] = []
    for i in range(M):
        for j in range(M):
            h = min(i * step_h + step_h // 2, T - 1)
            w = min(j * step_w + step_w // 2, T - 1)
            anchor_coords.append((h, w))

    anchor_indices = [h * T + w for h, w in anchor_coords]

    # ── 2. Initialize groups: each anchor is its own group ────────────────────
    # Each group stores an *averaged* probability map
    group_maps: List[torch.Tensor] = [
        Af_flat[idx].clone() for idx in anchor_indices
    ]
    group_members: List[List[int]] = [[idx] for idx in anchor_indices]

    # ── 3. Iterative merging ──────────────────────────────────────────────────
    for _ in range(n_iter):
        merged_this_round = True
        while merged_this_round:
            merged_this_round = False
            n_groups = len(group_maps)
            if n_groups <= 1:
                break

            merged = [False] * n_groups
            new_maps: List[torch.Tensor] = []
            new_members: List[List[int]] = []

            i = 0
            while i < n_groups:
                if merged[i]:
                    i += 1
                    continue

                # Find the most similar unmerged group to group i
                best_j = -1
                best_kl = float("inf")
                for j in range(i + 1, n_groups):
                    if merged[j]:
                        continue
                    kl = kl_divergence(group_maps[i], group_maps[j])
                    if kl < best_kl:
                        best_kl = kl
                        best_j = j

                if best_j != -1 and best_kl < kl_threshold:
                    # Merge group i and group best_j
                    members_i = group_members[i]
                    members_j = group_members[best_j]
                    # New map = weighted average by group size
                    ni = len(members_i)
                    nj = len(members_j)
                    merged_map = (ni * group_maps[i] + nj * group_maps[best_j]) / (ni + nj)
                    # Re-normalize
                    merged_map = merged_map / (merged_map.sum() + 1e-8)
                    new_maps.append(merged_map)
                    new_members.append(members_i + members_j)
                    merged[i] = True
                    merged[best_j] = True
                    merged_this_round = True
                else:
                    # No merge for group i this pass
                    new_maps.append(group_maps[i])
                    new_members.append(group_members[i])
                    merged[i] = True

                i += 1

            group_maps = new_maps
            group_members = new_members

    return group_maps


# ── fast vectorized KL for all pairs (used in large-M settings) ───────────────

def batch_kl_matrix(maps: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute pairwise symmetric KL divergence for a batch of distributions.

    Args:
        maps: (K, N) tensor of K probability distributions over N bins.

    Returns:
        kl_mat: (K, K) symmetric KL divergence matrix.
    """
    maps = maps + eps
    log_maps = maps.log()  # (K, N)

    # KL(p||q) = sum p * (log p - log q)
    # kl_pq[i,j] = sum_n maps[i,n] * (log_maps[i,n] - log_maps[j,n])
    # = (maps * log_maps).sum(1) - maps @ log_maps.T
    entropy = (maps * log_maps).sum(dim=1)  # (K,)
    cross = maps @ log_maps.T               # (K, K): cross[i,j] = sum_n maps[i] * log_maps[j]
    kl_pq = entropy.unsqueeze(1) - cross    # (K, K)
    kl_qp = kl_pq.T

    return 0.5 * (kl_pq + kl_qp)
