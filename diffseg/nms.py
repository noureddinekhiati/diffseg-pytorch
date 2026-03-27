"""
nms.py — Non-maximum suppression to convert object proposals into a segmentation mask.

After iterative merging, we have a list of probability maps (proposals).
Each proposal assigns to every pixel a probability that it belongs to that segment.
NMS converts this into a hard label map by:
    1. For each pixel, find the proposal with highest activation.
    2. Suppress proposals that are subsets of larger proposals.
    3. Remove tiny proposals (noise).
    4. Re-index surviving proposals into a clean integer label map.
"""

from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn.functional as F


def nms_proposals(
    proposals: List[torch.Tensor],
    target_hw: Tuple[int, int],
    min_area_ratio: float = 0.01,
    iou_threshold: float = 0.85,
) -> torch.Tensor:
    """
    Apply NMS to proposal probability maps and produce a final label mask.

    Args:
        proposals:       List of 1D probability maps, each shape (N,) where N = T*T.
        target_hw:       (H, W) — final output spatial resolution (usually 64×64,
                         then upscaled to original image size by caller).
        min_area_ratio:  Proposals covering less than this fraction of total pixels
                         are dropped as noise.
        iou_threshold:   Proposals with IoU > this against a larger proposal are
                         suppressed.

    Returns:
        label_map: LongTensor of shape (H, W) with integer labels 0, 1, 2, ...
                   Label 0 = background (unclaimed or suppressed pixels).
    """
    H, W = target_hw
    N = H * W

    if not proposals:
        return torch.zeros(H, W, dtype=torch.long)

    # ── Stack into matrix ─────────────────────────────────────────────────────
    # (K, N) — each row is one proposal's probability map
    prob_mat = torch.stack(proposals, dim=0)  # (K, N)

    # ── Hard assignment: argmax over proposals per pixel ─────────────────────
    # Each pixel gets the proposal that "claims" it most strongly
    hard_assignment = prob_mat.argmax(dim=0)  # (N,)

    # ── Build binary masks from assignment ───────────────────────────────────
    K = prob_mat.shape[0]
    binary_masks = torch.zeros(K, N, dtype=torch.bool)
    for k in range(K):
        binary_masks[k] = (hard_assignment == k)

    # ── Filter tiny proposals ─────────────────────────────────────────────────
    areas = binary_masks.float().sum(dim=1)  # (K,)
    min_area = min_area_ratio * N
    valid = areas >= min_area  # (K,) bool

    # ── IoU-based suppression: remove proposals that heavily overlap a larger one
    valid_indices = valid.nonzero(as_tuple=False).squeeze(1).tolist()

    # Sort by area descending (largest first, keep large, suppress subsets)
    valid_indices.sort(key=lambda i: areas[i].item(), reverse=True)

    keep = []
    suppressed = set()

    for i in valid_indices:
        if i in suppressed:
            continue
        keep.append(i)
        mask_i = binary_masks[i].float()
        area_i = areas[i].item()

        for j in valid_indices:
            if j == i or j in suppressed or j in keep:
                continue
            mask_j = binary_masks[j].float()
            area_j = areas[j].item()

            intersection = (mask_i * mask_j).sum().item()
            union = area_i + area_j - intersection
            iou = intersection / (union + 1e-8)

            if iou > iou_threshold:
                suppressed.add(j)

    # ── Build final label map ─────────────────────────────────────────────────
    label_map = torch.zeros(N, dtype=torch.long)
    for new_label, orig_idx in enumerate(keep, start=1):
        label_map[binary_masks[orig_idx]] = new_label

    label_map = label_map.reshape(H, W)
    return label_map


def upsample_label_map(
    label_map: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """
    Upsample an integer label map to the original image resolution using
    nearest-neighbor interpolation (to preserve crisp segment boundaries).

    Args:
        label_map: LongTensor (H, W) with integer labels.
        target_h:  Target height in pixels.
        target_w:  Target width in pixels.

    Returns:
        Upsampled LongTensor of shape (target_h, target_w).
    """
    # F.interpolate requires float and (N, C, H, W) format
    lm_float = label_map.unsqueeze(0).unsqueeze(0).float()
    upsampled = F.interpolate(
        lm_float,
        size=(target_h, target_w),
        mode="nearest",
    )
    return upsampled.squeeze(0).squeeze(0).long()
