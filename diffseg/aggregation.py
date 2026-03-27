"""
aggregation.py — Aggregate multi-resolution attention tensors into one 64×64 tensor.

CORRECT implementation matching the paper formula exactly:
    Ak = Bilinear-upsample(Ak) ∈ R^(hk × wk × 64 × 64)

The paper treats the 4D attention tensor asymmetrically:
  - dims 0,1 (query/reference location): TILE/REPEAT to match 64×64
    (each low-res query position corresponds to a patch of high-res positions)
  - dims 2,3 (key/value spatial map):    BILINEAR UPSAMPLE to 64×64
    (the probability distribution over spatial locations is interpolated)

"""

from __future__ import annotations
from typing import Dict, List, Union
import torch
import torch.nn.functional as F


# ── resolution weight presets (matches paper Figure 4) ───────────────────────
RESOLUTION_MODES: Dict[str, Dict[str, float]] = {
    "proportional (paper default)": {"64": 1.0,  "32": 0.5,  "16": 0.25, "8": 0.125},
    "only 64×64":                   {"64": 1.0,  "32": 0.0,  "16": 0.0,  "8": 0.0},
    "only 32×32":                   {"64": 0.0,  "32": 1.0,  "16": 0.0,  "8": 0.0},
    "only 16×16 + 8×8":             {"64": 0.0,  "32": 0.0,  "16": 1.0,  "8": 1.0},
    "64×64 + 32×32":                {"64": 1.0,  "32": 1.0,  "16": 0.0,  "8": 0.0},
    "equal all":                    {"64": 1.0,  "32": 1.0,  "16": 1.0,  "8": 1.0},
}

RESOLUTION_MODE_NAMES = list(RESOLUTION_MODES.keys())


def _upsample_attention_map(
    attn: torch.Tensor,
    src_res: int,
    tgt_res: int,
) -> torch.Tensor:
    """
    Upsample a single attention map from (src_res², src_res²) to (tgt_res², tgt_res²).

    Paper formula: Ak = Bilinear-upsample(Ak) ∈ R^(hk×wk × 64×64)
    - Key axis (cols, dims 2-3): bilinear upsample — the probability distribution
      over spatial locations is smoothly interpolated.
    - Query axis (rows, dims 0-1): nearest/tile repeat — each low-res query
      position maps to a (tgt/src × tgt/src) patch of high-res positions.

    Args:
        attn:    Tensor (n, n) where n = src_res²
        src_res: Source spatial resolution (e.g. 8, 16, 32)
        tgt_res: Target spatial resolution (64)

    Returns:
        Tensor (N, N) where N = tgt_res²
    """
    n = src_res * src_res
    N = tgt_res * tgt_res
    scale = tgt_res // src_res   # integer scale factor (8, 4, 2)

    # ── Step 1: Bilinear upsample the key/value axis (cols) ──────────────────
    # attn: (n, n) → treat cols as a spatial map → upsample to tgt_res²
    # Reshape cols: (n, src_res, src_res) then interpolate → (n, tgt_res, tgt_res)
    attn_4d = attn.reshape(n, 1, src_res, src_res)           # (n, 1, r, r)
    key_up = F.interpolate(
        attn_4d,
        size=(tgt_res, tgt_res),
        mode="bilinear",
        align_corners=False,
    ).reshape(n, N)                                           # (n, N)

    # ── Step 2: Tile/repeat the query axis (rows) ─────────────────────────────
    # Each low-res query position corresponds to a (scale×scale) patch of
    # high-res positions — they all share the same attention distribution.
    # key_up: (n, N) → reshape to (src_res, src_res, N)
    key_up_2d = key_up.reshape(src_res, src_res, N)          # (r, r, N)

    # Repeat each row and column by scale factor
    # torch.repeat_interleave repeats each element scale times along a dim
    query_up = key_up_2d.repeat_interleave(scale, dim=0)     # (T, r, N)
    query_up = query_up.repeat_interleave(scale, dim=1)      # (T, T, N)
    result = query_up.reshape(N, N)                          # (N, N)

    # Row-normalize → valid probability distribution per query position
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-8)

    return result


def aggregate_attention(
    maps: Dict[str, List[torch.Tensor]],
    target_res: int = 64,
    device: Union[str, torch.device] = "cpu",
    resolution_mode: str = "proportional (paper default)",
    custom_weights: Union[Dict[str, float], None] = None,
) -> torch.Tensor:
    """
    Aggregate all collected attention maps into a single 4D tensor.

    Args:
        maps:            {"64": [T,...], "32": [...], "16": [...], "8": [...]}
                         Each T has shape (N, N) where N = h*w.
        target_res:      64 for SD 1.x.
        device:          Torch device.
        resolution_mode: Named preset (ignored if custom_weights given).
        custom_weights:  {"64": w, "32": w, "16": w, "8": w} overrides preset.

    Returns:
        Af: (target_res, target_res, target_res, target_res),
            Af[i,j].sum() == 1 for all (i,j).
    """
    weights = custom_weights if custom_weights is not None else \
              RESOLUTION_MODES.get(resolution_mode,
                                   RESOLUTION_MODES["proportional (paper default)"])

    N = target_res * target_res
    accumulator = torch.zeros(N, N, device=device, dtype=torch.float32)
    total_weight = 0.0

    for res_key, tensor_list in maps.items():
        res = int(res_key)
        if res > target_res:
            continue
        w = weights.get(res_key, 0.0)
        if w == 0.0:
            continue

        for attn in tensor_list:
            attn = attn.to(device=device, dtype=torch.float32)

            if res < target_res:
                attn_up = _upsample_attention_map(attn, src_res=res, tgt_res=target_res)
            else:
                # Already 64×64 — just normalize rows
                attn_up = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

            accumulator += w * attn_up
            total_weight += w

    if total_weight == 0:
        raise ValueError("All resolution weights are zero or no maps were collected.")

    Af_flat = accumulator / total_weight
    Af_flat = Af_flat / (Af_flat.sum(dim=-1, keepdim=True) + 1e-8)
    return Af_flat.reshape(target_res, target_res, target_res, target_res)