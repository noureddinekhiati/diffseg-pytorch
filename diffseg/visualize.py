"""
visualize.py — Render a segmentation label map as a coloured overlay.

Produces:
    - overlay: original image blended with a false-colour mask
    - coloured_mask: pure RGBA mask (transparent background)
    - palette: list of (R, G, B) colours assigned to each label
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image


# ── Distinct colour palette (22 visually separable colours) ──────────────────
_PALETTE: List[Tuple[int, int, int]] = [
    (255,  64,  64),  # red
    ( 64, 160, 255),  # blue
    ( 64, 210,  64),  # green
    (255, 200,  40),  # amber
    (200,  80, 255),  # purple
    ( 40, 210, 200),  # teal
    (255, 128,  40),  # orange
    (255,  80, 180),  # pink
    (140, 200,  60),  # lime
    ( 60, 120, 200),  # steel blue
    (220,  60, 120),  # crimson
    ( 80, 220, 160),  # seafoam
    (200, 160,  40),  # gold
    (120,  60, 200),  # indigo
    ( 60, 180, 220),  # sky
    (200,  80,  60),  # coral
    (160, 220,  80),  # yellow-green
    ( 80, 160, 240),  # periwinkle
    (240, 120,  80),  # salmon
    (100, 240, 100),  # mint
    (180,  80, 160),  # mauve
    (240, 200,  80),  # pale gold
]


def label_map_to_overlay(
    label_map: np.ndarray,
    original_image: Image.Image,
    alpha: float = 0.55,
    background_label: int = 0,
) -> Tuple[Image.Image, Image.Image, List[Tuple[int, int, int]]]:
    """
    Blend a segmentation label map onto the original image.

    Args:
        label_map:        H×W numpy int array with integer labels (0 = background).
        original_image:   PIL Image (RGB), must match label_map spatial dims.
        alpha:            Opacity of the coloured mask overlay (0=invisible, 1=opaque).
        background_label: Label value treated as background (transparent in mask).

    Returns:
        (overlay, coloured_mask, palette_used)
        - overlay:        PIL Image (RGB) — original + coloured segments
        - coloured_mask:  PIL Image (RGBA) — transparent bg, coloured segments
        - palette_used:   List of (R,G,B) for labels 1, 2, 3, ...
    """
    H, W = label_map.shape
    orig = original_image.convert("RGB").resize((W, H), Image.LANCZOS)
    orig_arr = np.array(orig, dtype=np.float32)

    unique_labels = sorted(set(label_map.flatten().tolist()))
    # Drop background
    segment_labels = [l for l in unique_labels if l != background_label]

    # Assign colours
    colour_map: dict[int, Tuple[int, int, int]] = {}
    palette_used: List[Tuple[int, int, int]] = []
    for i, label in enumerate(segment_labels):
        colour = _PALETTE[i % len(_PALETTE)]
        colour_map[label] = colour
        palette_used.append(colour)

    # Build RGBA mask
    mask_rgba = np.zeros((H, W, 4), dtype=np.uint8)
    for label, (r, g, b) in colour_map.items():
        region = label_map == label
        mask_rgba[region] = [r, g, b, 200]   # ~78% opacity in mask image

    # Build overlay by blending
    overlay_arr = orig_arr.copy()
    for label, (r, g, b) in colour_map.items():
        region = label_map == label
        overlay_arr[region] = (
            (1 - alpha) * orig_arr[region]
            + alpha * np.array([r, g, b], dtype=np.float32)
        )

    overlay = Image.fromarray(overlay_arr.clip(0, 255).astype(np.uint8), "RGB")
    coloured_mask = Image.fromarray(mask_rgba, "RGBA")

    return overlay, coloured_mask, palette_used


def draw_legend(
    palette: List[Tuple[int, int, int]],
    labels: Optional[List[str]] = None,
    swatch_size: int = 20,
    padding: int = 6,
    font_size: int = 13,
) -> Image.Image:
    """
    Render a small legend image: coloured swatches + optional text labels.

    Args:
        palette:    List of (R, G, B) colours, one per segment.
        labels:     Optional list of string names (e.g. from BLIP semantics).
        swatch_size: Pixel size of each colour square.
        padding:    Spacing between rows and around swatches.
        font_size:  Font size for label text.

    Returns:
        PIL Image (RGBA) — the legend panel.
    """
    from PIL import ImageDraw, ImageFont
    n = len(palette)
    if n == 0:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    row_h = swatch_size + padding
    text_w = 180
    total_h = n * row_h + padding
    total_w = swatch_size + padding + text_w + padding

    img = Image.new("RGBA", (total_w, total_h), (30, 30, 30, 200))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    for i, (r, g, b) in enumerate(palette):
        y = padding + i * row_h
        # Swatch
        draw.rectangle(
            [padding, y, padding + swatch_size, y + swatch_size],
            fill=(r, g, b, 255),
        )
        # Label text
        if labels and i < len(labels):
            txt = labels[i]
        else:
            txt = f"segment {i + 1}"
        draw.text(
            (padding + swatch_size + padding, y + 2),
            txt,
            fill=(240, 240, 240, 255),
            font=font,
        )

    return img
