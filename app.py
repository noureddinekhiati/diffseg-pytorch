# Copyright (c) 2025 KHIATI Rezkellah Noureddine
# GitHub: https://github.com/noureddinekhiati
#
# Unofficial PyTorch implementation of:
#   "Diffuse, Attend, and Segment: Unsupervised Zero-Shot Segmentation
#    using Stable Diffusion" — Tian et al., CVPR 2024
#   Original TF repo: https://github.com/google/diffseg
#
# Licensed under the MIT License.
# ──────────────────────────────────────────────────────────────────────────────

"""
app.py — DiffSeg PyTorch — Gradio live demo.

Controls:
  - Timestep slider         → re-encodes (runs SD forward pass)
  - KL threshold slider     → re-segments only (instant, no SD re-run)
  - Resolution mode preset  → re-segments only (instant)
  - Per-resolution sliders  → re-segments only (instant)
  - Semantic labels toggle  → adds BLIP caption + noun assignment
"""

import os
import torch
import gradio as gr
import numpy as np
from PIL import Image
from typing import Optional

from model import DiffSegModel, AttentionBundle
from diffseg.aggregation import RESOLUTION_MODE_NAMES
from diffseg.semantics import generate_caption, extract_nouns, assign_semantic_labels
from diffseg.visualize import label_map_to_overlay, draw_legend

# ── Global model ──────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL: Optional[DiffSegModel] = None

def get_model() -> DiffSegModel:
    global MODEL
    if MODEL is None:
        MODEL = DiffSegModel.load(device=DEVICE)
    return MODEL


# ── Core functions ─────────────────────────────────────────────────────────────

def encode_image(image, timestep):
    if image is None:
        return None, " Please upload an image first."
    try:
        model = get_model()
        image_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        bundle = model.encode(image_pil, timestep=int(timestep))
        resolutions = {k: len(v) for k, v in bundle.maps.items()}
        return bundle, f" Encoded at timestep **{int(timestep)}** — attention maps: {resolutions}"
    except Exception as e:
        import traceback
        return None, f" Encoding failed: {e}\n```\n{traceback.format_exc()}\n```"


def segment_from_bundle(
    bundle, kl_threshold, resolution_mode,
    use_custom_weights, w64, w32, w16, w8,
    use_semantics,
):
    if bundle is None:
        return Image.new("RGB", (512, 512), (30, 30, 40)), "Upload an image and click **Encode** first.", None

    try:
        model = get_model()
        result = model.segment(
            bundle,
            kl_threshold=float(kl_threshold),
            resolution_mode=resolution_mode,
            use_custom_weights=bool(use_custom_weights),
            w64=float(w64), w32=float(w32), w16=float(w16), w8=float(w8),
        )

        overlay    = result["overlay"]
        n_seg      = result["n_segments"]
        palette    = result["palette"]
        label_map  = result["label_map"]

        # Optional semantic labels
        seg_labels = None
        caption    = ""
        if use_semantics and n_seg > 0:
            try:
                caption = generate_caption(bundle.original_image, device=DEVICE)
                nouns = extract_nouns(caption)
                label_to_name = assign_semantic_labels(label_map, bundle.original_image, nouns, device=DEVICE)
                unique_labels = sorted(set(label_map.flatten().tolist()) - {0})
                seg_labels = [label_to_name.get(l, f"region {l}") for l in unique_labels]
            except Exception:
                seg_labels = None

        legend = draw_legend(palette, labels=seg_labels)

        weight_info = (
            f"w64={w64:.2f} w32={w32:.2f} w16={w16:.2f} w8={w8:.2f}"
            if use_custom_weights else resolution_mode
        )
        info = (
            f"**{n_seg} segment{'s' if n_seg != 1 else ''} found**  ·  "
            f"KL `{kl_threshold:.2f}`  ·  {weight_info}  ·  device `{DEVICE}`"
        )
        if caption:
            info += f"\n\n Caption: *{caption}*"

        return overlay, info, legend

    except Exception as e:
        import traceback
        return Image.new("RGB", (512, 512), (60, 20, 20)), f" {e}\n```\n{traceback.format_exc()}\n```", None


def encode_then_segment(image, timestep, kl_threshold, resolution_mode,
                        use_custom_weights, w64, w32, w16, w8, use_semantics):
    bundle, status = encode_image(image, timestep)
    overlay, info, legend = segment_from_bundle(
        bundle, kl_threshold, resolution_mode,
        use_custom_weights, w64, w32, w16, w8, use_semantics
    )
    return bundle, status, overlay, info, legend


# ── Examples ──────────────────────────────────────────────────────────────────
EXAMPLES = [
    ["examples/airplane.jpg"],
    ["examples/cat.jpg"],
    ["examples/motorcycle.jpg"],
    ["examples/fruits.jpg"],
]

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
#title { text-align: center; }
#subtitle { text-align: center; color: #aaa; font-size: 0.85rem; margin-top: -10px; }
#unofficial { text-align: center; font-size: 1.05rem; color: #f0a500;
              border: 1px solid #f0a500; border-radius: 8px;
              padding: 6px 14px; display: inline-block; margin: 6px auto; }
.encode-note { font-size: 0.78rem; color: #888; margin-top: 2px; }
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="DiffSeg PyTorch") as demo:

    bundle_state = gr.State(None)

    # ── Header ────────────────────────────────────────────────────────────────
    gr.Markdown("# 🔍 DiffSeg — Unsupervised Zero-Shot Segmentation", elem_id="title")
    gr.HTML(
        '<div id="unofficial">⚠️ Unofficial PyTorch Implementation</div>'
        '<div style="text-align:center;margin-top:4px;font-size:0.82rem;color:#aaa;">'
        'Based on: Tian et al., <em>"Diffuse, Attend, and Segment"</em> (CVPR 2024) · '
        'PyTorch port by <strong>KHIATI Rezkellah Noureddine</strong>'
        '</div>'
    )
    gr.Markdown(
        "Upload any image — the model uses **self-attention maps** from a frozen "
        "Stable Diffusion UNet (no text prompt, no labels, no training) to segment it.",
        elem_id="subtitle"
    )

    with gr.Row():

        # ── Left: inputs & controls ───────────────────────────────────────────
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input image", type="pil", height=360)

            # ── Timestep (requires re-encode) ─────────────────────────────────
            with gr.Group():
                gr.Markdown("### 1 · Timestep  *(re-encodes on change)*")
                timestep_slider = gr.Slider(
                    minimum=1, maximum=999, value=50, step=1,
                    label="Timestep  — lower = early denoising (coarser), ~50–200 = sweet spot",
                )
                gr.HTML('<div class="encode-note">⚡ Changes here require clicking Encode again.</div>')

            encode_btn = gr.Button("🔍 Encode image", variant="primary")
            encode_status = gr.Markdown("*Upload an image, then click Encode.*")

            # ── Segmentation controls (instant, no re-encode) ─────────────────
            with gr.Group():
                gr.Markdown("### 2 · Segmentation  *(instant — no re-encode)*")

                kl_slider = gr.Slider(
                    minimum=0.10, maximum=3.00, value=0.50, step=0.05,
                    label="KL threshold — lower = more segments, higher = fewer",
                )

                gr.Markdown("**Attention resolution weights**")
                resolution_mode = gr.Dropdown(
                    choices=RESOLUTION_MODE_NAMES,
                    value="proportional (paper default)",
                    label="Resolution preset",
                )

                use_custom_weights = gr.Checkbox(
                    label="Use custom per-resolution weights (overrides preset above)",
                    value=False,
                )
                with gr.Row():
                    w64 = gr.Slider(0.0, 1.0, value=1.000, step=0.05, label="64×64 weight")
                    w32 = gr.Slider(0.0, 1.0, value=0.500, step=0.05, label="32×32 weight")
                with gr.Row():
                    w16 = gr.Slider(0.0, 1.0, value=0.250, step=0.05, label="16×16 weight")
                    w8  = gr.Slider(0.0, 1.0, value=0.125, step=0.05, label="8×8 weight")

                use_semantics = gr.Checkbox(
                    label="Add semantic labels via BLIP (~5s extra)",
                    value=False,
                )

            gr.Examples(examples=EXAMPLES, inputs=input_image, label="Example images")

        # ── Right: outputs ────────────────────────────────────────────────────
        with gr.Column(scale=1):
            output_overlay = gr.Image(label="Segmentation overlay", type="pil", height=420)
            seg_info = gr.Markdown("*Segmentation info will appear here.*")
            output_legend = gr.Image(label="Segment legend", type="pil", height=220)

    # ── Wiring ────────────────────────────────────────────────────────────────

    seg_inputs = [bundle_state, kl_slider, resolution_mode,
                  use_custom_weights, w64, w32, w16, w8, use_semantics]
    seg_outputs = [output_overlay, seg_info, output_legend]

    # Encode button → encode then immediately segment
    encode_btn.click(
        fn=encode_then_segment,
        inputs=[input_image, timestep_slider, kl_slider, resolution_mode,
                use_custom_weights, w64, w32, w16, w8, use_semantics],
        outputs=[bundle_state, encode_status, output_overlay, seg_info, output_legend],
    )

    # All cheap sliders → re-segment only (no re-encode)
    for component in [kl_slider, resolution_mode, use_custom_weights,
                      w64, w32, w16, w8, use_semantics]:
        component.change(
            fn=segment_from_bundle,
            inputs=seg_inputs,
            outputs=seg_outputs,
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)