# DiffSeg — PyTorch Implementation

> **Unsupervised Zero-Shot Segmentation using Stable Diffusion**
> A clean PyTorch reimplementation of [Tian et al., CVPR 2024](https://arxiv.org/abs/2308.12469)
> Original TensorFlow/KerasCV repo: [google/diffseg](https://github.com/google/diffseg)

**Author:** KHIATI Rezkellah Noureddine — [@noureddinekhiati](https://github.com/noureddinekhiati)

[![HuggingFace Space](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/noureddinekhiati/diffseg-pytorch)
[![GitHub](https://img.shields.io/badge/GitHub-noureddinekhiati-black)](https://github.com/noureddinekhiati/diffseg-pytorch)

> ⚠️ **Unofficial PyTorch implementation** — not affiliated with the original authors (Google / Georgia Tech).

---

## What is DiffSeg?

DiffSeg segments any image **without any labels, training, or text prompts**.
It exploits the fact that a pre-trained Stable Diffusion model has already
learned to group objects — this grouping information lives inside its
self-attention layers.

**Key insight:** Pixels belonging to the same object have similar self-attention
maps (they attend to the same other pixels). DiffSeg makes this explicit
by iteratively merging attention maps whose KL divergence is below a threshold.

---

## Algorithm Overview

```
Input image
    │
    ▼
┌─────────────────────────────┐
│ Stable Diffusion 1.5 UNet   │  ← 1 denoising step (no prompt)
│ + AttnProcessor hooks       │  ← tap 16 self-attention maps
└─────────────────────────────┘
    │  4D attention tensor Af ∈ R^(64×64×64×64)
    ▼
┌─────────────────────────────┐
│ Attention Aggregation       │  ← bilinear upsample key axis
│                             │  ← tile/repeat query axis (paper-exact)
│                             │  ← weighted sum (weight ∝ resolution)
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│ Iterative KL Merging        │  ← sample M×M anchor grid
│  KL(pᵢ ‖ pⱼ) < threshold? │  ← merge → N iterations
│  ← KL THRESHOLD SLIDER →   │
└─────────────────────────────┘
    │  object proposal maps
    ▼
┌─────────────────────────────┐
│ Non-Maximum Suppression     │  → integer label map
└─────────────────────────────┘
    │
    ▼
Coloured overlay + optional semantic labels (BLIP)
```

---

## Live Demo Controls

| Control | Type | Effect |
|---|---|---|
| **Timestep** | Slider 1–999 | Which denoising step to use. ~50–200 is the sweet spot. Requires re-encode. |
| **KL threshold** | Slider 0.1–3.0 | Lower = more segments. Higher = fewer coarser regions. Instant. |
| **Resolution preset** | Dropdown | Which attention resolutions to aggregate (matches paper Fig. 4). Instant. |
| **w64 / w32 / w16 / w8** | Sliders | Custom per-resolution weights. Overrides preset. Instant. |
| **Semantic labels** | Checkbox | Uses BLIP to name each segment. Adds ~5s. |

### KL threshold guide

| Value | Effect |
|---|---|
| 0.1 – 0.3 | Very fine-grained — may over-segment textures |
| 0.4 – 0.7 | Good for objects with clear boundaries **(default: 0.5)** |
| 0.8 – 1.5 | Coarser — merges similar objects |
| 1.5 – 3.0 | Very coarse — only major scene regions |

---

## Project Structure

```
diffseg-pytorch/
├── diffseg/
│   ├── __init__.py       # public API
│   ├── hooks.py          # AttnProcessor-based attention extraction
│   ├── aggregation.py    # paper-exact multi-resolution aggregation
│   ├── merging.py        # iterative KL-divergence merge
│   ├── nms.py            # NMS → integer label map
│   ├── visualize.py      # colour overlay + legend rendering
│   └── semantics.py      # BLIP caption + noun → segment label assignment
├── model.py              # DiffSegModel: encode() + segment()
├── app.py                # Gradio live demo
├── make_examples.py      # download example images
├── requirements.txt
└── README.md
```

---

## Local Setup

```bash
git clone https://github.com/noureddinekhiati/diffseg-pytorch
cd diffseg-pytorch

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate gradio Pillow numpy nltk huggingface-hub

# NLTK data for semantic labeling
python -c "import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('punkt'); nltk.download('averaged_perceptron_tagger_eng')"

# Download example images
python make_examples.py

# Run
python app.py
```

Open http://localhost:7860

**GPU strongly recommended.** On an RTX A6000 encode takes ~3s and re-segment ~0.1s.
CPU-only works but encode takes ~60–90s.

---

## Python API

```python
from PIL import Image
from model import DiffSegModel

model = DiffSegModel.load(device="cuda")

# Encode once — expensive (~3–5s on GPU)
image = Image.open("my_photo.jpg")
bundle = model.encode(image, timestep=100)

# Segment many times with different parameters — cheap (~0.1s each)
result_fine   = model.segment(bundle, kl_threshold=0.3)
result_medium = model.segment(bundle, kl_threshold=0.7)
result_coarse = model.segment(bundle, kl_threshold=1.5)

result_fine["overlay"].save("segmented.png")
print(f"Found {result_fine['n_segments']} segments")
```

---

## Differences from Original

| | Original (google/diffseg) | This repo |
|---|---|---|
| Framework | TensorFlow 2.14 + KerasCV | **PyTorch 2.x + diffusers** |
| Attention extraction | KerasCV layer hooks | **diffusers AttnProcessor** |
| Aggregation upsampling | Bilinear key + tile query | **Same — paper-exact** |
| Semantic labels | BLIP (TF) | **BLIP via HuggingFace transformers** |
| Live demo | Jupyter notebook | **Gradio app (HF Spaces)** |
| KL slider | Hardcoded | **Interactive live slider** |
| Timestep | Hardcoded | **Interactive slider** |
| Resolution weights | Hardcoded | **Interactive sliders + presets** |

---

## Deploy to HuggingFace Spaces

1. Create a new Space at https://huggingface.co/new-space
   - SDK: **Gradio** · Hardware: **ZeroGPU**

2. Re-enable the `spaces` decorator in `app.py`:
   ```python
   import spaces
   # and add @spaces.GPU(duration=60) above encode_image()
   ```

3. Add HF metadata block to the **very top** of `README.md`:
   ```
   ---
   title: DiffSeg PyTorch
   emoji: 🔍
   colorFrom: purple
   colorTo: teal
   sdk: gradio
   sdk_version: "4.19.0"
   app_file: app.py
   license: mit
   hardware: zero-gpu
   ---
   ```

4. Push:
   ```bash
   git remote add space https://huggingface.co/spaces/noureddinekhiati/diffseg-pytorch
   git push space main
   ```

---

## Citation

If you use this code, please also cite the original paper:

```bibtex
@inproceedings{tian2024diffuse,
  title={Diffuse, Attend, and Segment: Unsupervised Zero-Shot Segmentation using Stable Diffusion},
  author={Tian, Junjiao and Aggarwal, Lavisha and Colaco, Andrea and Kira, Zsolt and Gonzalez-Franco, Mar},
  booktitle={CVPR},
  year={2024}
}
```

---

## License

MIT © 2025 KHIATI Rezkellah Noureddine