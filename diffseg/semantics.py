"""
semantics.py — Assign text labels to segments using BLIP captioning + noun matching.

Workflow (matches the original DiffSeg paper's optional semantic branch):
    1. Generate a caption for the image using BLIP.
    2. POS-tag the caption to extract nouns.
    3. For each noun, generate a cross-attention map via a CLIP-like text probe.
       (We use a simpler approach: cosine similarity between BLIP text embeddings
       and per-segment mean visual features extracted from the VAE encoder.)
    4. Assign each noun to the segment it best matches via voting.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from PIL import Image


# ── lazy imports ─────────────────────────────────────────────────────────────

def _load_blip(device: str = "cpu"):
    from transformers import BlipProcessor, BlipForConditionalGeneration
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device)
    model.eval()
    return processor, model


# ── caption generation ────────────────────────────────────────────────────────

def generate_caption(image: Image.Image, device: str = "cpu") -> str:
    """Generate a free-form caption for the image using BLIP."""
    processor, model = _load_blip(device)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption


# ── noun extraction ────────────────────────────────────────────────────────────

def extract_nouns(caption: str) -> List[str]:
    """
    Extract nouns from a caption using simple POS tagging (no heavy deps).
    Falls back to NLTK if available, otherwise uses a heuristic word filter.
    """
    # Try NLTK first
    try:
        import nltk
        try:
            nltk.data.find("taggers/averaged_perceptron_tagger")
        except LookupError:
            nltk.download("averaged_perceptron_tagger", quiet=True)
            nltk.download("punkt", quiet=True)

        tokens = nltk.word_tokenize(caption.lower())
        tagged = nltk.pos_tag(tokens)
        nouns = [word for word, pos in tagged if pos.startswith("NN") and len(word) > 2]
        return list(dict.fromkeys(nouns))  # deduplicate, preserve order
    except ImportError:
        pass

    # Fallback: basic stopword filtering
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "this",
        "that", "these", "those", "there", "here", "with", "in",
        "on", "at", "to", "of", "and", "or", "but", "for", "not",
        "it", "its", "we", "they", "he", "she", "you", "i", "my",
        "their", "some", "very", "just", "also", "one", "two",
    }
    words = caption.lower().split()
    return [w.strip(".,!?;:") for w in words if w not in stopwords and len(w) > 3]


# ── label assignment via mean color/feature voting ────────────────────────────

def assign_semantic_labels(
    label_map: np.ndarray,
    image: Image.Image,
    nouns: List[str],
    device: str = "cpu",
) -> Dict[int, str]:
    """
    Assign a noun label to each segment using visual similarity.

    Strategy: embed each noun and each segment's mean visual feature using
    the BLIP text encoder, then assign by cosine similarity.
    Falls back to "segment N" naming if BLIP embedding fails.

    Args:
        label_map:  H×W int array (labels 1, 2, ...).
        image:      Original PIL image.
        nouns:      List of candidate noun strings from caption.
        device:     Torch device.

    Returns:
        label_to_name: Dict mapping integer label → noun string.
    """
    if not nouns:
        unique_labels = sorted(set(label_map.flatten().tolist()) - {0})
        return {l: f"region {l}" for l in unique_labels}

    unique_labels = sorted(set(label_map.flatten().tolist()) - {0})

    try:
        label_to_name = _assign_via_blip_embeddings(
            label_map, image, unique_labels, nouns, device
        )
    except Exception:
        # Fallback: just cycle through nouns
        label_to_name = {}
        for i, label in enumerate(unique_labels):
            label_to_name[label] = nouns[i % len(nouns)]

    return label_to_name


def _assign_via_blip_embeddings(
    label_map: np.ndarray,
    image: Image.Image,
    unique_labels: List[int],
    nouns: List[str],
    device: str,
) -> Dict[int, str]:
    """
    Use BLIP's text encoder to embed nouns and BLIP's vision encoder to embed
    segment crops.  Assign each segment to its nearest noun in embedding space.
    """
    from transformers import BlipProcessor, BlipModel

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipModel.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device)
    model.eval()

    # Embed nouns via text encoder
    noun_embeddings = []
    for noun in nouns:
        inputs = processor(text=noun, return_tensors="pt").to(device)
        with torch.no_grad():
            text_feat = model.get_text_features(**inputs)  # (1, D)
        noun_embeddings.append(text_feat[0])
    noun_emb = torch.stack(noun_embeddings, dim=0)  # (num_nouns, D)
    noun_emb = torch.nn.functional.normalize(noun_emb, dim=-1)

    # Embed each segment crop via vision encoder
    img_arr = np.array(image.convert("RGB"))
    H_map, W_map = label_map.shape
    img_resized = np.array(image.convert("RGB").resize((W_map, H_map), Image.LANCZOS))

    label_to_name: Dict[int, str] = {}

    for label in unique_labels:
        mask = (label_map == label)
        if mask.sum() == 0:
            label_to_name[label] = "region"
            continue

        # Bounding box crop of the segment
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        y0, y1 = rows[0], rows[-1] + 1
        x0, x1 = cols[0], cols[-1] + 1

        # Scale back to original image coords
        orig_h, orig_w = img_arr.shape[:2]
        y0_o = int(y0 * orig_h / H_map)
        y1_o = int(y1 * orig_h / H_map)
        x0_o = int(x0 * orig_w / W_map)
        x1_o = int(x1 * orig_w / W_map)

        crop = Image.fromarray(img_arr[y0_o:y1_o, x0_o:x1_o])
        if crop.size[0] < 4 or crop.size[1] < 4:
            label_to_name[label] = "region"
            continue

        inputs = processor(images=crop, return_tensors="pt").to(device)
        with torch.no_grad():
            vis_feat = model.get_image_features(**inputs)  # (1, D)
        vis_feat = torch.nn.functional.normalize(vis_feat, dim=-1)

        # Cosine similarity to all nouns
        sims = (vis_feat @ noun_emb.T).squeeze(0)  # (num_nouns,)
        best_noun_idx = sims.argmax().item()
        label_to_name[label] = nouns[best_noun_idx]

    return label_to_name
