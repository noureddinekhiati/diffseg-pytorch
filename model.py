"""
model.py — DiffSeg PyTorch model.

Exposes two operations:

    encode(image, timestep)
        Runs VAE encode + one UNet forward pass at the given timestep,
        collecting all self-attention maps.  Returns an AttentionBundle.
        *** Re-run this when timestep changes. ***

    segment(bundle, kl_threshold, resolution_mode, w64, w32, w16, w8, ...)
        Runs aggregation + iterative KL merge + NMS on the cached bundle.
        *** This is cheap — re-run on every slider change. ***
        resolution_mode selects a preset; OR custom weights w64/w32/w16/w8
        override it when the user uses the individual sliders.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from PIL import Image

from diffseg.hooks import AttentionCollector, patch_attention_for_hooks
from diffseg.aggregation import aggregate_attention, RESOLUTION_MODES, RESOLUTION_MODE_NAMES
from diffseg.merging import iterative_merge
from diffseg.nms import nms_proposals, upsample_label_map
from diffseg.visualize import label_map_to_overlay, draw_legend


# ── attention bundle ──────────────────────────────────────────────────────────

@dataclass
class AttentionBundle:
    """Cached result of one SD forward pass. Re-segment cheaply from this."""
    maps: Dict[str, List[torch.Tensor]]
    original_image: Image.Image
    latent_size: Tuple[int, int]
    timestep: int
    device: str = "cpu"

    # Cached aggregated tensors keyed by (resolution_mode, w64, w32, w16, w8)
    _Af_cache: Dict = field(default_factory=dict, repr=False)

    def get_Af(
        self,
        target_res: int = 64,
        resolution_mode: str = "proportional (paper default)",
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """Return aggregated 4D tensor, cached per unique weight config."""
        if custom_weights is not None:
            cache_key = ("custom", tuple(sorted(custom_weights.items())))
        else:
            cache_key = ("preset", resolution_mode)

        if cache_key not in self._Af_cache:
            self._Af_cache[cache_key] = aggregate_attention(
                self.maps,
                target_res=target_res,
                device=self.device,
                resolution_mode=resolution_mode,
                custom_weights=custom_weights,
            )
        return self._Af_cache[cache_key]


# ── main model ────────────────────────────────────────────────────────────────

class DiffSegModel:
    """
    DiffSeg segmentation model backed by Stable Diffusion 1.5.

    Usage:
        model = DiffSegModel.load(device="cuda")
        bundle = model.encode(image, timestep=50)       # slow (~5s on GPU)
        result = model.segment(bundle, kl_threshold=0.5) # fast (~0.2s)
    """

    SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"

    def __init__(self, pipe, device: str):
        self.pipe = pipe
        self.device = device

    @classmethod
    def load(cls, device: str = "cuda") -> "DiffSegModel":
        from diffusers import StableDiffusionPipeline
        print(f"[DiffSeg] Loading Stable Diffusion 1.5 on {device}...")
        dtype = torch.float16 if device != "cpu" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            cls.SD_MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
        patch_attention_for_hooks(pipe.unet)  # no-op now, kept for compat
        print("[DiffSeg] Model loaded.")
        return cls(pipe, device)

    # ── encode: expensive, only re-run when image or timestep changes ─────────

    @torch.no_grad()
    def encode(
        self,
        image: Image.Image,
        timestep: int = 50,
        image_size: int = 512,
    ) -> AttentionBundle:
        """
        Run one SD forward pass and collect all self-attention maps.

        Args:
            image:      Input PIL image.
            timestep:   Denoising timestep in [1, 1000].
                        Lower (e.g. 10-50):  early denoising, coarser features.
                        Mid   (e.g. 50-200): balanced detail and consistency ← sweet spot.
                        High  (e.g. 500+):   noisy, less stable attention maps.
            image_size: Resolution fed to SD (512 recommended).
        """
        img = image.convert("RGB").resize((image_size, image_size), Image.LANCZOS)
        img_tensor = self._pil_to_tensor(img)

        # VAE encode
        latents = self.pipe.vae.encode(img_tensor).latent_dist.sample()
        latents = latents * self.pipe.vae.config.scaling_factor

        # Unconditional text embeddings (empty prompt — no text guidance)
        uncond_input = self.pipe.tokenizer(
            [""],
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.pipe.text_encoder(
            uncond_input.input_ids.to(self.device)
        )[0]

        # Add noise at the requested timestep
        noise = torch.randn_like(latents)
        t = torch.tensor([timestep], device=self.device, dtype=torch.long)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, t)

        # One UNet forward pass — collect all self-attention maps
        collector = AttentionCollector(self.pipe.unet)
        with collector:
            _ = self.pipe.unet(
                noisy_latents,
                t,
                encoder_hidden_states=uncond_embeddings,
            ).sample

        print(f"[DiffSeg] Collected attention maps: { {k: len(v) for k,v in collector.maps.items()} }")

        return AttentionBundle(
            maps=collector.maps,
            original_image=image,
            latent_size=(image_size // 8, image_size // 8),
            timestep=timestep,
            device=self.device,
        )

    # ── segment: cheap, re-run on every slider change ────────────────────────

    def segment(
        self,
        bundle: AttentionBundle,
        kl_threshold: float = 0.5,
        M: int = 8,
        n_iter: int = 10,
        alpha: float = 0.55,
        min_area_ratio: float = 0.01,
        resolution_mode: str = "proportional (paper default)",
        # Individual resolution weight sliders (0.0 = off, 1.0 = full weight)
        w64: float = 1.0,
        w32: float = 0.5,
        w16: float = 0.25,
        w8:  float = 0.125,
        use_custom_weights: bool = False,
    ) -> dict:
        """
        Produce a segmentation from a cached AttentionBundle.

        Args:
            bundle:             Cached result from encode().
            kl_threshold:       KL divergence merge threshold.
            resolution_mode:    Preset aggregation mode (used when use_custom_weights=False).
            w64/w32/w16/w8:     Per-resolution weights (used when use_custom_weights=True).
            use_custom_weights: If True, use w64/w32/w16/w8 instead of the preset.
        """
        target_res = bundle.latent_size[0]

        custom_weights = None
        if use_custom_weights:
            custom_weights = {"64": w64, "32": w32, "16": w16, "8": w8}

        Af = bundle.get_Af(
            target_res=target_res,
            resolution_mode=resolution_mode,
            custom_weights=custom_weights,
        )

        proposals = iterative_merge(Af, kl_threshold=kl_threshold, M=M, n_iter=n_iter)

        label_map_small = nms_proposals(
            proposals,
            target_hw=(target_res, target_res),
            min_area_ratio=min_area_ratio,
        )

        orig_w, orig_h = bundle.original_image.size
        label_map_full = upsample_label_map(label_map_small, orig_h, orig_w)
        label_map_np = label_map_full.cpu().numpy()

        overlay, _, palette = label_map_to_overlay(label_map_np, bundle.original_image, alpha=alpha)
        n_segments = len(set(label_map_np.flatten().tolist()) - {0})

        return {
            "overlay": overlay,
            "label_map": label_map_np,
            "n_segments": n_segments,
            "palette": palette,
        }

    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        arr = np.array(image).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(device=self.device, dtype=next(self.pipe.vae.parameters()).dtype)