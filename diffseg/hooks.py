"""
hooks.py — PyTorch attention extraction from Stable Diffusion UNet.

Uses diffusers' AttnProcessor API (correct pattern for diffusers >= 0.18).
The previous monkey-patch approach caused a double-computation bug:
    mat1 (32768x64) cannot be multiplied with mat2 (512x512)
because Q/K were computed twice with mismatched shapes.

This version installs a CapturingAttnProcessor that computes attention
once, captures the weight matrix, and returns the correct output.

The UNet produces self-attention at 4 spatial resolutions: 64², 32², 16², 8².
We average over heads and collect all maps per resolution.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import torch


# ── Custom attention processor that captures weights ─────────────────────────

class CapturingAttnProcessor:
    """
    Drop-in replacement for diffusers AttnProcessor that additionally
    stores the self-attention weight matrix on the module after each forward.

    Handles both self-attention (encoder_hidden_states=None) and
    cross-attention — only self-attention maps are captured.
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        is_self_attn = encoder_hidden_states is None
        context = hidden_states if is_self_attn else encoder_hidden_states

        # Project Q, K, V
        q = attn.to_q(hidden_states)
        k = attn.to_k(context)
        v = attn.to_v(context)

        # Reshape to (B*heads, seq, head_dim)
        q = attn.head_to_batch_dim(q)
        k = attn.head_to_batch_dim(k)
        v = attn.head_to_batch_dim(v)

        # Attention weights: (B*heads, N_q, N_k)
        scale = q.shape[-1] ** -0.5
        sim = torch.bmm(q * scale, k.transpose(-1, -2))

        if attention_mask is not None:
            sim = sim + attention_mask

        weights = sim.softmax(dim=-1)

        # Capture self-attention weights only
        if is_self_attn:
            attn._captured_weights = weights.detach()
        else:
            attn._captured_weights = None

        # Compute output
        out = torch.bmm(weights, v)
        out = attn.batch_to_head_dim(out)

        # Linear projection + dropout
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        return out


# ── Collector ─────────────────────────────────────────────────────────────────

class AttentionCollector:
    """
    Installs CapturingAttnProcessor on every self-attention layer in the UNet,
    runs as a context manager, then reads back the captured weight tensors.

    Usage:
        collector = AttentionCollector(unet)
        with collector:
            unet(latent, t, encoder_hidden_states=uncond_embeds)
        maps = collector.maps   # Dict[str, List[Tensor(N,N)]]
    """

    def __init__(self, unet: torch.nn.Module):
        self.unet = unet
        self._original_processors: Dict = {}
        self._hooks: List = []
        self.maps: Dict[str, List[torch.Tensor]] = {}

    def __enter__(self):
        self._install()
        return self

    def __exit__(self, *_):
        self._uninstall()

    # ── install capturing processors ─────────────────────────────────────────

    def _install(self):
        self.maps = {}
        self._original_processors = {}
        capturing = CapturingAttnProcessor()

        for name, module in self.unet.named_modules():
            if _is_self_attn_module(module):
                self._original_processors[name] = getattr(module, "processor", None)
                module.processor = capturing
                hook = module.register_forward_hook(self._make_hook())
                self._hooks.append(hook)

    # ── forward hook reads captured weights after each layer ─────────────────

    def _make_hook(self):
        def hook(module, inputs, output):
            weights = getattr(module, "_captured_weights", None)
            if weights is None:
                return

            # weights: (B*heads, N, N)
            B_heads, N_q, N_k = weights.shape
            if N_q != N_k:
                return  # skip non-square

            heads = module.heads
            B = max(1, B_heads // heads)

            try:
                # Average over heads → (B, N, N), keep first batch item
                attn = weights.reshape(B, heads, N_q, N_q).mean(dim=1)
            except RuntimeError:
                return

            # Only keep valid square spatial resolutions
            h = int(N_q ** 0.5)
            if h * h != N_q or h not in (8, 16, 32, 64):
                return

            key = str(h)
            if key not in self.maps:
                self.maps[key] = []
            self.maps[key].append(attn[0].float().cpu())  # (N,N) on CPU

            # Free GPU memory immediately
            module._captured_weights = None

        return hook

    # ── restore original processors ───────────────────────────────────────────

    def _uninstall(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

        for name, module in self.unet.named_modules():
            if name in self._original_processors:
                orig = self._original_processors[name]
                if orig is not None:
                    module.processor = orig
        self._original_processors = {}


# ── helpers ───────────────────────────────────────────────────────────────────

def _is_self_attn_module(module: torch.nn.Module) -> bool:
    """True for diffusers Attention modules that are self-attention (not cross)."""
    if type(module).__name__ != "Attention":
        return False
    return not getattr(module, "is_cross_attention", True)


def patch_attention_for_hooks(unet: torch.nn.Module):
    """No-op — kept for API compatibility. AttentionCollector handles everything."""
    pass