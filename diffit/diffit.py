#!/usr/bin/env python3

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
DiffiT: Diffusion Vision Transformers for Image Generation.
Code by Ali Hatamizadeh.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from timm.models.vision_transformer import PatchEmbed


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-mean-square layer normalization over the last dim."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        x = (x32 * rms).to(in_dtype)
        if self.weight is not None:
            x = x * self.weight
        return x


# ---------------------------------------------------------------------------
# 2D axial Rotary Position Embedding
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def build_axial_rope_cache(grid_size: int, head_dim: int, base: float = 10000.0):
    """
    Precompute cos/sin tables for axial 2D RoPE on a ``grid_size × grid_size`` grid.

    The first half of ``head_dim`` encodes the row (y) position, the second half
    encodes the column (x). Each returned tensor has shape
    ``(1, 1, grid_size**2, head_dim/2)``.
    """
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for axial 2D RoPE"
    quarter = head_dim // 4
    inv_freq = 1.0 / (
        base ** (torch.arange(0, quarter, dtype=torch.float32) / quarter)
    )

    pos = torch.arange(grid_size, dtype=torch.float32)
    freqs_1d = torch.einsum("p,f->pf", pos, inv_freq)  # (G, quarter)

    freqs_y = (
        freqs_1d[:, None, :].expand(grid_size, grid_size, quarter).reshape(-1, quarter)
    )
    freqs_x = (
        freqs_1d[None, :, :].expand(grid_size, grid_size, quarter).reshape(-1, quarter)
    )

    def _to_cs(freqs: torch.Tensor):
        emb = torch.cat([freqs, freqs], dim=-1)  # (N, head_dim/2) — rotate_half layout
        return emb.cos()[None, None], emb.sin()[None, None]

    cos_y, sin_y = _to_cs(freqs_y)
    cos_x, sin_x = _to_cs(freqs_x)
    return cos_y, sin_y, cos_x, sin_x


def apply_axial_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_y: torch.Tensor,
    sin_y: torch.Tensor,
    cos_x: torch.Tensor,
    sin_x: torch.Tensor,
):
    """Apply axial 2D RoPE to ``q`` and ``k`` of shape ``(B, heads, N, head_dim)``."""
    qy, qx = q.chunk(2, dim=-1)
    ky, kx = k.chunk(2, dim=-1)

    cos_y = cos_y.to(q.dtype)
    sin_y = sin_y.to(q.dtype)
    cos_x = cos_x.to(q.dtype)
    sin_x = sin_x.to(q.dtype)

    qy = qy * cos_y + _rotate_half(qy) * sin_y
    ky = ky * cos_y + _rotate_half(ky) * sin_y
    qx = qx * cos_x + _rotate_half(qx) * sin_x
    kx = kx * cos_x + _rotate_half(kx) * sin_x

    return torch.cat([qy, qx], dim=-1), torch.cat([ky, kx], dim=-1)


# ---------------------------------------------------------------------------
# Timestep & label embedders
# ---------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """Embeds scalar diffusion timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    """Embeds class labels; supports label-dropout for classifier-free guidance."""

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float):
        super().__init__()
        use_cfg_embedding = int(dropout_prob > 0)
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(
        self,
        labels: torch.Tensor,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(
        self,
        labels: torch.Tensor,
        train: bool,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU feed-forward: ``down( silu(gate(x)) * up(x) )``."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
    ):
        super().__init__()
        out_features = out_features or in_features
        self.gate = nn.Linear(in_features, hidden_features, bias=True)
        self.up = nn.Linear(in_features, hidden_features, bias=True)
        self.down = nn.Linear(hidden_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def _swiglu_hidden(hidden: int, mlp_ratio: float, multiple_of: int = 64) -> int:
    """Matched-param inner width: ``hidden * mlp_ratio * 2/3`` rounded up."""
    raw = int(hidden * mlp_ratio * 2 / 3)
    return ((raw + multiple_of - 1) // multiple_of) * multiple_of


# ---------------------------------------------------------------------------
# DiffiT attention (TMSA + RoPE-2D + QK-norm, FlashAttention-compatible)
# ---------------------------------------------------------------------------

class DiffiTAttention(nn.Module):
    """
    Time-dependent Multihead Self-Attention (TMSA) with axial 2D RoPE and
    QK-norm. Implements DiffiT's additive time-conditioning on the QKV
    projections (Eqs. 3-5 of the DiffiT paper):

        qs = xs · Wqs + xt · Wqt
        ks = xs · Wks + xt · Wkt
        vs = xs · Wvs + xt · Wvt
    """

    def __init__(
        self,
        dim: int,
        temb_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim % 4 == 0, (
            "head_dim must be divisible by 4 for axial 2D RoPE"
        )

        # Spatial and temporal QKV projections (TMSA)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_temb = nn.Linear(temb_dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.q_norm = RMSNorm(self.head_dim, elementwise_affine=True)
        self.k_norm = RMSNorm(self.head_dim, elementwise_affine=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        cos_y: torch.Tensor,
        sin_y: torch.Tensor,
        cos_x: torch.Tensor,
        sin_x: torch.Tensor,
    ) -> torch.Tensor:
        B, N, C = x.shape

        # TMSA: add projected time token to spatial QKV before splitting heads.
        qkv_temb = self.qkv_temb(temb).unsqueeze(1).to(x.dtype)
        qkv = self.qkv(x) + qkv_temb
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B, heads, N, head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_axial_rope(q, k, cos_y, sin_y, cos_x, sin_x)

        # No attn_mask → SDPA can dispatch to FlashAttention.
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


# ---------------------------------------------------------------------------
# DiffiT transformer block (TMSA + SwiGLU)
# ---------------------------------------------------------------------------

class DiffiTBlock(nn.Module):
    """
    DiffiT transformer block (Eqs. 7-8 of the paper):

        x̂ = TMSA(LN(x), xt) + x
        x  = MLP(LN(x̂)) + x̂
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, elementwise_affine=False)
        self.attn = DiffiTAttention(
            dim=hidden_size,
            temb_dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
        )
        self.norm2 = RMSNorm(hidden_size, elementwise_affine=False)
        self.mlp = SwiGLU(hidden_size, _swiglu_hidden(hidden_size, mlp_ratio))

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        cos_y: torch.Tensor,
        sin_y: torch.Tensor,
        cos_x: torch.Tensor,
        sin_x: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), temb, cos_y, sin_y, cos_x, sin_x)
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Final projection layer
# ---------------------------------------------------------------------------

class FinalLayer(nn.Module):
    """Final projection: RMSNorm → SiLU → Linear → patch-level predictions."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, elementwise_affine=False)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.silu(self.norm_final(x)))


# ---------------------------------------------------------------------------
# DiffiT model
# ---------------------------------------------------------------------------

# Latent-size threshold that separates the two CFG strategies.
# image_size 256  → latent_size 32   (power-cosine CFG schedule)
# image_size 512  → latent_size 64   (constant CFG scale)
# image_size 1024 → latent_size 128  (constant CFG scale)
_LATENT_SIZE_THRESHOLD = 32


class DiffiT(nn.Module):
    """
    DiffiT: Diffusion Vision Transformers for Image Generation.

    A class-conditional latent diffusion model built from DiffiTBlocks with
    Time-dependent Multihead Self-Attention (TMSA) conditioning, axial 2D
    rotary position embeddings, QK-normalized attention, and SwiGLU FFNs.
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        mask_ratio=None,
        decode_layer=None,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        # Embedders -----------------------------------------------------------
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        # RoPE-2D cache (shared across all blocks, non-persistent buffer) ----
        grid_size = input_size // patch_size
        head_dim = hidden_size // num_heads
        cos_y, sin_y, cos_x, sin_x = build_axial_rope_cache(grid_size, head_dim)
        self.register_buffer("rope_cos_y", cos_y, persistent=False)
        self.register_buffer("rope_sin_y", sin_y, persistent=False)
        self.register_buffer("rope_cos_x", cos_x, persistent=False)
        self.register_buffer("rope_sin_x", sin_x, persistent=False)

        # Transformer backbone ------------------------------------------------
        self.blocks = nn.ModuleList([
            DiffiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.gradient_checkpointing = False

        self._initialize_weights()

    # ----- weight init -------------------------------------------------------

    def _initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Patch embedding projection
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Label embedding
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Timestep MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-init final projection so the model starts near identity.
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # ----- helpers -----------------------------------------------------------

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Rearrange patch tokens ``(N, T, p*p*C)`` back into ``(N, C, H, W)``."""
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, h * p)

    # ----- forward -----------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        enable_mask: bool = False,
    ) -> torch.Tensor:
        x = self.x_embedder(x)
        # TMSA time token: combined timestep + class embedding, shared across
        # all blocks (DiffiT paper, Section 3.2).
        temb = self.t_embedder(t) + self.y_embedder(y, self.training)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = grad_checkpoint(
                    block, x, temb,
                    self.rope_cos_y, self.rope_sin_y,
                    self.rope_cos_x, self.rope_sin_x,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x, temb,
                    self.rope_cos_y, self.rope_sin_y,
                    self.rope_cos_x, self.rope_sin_x,
                )

        x = self.final_layer(x)
        return self.unpatchify(x)

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: Optional[float] = None,
        diffusion_steps: int = 1000,
        scale_pow: float = 4.0,
    ) -> torch.Tensor:
        """
        Forward pass with (optional) classifier-free guidance.

        CFG strategy varies by resolution:
            - **image_size 256** (``input_size <= 32``): power-cosine schedule
              that ramps the effective scale over the diffusion process.
            - **image_size 512 / 1024** (``input_size > 32``): constant CFG
              scale across all denoising steps.
        """
        split = self.in_channels

        if cfg_scale is None:
            model_out = self.forward(x, t, y)
            eps, rest = model_out[:, :split], model_out[:, split:]
            return torch.cat([eps, rest], dim=1)

        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)

        eps, rest = model_out[:, :split], model_out[:, split:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

        if self.input_size <= _LATENT_SIZE_THRESHOLD:
            phase = ((1 - t / diffusion_steps) ** scale_pow) * math.pi
            scale_step = 0.5 * (1 - torch.cos(phase))
            real_cfg_scale = (cfg_scale - 1) * scale_step + 1
            real_cfg_scale = real_cfg_scale[: len(x) // 2].view(-1, 1, 1, 1)
        else:
            real_cfg_scale = cfg_scale

        half_eps = uncond_eps + real_cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


# ---------------------------------------------------------------------------
# Model constructor
# ---------------------------------------------------------------------------

def Diffit(**kwargs):
    """DiffiT-XL/2 configuration (depth 28, dim 1152, patch 2, 16 heads)."""
    return DiffiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
