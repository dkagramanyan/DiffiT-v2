# Copyright (c) 2024, DiffiT authors.
# Attention mechanisms for DiffiT.

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings (matching diffusers implementation).

    Args:
        timesteps: 1D tensor of timesteps.
        embedding_dim: Dimension of the embedding.

    Returns:
        Tensor of shape (batch_size, embedding_dim).
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimestepEmbedding(nn.Module):
    """Timestep embedding MLP."""

    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class TDMHSA(nn.Module):
    """Time-Dependent Multi-Head Self-Attention.

    Uses torch.nn.functional.scaled_dot_product_attention for efficiency.
    """

    def __init__(self, feature_dim: int, time_dim: int, n_heads: int, max_len: int = 1024) -> None:
        super().__init__()
        assert feature_dim % n_heads == 0, f"Feature dim {feature_dim} not divisible by {n_heads} heads"

        self.num_heads = n_heads
        self.head_dim = feature_dim // n_heads
        self.max_len = max_len

        # Combined QKV projections for spatial features
        self.qkv_proj = nn.Linear(feature_dim, feature_dim * 3)

        # Time projections
        self.time_proj = nn.Linear(time_dim, feature_dim * 3)

        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        # Relative positional embeddings
        self.Er = nn.Parameter(torch.randn(max_len, self.head_dim) * 0.02)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Compute spatial QKV
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each: (B, heads, N, head_dim)

        # Compute time-based QKV offset
        time_qkv = self.time_proj(time_emb).reshape(B, 1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qt, kt, vt = time_qkv.unbind(0)

        # Add time conditioning
        q = q + qt
        k = k + kt
        v = v + vt

        # Compute relative positional bias
        Er = self.Er[self.max_len - N:]
        rel_pos_bias = torch.einsum("bhnd,md->bhnm", q, Er)
        rel_pos_bias = self._skew(rel_pos_bias)

        # Scaled dot-product attention with relative position bias
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale + rel_pos_bias
        attn = F.softmax(attn, dim=-1)
        out = attn @ v

        # Reshape and project output
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)

    def _skew(self, x: torch.Tensor) -> torch.Tensor:
        """Skew operation for relative positional embeddings."""
        B, H, N, M = x.shape
        x = F.pad(x, (1, 0))
        x = x.view(B, H, M + 1, N)
        return x[:, :, 1:, :]


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer block with time-dependent attention."""

    def __init__(self, time_emb_dim: int, hidden_dim: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = TDMHSA(hidden_dim, time_emb_dim, num_heads)
        self.ffn = FeedForward(hidden_dim, mlp_ratio)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), t)
        x = x + self.ffn(x)
        return x
