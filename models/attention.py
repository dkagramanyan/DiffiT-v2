# Copyright (c) 2024, DiffiT authors.
# Attention mechanisms for DiffiT.

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time conditioning."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Head(nn.Module):
    """Output head for the U-Net."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        conv_class: type[nn.Module] = nn.Conv2d,
        gn_groups: int = 8,
    ) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(gn_groups, dim_in)
        self.swish = nn.SiLU()
        self.conv = conv_class(dim_in, dim_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        x = self.gn(x)
        x = self.swish(x)
        x = self.conv(x)
        return x


class Tokenizer(nn.Module):
    """Initial tokenization layer for images."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        conv_class: type[nn.Module] = nn.Conv2d,
    ) -> None:
        super().__init__()
        self.conv = conv_class(dim_in, dim_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        return self.conv(x)


class TDMHSA(nn.Module):
    """Time-Dependent Multi-Head Self-Attention.

    Implements the attention mechanism from DiffiT paper with relative positional embeddings.
    """

    def __init__(
        self,
        feature_dim: int,
        time_dim: int,
        n_heads: int,
        max_len: int = 1024,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.time_dim = time_dim
        self.num_heads = n_heads
        self.max_len = max_len

        assert feature_dim % n_heads == 0, f"Feature dim {feature_dim} not divisible by {n_heads} heads"
        assert time_dim % n_heads == 0, f"Time dim {time_dim} not divisible by {n_heads} heads"

        self.spatial_head_dim = feature_dim // n_heads
        self.time_head_dim = time_dim

        # Spatial projections
        self.qs_projections = nn.Linear(feature_dim, feature_dim)
        self.ks_projections = nn.Linear(feature_dim, feature_dim)
        self.vs_projections = nn.Linear(feature_dim, feature_dim)

        # Time projections
        self.qt_projections = nn.Linear(time_dim, feature_dim)
        self.kt_projections = nn.Linear(time_dim, feature_dim)
        self.vt_projections = nn.Linear(time_dim, feature_dim)

        # Relative positional embeddings
        self.Er = nn.Parameter(torch.randn(max_len, self.spatial_head_dim))

        self.scale = self.spatial_head_dim ** -0.5

    def forward(self, patches: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, embed_dim = patches.size()

        # Compute spatial Q, K, V
        qs = self.qs_projections(patches)
        ks = self.ks_projections(patches)
        vs = self.vs_projections(patches)

        # Compute time-based Q, K, V
        qt = self.qt_projections(time_emb)
        kt = self.kt_projections(time_emb)
        vt = self.vt_projections(time_emb)

        # Reshape for multi-head attention
        qs = qs.view(batch_size, seq_length, self.num_heads, self.spatial_head_dim).permute(0, 2, 1, 3)
        ks = ks.view(batch_size, seq_length, self.num_heads, self.spatial_head_dim).permute(0, 2, 1, 3)
        vs = vs.view(batch_size, seq_length, self.num_heads, self.spatial_head_dim).permute(0, 2, 1, 3)

        qt = qt.view(batch_size, 1, self.num_heads, self.spatial_head_dim).permute(0, 2, 1, 3)
        kt = kt.view(batch_size, 1, self.num_heads, self.spatial_head_dim).permute(0, 2, 1, 3)
        vt = vt.view(batch_size, 1, self.num_heads, self.spatial_head_dim).permute(0, 2, 1, 3)

        # Combine spatial and temporal
        q = qs + qt
        k = ks + kt
        v = vs + vt

        # Compute relative positional embeddings
        start = self.max_len - seq_length
        Er_t = self.Er[start:, :].transpose(0, 1)
        QEr = torch.matmul(qs, Er_t)
        Srel = self._skew(QEr)

        # Compute attention scores
        scores = (torch.matmul(q, k.transpose(-2, -1)) + Srel) * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, v)

        # Reshape output
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.num_heads * self.spatial_head_dim)

        return attention_output

    def _skew(self, QEr: torch.Tensor) -> torch.Tensor:
        """Apply skewing operation for relative positional embeddings."""
        padded = F.pad(QEr, (1, 0))
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        Srel = reshaped[:, :, 1:, :]
        return Srel


class VisionTransformerBlock(nn.Module):
    """Vision Transformer block with time-dependent attention."""

    def __init__(
        self,
        time_emb_dim: int,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: int,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.tdmhsa = TDMHSA(hidden_dim, time_emb_dim, num_heads)

        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mlp_ratio * hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_dim, hidden_dim),
        )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        out = x + self.tdmhsa(self.norm(x), t)
        out = out + self.mlp(out)
        return self.relu(out)
