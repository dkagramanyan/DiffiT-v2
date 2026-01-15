# Copyright (c) 2024, DiffiT authors.
# Vision Transformer components for DiffiT.

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import TransformerBlock


class PatchEmbed(nn.Module):
    """Convert image patches to embeddings."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x: torch.Tensor, num_patches: int) -> torch.Tensor:
        B, C, H, W = x.shape
        P = self.patch_size

        # Patchify: (B, C, H, W) -> (B, num_patches^2, C*P*P)
        x = x.reshape(B, C, num_patches, P, num_patches, P)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, num_patches * num_patches, C * P * P)
        return self.proj(x)


class PatchUnembed(nn.Module):
    """Convert embeddings back to image patches."""

    def __init__(self, embed_dim: int, out_channels: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.proj = nn.Linear(embed_dim, out_channels * patch_size * patch_size)

    def forward(self, x: torch.Tensor, num_patches: int, H: int, W: int) -> torch.Tensor:
        B, N, _ = x.shape
        P = self.patch_size
        C = self.out_channels

        x = self.proj(x)
        x = x.reshape(B, num_patches, num_patches, C, P, P)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for processing image patches with time conditioning."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        time_emb_dim: int,
        image_dimension: int,
        hidden_dim: int = 64,
        num_patches: int = 2,
        num_heads: int = 4,
        num_blocks: int = 1,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches

        assert image_dimension % num_patches == 0, f"Image dim {image_dimension} not divisible by {num_patches}"
        patch_size = image_dimension // num_patches

        # Patch embedding
        self.patch_embed = PatchEmbed(dim_in, hidden_dim, patch_size)

        # Positional embedding (learnable)
        num_tokens = num_patches * num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(time_emb_dim, hidden_dim, num_heads, mlp_ratio)
            for _ in range(num_blocks)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        # Patch unembedding
        self.patch_unembed = PatchUnembed(hidden_dim, dim_in, patch_size)

        # Channel alignment if needed
        self.align = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Embed patches
        tokens = self.patch_embed(x, self.num_patches)
        tokens = tokens + self.pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            tokens = block(tokens, time_emb)

        tokens = self.norm(tokens)

        # Reconstruct image
        out = self.patch_unembed(tokens, self.num_patches, H, W)
        return self.align(out)
