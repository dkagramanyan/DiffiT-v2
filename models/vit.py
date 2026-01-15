# Copyright (c) 2024, DiffiT authors.
# Vision Transformer components for DiffiT.

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models.attention import VisionTransformerBlock


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
        vit_block_class: type[nn.Module] = VisionTransformerBlock,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches

        c, h, w = (dim_in, image_dimension, image_dimension)
        assert h % num_patches == 0, f"Height {h} not divisible by {num_patches}"
        assert w % num_patches == 0, f"Width {w} not divisible by {num_patches}"

        self.patch_size = (h // num_patches, w // num_patches)
        self.input_d = c * self.patch_size[0] * self.patch_size[1]

        self.initial_linear_layer = nn.Linear(self.input_d, hidden_dim)

        pos_emb = self._get_positional_embeddings(num_patches ** 2, hidden_dim)
        self.register_buffer("pos_embed", pos_emb)

        self.vit_blocks = nn.ModuleList([
            vit_block_class(time_emb_dim, hidden_dim, num_heads, mlp_ratio)
            for _ in range(num_blocks)
        ])

        self.final_layer_norm = nn.LayerNorm(hidden_dim)
        self.final_linear_layer = nn.Linear(hidden_dim, self.input_d)

        self.align = nn.Conv2d(dim_in, dim_out, kernel_size=1) if dim_in != dim_out else nn.Identity()

    def forward(self, images: torch.Tensor, time_embeddings: torch.Tensor) -> torch.Tensor:
        patches = self._patchify(images, self.num_patches)
        tokens = self.initial_linear_layer(patches)

        pos_embed = self.pos_embed.repeat(images.size(0), 1, 1)
        out = tokens + pos_embed

        for block in self.vit_blocks:
            out = block(out, time_embeddings)

        out = self.final_layer_norm(out)
        out = self.final_linear_layer(out)

        original_images = self._depatchify(out, self.num_patches, images.shape)
        return self.align(original_images)

    def _patchify(self, images: torch.Tensor, num_patches: int) -> torch.Tensor:
        """Convert images to patches."""
        b, c, h, w = images.shape
        patch_size = h // num_patches

        patches = images.reshape(b, c, num_patches, patch_size, num_patches, patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5)
        patches = patches.reshape(b, num_patches ** 2, c * patch_size * patch_size)
        return patches

    def _depatchify(self, patches: torch.Tensor, num_patches: int, data_shape: tuple) -> torch.Tensor:
        """Convert patches back to images."""
        b, c, h, w = data_shape
        patch_size = h // num_patches

        images = patches.reshape(b, num_patches, num_patches, c, patch_size, patch_size)
        images = images.permute(0, 3, 1, 4, 2, 5)
        images = images.reshape(b, c, h, w)
        return images

    @staticmethod
    def _get_positional_embeddings(sequence_length: int, d: int) -> torch.Tensor:
        """Generate sinusoidal positional embeddings."""
        result = torch.zeros(sequence_length, d)
        position = torch.arange(sequence_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))

        result[:, 0::2] = torch.sin(position * div_term)
        result[:, 1::2] = torch.cos(position * div_term)
        return result.unsqueeze(0)
