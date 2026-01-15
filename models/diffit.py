# Copyright (c) 2024, DiffiT authors.
# DiffiT model architecture.

from __future__ import annotations

import torch
import torch.nn as nn

from models.attention import get_timestep_embedding, TimestepEmbedding, TransformerBlock
from models.vit import VisionTransformer


class ResBlock(nn.Module):
    """Residual block with Vision Transformer."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        time_emb_dim: int,
        image_size: int,
        gn_groups: int = 8,
        hidden_dim: int = 64,
        num_patches: int = 2,
        num_heads: int = 4,
        num_blocks: int = 1,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(gn_groups, dim_in)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(dim_in, dim_in, 3, padding=1)
        self.transformer = VisionTransformer(
            dim_in, dim_in, time_emb_dim, image_size,
            hidden_dim=hidden_dim,
            num_patches=num_patches,
            num_heads=num_heads,
            num_blocks=num_blocks,
            mlp_ratio=mlp_ratio,
        )
        self.out_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.act(h)
        h = self.conv(h)
        h = self.transformer(h, t)
        h = h + x
        h = self.act(h)
        return self.out_conv(h)


class Downsample(nn.Module):
    """Downsample using strided convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsample using transposed convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DiffiT(nn.Module):
    """DiffiT: Diffusion Vision Transformers for Image Generation.

    A U-Net architecture with Vision Transformer blocks.
    """

    def __init__(
        self,
        image_shape: list[int],
        base_dim: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 2, 2),
        num_res_blocks: int = 1,
        time_embed_mult: int = 4,
        gn_groups: int = 8,
        hidden_dim: int = 64,
        num_patches: int = 2,
        num_heads: int = 4,
        num_transformer_blocks: int = 1,
        mlp_ratio: int = 4,
        # Unused parameters kept for backward compatibility
        resolutions_list: list[int] | None = None,
        num_resolutions: int = 4,
        num_res_blocks_tuple: tuple[int, ...] | None = None,
        residual_connection_level: tuple[bool, ...] | None = None,
        time_mlp_ratio: int = 2,
        vit_block_class: type[nn.Module] | None = None,
        downsampling_class: type[nn.Module] | None = None,
        upsampling_class: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()

        in_channels = image_shape[0]
        image_size = image_shape[1]

        # Time embedding
        time_embed_dim = base_dim * time_embed_mult
        self.time_proj = nn.Sequential(
            nn.Linear(base_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.time_embed_dim_in = base_dim

        # Compute channel dimensions per level
        dims = [base_dim * m for m in channel_mults]
        num_levels = len(dims)

        # Input convolution
        self.conv_in = nn.Conv2d(in_channels, dims[0], 3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        curr_res = image_size

        for i, dim in enumerate(dims[:-1]):
            next_dim = dims[i + 1]

            # ResBlocks at this level
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    ResBlock(dim, dim, time_embed_dim, curr_res, gn_groups, hidden_dim, num_patches, num_heads, num_transformer_blocks, mlp_ratio)
                )

            # Downsample
            self.down_samples.append(nn.Conv2d(dim, next_dim, 3, stride=2, padding=1))
            curr_res //= 2

        # Bottleneck
        bottleneck_dim = dims[-1]
        self.mid_block1 = ResBlock(bottleneck_dim, bottleneck_dim, time_embed_dim, curr_res, gn_groups, hidden_dim, num_patches, num_heads, num_transformer_blocks, mlp_ratio)
        self.mid_block2 = ResBlock(bottleneck_dim, bottleneck_dim, time_embed_dim, curr_res, gn_groups, hidden_dim, num_patches, num_heads, num_transformer_blocks, mlp_ratio)

        # Upsampling path
        self.up_samples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        for i in range(num_levels - 2, -1, -1):
            dim = dims[i]
            prev_dim = dims[i + 1] if i < num_levels - 1 else bottleneck_dim
            curr_res *= 2

            # Upsample
            self.up_samples.append(nn.ConvTranspose2d(prev_dim, dim, 4, stride=2, padding=1))

            # Skip connection conv (to handle channel mismatch)
            self.skip_convs.append(nn.Conv2d(dim, dim, 1))

            # ResBlocks at this level
            for _ in range(num_res_blocks):
                self.up_blocks.append(
                    ResBlock(dim * 2, dim, time_embed_dim, curr_res, gn_groups, hidden_dim, num_patches, num_heads, num_transformer_blocks, mlp_ratio)
                )

        # Output
        self.norm_out = nn.GroupNorm(gn_groups, dims[0])
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(dims[0], in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t_emb = get_timestep_embedding(timesteps, self.time_embed_dim_in)
        t_emb = self.time_proj(t_emb)

        # Input
        h = self.conv_in(x)
        skips = [h]

        # Downsampling
        block_idx = 0
        for i, downsample in enumerate(self.down_samples):
            for _ in range(len(self.down_blocks) // len(self.down_samples)):
                h = self.down_blocks[block_idx](h, t_emb)
                block_idx += 1
            skips.append(h)
            h = downsample(h)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        # Upsampling
        block_idx = 0
        for i, (upsample, skip_conv) in enumerate(zip(self.up_samples, self.skip_convs)):
            h = upsample(h)
            skip = skips.pop()
            skip = skip_conv(skip)
            h = torch.cat([h, skip], dim=1)

            for _ in range(len(self.up_blocks) // len(self.up_samples)):
                h = self.up_blocks[block_idx](h, t_emb)
                block_idx += 1

        # Output
        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)

        return h
