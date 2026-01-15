# Copyright (c) 2024, DiffiT authors.
# DiffiT model architecture.

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models.attention import Head, SinusoidalPositionEmbeddings, Tokenizer, VisionTransformerBlock
from models.vit import VisionTransformer


class ResBlock(nn.Module):
    """Residual block with Vision Transformer."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        time_emb_dim: int,
        image_shape: int,
        gn_groups: int = 8,
        hidden_dim: int = 64,
        num_patches: int = 2,
        num_heads: int = 4,
        num_blocks: int = 1,
        mlp_ratio: int = 4,
        conv_class: type[nn.Module] = nn.Conv2d,
        vit_block_class: type[nn.Module] = VisionTransformerBlock,
    ) -> None:
        super().__init__()
        self.swish = nn.SiLU()
        self.swish2 = nn.SiLU()
        self.gn = nn.GroupNorm(gn_groups, dim_in)
        self.conv = conv_class(dim_in, dim_in, kernel_size=3, stride=1, padding=1)
        self.transformer = VisionTransformer(
            dim_in, dim_in, time_emb_dim, image_shape,
            hidden_dim=hidden_dim,
            num_patches=num_patches,
            num_heads=num_heads,
            num_blocks=num_blocks,
            mlp_ratio=mlp_ratio,
            vit_block_class=vit_block_class,
        )
        self.resize = (
            conv_class(dim_in, dim_out, kernel_size=1, stride=1, padding=0)
            if dim_in != dim_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.gn(x)
        x = self.swish(x)
        x = self.conv(x)
        x = self.transformer(x, t)
        x = x + residual
        x = self.swish2(x)
        return self.resize(x)


class Downsampling(nn.Module):
    """Downsampling layer using strided convolution."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        return self.conv(x)


class Upsampling(nn.Module):
    """Upsampling layer using transposed convolution."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        return self.conv(x)


class Resizing(nn.Module):
    """1x1 convolution for channel resizing."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        return self.conv(x)


class DiffiT(nn.Module):
    """DiffiT: Diffusion Vision Transformers for Image Generation.

    A U-Net style architecture with Vision Transformer blocks for denoising
    in diffusion models.
    """

    def __init__(
        self,
        image_shape: list[int],
        base_dim: int = 128,
        resolutions_list: list[int] | None = None,
        num_resolutions: int = 4,
        num_res_blocks: tuple[int, ...] = (1, 1, 1, 1),
        residual_connection_level: tuple[bool, ...] | None = None,
        time_mlp_ratio: int = 2,
        gn_groups: int = 8,
        hidden_dim: int = 64,
        num_patches: int = 2,
        num_heads: int = 4,
        num_transformer_blocks: int = 1,
        mlp_ratio: int = 4,
        vit_block_class: type[nn.Module] = VisionTransformerBlock,
        downsampling_class: type[nn.Module] = Downsampling,
        upsampling_class: type[nn.Module] = Upsampling,
    ) -> None:
        super().__init__()
        self.downsampling_class = downsampling_class
        self.upsampling_class = upsampling_class

        image_size = image_shape[1]
        if resolutions_list is None:
            resolutions_list = [image_size // (2 ** i) for i in range(num_resolutions)]

        dim_in_out = self._compute_per_layer_channels(base_dim, image_shape[0])

        # Time embedding MLP
        time_emb_dim = image_size * time_mlp_ratio
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(image_size),
            nn.Linear(image_size, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Downward path
        self.downward_layers = nn.ModuleList([])
        for i in range(num_resolutions - 1):
            dim_in = dim_in_out[0][i][0]
            dim_mid = dim_in_out[0][i][1]
            dim_out = dim_in_out[0][i][2]

            if i == 0:
                self.downward_layers.append(Tokenizer(dim_in, dim_mid, downsampling_class))
            else:
                self.downward_layers.append(downsampling_class(dim_in, dim_mid))

            for j in range(num_res_blocks[i]):
                out_dim = dim_out if j == num_res_blocks[i] - 1 else dim_mid
                self.downward_layers.append(
                    ResBlock(
                        dim_mid, out_dim, time_emb_dim, resolutions_list[i],
                        gn_groups=gn_groups,
                        hidden_dim=hidden_dim,
                        num_patches=num_patches,
                        num_heads=num_heads,
                        num_blocks=num_transformer_blocks,
                        mlp_ratio=mlp_ratio,
                        conv_class=downsampling_class,
                        vit_block_class=vit_block_class,
                    )
                )

        # Bottleneck
        self.bottleneck_layers = nn.ModuleList([])
        self.bottleneck_layers.append(downsampling_class(dim_in_out[1][0], dim_in_out[1][1]))
        for j in range(num_res_blocks[-1]):
            out_dim = dim_in_out[1][2] if j == num_res_blocks[-1] - 1 else dim_in_out[1][1]
            self.bottleneck_layers.append(
                ResBlock(
                    dim_in_out[1][1], out_dim, time_emb_dim, resolutions_list[-1],
                    gn_groups=gn_groups,
                    hidden_dim=hidden_dim,
                    num_patches=num_patches,
                    num_heads=num_heads,
                    num_blocks=num_transformer_blocks,
                    mlp_ratio=mlp_ratio,
                    conv_class=downsampling_class,
                    vit_block_class=vit_block_class,
                )
            )
        self.bottleneck_layers.append(upsampling_class(dim_in_out[1][2], dim_in_out[1][3]))

        # Resizing layers for skip connections
        self.resizing_layers = nn.ModuleList([])
        for i in range(num_resolutions - 1):
            from_dim = dim_in_out[0][i][1]
            to_dim = dim_in_out[2][i][1]
            self.resizing_layers.append(Resizing(from_dim, to_dim))

        # Upward path
        self.upward_layers = nn.ModuleList([])
        for i in range(num_resolutions - 1):
            dim_in = dim_in_out[2][i][0]
            dim_mid = dim_in_out[2][i][1]
            dim_out = dim_in_out[2][i][2]

            for j in range(num_res_blocks[i]):
                out_dim = dim_mid if j == num_res_blocks[i] - 1 else dim_in
                self.upward_layers.append(
                    ResBlock(
                        dim_in, out_dim, time_emb_dim, np.flip(resolutions_list)[i + 1],
                        gn_groups=gn_groups,
                        hidden_dim=hidden_dim,
                        num_patches=num_patches,
                        num_heads=num_heads,
                        num_blocks=num_transformer_blocks,
                        mlp_ratio=mlp_ratio,
                        conv_class=downsampling_class,
                        vit_block_class=vit_block_class,
                    )
                )

            if i == num_resolutions - 2:
                self.upward_layers.append(Head(dim_mid, dim_out, downsampling_class, gn_groups))
            else:
                self.upward_layers.append(upsampling_class(dim_mid, dim_out))

        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        t = self.time_mlp(time)
        h = []

        # Downward path
        for layer in self.downward_layers:
            x = layer(x, t)
            if isinstance(layer, self.downsampling_class):
                h.append(x)

        # Bottleneck
        for layer in self.bottleneck_layers:
            if isinstance(layer, self.upsampling_class):
                x = x + h.pop()
            x = layer(x, t)
            if isinstance(layer, self.downsampling_class):
                h.append(x)

        # Upward path
        counter = 0
        for layer in self.upward_layers:
            if isinstance(layer, self.upsampling_class):
                skip = h.pop()
                if x.size(1) != skip.size(1):
                    skip = self.resizing_layers[counter](skip)
                x = x + skip
                counter += 1
                x = self.act(x)
            x = layer(x, t)

        return x

    def _compute_per_layer_channels(self, base_dim: int, image_channels: int):
        """Compute channel dimensions for each layer."""
        downward_dims_in = [image_channels, *[base_dim * m for m in (1, 2)]]
        downward_dims_downsample = [base_dim * m for m in (1, 1, 1)]
        downward_dims_out = [base_dim * m for m in (1, 2, 2)]
        downward_dim_in_out = list(zip(downward_dims_in, downward_dims_downsample, downward_dims_out))

        bottleneck_dim_in_out = [base_dim * m for m in (2, 2, 2, 2)]

        upward_dims_in = [base_dim * m for m in (2, 2, 2)]
        upward_dims_upsample = [base_dim * m for m in (2, 2, 1)]
        upward_dims_out = [base_dim * m for m in (2, 2)] + [image_channels]
        upward_dim_in_out = list(zip(upward_dims_in, upward_dims_upsample, upward_dims_out))

        return (downward_dim_in_out, bottleneck_dim_in_out, upward_dim_in_out)
