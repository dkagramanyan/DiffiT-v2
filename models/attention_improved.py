# Copyright (c) 2024, DiffiT authors.
# Improved attention mechanisms for DiffiT matching the paper specifications.
#
# Reference: DiffiT paper Section 3.2, Equations 3-6

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


def window_partition(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, int, int]:
    """Partition tokens into non-overlapping windows.

    Args:
        x: Input tensor (B, H*W, C) where H=W=resolution
        window_size: Window size for local attention

    Returns:
        Windowed tensor (B*num_windows, window_size*window_size, C)
        Number of windows in H dimension
        Number of windows in W dimension
    """
    B, N, C = x.shape
    H = W = int(math.sqrt(N))
    assert H * W == N, f"Sequence length {N} is not a perfect square"
    assert H % window_size == 0, f"Resolution {H} not divisible by window size {window_size}"

    nH = H // window_size
    nW = W // window_size

    # Reshape to (B, nH, window_size, nW, window_size, C)
    x = x.view(B, nH, window_size, nW, window_size, C)
    # Permute to (B, nH, nW, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    # Reshape to (B*nH*nW, window_size*window_size, C)
    x = x.view(-1, window_size * window_size, C)

    return x, nH, nW


def window_unpartition(x: torch.Tensor, window_size: int, H: int, W: int, B: int) -> torch.Tensor:
    """Reverse of window_partition.

    Args:
        x: Windowed tensor (B*num_windows, window_size*window_size, C)
        window_size: Window size
        H, W: Original spatial dimensions
        B: Original batch size

    Returns:
        Unwindowed tensor (B, H*W, C)
    """
    nH = H // window_size
    nW = W // window_size
    C = x.shape[-1]

    # Reshape to (B, nH, nW, window_size, window_size, C)
    x = x.view(B, nH, nW, window_size, window_size, C)
    # Permute to (B, nH, window_size, nW, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    # Reshape to (B, H*W, C)
    x = x.view(B, H * W, C)

    return x


class TDMHSA(nn.Module):
    """Time-Dependent Multi-Head Self-Attention (TMSA).

    Implements Equations 3-6 from the DiffiT paper:
    - q_s = x_s · W_qs + x_t · W_qt  (Eq. 3)
    - k_s = x_s · W_ks + x_t · W_kt  (Eq. 4)
    - v_s = x_s · W_vs + x_t · W_vt  (Eq. 5)
    - Attention(Q,K,V) = Softmax(QK^T/√d + B)V  (Eq. 6)

    Args:
        feature_dim: Feature dimension (must be divisible by n_heads)
        time_dim: Time embedding dimension
        n_heads: Number of attention heads
        window_size: Optional window size for local attention (paper Sec. 3.2)
                     If None, uses global attention. If set, uses windowed attention.
        max_len: Maximum sequence length for relative position embeddings
    """

    def __init__(
        self,
        feature_dim: int,
        time_dim: int,
        n_heads: int,
        window_size: int | None = None,
        max_len: int = 1024,
    ) -> None:
        super().__init__()
        assert feature_dim % n_heads == 0, f"Feature dim {feature_dim} not divisible by {n_heads} heads"

        self.num_heads = n_heads
        self.head_dim = feature_dim // n_heads
        self.window_size = window_size
        self.max_len = max_len
        self.scale = self.head_dim ** -0.5

        # Spatial QKV projections (W_qs, W_ks, W_vs)
        self.qkv_proj = nn.Linear(feature_dim, feature_dim * 3)

        # Time QKV projections (W_qt, W_kt, W_vt)
        self.time_proj = nn.Linear(time_dim, feature_dim * 3)

        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        # Relative positional embeddings (bias B in Eq. 6)
        # Use learnable bias table as in Swin Transformer
        if window_size is not None:
            # For windowed attention, bias table size is (2*window_size-1)^2
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), n_heads)
            )
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

            # Pre-compute relative position index
            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            self.register_buffer("relative_position_index", relative_coords.sum(-1))  # Wh*Ww, Wh*Ww
        else:
            # For global attention, use simpler learnable bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(max_len, n_heads)
            )
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
            self.register_buffer("relative_position_index", None)

    def get_relative_position_bias(self, N: int) -> torch.Tensor:
        """Get relative position bias for attention computation.

        Args:
            N: Sequence length

        Returns:
            Relative position bias tensor of shape (num_heads, N, N)
        """
        if self.window_size is not None:
            # Window-based attention: use pre-computed index
            bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
            bias = bias.view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
            bias = bias.permute(2, 0, 1).contiguous()  # (num_heads, N, N)
            return bias
        else:
            # Global attention: use learnable bias
            bias = self.relative_position_bias_table[:N, :]  # (N, num_heads)
            # Create pairwise bias (simplified version)
            bias = bias.unsqueeze(1).expand(-1, N, -1)  # (N, N, num_heads)
            bias = bias.permute(2, 0, 1).contiguous()  # (num_heads, N, N)
            return bias

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with time-dependent attention.

        Args:
            x: Input features (B, N, C) where N is sequence length
            time_emb: Time embedding (B, time_dim)

        Returns:
            Output features (B, N, C)
        """
        B, N, C = x.shape
        orig_B = B  # Store original batch size for windowing

        # Window partition if using local attention
        if self.window_size is not None and N > self.window_size * self.window_size:
            H = W = int(math.sqrt(N))
            x, nH, nW = window_partition(x, self.window_size)
            B = x.shape[0]  # Update B to include window dimension
            N = self.window_size * self.window_size  # Update N to window size
            # Expand time_emb for each window
            time_emb = time_emb.unsqueeze(1).expand(-1, nH * nW, -1).reshape(B, -1)

        # Compute spatial QKV (x_s · W_q/k/v in Eq. 3-5)
        qkv_spatial = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv_spatial = qkv_spatial.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q_s, k_s, v_s = qkv_spatial.unbind(0)

        # Compute time-based QKV offset (x_t · W_qt/kt/vt in Eq. 3-5)
        qkv_time = self.time_proj(time_emb).reshape(B, 1, 3, self.num_heads, self.head_dim)
        qkv_time = qkv_time.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, 1, head_dim)
        q_t, k_t, v_t = qkv_time.unbind(0)

        # Time-dependent Q, K, V (Eq. 3-5)
        q = q_s + q_t  # (B, num_heads, N, head_dim)
        k = k_s + k_t
        v = v_s + v_t

        # Scaled dot-product attention with relative position bias (Eq. 6)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)

        # Add relative position bias B
        rel_pos_bias = self.get_relative_position_bias(N)  # (num_heads, N, N)
        attn = attn + rel_pos_bias.unsqueeze(0)  # (B, num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # (B, num_heads, N, head_dim)

        # Reshape and project output
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)

        # Window unpartition if using local attention
        if self.window_size is not None and self.window_size * self.window_size < int(math.sqrt(N)) ** 2:
            H = W = int(math.sqrt(N))
            out = window_unpartition(out, self.window_size, H, W, orig_B)

        return out


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
    """Transformer block with time-dependent attention.

    Implements DiffiT Transformer Block (Eq. 7-8 from paper):
    - x̂ = TMSA(LN(x), x_t) + x   (Eq. 7)
    - x = MLP(LN(x̂)) + x̂          (Eq. 8)
    """

    def __init__(
        self,
        time_emb_dim: int,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        window_size: int | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = TDMHSA(hidden_dim, time_emb_dim, num_heads, window_size=window_size)
        self.ffn = FeedForward(hidden_dim, mlp_ratio)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass following Eq. 7-8 from the paper."""
        x = x + self.attn(self.norm1(x), t)  # Eq. 7
        x = x + self.ffn(x)  # Eq. 8
        return x
