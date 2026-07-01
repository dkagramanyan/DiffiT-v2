"""Lightweight CPU smoke tests for DiffiT.

These run on tiny dimensions so they are fast and need no GPU, dataset, or
external weights. They guard the model's forward contract and the diffusion
math against accidental regressions during refactors. Run with:

    pytest tests/ -q
"""

import pytest
import torch

from diffit import create_diffusion, diffusion_defaults
from diffit.diffit import (
    DiffiT,
    DiffiTAttention,
    build_axial_rope_cache,
)


def _tiny_model(**kwargs):
    """A small DiffiT instance: head_dim = 64 / 4 = 16 (divisible by 4)."""
    defaults = dict(
        input_size=8, patch_size=2, in_channels=4,
        hidden_size=64, depth=2, num_heads=4, num_classes=10,
    )
    defaults.update(kwargs)
    torch.manual_seed(0)
    return DiffiT(**defaults)


# --- model forward -----------------------------------------------------------

def test_forward_output_shape_learn_sigma():
    model = _tiny_model(learn_sigma=True).eval()
    x = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 1000, (2,))
    y = torch.randint(0, 10, (2,))
    with torch.no_grad():
        out = model(x, t, y)
    # learn_sigma doubles the output channels (mean + variance).
    assert out.shape == (2, 8, 8, 8)


def test_forward_output_shape_no_sigma():
    model = _tiny_model(learn_sigma=False).eval()
    x = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 1000, (2,))
    y = torch.randint(0, 10, (2,))
    with torch.no_grad():
        out = model(x, t, y)
    assert out.shape == (2, 4, 8, 8)


def test_unpatchify_roundtrip_shape():
    model = _tiny_model().eval()
    n_tokens = (8 // 2) ** 2  # grid 4x4 = 16 tokens
    p = model.patch_size
    c = model.out_channels
    x = torch.randn(2, n_tokens, p * p * c)
    out = model.unpatchify(x)
    assert out.shape == (2, c, 8, 8)


# --- RoPE cache --------------------------------------------------------------

def test_rope_cache_shapes():
    grid_size, head_dim = 4, 16
    cos_y, sin_y, cos_x, sin_x = build_axial_rope_cache(grid_size, head_dim)
    expected = (1, 1, grid_size ** 2, head_dim // 2)
    for t in (cos_y, sin_y, cos_x, sin_x):
        assert t.shape == expected


def test_rope_requires_head_dim_divisible_by_4():
    with pytest.raises(AssertionError):
        build_axial_rope_cache(grid_size=4, head_dim=6)
    with pytest.raises(AssertionError):
        # dim=6, num_heads=1 -> head_dim=6, not divisible by 4
        DiffiTAttention(dim=6, temb_dim=8, num_heads=1)


# --- diffusion math ----------------------------------------------------------

def test_q_sample_at_t0_is_scaled_x():
    diff = create_diffusion(**diffusion_defaults())
    x = torch.randn(3, 4, 8, 8)
    t0 = torch.zeros(3, dtype=torch.long)
    qs = diff.q_sample(x, t0, noise=torch.zeros_like(x))
    scale = float(diff.sqrt_alphas_cumprod[0])
    assert torch.allclose(qs, scale * x, atol=1e-6)


def test_training_losses_finite():
    model = _tiny_model().eval()
    diff = create_diffusion(**diffusion_defaults())
    x = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 1000, (2,))
    y = torch.randint(0, 10, (2,))
    losses = diff.training_losses(model, x, t, {"y": y})
    for key in ("loss", "mse", "vb"):
        assert key in losses
        assert torch.isfinite(losses[key]).all()


# --- samplers ----------------------------------------------------------------

def test_unipc_sample_shape_and_finite():
    from diffit.unipc_solver import unipc_sample

    model = _tiny_model(learn_sigma=True).eval()

    def model_fn(x, t):
        return model(x, t, torch.zeros(x.shape[0], dtype=torch.long))

    diff = create_diffusion(**diffusion_defaults())
    shape = (2, 4, 8, 8)
    out = unipc_sample(model_fn, diff, shape, torch.device("cpu"), num_steps=3)
    assert out.shape == shape
    assert torch.isfinite(out).all()
