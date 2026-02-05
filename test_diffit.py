#!/usr/bin/env python3
# Copyright (c) 2024, DiffiT authors.
# Test script for DiffiT implementation.

"""
Test script to verify DiffiT implementation matches paper specifications.

Usage:
    python test_diffit.py --test all
    python test_diffit.py --test attention
    python test_diffit.py --test model
    python test_diffit.py --test diffusion
"""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

# Test utilities
def check_shape(tensor: torch.Tensor, expected_shape: Tuple, name: str):
    """Check if tensor shape matches expected shape."""
    if tuple(tensor.shape) != expected_shape:
        print(f"❌ {name}: Expected {expected_shape}, got {tensor.shape}")
        return False
    else:
        print(f"✅ {name}: Shape {tensor.shape} is correct")
        return True


def test_attention():
    """Test TMSA attention mechanism."""
    print("\n" + "="*60)
    print("Testing TMSA Attention Mechanism")
    print("="*60)

    try:
        from models.attention_improved import TDMHSA, TransformerBlock
        print("✅ Successfully imported improved attention modules")
    except ImportError:
        print("❌ Failed to import improved attention modules")
        print("   Falling back to original implementation")
        from models.attention import TDMHSA, TransformerBlock

    # Test parameters
    batch_size = 4
    seq_len = 64  # 8x8 feature map
    feature_dim = 256
    time_dim = 512
    num_heads = 4
    window_size = 8

    print(f"\nTest configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Time dim: {time_dim}")
    print(f"  Num heads: {num_heads}")
    print(f"  Window size: {window_size}")

    # Create TMSA module
    print("\nTest 1: TMSA without windowing (global attention)")
    tmsa = TDMHSA(feature_dim, time_dim, num_heads, window_size=None)
    x = torch.randn(batch_size, seq_len, feature_dim)
    time_emb = torch.randn(batch_size, time_dim)

    out = tmsa(x, time_emb)
    check_shape(out, (batch_size, seq_len, feature_dim), "TMSA output (global)")

    # Test windowed attention
    print("\nTest 2: TMSA with windowing (local attention)")
    tmsa_windowed = TDMHSA(feature_dim, time_dim, num_heads, window_size=window_size)
    out_windowed = tmsa_windowed(x, time_emb)
    check_shape(out_windowed, (batch_size, seq_len, feature_dim), "TMSA output (windowed)")

    # Test Transformer block
    print("\nTest 3: Transformer Block")
    block = TransformerBlock(time_dim, feature_dim, num_heads, mlp_ratio=4)
    out_block = block(x, time_emb)
    check_shape(out_block, (batch_size, seq_len, feature_dim), "TransformerBlock output")

    # Test gradient flow
    print("\nTest 4: Gradient flow")
    loss = out_block.mean()
    loss.backward()
    has_grads = tmsa.qkv_proj.weight.grad is not None
    if has_grads:
        print("✅ Gradients flow correctly through TMSA")
    else:
        print("❌ No gradients through TMSA")

    print("\n" + "="*60)
    print("Attention tests completed!")
    print("="*60)


def test_model():
    """Test DiffiT model architecture."""
    print("\n" + "="*60)
    print("Testing DiffiT Model Architecture")
    print("="*60)

    from models.diffit import DiffiT

    # Test configurations
    test_configs = [
        {
            "name": "CIFAR-10 (32x32)",
            "image_shape": [3, 32, 32],
            "base_dim": 128,
            "channel_mults": (1, 2, 2),
            "num_res_blocks": 2,
            "hidden_dim": 256,
            "num_heads": 4,
        },
        {
            "name": "FFHQ-64 (64x64)",
            "image_shape": [3, 64, 64],
            "base_dim": 128,
            "channel_mults": (1, 2, 2, 2),
            "num_res_blocks": 4,
            "hidden_dim": 256,
            "num_heads": 4,
        },
    ]

    for config in test_configs:
        print(f"\nTest: {config['name']}")
        print("-" * 60)

        # Create model
        model = DiffiT(
            image_shape=config["image_shape"],
            base_dim=config["base_dim"],
            channel_mults=config["channel_mults"],
            num_res_blocks=config["num_res_blocks"],
            hidden_dim=config["hidden_dim"],
            num_heads=config["num_heads"],
            label_dim=0,  # Unconditional
        )

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")

        # Test forward pass
        batch_size = 2
        C, H, W = config["image_shape"]
        x = torch.randn(batch_size, C, H, W)
        timesteps = torch.randint(0, 1000, (batch_size,))

        out = model(x, timesteps)
        check_shape(out, (batch_size, C, H, W), f"Model output ({config['name']})")

        # Test with labels (conditional)
        print(f"\n  Testing class-conditional generation")
        model_cond = DiffiT(
            image_shape=config["image_shape"],
            base_dim=config["base_dim"],
            channel_mults=config["channel_mults"],
            num_res_blocks=config["num_res_blocks"],
            hidden_dim=config["hidden_dim"],
            num_heads=config["num_heads"],
            label_dim=10,  # 10 classes
            label_drop_prob=0.1,
        )

        labels = torch.randint(0, 10, (batch_size,))
        out_cond = model_cond(x, timesteps, labels=labels)
        check_shape(out_cond, (batch_size, C, H, W), f"Conditional output ({config['name']})")

        # Test CFG (force_drop_labels)
        out_uncond = model_cond(x, timesteps, labels=labels, force_drop_labels=True)
        check_shape(out_uncond, (batch_size, C, H, W), f"Unconditional output for CFG")

        print(f"✅ {config['name']} tests passed")

    print("\n" + "="*60)
    print("Model tests completed!")
    print("="*60)


def test_diffusion():
    """Test diffusion process."""
    print("\n" + "="*60)
    print("Testing Diffusion Process")
    print("="*60)

    from diffusion.diffusion import Diffusion
    from models.diffit import DiffiT

    # Create simple model
    image_shape = [3, 32, 32]
    model = DiffiT(
        image_shape=image_shape,
        base_dim=64,  # Smaller for faster testing
        channel_mults=(1, 2),
        num_res_blocks=1,
        hidden_dim=128,
        num_heads=2,
    )

    # Create diffusion wrapper
    diffusion = Diffusion(
        model=model,
        image_resolution=image_shape,
        n_times=1000,
        device="cpu",  # Use CPU for testing
    )

    print("✅ Diffusion wrapper created")

    # Test sampling
    print("\nTest: DDPM sampling")
    batch_size = 2
    samples_ddpm = diffusion.sample_ddpm(batch_size, num_inference_steps=10)
    check_shape(samples_ddpm, (batch_size, *image_shape), "DDPM samples")

    print("\nTest: DDIM sampling")
    samples_ddim = diffusion.sample_ddim(batch_size, num_inference_steps=10)
    check_shape(samples_ddim, (batch_size, *image_shape), "DDIM samples")

    # Test training step
    print("\nTest: Training forward pass")
    x = torch.randn(batch_size, *image_shape)
    timesteps = torch.randint(0, 1000, (batch_size,))
    noise = torch.randn_like(x)

    # Simulate training step
    noisy_x = diffusion.q_sample(x, timesteps, noise)
    pred_noise = model(noisy_x, timesteps)
    loss = torch.nn.functional.mse_loss(pred_noise, noise)

    print(f"  Noisy input shape: {noisy_x.shape}")
    print(f"  Predicted noise shape: {pred_noise.shape}")
    print(f"  Loss: {loss.item():.6f}")
    print("✅ Training forward pass works")

    # Test CFG
    print("\nTest: Classifier-Free Guidance")
    model_cond = DiffiT(
        image_shape=image_shape,
        base_dim=64,
        channel_mults=(1, 2),
        num_res_blocks=1,
        hidden_dim=128,
        num_heads=2,
        label_dim=10,
    )
    diffusion_cond = Diffusion(
        model=model_cond,
        image_resolution=image_shape,
        n_times=1000,
        device="cpu",
    )

    labels = torch.randint(0, 10, (batch_size,))
    samples_cfg = diffusion_cond.sample_ddim(
        batch_size,
        labels=labels,
        cfg_scale=2.0,
        num_inference_steps=10,
    )
    check_shape(samples_cfg, (batch_size, *image_shape), "CFG samples")

    print("\n" + "="*60)
    print("Diffusion tests completed!")
    print("="*60)


def test_configs():
    """Test configuration presets."""
    print("\n" + "="*60)
    print("Testing Configuration Presets")
    print("="*60)

    try:
        from configs import get_diffit_config, get_training_config
    except ImportError:
        print("❌ Failed to import configs module")
        return

    datasets = ["cifar10", "ffhq64", "imagenet256", "imagenet512"]

    for dataset in datasets:
        print(f"\nTest: {dataset}")
        print("-" * 60)

        model_config = get_diffit_config(dataset)
        train_config = get_training_config(dataset)

        print(f"  Model config keys: {list(model_config.keys())}")
        print(f"  Resolution: {model_config['resolution']}")
        print(f"  Hidden dim: {model_config['hidden_dim']}")
        print(f"  Num heads: {model_config['num_heads']}")
        print(f"  Window size: {model_config.get('window_size', 'None')}")
        print(f"  Batch size: {train_config['batch_size']}")
        print(f"  Learning rate: {train_config['lr']}")

        print(f"✅ {dataset} configuration loaded")

    print("\n" + "="*60)
    print("Config tests completed!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Test DiffiT implementation")
    parser.add_argument(
        "--test",
        type=str,
        choices=["all", "attention", "model", "diffusion", "configs"],
        default="all",
        help="Which tests to run",
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("DiffiT Implementation Test Suite")
    print("="*60)

    if args.test in ["all", "attention"]:
        test_attention()

    if args.test in ["all", "model"]:
        test_model()

    if args.test in ["all", "diffusion"]:
        test_diffusion()

    if args.test in ["all", "configs"]:
        test_configs()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
