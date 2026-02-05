# Copyright (c) 2024, DiffiT authors.
# Model configuration presets matching the DiffiT paper.

from __future__ import annotations


def get_diffit_config(dataset: str = "cifar10", resolution: int = None) -> dict:
    """Get DiffiT model configuration matching the paper specifications.

    Args:
        dataset: Dataset name ('cifar10', 'ffhq64', 'imagenet256', 'imagenet512')
        resolution: Image resolution (overrides dataset default)

    Returns:
        Dictionary with model configuration parameters
    """
    configs = {
        # CIFAR-10 configuration (32x32) - Paper Table S.2-S.3
        "cifar10": {
            "resolution": 32,
            "base_dim": 128,  # Base channel dimension
            "channel_mults": (1, 2, 2),  # 3 stages: 128, 256, 256
            "num_res_blocks": 2,  # 2 ResBlocks per stage
            "hidden_dim": 256,  # ViT hidden dimension
            "num_patches": 4,  # Patch size = 32/4 = 8, 16/4 = 4, 8/4 = 2
            "num_heads": 4,  # Number of attention heads
            "num_transformer_blocks": 2,  # Transformer blocks per ResBlock
            "window_size": 4,  # Local attention window size
            "timesteps": 1000,
            "beta_schedule": "squaredcos_cap_v2",
        },
        # FFHQ-64 configuration (64x64) - Paper Table S.2-S.3
        "ffhq64": {
            "resolution": 64,
            "base_dim": 128,
            "channel_mults": (1, 2, 2, 2),  # 4 stages: 128, 256, 256, 256
            "num_res_blocks": 4,  # 4 ResBlocks per stage (as per Table S.2)
            "hidden_dim": 256,
            "num_patches": 2,  # Patch size = 64/2=32, 32/2=16, 16/2=8, 8/2=4
            "num_heads": 4,
            "num_transformer_blocks": 1,
            "window_size": 8,  # Larger window for 64x64
            "timesteps": 1000,
            "beta_schedule": "squaredcos_cap_v2",
        },
        # ImageNet-256 latent configuration
        "imagenet256": {
            "resolution": 256,
            "base_dim": 128,
            "channel_mults": (1, 2, 2, 2),
            "num_res_blocks": 2,
            "hidden_dim": 1152,  # Larger for ImageNet
            "num_patches": 2,
            "num_heads": 16,  # More heads for larger model
            "num_transformer_blocks": 30,  # Depth of 30 layers (paper Section H.2)
            "window_size": None,  # Global attention for latent space
            "timesteps": 1000,
            "beta_schedule": "squaredcos_cap_v2",
        },
        # ImageNet-512 latent configuration
        "imagenet512": {
            "resolution": 512,
            "base_dim": 128,
            "channel_mults": (1, 2, 2, 2),
            "num_res_blocks": 2,
            "hidden_dim": 1152,
            "num_patches": 2,
            "num_heads": 16,
            "num_transformer_blocks": 30,
            "window_size": None,
            "timesteps": 1000,
            "beta_schedule": "squaredcos_cap_v2",
        },
    }

    if dataset not in configs:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available: {list(configs.keys())}"
        )

    config = configs[dataset].copy()

    # Override resolution if provided
    if resolution is not None:
        config["resolution"] = resolution

        # Adjust window size based on resolution
        if resolution <= 32:
            config["window_size"] = 4
        elif resolution <= 64:
            config["window_size"] = 8
        else:
            config["window_size"] = None  # Use global attention for large resolutions

    return config


def get_training_config(dataset: str = "cifar10") -> dict:
    """Get training configuration matching the paper.

    Args:
        dataset: Dataset name

    Returns:
        Dictionary with training hyperparameters
    """
    # Based on paper Section I: Implementation Details
    configs = {
        "cifar10": {
            "batch_size": 512,  # Total batch size (paper Section I.1)
            "lr": 1e-3,  # Learning rate
            "total_kimg": 200000,  # Training duration (200K iterations)
            "ema_kimg": 10.0,  # EMA half-life
            "fp32": False,  # Use mixed precision
        },
        "ffhq64": {
            "batch_size": 256,
            "lr": 2e-4,
            "total_kimg": 200000,
            "ema_kimg": 10.0,
            "fp32": False,
        },
        "imagenet256": {
            "batch_size": 256,  # Paper Section I.2
            "lr": 3e-4,
            "total_kimg": 1000000,  # 1M iterations for ImageNet
            "ema_kimg": 10.0,
            "fp32": False,
        },
        "imagenet512": {
            "batch_size": 512,
            "lr": 1e-4,
            "total_kimg": 1000000,
            "ema_kimg": 10.0,
            "fp32": False,
        },
    }

    return configs.get(dataset, configs["cifar10"])


# Example usage:
# model_config = get_diffit_config("ffhq64")
# training_config = get_training_config("ffhq64")
