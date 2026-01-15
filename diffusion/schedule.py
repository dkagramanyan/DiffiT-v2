# Copyright (c) 2024, DiffiT authors.
# Noise schedules for diffusion models.

from __future__ import annotations

import torch


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule as proposed in 'Improved Denoising Diffusion Probabilistic Models'.

    Args:
        timesteps: Number of diffusion timesteps.
        s: Small offset to prevent singularities at t=0.

    Returns:
        Beta values for each timestep.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """Linear beta schedule.

    Args:
        timesteps: Number of diffusion timesteps.
        beta_start: Starting beta value.
        beta_end: Ending beta value.

    Returns:
        Beta values for each timestep.
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """Quadratic beta schedule.

    Args:
        timesteps: Number of diffusion timesteps.
        beta_start: Starting beta value.
        beta_end: Ending beta value.

    Returns:
        Beta values for each timestep.
    """
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """Sigmoid beta schedule.

    Args:
        timesteps: Number of diffusion timesteps.
        beta_start: Starting beta value.
        beta_end: Ending beta value.

    Returns:
        Beta values for each timestep.
    """
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
