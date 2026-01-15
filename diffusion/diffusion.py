# Copyright (c) 2024, DiffiT authors.
# Diffusion model for image generation.

from __future__ import annotations

import torch

from diffusion.schedule import cosine_beta_schedule


class Diffusion:
    """Denoising Diffusion Probabilistic Model.

    Implements the forward and reverse diffusion process for image generation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        image_resolution: tuple[int, int, int] | list[int] = (3, 64, 64),
        n_times: int = 1000,
        device: str | torch.device = "cuda",
        schedule: str = "cosine",
    ) -> None:
        """Initialize the diffusion model.

        Args:
            model: The denoising network (predicts noise).
            image_resolution: Tuple of (channels, height, width).
            n_times: Number of diffusion timesteps.
            device: Device to run computations on.
            schedule: Noise schedule type ('cosine' or 'linear').
        """
        self.n_times = n_times
        self.img_C, self.img_H, self.img_W = image_resolution
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device

        # Compute noise schedule
        betas = cosine_beta_schedule(timesteps=n_times).to(self.device)
        self.register_buffer("betas", betas)
        self.register_buffer("sqrt_betas", torch.sqrt(betas))

        alphas = 1 - betas
        self.register_buffer("alphas", alphas)
        self.register_buffer("sqrt_alphas", torch.sqrt(alphas))

        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1 - alpha_bars))

    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        """Register a buffer (non-learnable parameter)."""
        setattr(self, name, tensor)

    def extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        """Extract values from a at indices t, and reshape for broadcasting."""
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def scale_to_minus_one_to_one(self, x: torch.Tensor) -> torch.Tensor:
        """Scale input from [0, 1] to [-1, 1]."""
        return x * 2 - 1

    def reverse_scale_to_zero_to_one(self, x: torch.Tensor) -> torch.Tensor:
        """Scale output from [-1, 1] to [0, 1]."""
        return (x + 1) * 0.5

    def make_noisy(self, x_zeros: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise to clean images using the forward diffusion process.

        Args:
            x_zeros: Clean images in [-1, 1] range.
            t: Timestep indices.

        Returns:
            Tuple of (noisy_images, noise).
        """
        epsilon = torch.randn_like(x_zeros, device=self.device)

        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t, x_zeros.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t, x_zeros.shape)

        noisy_sample = x_zeros * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar
        return noisy_sample.detach(), epsilon

    def perturb_and_predict(self, x_zeros: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: add noise and predict it back.

        Args:
            x_zeros: Clean images in [0, 1] range.

        Returns:
            Tuple of (perturbed_images, true_noise, predicted_noise).
        """
        x_zeros = self.scale_to_minus_one_to_one(x_zeros)

        b = x_zeros.shape[0]
        t = torch.randint(low=0, high=self.n_times, size=(b,), device=self.device).long()

        perturbed_images, epsilon = self.make_noisy(x_zeros, t)
        pred_epsilon = self.model(perturbed_images, t)

        return perturbed_images, epsilon, pred_epsilon

    def denoise_at_t(self, x_t: torch.Tensor, timestep: torch.Tensor, t: int) -> torch.Tensor:
        """Single denoising step at timestep t.

        Args:
            x_t: Noisy images at timestep t.
            timestep: Timestep tensor.
            t: Integer timestep value.

        Returns:
            Denoised images at timestep t-1.
        """
        if t > 1:
            z = torch.randn_like(x_t, device=self.device)
        else:
            z = torch.zeros_like(x_t, device=self.device)

        epsilon_pred = self.model(x_t, timestep)

        alpha = self.extract(self.alphas, timestep, x_t.shape)
        sqrt_alpha = self.extract(self.sqrt_alphas, timestep, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep, x_t.shape)
        sqrt_beta = self.extract(self.sqrt_betas, timestep, x_t.shape)

        x_t_minus_1 = (1 / sqrt_alpha) * (
            x_t - (1 - alpha) / sqrt_one_minus_alpha_bar * epsilon_pred
        ) + sqrt_beta * z

        return x_t_minus_1.clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample(self, n_samples: int, return_intermediate: bool = False) -> torch.Tensor:
        """Generate samples using the reverse diffusion process.

        Args:
            n_samples: Number of samples to generate.
            return_intermediate: If True, return all intermediate denoising steps.

        Returns:
            Generated images in [0, 1] range.
        """
        self.model.eval()

        x_t = torch.randn((n_samples, self.img_C, self.img_H, self.img_W), device=self.device)

        if return_intermediate:
            intermediates = [x_t.clone()]

        for t in range(self.n_times - 1, -1, -1):
            timestep = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
            x_t = self.denoise_at_t(x_t, timestep, t)

            if return_intermediate:
                intermediates.append(x_t.clone())

        x_0 = self.reverse_scale_to_zero_to_one(x_t)

        if return_intermediate:
            return x_0, intermediates

        return x_0

    @torch.no_grad()
    def sample_ddim(
        self,
        n_samples: int,
        ddim_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Generate samples using DDIM (faster sampling).

        Args:
            n_samples: Number of samples to generate.
            ddim_steps: Number of DDIM sampling steps.
            eta: DDIM stochasticity parameter (0 = deterministic, 1 = DDPM).

        Returns:
            Generated images in [0, 1] range.
        """
        self.model.eval()

        # Compute DDIM timesteps
        step_ratio = self.n_times // ddim_steps
        timesteps = torch.arange(0, self.n_times, step_ratio, device=self.device).long()
        timesteps = torch.flip(timesteps, [0])

        x_t = torch.randn((n_samples, self.img_C, self.img_H, self.img_W), device=self.device)

        for i, t in enumerate(timesteps):
            t_batch = t.repeat(n_samples)
            epsilon_pred = self.model(x_t, t_batch)

            alpha_bar_t = self.alpha_bars[t]
            alpha_bar_t_prev = self.alpha_bars[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0, device=self.device)

            # DDIM sampling formula
            pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * epsilon_pred) / torch.sqrt(alpha_bar_t)
            pred_x0 = pred_x0.clamp(-1, 1)

            sigma = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * torch.sqrt(1 - alpha_bar_t / alpha_bar_t_prev)
            noise = torch.randn_like(x_t) if eta > 0 else 0

            x_t = torch.sqrt(alpha_bar_t_prev) * pred_x0 + torch.sqrt(1 - alpha_bar_t_prev - sigma ** 2) * epsilon_pred + sigma * noise

        x_0 = self.reverse_scale_to_zero_to_one(x_t)
        return x_0
