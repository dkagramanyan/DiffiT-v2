# Copyright (c) 2024, DiffiT authors.
# Diffusion model for image generation.
#
# Implements the diffusion framework from the DiffiT paper (Section 3.1).
#
# The paper uses Variance-Exploding (VE) notation:
#   q(z_t | z_0) = N(z_0, σ_t² I)
#
# Training objective (Eq. 1):
#   L = E[λ(t) * ||ε - ε_θ(z_0 + σ_t·ε, t)||²]
#
# The score network relates to epsilon prediction via:
#   s_θ(z_t, t) = -ε_θ(z_t, t) / σ_t

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

# Try to import diffusers, fallback to minimal implementation
try:
    from diffusers import DDPMScheduler, DDIMScheduler
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False


class Diffusion:
    """Denoising Diffusion Probabilistic Model.

    Implements the diffusion framework described in the DiffiT paper (Section 3.1).
    Uses diffusers schedulers when available for efficient and well-tested implementations.
    
    The paper uses Variance-Exploding (VE) notation but notes it can be converted to
    Variance-Preserving (VP) by rescaling. This implementation uses VP (DDPM) scheduling
    which is equivalent after the appropriate rescaling.
    """

    def __init__(
        self,
        model: nn.Module,
        image_resolution: tuple[int, int, int] | list[int] = (3, 64, 64),
        n_times: int = 1000,
        device: str | torch.device = "cuda",
        beta_schedule: str = "squaredcos_cap_v2",
    ) -> None:
        """Initialize the diffusion model.

        Args:
            model: The denoising network (predicts noise ε_θ as per paper Eq. 1).
            image_resolution: Tuple of (channels, height, width).
            n_times: Number of diffusion timesteps T.
            device: Device to run computations on.
            beta_schedule: Noise schedule type ('linear', 'squaredcos_cap_v2').
                'squaredcos_cap_v2' corresponds to the improved cosine schedule.
        """
        self.model = model
        self.img_C, self.img_H, self.img_W = image_resolution
        self.device = torch.device(device) if isinstance(device, str) else device
        self.n_times = n_times

        if HAS_DIFFUSERS:
            # Use diffusers schedulers (DDPM/DDIM implementations)
            # These implement the sampling SDEs from the paper (Eq. 2)
            self.ddpm_scheduler = DDPMScheduler(
                num_train_timesteps=n_times,
                beta_schedule=beta_schedule,
                prediction_type="epsilon",
                clip_sample=True,
            )
            self.ddim_scheduler = DDIMScheduler(
                num_train_timesteps=n_times,
                beta_schedule=beta_schedule,
                prediction_type="epsilon",
                clip_sample=True,
            )
        else:
            # Fallback: compute schedules manually
            self._init_schedules(n_times, beta_schedule)

    def _init_schedules(self, n_times: int, schedule: str) -> None:
        """Initialize noise schedules when diffusers is not available."""
        if schedule == "squaredcos_cap_v2":
            # Cosine schedule
            s = 0.008
            steps = n_times + 1
            t = torch.linspace(0, n_times, steps)
            alphas_cumprod = torch.cos(((t / n_times) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            # Linear schedule
            betas = torch.linspace(0.0001, 0.02, n_times)

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas.to(self.device)
        self.alphas = alphas.to(self.device)
        self.alphas_cumprod = alphas_cumprod.to(self.device)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod).to(self.device)

    def add_noise(self, x: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to samples using the forward diffusion process."""
        if HAS_DIFFUSERS:
            return self.ddpm_scheduler.add_noise(x, noise, timesteps)

        # Manual implementation
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        return sqrt_alpha * x + sqrt_one_minus_alpha * noise

    def perturb_and_predict(
        self, 
        x: torch.Tensor, 
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add noise and predict it back (training forward pass).
        
        Implements the training objective from DiffiT paper Eq. 1:
            L = E[λ(t) * ||ε - ε_θ(z_0 + σ_t·ε, t)||²]
        
        This function:
        1. Samples random timesteps t ~ p(t) (uniform)
        2. Samples noise ε ~ N(0, I)
        3. Creates noisy samples: z_t = z_0 + σ_t·ε (in VE) or equivalent VP formulation
        4. Predicts noise: ε̂ = ε_θ(z_t, t, c) where c is the class label

        Args:
            x: Clean images (z_0) in [0, 1] range.
            labels: Optional class labels (B,) as integer indices.

        Returns:
            Tuple of (noisy_images z_t, true_noise ε, predicted_noise ε̂).
        """
        # Scale to [-1, 1] (standard preprocessing for diffusion models)
        x = x * 2 - 1

        B = x.shape[0]
        # Sample timesteps uniformly: t ~ p(t) where p(t) is uniform over [0, T)
        timesteps = torch.randint(0, self.n_times, (B,), device=self.device, dtype=torch.long)
        # Sample noise: ε ~ N(0, I)
        noise = torch.randn_like(x)

        # Create noisy samples using the forward diffusion process
        # z_t = √α̅_t · z_0 + √(1-α̅_t) · ε (VP formulation, equivalent to VE after rescaling)
        noisy_x = self.add_noise(x, noise, timesteps)
        
        # Predict noise: ε̂ = ε_θ(z_t, t, c)
        # Note: model handles label dropout internally for CFG training
        pred_noise = self.model(noisy_x, timesteps, labels=labels)

        return noisy_x, noise, pred_noise

    @torch.no_grad()
    def sample(
        self, 
        n_samples: int, 
        labels: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        """Generate samples using DDPM reverse process with optional CFG.
        
        Implements the sampling SDE from DiffiT paper Eq. 2:
            dz = -(σ̇_t + β_t)σ_t·s_θ(z,t)dt + √(2β_t·σ_t)·dω_t
        
        When β_t > 0, this is the stochastic DDPM sampler.
        The score network s_θ relates to ε_θ via: s_θ = -ε_θ/σ_t
        
        Classifier-free guidance (CFG) combines conditional and unconditional:
            ε̂ = ε_uncond + cfg_scale * (ε_cond - ε_uncond)

        Args:
            n_samples: Number of samples to generate.
            labels: Optional class labels (n_samples,) for conditional generation.
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance).
            return_intermediates: Whether to return intermediate steps.

        Returns:
            Generated images in [0, 1] range.
        """
        self.model.eval()
        use_cfg = labels is not None and cfg_scale != 1.0

        x = torch.randn(n_samples, self.img_C, self.img_H, self.img_W, device=self.device)
        intermediates = [x.clone()] if return_intermediates else None

        if HAS_DIFFUSERS:
            self.ddpm_scheduler.set_timesteps(self.n_times)
            for t in self.ddpm_scheduler.timesteps:
                t_batch = t.expand(n_samples).to(self.device)
                
                if use_cfg:
                    # Classifier-free guidance: compute both conditional and unconditional
                    noise_pred_cond = self.model(x, t_batch, labels=labels)
                    noise_pred_uncond = self.model(x, t_batch, labels=labels, force_drop_labels=True)
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = self.model(x, t_batch, labels=labels)
                
                x = self.ddpm_scheduler.step(noise_pred, t, x).prev_sample
                if return_intermediates:
                    intermediates.append(x.clone())
        else:
            for t in reversed(range(self.n_times)):
                t_batch = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
                
                if use_cfg:
                    # Classifier-free guidance
                    noise_pred_cond = self.model(x, t_batch, labels=labels)
                    noise_pred_uncond = self.model(x, t_batch, labels=labels, force_drop_labels=True)
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = self.model(x, t_batch, labels=labels)

                alpha = self.alphas[t]
                alpha_bar = self.alphas_cumprod[t]
                beta = self.betas[t]

                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = 0

                x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * noise_pred)
                x = x + torch.sqrt(beta) * noise
                x = x.clamp(-1, 1)

                if return_intermediates:
                    intermediates.append(x.clone())

        # Scale to [0, 1]
        x = (x + 1) / 2
        x = x.clamp(0, 1)

        if return_intermediates:
            return x, intermediates
        return x

    @torch.no_grad()
    def sample_ddim(
        self, 
        n_samples: int, 
        labels: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        num_inference_steps: int = 50, 
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Generate samples using DDIM (faster sampling) with optional CFG.
        
        When β_t = 0 in paper Eq. 2, the sampling becomes a probability flow ODE
        that can be solved with DDIM. This allows for faster sampling with fewer steps.
        
        Classifier-free guidance (CFG) combines conditional and unconditional:
            ε̂ = ε_uncond + cfg_scale * (ε_cond - ε_uncond)

        Args:
            n_samples: Number of samples to generate.
            labels: Optional class labels (n_samples,) for conditional generation.
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance).
            num_inference_steps: Number of denoising steps (can be << n_times).
            eta: DDIM eta parameter (0 = deterministic ODE, 1 = stochastic like DDPM).

        Returns:
            Generated images in [0, 1] range.
        """
        self.model.eval()
        use_cfg = labels is not None and cfg_scale != 1.0

        x = torch.randn(n_samples, self.img_C, self.img_H, self.img_W, device=self.device)

        if HAS_DIFFUSERS:
            self.ddim_scheduler.set_timesteps(num_inference_steps)
            for t in self.ddim_scheduler.timesteps:
                t_batch = t.expand(n_samples).to(self.device)
                
                if use_cfg:
                    # Classifier-free guidance
                    noise_pred_cond = self.model(x, t_batch, labels=labels)
                    noise_pred_uncond = self.model(x, t_batch, labels=labels, force_drop_labels=True)
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = self.model(x, t_batch, labels=labels)
                
                x = self.ddim_scheduler.step(noise_pred, t, x, eta=eta).prev_sample
        else:
            # Simple DDIM without diffusers
            step_ratio = self.n_times // num_inference_steps
            timesteps = list(range(0, self.n_times, step_ratio))[::-1]

            for i, t in enumerate(timesteps):
                t_batch = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
                
                if use_cfg:
                    # Classifier-free guidance
                    noise_pred_cond = self.model(x, t_batch, labels=labels)
                    noise_pred_uncond = self.model(x, t_batch, labels=labels, force_drop_labels=True)
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = self.model(x, t_batch, labels=labels)

                alpha_bar_t = self.alphas_cumprod[t]
                alpha_bar_prev = self.alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0, device=self.device)

                # Predict x0
                pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
                pred_x0 = pred_x0.clamp(-1, 1)

                # DDIM step
                sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev))
                dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * noise_pred
                noise = torch.randn_like(x) if eta > 0 else 0
                x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + sigma * noise

        # Scale to [0, 1]
        x = (x + 1) / 2
        x = x.clamp(0, 1)

        return x
