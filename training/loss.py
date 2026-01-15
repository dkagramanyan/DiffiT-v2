# Copyright (c) 2024, DiffiT authors.
# Loss functions for DiffiT diffusion model.

from __future__ import annotations

import torch
import torch.nn.functional as F

from torch_utils import training_stats


class DiffusionLoss:
    """Loss function for diffusion model training."""

    def __init__(
        self,
        device: torch.device,
        model: torch.nn.Module,
        diffusion,
        loss_type: str = "smooth_l1",
    ):
        self.device = device
        self.model = model
        self.diffusion = diffusion
        self.loss_type = loss_type

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the diffusion training loss.

        Args:
            x: Input images tensor [B, C, H, W] in range [0, 1].

        Returns:
            Scalar loss value.
        """
        # Perturb images and predict noise
        _, epsilon, pred_epsilon = self.diffusion.perturb_and_predict(x)

        # Compute loss
        if self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(epsilon, pred_epsilon)
        elif self.loss_type == "l1":
            loss = F.l1_loss(epsilon, pred_epsilon)
        elif self.loss_type == "l2":
            loss = F.mse_loss(epsilon, pred_epsilon)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Report statistics
        training_stats.report("Loss/diffusion", loss)

        return loss

    def accumulate_gradients(self, x: torch.Tensor, gain: float = 1.0) -> None:
        """Accumulate gradients for a batch.

        Args:
            x: Input images tensor [B, C, H, W] in range [0, 1].
            gain: Gradient scaling factor.
        """
        loss = self.compute_loss(x)
        (loss * gain).backward()
