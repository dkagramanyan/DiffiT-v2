# Copyright (c) 2024, DiffiT authors.
# Fused multiply-add operations.

from __future__ import annotations

import torch


def fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Fused multiply-add: a * b + c."""
    return torch.addcmul(c, a, b)
