# Copyright (c) 2024, DiffiT authors.
# Facilities for reporting and collecting training statistics.

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import torch

# Global state
_num_moments = 3
_reduce_dtype = torch.float32
_counters = dict()
_cumulative = dict()
_sync_device = None
_sync_called = False
_rank = 0


def init_multiprocessing(rank: int, sync_device: torch.device | None = None):
    """Initialize for multiprocessing (called once per process)."""
    global _rank, _sync_device
    _rank = rank
    _sync_device = sync_device


def reset():
    """Reset all counters to zero."""
    global _counters, _cumulative, _sync_called
    _counters = dict()
    _cumulative = dict()
    _sync_called = False


def report(name: str, value: torch.Tensor | float | int):
    """Report a scalar value to be collected during training."""
    _report(name, value)


def report0(name: str, value: torch.Tensor | float | int):
    """Report a scalar value only from rank 0."""
    _report(name, value if _rank == 0 else [])
    return value


def _report(name: str, value):
    """Internal: add value to the named counter."""
    global _sync_called
    if name not in _counters:
        _counters[name] = []
    if isinstance(value, torch.Tensor):
        _counters[name].append(value.detach().flatten().to(_reduce_dtype))
    elif isinstance(value, (int, float)):
        _counters[name].append(torch.tensor([value], dtype=_reduce_dtype))
    elif isinstance(value, list) and len(value) == 0:
        pass
    else:
        raise ValueError(f"Unsupported value type: {type(value)}")
    _sync_called = False


class Collector:
    """Collects training statistics after each training tick."""

    def __init__(self, regex: str = ".*"):
        self._regex = re.compile(regex)
        self._cumulative = None
        self._value = dict()

    def names(self) -> list[str]:
        """Get list of all statistic names."""
        return list(self._value.keys())

    def __getitem__(self, name: str) -> float:
        """Get the mean value of a statistic."""
        return self._value.get(name, 0)

    def update(self):
        """Collect all pending statistics, compute means, and store for retrieval."""
        global _cumulative, _sync_called

        if not _sync_called:
            _sync()

        # Copy current cumulative state
        cumulative = {name: moments.clone() for name, moments in _cumulative.items()}

        # Compute deltas
        for name in list(cumulative.keys()):
            if self._cumulative is not None and name in self._cumulative:
                cumulative[name] -= self._cumulative[name]

        # Compute values
        self._value.clear()
        for name, moments in cumulative.items():
            if self._regex.fullmatch(name):
                _check_moments(moments)
                self._value[name] = moments[0].item() / max(moments[2].item(), 1)

        self._cumulative = {name: moments.clone() for name, moments in _cumulative.items()}

    def as_dict(self) -> dict:
        """Return statistics as dict with mean/std for each name."""
        result = dict()
        for name, moments in (self._cumulative or {}).items():
            if self._regex.fullmatch(name):
                _check_moments(moments)
                count = max(moments[2].item(), 1)
                mean = moments[0].item() / count
                raw_var = moments[1].item() / count - mean ** 2
                std = np.sqrt(max(raw_var, 0))
                result[name] = dnnlib.EasyDict(mean=mean, std=std, count=moments[2].item())
        return result


def _check_moments(moments: torch.Tensor):
    """Validate moments tensor."""
    assert moments.ndim == 1 and moments.shape[0] == _num_moments


def _sync():
    """Synchronize counters across all ranks."""
    global _counters, _cumulative, _sync_called

    # Gather counters locally
    for name, values in _counters.items():
        if name not in _cumulative:
            _cumulative[name] = torch.zeros([_num_moments], dtype=_reduce_dtype, device="cpu")

        if len(values) > 0:
            flat = torch.cat(values)
            moments = torch.stack([flat.sum(), flat.square().sum(), torch.tensor(flat.numel(), dtype=_reduce_dtype, device=flat.device)])
            _cumulative[name] += moments.cpu()

    # Clear counters
    _counters.clear()

    # Sync across ranks if distributed
    if _sync_device is not None:
        for name, moments in _cumulative.items():
            moments_gpu = moments.to(_sync_device)
            torch.distributed.all_reduce(moments_gpu)
            _cumulative[name] = moments_gpu.cpu()

    _sync_called = True


import dnnlib
