# Copyright (c) 2024, DiffiT authors.
# Adapted from NVIDIA StyleGAN3 for diffusion models.
#
# Main API for computing and reporting quality metrics.

from __future__ import annotations

import os
import time
import json
import torch

import dnnlib
from . import metric_utils
from . import frechet_inception_distance


# Registry of available metrics
_metric_dict = dict()


def register_metric(fn):
    """Decorator to register a metric function."""
    assert callable(fn)
    _metric_dict[fn.__name__] = fn
    return fn


def is_valid_metric(metric):
    """Check if a metric name is valid."""
    return metric in _metric_dict


def list_valid_metrics():
    """List all valid metric names."""
    return list(_metric_dict.keys())


def calc_metric(metric, **kwargs):
    """Calculate a metric.
    
    Args:
        metric: Name of the metric to compute.
        **kwargs: Arguments passed to MetricOptions.
    
    Returns:
        EasyDict with results, metric name, timing, etc.
    """
    assert is_valid_metric(metric), f"Unknown metric: {metric}. Valid metrics: {list_valid_metrics()}"
    opts = metric_utils.MetricOptions(**kwargs)

    # Calculate
    start_time = time.time()
    results = _metric_dict[metric](opts)
    total_time = time.time() - start_time

    # Broadcast results across GPUs
    for key, value in list(results.items()):
        if opts.num_gpus > 1:
            value = torch.as_tensor(value, dtype=torch.float64, device=opts.device)
            torch.distributed.broadcast(tensor=value, src=0)
            value = float(value.cpu())
        results[key] = value

    # Decorate with metadata
    return dnnlib.EasyDict(
        results=dnnlib.EasyDict(results),
        metric=metric,
        total_time=total_time,
        total_time_str=dnnlib.util.format_time(total_time),
        num_gpus=opts.num_gpus,
    )


def report_metric(result_dict, run_dir=None, snapshot_pkl=None):
    """Report metric results to console and file.
    
    Args:
        result_dict: Result from calc_metric().
        run_dir: Directory to save results.
        snapshot_pkl: Name of the model snapshot.
    """
    metric = result_dict['metric']
    assert is_valid_metric(metric)
    
    if run_dir is not None and snapshot_pkl is not None:
        snapshot_pkl = os.path.relpath(snapshot_pkl, run_dir)

    jsonl_line = json.dumps(dict(result_dict, snapshot_pkl=snapshot_pkl, timestamp=time.time()))
    print(jsonl_line)
    
    if run_dir is not None and os.path.isdir(run_dir):
        with open(os.path.join(run_dir, f'metric-{metric}.jsonl'), 'at') as f:
            f.write(jsonl_line + '\n')


# ============================================================================
# Registered Metrics
# ============================================================================

@register_metric
def fid50k_full(opts):
    """FID with 50k generated images against full dataset."""
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=50000)
    return dict(fid50k_full=fid)


@register_metric
def fid10k_full(opts):
    """FID with 10k generated images against full dataset."""
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=10000)
    return dict(fid10k_full=fid)


@register_metric
def fid5k(opts):
    """FID with 5k generated images (faster for validation)."""
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=5000, num_gen=5000)
    return dict(fid5k=fid)


@register_metric
def fid2k(opts):
    """FID with 2k generated images (fast validation)."""
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=2000, num_gen=2000)
    return dict(fid2k=fid)


@register_metric
def fid1k(opts):
    """FID with 1k generated images (very fast validation)."""
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=1000, num_gen=1000)
    return dict(fid1k=fid)
