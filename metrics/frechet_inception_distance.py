# Copyright (c) 2024, DiffiT authors.
# Adapted from NVIDIA StyleGAN3 for diffusion models.
#
# Frechet Inception Distance (FID) from the paper
# "GANs trained by a two time-scale update rule converge to a local Nash equilibrium"
# Matches the original implementation by Heusel et al.

from __future__ import annotations

import numpy as np
import scipy.linalg
from . import metric_utils


# Inception network URL (TorchScript version)
INCEPTION_URL = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'


def compute_fid(opts, max_real, num_gen):
    """Compute Frechet Inception Distance between real and generated images.
    
    Args:
        opts: MetricOptions with diffusion model and dataset settings.
        max_real: Maximum number of real images to use (None = all).
        num_gen: Number of generated images to use.
    
    Returns:
        FID score (lower is better, 0 = identical distributions).
    """
    detector_url = INCEPTION_URL
    detector_kwargs = dict(return_features=True)  # Return raw features before softmax

    # Compute statistics for real images
    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, 
        detector_url=detector_url, 
        detector_kwargs=detector_kwargs,
        rel_lo=0, 
        rel_hi=0.5, 
        capture_mean_cov=True, 
        max_items=max_real
    ).get_mean_cov()

    # Compute statistics for generated images
    mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_diffusion(
        opts=opts, 
        detector_url=detector_url, 
        detector_kwargs=detector_kwargs,
        rel_lo=0.5, 
        rel_hi=1.0, 
        capture_mean_cov=True, 
        max_items=num_gen
    ).get_mean_cov()

    if opts.rank != 0:
        return float('nan')

    # Compute FID
    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    
    return float(fid)
