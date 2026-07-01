"""
UniPC (Unified Predictor-Corrector) sampler for GaussianDiffusion noise schedules.

Thin adapter around diffusers' ``UniPCMultistepScheduler`` (Zhao et al., "UniPC:
A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models",
NeurIPS 2023). Like DPM-Solver++, UniPC subsamples the full 1000-step schedule
itself, so it is driven from the diffusion object's ``betas`` and only needs the
requested number of solver steps.

The DiffiT model predicts noise (eps); with a learned-sigma model the extra
variance channels are sliced off before the scheduler step, matching the eps
convention used in ``dpm_solver.py``.
"""

import numpy as np
import torch
from diffusers import UniPCMultistepScheduler
from tqdm import tqdm


def unipc_sample(
    model,
    diffusion,
    shape,
    device,
    num_steps=25,
    model_kwargs=None,
    noise=None,
    progress=False,
):
    """Sample from a diffusion model using UniPC.

    Parameters
    ----------
    model : callable
        The denoising model.  Called as ``model(x, t, **model_kwargs)`` where
        *t* is an integer timestep tensor (values in 0 .. T-1).
    diffusion : GaussianDiffusion or SpacedDiffusion
        Diffusion object that carries ``betas`` (numpy, length T).
    shape : tuple
        Shape of the sample tensor, e.g. ``(B, C, H, W)``.
    device : torch.device
        Device on which to run.
    num_steps : int
        Number of UniPC steps (default 25).
    model_kwargs : dict or None
        Extra kwargs forwarded to *model*.
    noise : Tensor or None
        Starting noise; if *None* a fresh sample is drawn.
    progress : bool
        If *True*, wrap the step loop with tqdm.

    Returns
    -------
    x : Tensor
        The final denoised sample, shape ``shape``.
    """
    if model_kwargs is None:
        model_kwargs = {}

    B = shape[0]
    C_in = shape[1]  # input channels (before learned sigma doubling)

    betas = np.asarray(diffusion.betas, dtype=np.float64)
    T = len(betas)

    scheduler = UniPCMultistepScheduler(
        num_train_timesteps=T,
        trained_betas=betas,
        prediction_type="epsilon",
        solver_order=2,
    )
    scheduler.set_timesteps(num_steps, device=device)

    if noise is not None:
        x = noise.to(device)
    else:
        x = torch.randn(*shape, device=device)
    # UniPC assumes the sample starts at the scheduler's init noise sigma.
    x = x * scheduler.init_noise_sigma

    timesteps = scheduler.timesteps
    if progress:
        timesteps = tqdm(timesteps, desc="UniPC", total=len(timesteps))

    for t in timesteps:
        t_int = int(t.item())
        t_tensor = torch.full((B,), t_int, device=device, dtype=torch.long)

        with torch.no_grad():
            model_output = model(x, t_tensor, **model_kwargs)

        eps = model_output[:, :C_in]
        x = scheduler.step(eps, t, x).prev_sample

    return x
