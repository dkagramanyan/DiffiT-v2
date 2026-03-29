"""
DPM-Solver++ (2nd-order multistep) sampler for GaussianDiffusion noise schedules.

Implements the DPM-Solver++(2M) algorithm from:
    Lu et al., "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion
    Probabilistic Models", NeurIPS 2022.

This uses the data-prediction (x0) formulation internally, converting from
the eps-prediction output of the model.
"""

import numpy as np
import torch
from tqdm import tqdm


def _log_snr_from_alpha_cumprod(alpha_cumprod):
    """Compute log-SNR = log(alpha_cumprod / (1 - alpha_cumprod))."""
    return np.log(alpha_cumprod / (1.0 - alpha_cumprod))


def _build_time_schedule(alpha_cumprod, num_steps):
    """Build a time schedule with uniformly-spaced log-SNR values.

    Returns an array of *continuous* timesteps in [0, T-1] (float64)
    corresponding to the endpoints of ``num_steps`` intervals that are
    equally spaced in log-SNR space.  The schedule goes from high noise
    (t = T-1) to low noise (t = 0).
    """
    T = len(alpha_cumprod)
    log_snr = _log_snr_from_alpha_cumprod(alpha_cumprod)

    # log-SNR at the two extremes of the schedule
    log_snr_max = log_snr[0]    # low noise (t=0)
    log_snr_min = log_snr[-1]   # high noise (t=T-1)

    # Uniform grid in log-SNR, from high noise to low noise
    target_log_snr = np.linspace(log_snr_min, log_snr_max, num_steps + 1)

    # Map each target log-SNR to a continuous timestep via interpolation
    # log_snr is *decreasing* in t, so we flip for np.interp (needs increasing x)
    ts_float = np.interp(target_log_snr, np.flip(log_snr), np.flip(np.arange(T, dtype=np.float64)))

    return ts_float  # shape (num_steps + 1,), from t_max to t_min


def _alpha_sigma_at_t(alpha_cumprod_torch, t_continuous):
    """Interpolate alpha and sigma at a continuous timestep.

    ``t_continuous`` is a float tensor of shape (B,) with values in [0, T-1].
    Returns sqrt(alpha_cumprod) and sqrt(1 - alpha_cumprod) at those times.
    """
    # Linear interpolation between the two nearest discrete timesteps
    t_low = t_continuous.long().clamp(0, len(alpha_cumprod_torch) - 1)
    t_high = (t_low + 1).clamp(0, len(alpha_cumprod_torch) - 1)
    frac = (t_continuous - t_low.float()).clamp(0, 1)

    ac_low = alpha_cumprod_torch[t_low]
    ac_high = alpha_cumprod_torch[t_high]
    alpha_cumprod_t = ac_low + frac * (ac_high - ac_low)

    alpha_t = alpha_cumprod_t.sqrt()
    sigma_t = (1.0 - alpha_cumprod_t).sqrt()
    return alpha_t, sigma_t, alpha_cumprod_t


def _predict_x0(model_output, x_t, alpha_t, sigma_t):
    """Convert eps-prediction to x0-prediction.

    model_output may have extra variance channels (learned sigma); we only
    use the first C channels for the noise prediction.
    """
    C = x_t.shape[1]
    eps = model_output[:, :C]
    # x_t = alpha_t * x_0 + sigma_t * eps  =>  x_0 = (x_t - sigma_t * eps) / alpha_t
    dims = [-1] + [1] * (x_t.ndim - 1)  # broadcast shape for (B, 1, 1, 1)
    alpha = alpha_t.view(*dims)
    sigma = sigma_t.view(*dims)
    return (x_t - sigma * eps) / alpha


def dpm_solver_sample(
    model,
    diffusion,
    shape,
    device,
    num_steps=25,
    model_kwargs=None,
    noise=None,
    progress=False,
):
    """Sample from a diffusion model using DPM-Solver++(2M).

    Parameters
    ----------
    model : callable
        The denoising model.  Called as ``model(x, t, **model_kwargs)``
        where *t* is an integer timestep tensor (values in 0 .. T-1).
    diffusion : GaussianDiffusion or SpacedDiffusion
        Diffusion object that carries ``alphas_cumprod`` (numpy, length T)
        and ``num_timesteps``.
    shape : tuple
        Shape of the sample tensor, e.g. ``(B, C, H, W)``.
    device : torch.device
        Device on which to run.
    num_steps : int
        Number of DPM-Solver++ steps (default 25).
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

    # --- Noise schedule from the *base* (1000-step) diffusion object -------
    # SpacedDiffusion re-indexes alphas_cumprod to its sub-set of timesteps,
    # but we want the *original* 1000-step schedule so that the log-SNR grid
    # covers the full range.  We reconstruct it from betas in the base case,
    # or access it directly.
    alpha_cumprod_np = np.cumprod(1.0 - diffusion.betas, axis=0) if not hasattr(
        diffusion, '_base_alphas_cumprod'
    ) else diffusion._base_alphas_cumprod
    # Actually, diffusion.alphas_cumprod always stores the schedule for the
    # timesteps this object works with.  For the *unspaced* (training)
    # diffusion object this is the full 1000-step schedule.  We just use it.
    alpha_cumprod_np = np.asarray(diffusion.alphas_cumprod, dtype=np.float64)
    T = len(alpha_cumprod_np)

    alpha_cumprod_torch = torch.from_numpy(alpha_cumprod_np).float().to(device)

    # --- Time schedule (log-SNR uniform) -----------------------------------
    t_schedule = _build_time_schedule(alpha_cumprod_np, num_steps)
    # t_schedule[0] = high noise  ...  t_schedule[-1] = low noise

    # --- Initial noise -----------------------------------------------------
    if noise is not None:
        x = noise.to(device)
    else:
        x = torch.randn(*shape, device=device)

    # --- DPM-Solver++(2M) loop --------------------------------------------
    x0_prev = None  # previous x0 prediction (for 2nd-order multistep)

    steps = range(num_steps)
    if progress:
        steps = tqdm(steps, desc="DPM-Solver++", total=num_steps)

    for i in steps:
        t_cur = t_schedule[i]       # current (higher noise)
        t_next = t_schedule[i + 1]  # next    (lower noise)

        # Integer timestep for the model (round to nearest discrete step)
        t_int = max(0, min(T - 1, round(t_cur)))
        t_tensor = torch.full((shape[0],), t_int, device=device, dtype=torch.long)

        # --- Model evaluation ---
        model_output = model(x, t_tensor, **model_kwargs)

        # --- Convert eps -> x0 ---
        alpha_t, sigma_t, _ = _alpha_sigma_at_t(alpha_cumprod_torch, torch.full((shape[0],), t_cur, device=device))
        x0_pred = _predict_x0(model_output, x, alpha_t, sigma_t)

        # --- Coefficients at the *next* timestep ---
        alpha_next, sigma_next, _ = _alpha_sigma_at_t(alpha_cumprod_torch, torch.full((shape[0],), t_next, device=device))

        # log-SNR values (lambda in the DPM-Solver++ paper)
        lambda_cur = 0.5 * torch.log(alpha_t ** 2 / sigma_t ** 2 + 1e-12).mean()
        lambda_next = 0.5 * torch.log(alpha_next ** 2 / sigma_next ** 2 + 1e-12).mean()

        # Reshape for broadcasting
        dims = [-1] + [1] * (x.ndim - 1)
        a_next = alpha_next.view(*dims)
        s_next = sigma_next.view(*dims)

        if i == 0 or x0_prev is None:
            # --- 1st-order update (DPM-Solver++1) for the first step ---
            x = (a_next / alpha_t.view(*dims)) * x - s_next * (torch.expm1(lambda_cur - lambda_next)) * x0_pred
        else:
            # --- 2nd-order multistep update (DPM-Solver++2M) ---
            lambda_prev = 0.5 * torch.log(alpha_prev_saved ** 2 / sigma_prev_saved ** 2 + 1e-12).mean()
            h = lambda_next - lambda_cur
            h_prev = lambda_cur - lambda_prev
            r = h_prev / h

            # D0 = x0_pred,  D1 = (1 + 1/(2r)) * x0_pred - (1/(2r)) * x0_prev
            D0 = x0_pred
            D1 = (1.0 + 0.5 / r) * x0_pred - (0.5 / r) * x0_prev

            x = (a_next / alpha_t.view(*dims)) * x - s_next * torch.expm1(lambda_cur - lambda_next) * D1

        # Save for next multistep
        x0_prev = x0_pred
        alpha_prev_saved = alpha_t
        sigma_prev_saved = sigma_t

    return x
