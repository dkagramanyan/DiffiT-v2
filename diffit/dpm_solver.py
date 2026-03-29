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


def _build_time_schedule(alpha_cumprod, num_steps):
    """Build a time schedule with uniformly-spaced log-SNR values.

    Returns an array of integer timesteps of length ``num_steps + 1``,
    going from t = T-1 (high noise) to t = 0 (low noise).
    """
    T = len(alpha_cumprod)
    log_snr = np.log(alpha_cumprod / (1.0 - alpha_cumprod))

    # Uniform grid in log-SNR, from high noise (min log-SNR) to low noise (max log-SNR)
    target_log_snr = np.linspace(log_snr[-1], log_snr[0], num_steps + 1)

    # Map each target log-SNR to a continuous timestep via interpolation.
    # log_snr is *decreasing* in t, so flip for np.interp (needs increasing x).
    ts = np.interp(
        target_log_snr,
        np.flip(log_snr),
        np.flip(np.arange(T, dtype=np.float64)),
    )
    return ts  # shape (num_steps + 1,), from t_max to t_min


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
        Diffusion object that carries ``alphas_cumprod`` (numpy, length T).
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

    B = shape[0]
    C_in = shape[1]  # input channels (before learned sigma doubling)

    # --- Noise schedule (full 1000-step) ----------------------------------
    alpha_cumprod_np = np.asarray(diffusion.alphas_cumprod, dtype=np.float64)
    T = len(alpha_cumprod_np)
    ac = torch.from_numpy(alpha_cumprod_np).float().to(device)

    # Precompute useful quantities
    # lambda(t) = log(alpha(t) / sigma(t))
    log_alpha = 0.5 * torch.log(ac)
    log_sigma = 0.5 * torch.log(1.0 - ac)
    lam = log_alpha - log_sigma  # log-SNR, decreasing in t

    def get_coeffs(t_continuous):
        """Get alpha, sigma, lambda at a continuous timestep (scalar)."""
        t_low = int(max(0, min(T - 1, int(t_continuous))))
        t_high = min(t_low + 1, T - 1)
        frac = t_continuous - t_low
        frac = max(0.0, min(1.0, frac))

        ac_t = ac[t_low] + frac * (ac[t_high] - ac[t_low])
        la_t = log_alpha[t_low] + frac * (log_alpha[t_high] - log_alpha[t_low])
        ls_t = log_sigma[t_low] + frac * (log_sigma[t_high] - log_sigma[t_low])
        lam_t = la_t - ls_t
        alpha_t = la_t.exp()
        sigma_t = ls_t.exp()
        return alpha_t, sigma_t, lam_t

    def predict_x0(model_out, x_t, alpha_t, sigma_t):
        """Convert eps-prediction to x0-prediction."""
        eps = model_out[:, :C_in]
        return (x_t - sigma_t * eps) / alpha_t

    # --- Time schedule (log-SNR uniform) ----------------------------------
    t_schedule = _build_time_schedule(alpha_cumprod_np, num_steps)

    # --- Initial noise ----------------------------------------------------
    if noise is not None:
        x = noise.to(device)
    else:
        x = torch.randn(*shape, device=device)

    # --- DPM-Solver++(2M) loop -------------------------------------------
    x0_prev = None
    lam_prev = None

    steps = range(num_steps)
    if progress:
        steps = tqdm(steps, desc="DPM-Solver++", total=num_steps)

    for i in steps:
        t_cur = t_schedule[i]
        t_next = t_schedule[i + 1]

        # Integer timestep for the model
        t_int = max(0, min(T - 1, round(t_cur)))
        t_tensor = torch.full((B,), t_int, device=device, dtype=torch.long)

        # Model evaluation
        with torch.no_grad():
            model_output = model(x, t_tensor, **model_kwargs)

        # Get coefficients
        alpha_cur, sigma_cur, lam_cur = get_coeffs(t_cur)
        alpha_next, sigma_next, lam_next = get_coeffs(t_next)

        # Predict x0 from eps
        x0_pred = predict_x0(model_output, x, alpha_cur, sigma_cur)

        h = lam_next - lam_cur  # step size in lambda space (negative since lam decreases)

        if i == 0 or x0_prev is None:
            # 1st-order update (DPM-Solver++1):
            # x_{s} = (sigma_s / sigma_t) * x_t + alpha_s * (1 - exp(-(h))) * x0
            # where h = lam_next - lam_cur (negative)
            x = (sigma_next / sigma_cur) * x + alpha_next * (1.0 - torch.exp(h)) * x0_pred
        else:
            # 2nd-order multistep update (DPM-Solver++2M):
            h_prev = lam_cur - lam_prev
            r = h_prev / h

            # Corrected x0 prediction using linear extrapolation
            D1 = (1.0 + 0.5 / r) * x0_pred - (0.5 / r) * x0_prev

            x = (sigma_next / sigma_cur) * x + alpha_next * (1.0 - torch.exp(h)) * D1

        # Save for next step
        x0_prev = x0_pred
        lam_prev = lam_cur

    return x
