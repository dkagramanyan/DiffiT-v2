"""Inline evaluation metrics (IS, FID, sFID, Precision, Recall).

Shared by train.py and experiments/train_sample_split.py.
"""
import warnings

import numpy as np
import torch
import torch.distributed as dist
from scipy import linalg
from tqdm import tqdm

from diffit import NUM_CLASSES
from diffit.constants import PIXEL_NORM_HALF, UINT8_MAX, VAE_SCALE_FACTOR
from diffit.dpm_solver import dpm_solver_sample
from diffit.unipc_solver import unipc_sample

# Optional combra integration: score generated samples with combra's
# generative-quality metrics during training. The import is guarded so training
# runs unchanged when combra is not installed.
#
# Only what this module calls is imported: the sharded harness itself now lives in
# `combra.metrics.distributed`, so the feature extractors and distance helpers it
# needs are combra's dependency, covered by combra's own tests.
try:
    from combra.metrics.distributed import (
        all_ranks_ok as _combra_all_ranks_ok,
        distributed_metrics as _combra_distributed_metrics_impl,
        gather_generated as _combra_gather_generated,
        precompute_reference as _combra_precompute_reference,
    )

    HAS_COMBRA = True
    COMBRA_IMPORT_ERROR = None
except ImportError as _combra_exc:
    _combra_precompute_reference = _combra_gather_generated = None
    _combra_distributed_metrics_impl = _combra_all_ranks_ok = None
    HAS_COMBRA = False
    # Keep the reason. "combra is not installed" is the wrong diagnosis when combra
    # IS installed but has moved a symbol -- that misdirection is exactly how this
    # integration stayed broken for a whole combra release. train.py prints this.
    COMBRA_IMPORT_ERROR = _combra_exc

# combra metric keys are bare: combra_fid, combra_cmmd, combra_fd_dinov2. They used
# to carry a "10k" suffix that stayed 10k whatever --num-fid-samples said, so every
# chart built from them was mislabelled. The sample count is logged once instead, as
# Metrics/combra_num_fid_samples.


@torch.inference_mode()
def compute_activations(images_uint8_nchw, extractor, batch_size, device, desc="Computing Inception features"):
    """Compute inception activations from NCHW uint8 numpy array."""
    all_pool, all_spatial, all_logits = [], [], []
    for i in tqdm(range(0, len(images_uint8_nchw), batch_size), desc=desc, unit="batch"):
        batch = torch.from_numpy(images_uint8_nchw[i : i + batch_size]).float().to(device) / UINT8_MAX
        feats = extractor(batch)
        all_pool.append(feats["pool"].cpu().numpy())
        all_spatial.append(feats["spatial"].cpu().numpy())
        all_logits.append(feats["logits"].cpu().numpy())
    return {
        "pool": np.concatenate(all_pool),
        "spatial": np.concatenate(all_spatial),
        "logits": np.concatenate(all_logits),
    }


def sample_latents(model_fn, diffusion, shape, device, *, sampler, num_steps, model_kwargs, noise, progress=False):
    """Dispatch latent sampling to the chosen reverse-diffusion sampler.

    sampler: "dpm++" runs DPM-Solver++(2M) and "unipc" runs UniPC on the full
        schedule (``diffusion`` must carry the full 1000-step ``alphas_cumprod`` /
        ``betas``; ``num_steps`` sets the solver steps). "ddim" / "ddpm" use the
        native deterministic-DDIM / ancestral-DDPM loops, in which case
        ``diffusion`` must be a SpacedDiffusion whose ``num_timesteps`` already
        encodes the step count (``num_steps`` is ignored). DDIM uses the default
        ``eta=0`` (deterministic).
    """
    if sampler == "dpm++":
        return dpm_solver_sample(
            model_fn, diffusion, shape, device,
            num_steps=num_steps, model_kwargs=model_kwargs, noise=noise, progress=progress,
        )
    if sampler == "unipc":
        return unipc_sample(
            model_fn, diffusion, shape, device,
            num_steps=num_steps, model_kwargs=model_kwargs, noise=noise, progress=progress,
        )
    sample_fn = diffusion.ddim_sample_loop if sampler == "ddim" else diffusion.p_sample_loop
    return sample_fn(
        model_fn, shape, noise=noise, clip_denoised=False,
        model_kwargs=model_kwargs, device=device, progress=progress,
    )


@torch.inference_mode()
def _generate_local_shard(
    ema_model, vae, diffusion, num_samples, batch_gpu, latent_size, device,
    *,
    cfg_scale, num_sampling_steps=25, scale_pow=4.0, sampler="dpm++",
    rank=0, world_size=1,
    class_list=None, null_class_idx=None,
):
    """Generate this rank's shard of the eval samples (NCHW uint8 numpy).

    The ``num_samples`` are split evenly across ranks; each rank returns only the
    images it generated. Use :func:`_gather_eval_images` to collect them on rank 0.
    """
    if class_list is None:
        class_list = list(range(NUM_CLASSES))
    if null_class_idx is None:
        null_class_idx = NUM_CLASSES
    class_tensor = torch.tensor(class_list, device=device, dtype=torch.long)

    samples_per_rank = (num_samples + world_size - 1) // world_size
    local_target = min(samples_per_rank, num_samples - rank * samples_per_rank)
    local_target = max(local_target, 0)

    all_images = []
    generated = 0
    show_progress = (rank == 0)
    pbar = tqdm(total=local_target, desc=f"Generating eval samples ({sampler}×{num_sampling_steps})", unit="img") if show_progress else None
    while generated < local_target:
        bs = min(batch_gpu, local_target - generated)
        z = torch.randn(bs, 4, latent_size, latent_size, device=device)
        classes = class_tensor[torch.randint(0, len(class_tensor), (bs,), device=device)]

        z_cfg = torch.cat([z, z], 0)
        classes_null = torch.full((bs,), null_class_idx, device=device, dtype=torch.long)
        model_kwargs = {
            "y": torch.cat([classes, classes_null], 0),
            "cfg_scale": cfg_scale,
            "diffusion_steps": 1000,
            "scale_pow": scale_pow,
        }
        with torch.amp.autocast("cuda", dtype=torch.float16):
            sample = sample_latents(
                ema_model.forward_with_cfg,
                diffusion,
                z_cfg.shape,
                device,
                sampler=sampler,
                num_steps=num_sampling_steps,
                model_kwargs=model_kwargs,
                noise=z_cfg,
            )
            sample, _ = sample.chunk(2, dim=0)
            decoded = vae.decode(sample.float() / VAE_SCALE_FACTOR).sample
        decoded = ((decoded + 1) * PIXEL_NORM_HALF).clamp(0, UINT8_MAX).to(torch.uint8)
        all_images.append(decoded.cpu().numpy())
        generated += bs
        if pbar is not None:
            pbar.update(bs)
    if pbar is not None:
        pbar.close()

    return np.concatenate(all_images, axis=0)[:local_target] if all_images else np.empty((0, 3, 0, 0), dtype=np.uint8)


def _gather_eval_images(local_images, device, rank, world_size, num_samples):
    """Gather per-rank image shards to rank 0 (full set on rank 0, None elsewhere).

    Ranks may hold different counts, so each shard is padded to the max before the
    collective gather and trimmed back to its true length on rank 0.
    """
    if world_size == 1:
        return local_images[:num_samples]

    local_tensor = torch.from_numpy(local_images).to(device)
    local_count = torch.tensor([local_tensor.shape[0]], device=device, dtype=torch.long)
    all_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
    dist.all_gather(all_counts, local_count)

    max_count = max(c.item() for c in all_counts)
    if local_tensor.shape[0] < max_count:
        pad = torch.zeros(max_count - local_tensor.shape[0], *local_tensor.shape[1:], device=device, dtype=local_tensor.dtype)
        local_tensor = torch.cat([local_tensor, pad], 0)

    if rank == 0:
        gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]
        dist.gather(local_tensor, gathered, dst=0)
        result = [g[:all_counts[i].item()].cpu().numpy() for i, g in enumerate(gathered)]
        return np.concatenate(result, axis=0)[:num_samples]
    dist.gather(local_tensor, dst=0)
    return None


@torch.inference_mode()
def generate_eval_samples(
    ema_model, vae, diffusion, num_samples, batch_gpu, latent_size, device,
    *,
    cfg_scale, num_sampling_steps=25, scale_pow=4.0, sampler="dpm++",
    rank=0, world_size=1,
    class_list=None, null_class_idx=None,
):
    """Generate N random images for metric evaluation (NCHW uint8 numpy).

    Splits generation across ranks and gathers the full set to rank 0 (None on
    other ranks). class_list: list of int class IDs to sample from. Defaults to
    range(NUM_CLASSES). null_class_idx: CFG null-token embedding index. Defaults
    to NUM_CLASSES (matches a model built with num_classes=NUM_CLASSES).
    """
    local_images = _generate_local_shard(
        ema_model, vae, diffusion, num_samples, batch_gpu, latent_size, device,
        cfg_scale=cfg_scale, num_sampling_steps=num_sampling_steps, scale_pow=scale_pow,
        sampler=sampler, rank=rank, world_size=world_size,
        class_list=class_list, null_class_idx=null_class_idx,
    )
    return _gather_eval_images(local_images, device, rank, world_size, num_samples)


def compute_fid(acts1, acts2, eps=1e-6):
    """Frechet Inception Distance between two activation sets."""
    mu1, sigma1 = np.mean(acts1, axis=0), np.cov(acts1, rowvar=False)
    mu2, sigma2 = np.mean(acts2, axis=0), np.cov(acts2, rowvar=False)
    diff = mu1 - mu2
    # scipy >= 1.18 removed sqrtm's `disp` parameter. Calling it without disp
    # returns the matrix alone on every version, so this works either side of 1.18.
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        warnings.warn("FID: singular product, adding eps to diagonal")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def compute_inception_score(logits, split_size=5000):
    """Inception Score from classifier logits."""
    from scipy.special import softmax
    preds = softmax(logits, axis=1)
    scores = []
    for i in range(0, len(preds), split_size):
        part = preds[i : i + split_size]
        kl = part * (np.log(part + 1e-10) - np.log(np.mean(part, 0, keepdims=True) + 1e-10))
        scores.append(np.exp(np.mean(np.sum(kl, 1))))
    return float(np.mean(scores))


@torch.inference_mode()
def compute_precision_recall(ref_acts, sample_acts, k=3, rank=0, world_size=1, device=None):
    """Precision and Recall via k-NN manifold estimation (multi-GPU accelerated).

    ``device`` selects where the k-NN tensors live; defaults to CUDA to preserve
    the original behaviour, but can be overridden (e.g. CPU) for testing.
    """
    BATCH = 512
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if world_size > 1:
        if rank == 0:
            ref_t = torch.from_numpy(ref_acts).to(device)
            sample_t = torch.from_numpy(sample_acts).to(device)
            shapes = torch.tensor([ref_t.shape[0], ref_t.shape[1],
                                   sample_t.shape[0], sample_t.shape[1]], device=device)
        else:
            shapes = torch.zeros(4, dtype=torch.long, device=device)
        dist.broadcast(shapes, src=0)
        nr, d, ns, _ = shapes.tolist()
        if rank != 0:
            ref_t = torch.zeros(nr, d, device=device)
            sample_t = torch.zeros(ns, d, device=device)
        dist.broadcast(ref_t, src=0)
        dist.broadcast(sample_t, src=0)
    else:
        ref_t = torch.from_numpy(ref_acts).to(device)
        sample_t = torch.from_numpy(sample_acts).to(device)

    device = ref_t.device

    def knn_radii(feats, k, rank, world_size):
        n = feats.shape[0]
        radii = torch.zeros(n, device=device)
        indices = list(range(0, n, BATCH))
        my_indices = indices[rank::world_size]
        for i in my_indices:
            end = min(i + BATCH, n)
            dists_sq = torch.cdist(feats[i:end], feats).square()
            radii[i:end] = torch.kthvalue(dists_sq, k + 1, dim=1).values
        if world_size > 1:
            dist.all_reduce(radii, op=dist.ReduceOp.SUM)
        return radii

    def manifold_coverage(ref_f, ref_r, eval_f, rank, world_size):
        n = eval_f.shape[0]
        local_count = 0
        indices = list(range(0, n, BATCH))
        my_indices = indices[rank::world_size]
        for i in my_indices:
            end = min(i + BATCH, n)
            dists_sq = torch.cdist(eval_f[i:end], ref_f).square()
            local_count += int((dists_sq <= ref_r.unsqueeze(0)).any(dim=1).sum())
        count_t = torch.tensor([local_count], device=device, dtype=torch.long)
        if world_size > 1:
            dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
        return count_t.item() / n

    ref_r = knn_radii(ref_t, k, rank, world_size)
    sample_r = knn_radii(sample_t, k, rank, world_size)
    precision = manifold_coverage(ref_t, ref_r, sample_t, rank, world_size)
    recall = manifold_coverage(sample_t, sample_r, ref_t, rank, world_size)
    return precision, recall


# The shard -> extract -> gather -> distance harness lives in combra
# (`combra.metrics.distributed`), shared by all four model repos. What stays in this
# file is DiffiT-specific: the Inception suite, and generating a shard of fakes.

precompute_combra_reference = _combra_precompute_reference
# Re-exported so train.py can agree across ranks before a collective without importing
# combra directly (it must keep working with combra absent).
all_ranks_ok = _combra_all_ranks_ok




@torch.inference_mode()
def evaluate_metrics(
    ema_model, vae, diffusion, ref_acts, inception_extractor,
    num_fid_samples, batch_gpu, latent_size, device,
    *,
    cfg_scale,
    sampler="dpm++", num_sampling_steps=25,
    rank=0, world_size=1,
    log_fn=None,
    class_list=None, null_class_idx=None,
    combra_ref=None,
    inception_metrics=True,
):
    """Generate samples across all ranks, compute metrics on rank 0.

    class_list / null_class_idx: forwarded to generate_eval_samples. See there.

    combra_ref: the precomputed sharded reference from
        :func:`precompute_combra_reference` (rank 0 only; ``None`` elsewhere). When
        combra is active (``inception_metrics=False`` and combra installed) the
        generated batch is scored with combra's ``fid`` / ``cmmd`` / ``fd_dinov2``
        image-feature metrics plus the angle-density / Gaussian-fit metrics, merged
        into the returned dict under ``combra_*`` keys. Both the angle and the
        feature extraction are sharded: every rank extracts from its own generated
        shard and the rows are gathered to rank 0, so all GPUs share that work and
        the (cached) reference side never recomputes.

    inception_metrics: when True (default) compute the InceptionV3-based suite
        (IS / FID / sFID / Precision / Recall). Set False to skip it entirely --
        the skip is collective (it also bypasses the multi-rank
        ``compute_precision_recall`` all-reduce, so every rank must pass the same
        value) and leaves only the combra metrics. ``inception_extractor`` /
        ``ref_acts`` may be None in that case. ``inception_metrics`` is the uniform
        per-rank signal for "combra active" (combra_ref is rank-0 only), which the
        sharded angle/feature gathers rely on for matched collectives across ranks.

    Returns dict of metrics on rank 0, None on other ranks.
    """
    if log_fn is None:
        log_fn = lambda *a, **k: None

    if rank == 0:
        n_cls = len(class_list) if class_list is not None else NUM_CLASSES
        log_fn(
            f"Evaluating metrics ({num_fid_samples} samples across "
            f"{world_size} GPU(s), sampler={sampler}×{num_sampling_steps}, "
            f"cfg_scale={cfg_scale}, classes={n_cls})..."
        )
    local_fakes = _generate_local_shard(
        ema_model, vae, diffusion, num_fid_samples, batch_gpu, latent_size, device,
        cfg_scale=cfg_scale,
        num_sampling_steps=num_sampling_steps, sampler=sampler,
        rank=rank, world_size=world_size,
        class_list=class_list, null_class_idx=null_class_idx,
    )

    # combra metrics are sharded across ranks. The signal must be uniform across
    # ranks (combra_ref is rank-0 only), so key off inception_metrics, which train.py
    # sets collectively. Every rank extracts the image features and the pooled angles
    # from its own generated shard and the rows are gathered to rank 0.
    combra_active = (not inception_metrics) and HAS_COMBRA
    gen_feats, gen_angles = (
        _combra_gather_generated(local_fakes, device, rank, world_size)
        if combra_active else (None, None)
    )

    # Gather the full generated set to rank 0 only for the Inception suite (the
    # combra path uses the sharded gen angles/features above, so it needs no central
    # copy of the raw high-res images).
    fake_images = (
        _gather_eval_images(local_fakes, device, rank, world_size, num_fid_samples)
        if inception_metrics else None
    )

    if rank == 0:
        metrics = {}
        if inception_metrics:
            fake_acts = compute_activations(fake_images, inception_extractor, batch_size=64, device=device)
            metrics["IS"] = compute_inception_score(fake_acts["logits"])
            metrics["FID"] = compute_fid(ref_acts["pool"], fake_acts["pool"])
            metrics["sFID"] = compute_fid(ref_acts["spatial"], fake_acts["spatial"])
            ref_pool = ref_acts["pool"]
            fake_pool = fake_acts["pool"]
    else:
        metrics = {}
        ref_pool = None
        fake_pool = None

    if inception_metrics:
        prec, rec = compute_precision_recall(
            ref_pool, fake_pool, k=3, rank=rank, world_size=world_size, device=device,
        )
        if rank == 0:
            metrics["Precision"] = prec
            metrics["Recall"] = rec

    if rank == 0:
        if combra_active and combra_ref is not None:
            try:
                # device= so the CMMD reduction runs where the features were
                # extracted; left unset it resolves independently.
                combra_metrics = _combra_distributed_metrics_impl(
                    combra_ref, gen_angles, gen_feats, device=device,
                )
                for k, v in combra_metrics.items():
                    metrics[f"combra_{k}"] = float(v)
                metrics["combra_num_fid_samples"] = float(num_fid_samples)
            except Exception as e:  # combra failure must not abort the eval tick
                log_fn(f"  combra metrics failed: {e}")
        for k, v in metrics.items():
            log_fn(f"  {k}: {v:.4f}")
        return metrics
    return None
