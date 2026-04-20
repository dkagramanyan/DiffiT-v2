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
from diffit.dpm_solver import dpm_solver_sample


class InceptionFeatureExtractor(torch.nn.Module):
    """Extract pool, spatial and logit features from InceptionV3."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        m = self.model
        x = m.Conv2d_1a_3x3(x)
        x = m.Conv2d_2a_3x3(x)
        x = m.Conv2d_2b_3x3(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        x = m.Conv2d_3b_1x1(x)
        x = m.Conv2d_4a_3x3(x)
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        x = m.Mixed_5b(x)
        x = m.Mixed_5c(x)
        x = m.Mixed_5d(x)
        x = m.Mixed_6a(x)
        x = m.Mixed_6b(x)
        spatial = x
        x = m.Mixed_6c(x)
        x = m.Mixed_6d(x)
        x = m.Mixed_6e(x)
        x = m.Mixed_7a(x)
        x = m.Mixed_7b(x)
        x = m.Mixed_7c(x)

        pool = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        logits = m.fc(pool)
        spatial = spatial.mean(dim=[-2, -1])
        return {"pool": pool, "spatial": spatial, "logits": logits}


@torch.inference_mode()
def compute_activations(images_uint8_nchw, extractor, batch_size, device, desc="Computing Inception features"):
    """Compute inception activations from NCHW uint8 numpy array."""
    all_pool, all_spatial, all_logits = [], [], []
    for i in tqdm(range(0, len(images_uint8_nchw), batch_size), desc=desc, unit="batch"):
        batch = torch.from_numpy(images_uint8_nchw[i : i + batch_size]).float().to(device) / 255.0
        feats = extractor(batch)
        all_pool.append(feats["pool"].cpu().numpy())
        all_spatial.append(feats["spatial"].cpu().numpy())
        all_logits.append(feats["logits"].cpu().numpy())
    return {
        "pool": np.concatenate(all_pool),
        "spatial": np.concatenate(all_spatial),
        "logits": np.concatenate(all_logits),
    }


@torch.inference_mode()
def generate_eval_samples(
    ema_model, vae, diffusion, num_samples, batch_gpu, latent_size, device,
    *,
    cfg_scale, num_sampling_steps=25, scale_pow=4.0,
    rank=0, world_size=1,
):
    """Generate N random images for metric evaluation (NCHW uint8 numpy)."""
    samples_per_rank = (num_samples + world_size - 1) // world_size
    local_target = min(samples_per_rank, num_samples - rank * samples_per_rank)
    local_target = max(local_target, 0)

    all_images = []
    generated = 0
    show_progress = (rank == 0)
    pbar = tqdm(total=local_target, desc=f"Generating eval samples (dpm++{num_sampling_steps})", unit="img") if show_progress else None
    while generated < local_target:
        bs = min(batch_gpu, local_target - generated)
        z = torch.randn(bs, 4, latent_size, latent_size, device=device)
        classes = torch.randint(0, NUM_CLASSES, (bs,), device=device)

        z_cfg = torch.cat([z, z], 0)
        classes_null = torch.full((bs,), NUM_CLASSES, device=device, dtype=torch.long)
        model_kwargs = {
            "y": torch.cat([classes, classes_null], 0),
            "cfg_scale": cfg_scale,
            "diffusion_steps": 1000,
            "scale_pow": scale_pow,
        }
        with torch.amp.autocast("cuda", dtype=torch.float16):
            sample = dpm_solver_sample(
                ema_model.forward_with_cfg,
                diffusion,
                z_cfg.shape,
                device,
                num_steps=num_sampling_steps,
                model_kwargs=model_kwargs,
                noise=z_cfg,
            )
            sample, _ = sample.chunk(2, dim=0)
            decoded = vae.decode(sample.float() / 0.18215).sample
        decoded = ((decoded + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        all_images.append(decoded.cpu().numpy())
        generated += bs
        if pbar is not None:
            pbar.update(bs)
    if pbar is not None:
        pbar.close()

    local_images = np.concatenate(all_images, axis=0)[:local_target] if all_images else np.empty((0, 3, 0, 0), dtype=np.uint8)

    if world_size > 1:
        local_tensor = torch.from_numpy(local_images).to(device)
        local_count = torch.tensor([local_tensor.shape[0]], device=device, dtype=torch.long)
        all_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
        dist.all_gather(all_counts, local_count)

        if rank == 0:
            max_count = max(c.item() for c in all_counts)
            if local_tensor.shape[0] < max_count:
                pad = torch.zeros(max_count - local_tensor.shape[0], *local_tensor.shape[1:], device=device, dtype=local_tensor.dtype)
                local_tensor = torch.cat([local_tensor, pad], 0)
            gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]
            dist.gather(local_tensor, gathered, dst=0)
            result = []
            for i, g in enumerate(gathered):
                result.append(g[:all_counts[i].item()].cpu().numpy())
            return np.concatenate(result, axis=0)[:num_samples]
        else:
            max_count = max(c.item() for c in all_counts)
            if local_tensor.shape[0] < max_count:
                pad = torch.zeros(max_count - local_tensor.shape[0], *local_tensor.shape[1:], device=device, dtype=local_tensor.dtype)
                local_tensor = torch.cat([local_tensor, pad], 0)
            dist.gather(local_tensor, dst=0)
            return None

    return local_images[:num_samples]


def compute_fid(acts1, acts2, eps=1e-6):
    """Frechet Inception Distance between two activation sets."""
    mu1, sigma1 = np.mean(acts1, axis=0), np.cov(acts1, rowvar=False)
    mu2, sigma2 = np.mean(acts2, axis=0), np.cov(acts2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
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
def compute_precision_recall(ref_acts, sample_acts, k=3, rank=0, world_size=1):
    """Precision and Recall via k-NN manifold estimation (multi-GPU accelerated)."""
    BATCH = 512

    if world_size > 1:
        if rank == 0:
            ref_t = torch.from_numpy(ref_acts).cuda()
            sample_t = torch.from_numpy(sample_acts).cuda()
            shapes = torch.tensor([ref_t.shape[0], ref_t.shape[1],
                                   sample_t.shape[0], sample_t.shape[1]], device="cuda")
        else:
            shapes = torch.zeros(4, dtype=torch.long, device="cuda")
        dist.broadcast(shapes, src=0)
        nr, d, ns, _ = shapes.tolist()
        if rank != 0:
            ref_t = torch.zeros(nr, d, device="cuda")
            sample_t = torch.zeros(ns, d, device="cuda")
        dist.broadcast(ref_t, src=0)
        dist.broadcast(sample_t, src=0)
    else:
        ref_t = torch.from_numpy(ref_acts).cuda()
        sample_t = torch.from_numpy(sample_acts).cuda()

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


@torch.inference_mode()
def evaluate_metrics(
    ema_model, vae, diffusion, ref_acts, inception_extractor,
    num_fid_samples, batch_gpu, latent_size, device,
    *,
    cfg_scale,
    rank=0, world_size=1,
    log_fn=None,
):
    """Generate samples across all ranks, compute metrics on rank 0.

    log_fn: optional callable(str) -> None. Defaults to a no-op.
    Returns dict of metrics on rank 0, None on other ranks.
    """
    if log_fn is None:
        log_fn = lambda *a, **k: None

    if rank == 0:
        log_fn(
            f"Evaluating metrics ({num_fid_samples} samples across "
            f"{world_size} GPU(s), cfg_scale={cfg_scale})..."
        )
    fake_images = generate_eval_samples(
        ema_model, vae, diffusion, num_fid_samples, batch_gpu, latent_size, device,
        cfg_scale=cfg_scale,
        rank=rank, world_size=world_size,
    )

    if rank == 0:
        fake_acts = compute_activations(fake_images, inception_extractor, batch_size=64, device=device)
        metrics = {}
        metrics["IS"] = compute_inception_score(fake_acts["logits"])
        metrics["FID"] = compute_fid(ref_acts["pool"], fake_acts["pool"])
        metrics["sFID"] = compute_fid(ref_acts["spatial"], fake_acts["spatial"])
        ref_pool = ref_acts["pool"]
        fake_pool = fake_acts["pool"]
    else:
        metrics = {}
        ref_pool = None
        fake_pool = None

    prec, rec = compute_precision_recall(
        ref_pool, fake_pool, k=3, rank=rank, world_size=world_size,
    )

    if rank == 0:
        metrics["Precision"] = prec
        metrics["Recall"] = rec
        for k, v in metrics.items():
            log_fn(f"  {k}: {v:.4f}")
        return metrics
    return None
