"""
Train DiffiT: Diffusion Vision Transformers for Image Generation.

Uses PyTorch DDP for multi-GPU training with the same interface style
as StyleGAN-XL / SAN-v2.  The --cfg flag selects a base configuration
that sets model, resolution, learning rate, diffusion settings, etc.
Individual options can still override any preset value.

Usage (multi-GPU):
    python train.py --outdir ./training-runs \
        --cfg diffit-256 \
        --data ./datasets/imagenet_256x256.zip \
        --gpus 4 --batch-gpu 64

Single GPU:
    python train.py --outdir ./training-runs \
        --cfg diffit-256 \
        --data ./datasets/imagenet_256x256.zip \
        --gpus 1 --batch-gpu 64
"""

import copy
import json
import os
import re
import tempfile
import time
import warnings

import click
import numpy as np
import PIL.Image
import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers.models import AutoencoderKL
from scipy import linalg
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import diffit.diffit as diffit_module
from diffit import create_diffusion, diffusion_defaults, NUM_CLASSES
from diffit.dist_util import setup_dist, dev, get_rank, get_world_size, synchronize
from diffit.image_datasets import load_data
from diffit.fp16_util import MixedPrecisionTrainer
from diffit.nn import update_ema
from diffit import logger
from diffit.timestep_sampler import create_named_schedule_sampler


# ---------------------------------------------------------------------------
# Image grid utilities (matching SAN-v2 style)
# ---------------------------------------------------------------------------


def save_image_grid(img, fname, drange, grid_size):
    """Save a grid of images as a single PNG (SAN-v2 style)."""
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], "L").save(fname)
    if C == 3:
        PIL.Image.fromarray(img, "RGB").save(fname)


def setup_snapshot_image_grid(image_size, gw=None, gh=None):
    """Determine grid dimensions for snapshot images."""
    if gw is None:
        gw = max(np.clip(2560 // image_size, 7, 32), 1)
    if gh is None:
        gh = max(np.clip(1440 // image_size, 4, 32), 1)
    return gw, gh


@torch.no_grad()
def generate_snapshot_images(
    ema_model, vae, diffusion, grid_z, grid_classes, batch_gpu, device,
    cfg_scale=4.4, num_sampling_steps=50, scale_pow=4.0,
):
    """Generate a batch of images from the EMA model for snapshot grids."""
    diff_config = diffusion_defaults()
    diff_config["timestep_respacing"] = f"ddim{num_sampling_steps}"
    snap_diffusion = create_diffusion(**diff_config)

    all_samples = []
    for z_chunk, c_chunk in zip(grid_z.split(batch_gpu), grid_classes.split(batch_gpu)):
        bs = z_chunk.shape[0]
        z_cfg = torch.cat([z_chunk, z_chunk], 0)
        classes_null = torch.full((bs,), NUM_CLASSES, device=device, dtype=torch.long)
        model_kwargs = {
            "y": torch.cat([c_chunk, classes_null], 0),
            "cfg_scale": cfg_scale,
            "diffusion_steps": diff_config["diffusion_steps"],
            "scale_pow": scale_pow,
        }
        sample = snap_diffusion.ddim_sample_loop(
            ema_model.forward_with_cfg,
            z_cfg.shape,
            z_cfg,
            clip_denoised=False,
            progress=False,
            model_kwargs=model_kwargs,
            device=device,
        )
        sample, _ = sample.chunk(2, dim=0)
        # Decode latent -> image
        decoded = vae.decode(sample / 0.18215).sample
        decoded = ((decoded + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        all_samples.append(decoded.permute(0, 2, 3, 1).cpu().numpy())

    # Return as NCHW uint8 for save_image_grid
    images = np.concatenate(all_samples, axis=0)
    # Convert NHWC -> NCHW
    images = images.transpose(0, 3, 1, 2)
    return images


# ---------------------------------------------------------------------------
# Inline evaluation metrics (IS, FID, sFID, Precision, Recall)
# ---------------------------------------------------------------------------


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


@torch.no_grad()
def compute_activations(images_uint8_nchw, extractor, batch_size, device):
    """Compute inception activations from NCHW uint8 numpy array."""
    all_pool, all_spatial, all_logits = [], [], []
    for i in range(0, len(images_uint8_nchw), batch_size):
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


@torch.no_grad()
def generate_eval_samples(
    ema_model, vae, diffusion, num_samples, batch_gpu, latent_size, device,
    cfg_scale=4.4, num_sampling_steps=50, scale_pow=4.0,
):
    """Generate N random images for metric evaluation (NCHW uint8 numpy)."""
    diff_config = diffusion_defaults()
    diff_config["timestep_respacing"] = f"ddim{num_sampling_steps}"
    eval_diffusion = create_diffusion(**diff_config)

    all_images = []
    generated = 0
    while generated < num_samples:
        bs = min(batch_gpu, num_samples - generated)
        z = torch.randn(bs, 4, latent_size, latent_size, device=device)
        classes = torch.randint(0, NUM_CLASSES, (bs,), device=device)

        z_cfg = torch.cat([z, z], 0)
        classes_null = torch.full((bs,), NUM_CLASSES, device=device, dtype=torch.long)
        model_kwargs = {
            "y": torch.cat([classes, classes_null], 0),
            "cfg_scale": cfg_scale,
            "diffusion_steps": diff_config["diffusion_steps"],
            "scale_pow": scale_pow,
        }
        sample = eval_diffusion.ddim_sample_loop(
            ema_model.forward_with_cfg, z_cfg.shape, z_cfg,
            clip_denoised=False, progress=False,
            model_kwargs=model_kwargs, device=device,
        )
        sample, _ = sample.chunk(2, dim=0)
        decoded = vae.decode(sample / 0.18215).sample
        decoded = ((decoded + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        all_images.append(decoded.cpu().numpy())  # NCHW
        generated += bs

    return np.concatenate(all_images, axis=0)[:num_samples]


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


def compute_precision_recall(ref_acts, sample_acts, k=3):
    """Precision and Recall via k-NN manifold estimation."""
    def knn_radii(feats, k):
        n = len(feats)
        radii = np.zeros(n, dtype=np.float32)
        for i in range(0, n, 10000):
            batch = feats[i : i + 10000]
            dists = np.sum((batch[:, None, :] - feats[None, :, :]) ** 2, axis=-1)
            radii[i : i + 10000] = np.partition(dists, k + 1, axis=1)[:, k]
        return radii

    def manifold_coverage(ref_f, ref_r, eval_f):
        count = 0
        for i in range(0, len(eval_f), 10000):
            batch = eval_f[i : i + 10000]
            dists = np.sum((batch[:, None, :] - ref_f[None, :, :]) ** 2, axis=-1)
            count += np.sum(np.any(dists <= ref_r[None, :], axis=1))
        return count / len(eval_f)

    ref_r = knn_radii(ref_acts, k)
    sample_r = knn_radii(sample_acts, k)
    precision = manifold_coverage(ref_acts, ref_r, sample_acts)
    recall = manifold_coverage(sample_acts, sample_r, ref_acts)
    return precision, recall


def evaluate_metrics(
    ema_model, vae, diffusion, ref_acts, inception_extractor,
    num_fid_samples, batch_gpu, latent_size, device,
):
    """Generate samples, compute all metrics, return dict."""
    logger.log(f"Evaluating metrics ({num_fid_samples} samples)...")
    fake_images = generate_eval_samples(
        ema_model, vae, diffusion, num_fid_samples, batch_gpu, latent_size, device,
    )
    fake_acts = compute_activations(fake_images, inception_extractor, batch_size=64, device=device)

    metrics = {}
    metrics["IS"] = compute_inception_score(fake_acts["logits"])
    metrics["FID"] = compute_fid(ref_acts["pool"], fake_acts["pool"])
    metrics["sFID"] = compute_fid(ref_acts["spatial"], fake_acts["spatial"])
    prec, rec = compute_precision_recall(ref_acts["pool"], fake_acts["pool"])
    metrics["Precision"] = prec
    metrics["Recall"] = rec

    for k, v in metrics.items():
        logger.log(f"  {k}: {v:.4f}")
    return metrics


# ---------------------------------------------------------------------------


def subprocess_fn(rank, c, temp_dir):
    """Entry point for each DDP worker."""
    logger.configure(log_dir=c["run_dir"])

    if c["num_gpus"] > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, ".torch_distributed_init"))
        init_method = f"file://{init_file}"
        dist.init_process_group(
            backend="nccl", init_method=init_method, rank=rank, world_size=c["num_gpus"]
        )

    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = c.get("allow_tf32", True)

    training_loop(rank=rank, **c)

    if c["num_gpus"] > 1 and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------


def training_loop(
    rank,
    run_dir,
    data,
    image_size,
    num_gpus,
    batch_size,
    batch_gpu,
    total_kimg,
    kimg_per_tick,
    snap,
    seed,
    lr,
    use_fp16,
    ema_rate,
    log_interval,
    save_interval,
    resume,
    model_name,
    schedule_sampler_name,
    workers=4,
    num_fid_samples=2048,
    **_extra,
):
    """Main training loop for DiffiT."""
    device = torch.device("cuda", rank)
    is_main = rank == 0

    # Seeding
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    # Build model
    latent_size = image_size // 8
    model = diffit_module.__dict__[model_name](input_size=latent_size)
    model.to(device)

    # EMA model
    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)
    ema_model.eval()

    # Create diffusion
    diff_config = diffusion_defaults()
    diffusion = create_diffusion(**diff_config)
    schedule_sampler = create_named_schedule_sampler(schedule_sampler_name, diffusion)

    # Mixed precision
    mp_trainer = MixedPrecisionTrainer(model=model, use_fp16=use_fp16)

    # Optimizer
    opt = torch.optim.AdamW(mp_trainer.master_params, lr=lr, weight_decay=0.0)

    # DDP wrapper
    if num_gpus > 1:
        ddp_model = DDP(model, device_ids=[rank])
    else:
        ddp_model = model

    # VAE encoder (for latent diffusion)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()
    vae.requires_grad_(False)

    # Resume
    cur_nimg = 0
    if resume is not None:
        logger.log(f"Resuming from {resume}...")
        ckpt = torch.load(resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        cur_nimg = ckpt.get("cur_nimg", 0)
        logger.log(f"Resumed at {cur_nimg // 1000} kimg")

    # Data loader
    logger.log("Loading data...")
    data_iter = load_data(
        data_dir=data,
        batch_size=batch_gpu,
        image_size=image_size,
        class_cond=True,
        random_flip=True,
        num_workers=workers,
        distributed=(num_gpus > 1),
    )

    # --- Initialize logs: stats.jsonl + TensorBoard (SAN-v2 style) ---
    stats_jsonl = None
    stats_tfevents = None
    if is_main:
        stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "at")
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            logger.log(f"Skipping TensorBoard export: {err}")

    # --- Setup snapshot image grid (SAN-v2 style) ---
    start_time = time.time()
    gw, gh = setup_snapshot_image_grid(image_size)
    grid_size = (gw, gh)
    n_grid = gw * gh

    # --- Load Inception model + pre-compute reference features (rank 0 only) ---
    inception_extractor = None
    ref_acts = None
    if is_main and num_fid_samples > 0:
        logger.log("Loading InceptionV3 for metric evaluation...")
        from torchvision.models import inception_v3, Inception_V3_Weights
        inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT).eval().to(device)
        inception_extractor = InceptionFeatureExtractor(inception_model)

        logger.log(f"Pre-computing reference Inception features ({num_fid_samples} real images)...")
        ref_iter = load_data(
            data_dir=data, batch_size=min(num_fid_samples, 64), image_size=image_size,
            class_cond=True, random_flip=False, num_workers=workers,
            distributed=False,
        )
        ref_images = []
        while len(ref_images) < num_fid_samples:
            batch_ref, _ = next(ref_iter)
            batch_np = ((batch_ref + 1) * 127.5).clamp(0, 255).to(torch.uint8).numpy()
            ref_images.append(batch_np)
        ref_images = np.concatenate(ref_images, axis=0)[:num_fid_samples]
        ref_acts = compute_activations(ref_images, inception_extractor, batch_size=64, device=device)
        del ref_images
        logger.log("Reference features computed.")

    # Save real training images grid
    if is_main:
        logger.log("Exporting sample images (reals.png, fakes_init.png)...")
        real_iter = load_data(
            data_dir=data, batch_size=n_grid, image_size=image_size,
            class_cond=True, random_flip=False, num_workers=2,
            distributed=False,
        )
        real_batch, _ = next(real_iter)
        # real_batch is [-1, 1] float tensor NCHW
        real_np = ((real_batch + 1) * 127.5).clamp(0, 255).to(torch.uint8).numpy()
        save_image_grid(real_np, os.path.join(run_dir, "reals.png"), drange=[0, 255], grid_size=grid_size)

    # Fixed latents + classes for consistent snapshot generation
    grid_z = torch.randn(
        n_grid, 4, latent_size, latent_size, device=device,
        generator=torch.Generator(device=device).manual_seed(seed * max(num_gpus, 1)),
    )
    grid_classes = torch.randint(
        0, NUM_CLASSES, (n_grid,), device=device,
        generator=torch.Generator(device=device).manual_seed(seed * max(num_gpus, 1)),
    )

    # Save initial fakes
    if is_main:
        fakes_init = generate_snapshot_images(
            ema_model, vae, diffusion, grid_z, grid_classes,
            batch_gpu=batch_gpu, device=device,
        )
        save_image_grid(fakes_init, os.path.join(run_dir, "fakes_init.png"), drange=[0, 255], grid_size=grid_size)

    # Training
    logger.log(f"Training for {total_kimg} kimg...")
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    cur_tick = 0

    while cur_nimg < total_kimg * 1000:
        batch, cond = next(data_iter)
        batch = batch.to(device)

        # Encode to latent space
        with torch.no_grad():
            latent = vae.encode(batch).latent_dist.sample() * 0.18215

        # Sample timesteps
        t, weights = schedule_sampler.sample(latent.shape[0], device)

        # Model kwargs
        model_kwargs = {}
        if "y" in cond:
            model_kwargs["y"] = cond["y"].to(device)

        # Compute losses
        losses = diffusion.training_losses(ddp_model, latent, t, model_kwargs=model_kwargs)
        loss = (losses["loss"] * weights).mean()

        # Backward
        mp_trainer.zero_grad()
        mp_trainer.backward(loss)
        took_step = mp_trainer.optimize(opt)

        # Update EMA
        if took_step:
            update_ema(ema_model.parameters(), model.parameters(), rate=ema_rate)

        cur_nimg += batch_gpu * num_gpus

        # Logging
        logger.logkv("loss", loss.item())
        logger.logkv("step", cur_nimg // (batch_gpu * num_gpus))
        logger.logkv("kimg", cur_nimg / 1000)
        if "vb" in losses:
            logger.logkv("vb_loss", losses["vb"].mean().item())
        if "mse" in losses:
            logger.logkv("mse_loss", losses["mse"].mean().item())

        # Tick
        done_kimg = (cur_nimg - tick_start_nimg) / 1000
        if done_kimg >= kimg_per_tick or cur_nimg >= total_kimg * 1000:
            tick_end_time = time.time()
            elapsed = tick_end_time - tick_start_time
            kimg_per_sec = done_kimg / elapsed if elapsed > 0 else 0

            logger.logkv("kimg/s", kimg_per_sec)
            logger.logkv("elapsed_sec", elapsed)
            logged_kvs = logger.dumpkvs()

            # Write stats.jsonl + TensorBoard (SAN-v2 style)
            if is_main:
                timestamp = time.time()
                global_step = int(cur_nimg / 1e3)
                walltime = timestamp - start_time

                if stats_jsonl is not None:
                    fields = dict(logged_kvs, timestamp=timestamp)
                    stats_jsonl.write(json.dumps(fields) + "\n")
                    stats_jsonl.flush()

                if stats_tfevents is not None:
                    for name, value in logged_kvs.items():
                        if isinstance(value, (int, float)):
                            stats_tfevents.add_scalar(f"Train/{name}", value, global_step=global_step, walltime=walltime)
                    stats_tfevents.flush()

            cur_tick += 1
            tick_start_nimg = cur_nimg
            tick_start_time = time.time()

            # Save image snapshot + evaluate metrics (SAN-v2 style: every `snap` ticks)
            if is_main and cur_tick % snap == 0:
                logger.log(f"Saving image snapshot (kimg={cur_nimg / 1e3:.1f})...")
                fakes = generate_snapshot_images(
                    ema_model, vae, diffusion, grid_z, grid_classes,
                    batch_gpu=batch_gpu, device=device,
                )
                save_image_grid(
                    fakes, os.path.join(run_dir, f"fakes{cur_nimg // 1000:06d}.png"),
                    drange=[0, 255], grid_size=grid_size,
                )

                # Evaluate quality metrics
                if ref_acts is not None:
                    stats_metrics = evaluate_metrics(
                        ema_model, vae, diffusion, ref_acts, inception_extractor,
                        num_fid_samples, batch_gpu, latent_size, device,
                    )
                    timestamp = time.time()
                    global_step = int(cur_nimg / 1e3)
                    walltime = timestamp - start_time
                    if stats_jsonl is not None:
                        fields = dict(stats_metrics, timestamp=timestamp, kimg=cur_nimg / 1000)
                        stats_jsonl.write(json.dumps(fields) + "\n")
                        stats_jsonl.flush()
                    if stats_tfevents is not None:
                        for name, value in stats_metrics.items():
                            stats_tfevents.add_scalar(f"Metrics/{name}", value, global_step=global_step, walltime=walltime)
                        stats_tfevents.flush()

        # Snapshot (checkpoint)
        if (cur_nimg // 1000) % (snap * kimg_per_tick) == 0 and cur_nimg > 0 and is_main:
            save_path = os.path.join(run_dir, f"network-snapshot-{cur_nimg // 1000:06d}.pt")
            logger.log(f"Saving snapshot to {save_path}...")
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict(),
                    "opt": opt.state_dict(),
                    "cur_nimg": cur_nimg,
                    "config": {
                        "image_size": image_size,
                        "model_name": model_name,
                    },
                },
                save_path,
            )

    # Final save
    if is_main:
        save_path = os.path.join(run_dir, "network-final.pt")
        logger.log(f"Saving final model to {save_path}...")
        torch.save(
            {
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "opt": opt.state_dict(),
                "cur_nimg": cur_nimg,
                "config": {
                    "image_size": image_size,
                    "model_name": model_name,
                },
            },
            save_path,
        )

        # Final image snapshot
        logger.log("Saving final image snapshot...")
        fakes = generate_snapshot_images(
            ema_model, vae, diffusion, grid_z, grid_classes,
            batch_gpu=batch_gpu, device=device,
        )
        save_image_grid(
            fakes, os.path.join(run_dir, f"fakes{cur_nimg // 1000:06d}.png"),
            drange=[0, 255], grid_size=grid_size,
        )

    # Close logs
    if stats_jsonl is not None:
        stats_jsonl.close()
    if stats_tfevents is not None:
        stats_tfevents.close()

    logger.log("Training complete.")


# ---------------------------------------------------------------------------


def launch_training(c, desc, outdir, dry_run):
    """Set up run directory and launch training processes."""
    # Pick output directory
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]

    prev_run_ids = [re.match(r"^\d+", x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1

    if c.get("resume") is not None:
        # Try to find matching directory
        matching = [x for x in prev_run_dirs if re.fullmatch(r"\d{5}-" + re.escape(desc), x)]
        if matching:
            c["run_dir"] = os.path.join(outdir, matching[0])
        else:
            c["run_dir"] = os.path.join(outdir, f"{cur_run_id:05d}-{desc}")
    else:
        c["run_dir"] = os.path.join(outdir, f"{cur_run_id:05d}-{desc}")

    # Print options
    print()
    print("Training options:")
    print(json.dumps(c, indent=2, default=str))
    print()
    print(f"Output directory:    {c['run_dir']}")
    print(f"Number of GPUs:      {c['num_gpus']}")
    print(f"Batch size:          {c['batch_size']} images")
    print(f"Training duration:   {c['total_kimg']} kimg")
    print(f"Image size:          {c['image_size']}")
    print()

    if dry_run:
        print("Dry run; exiting.")
        return

    # Create output directory
    print("Creating output directory...")
    os.makedirs(c["run_dir"], exist_ok=True)
    with open(os.path.join(c["run_dir"], "training_options.json"), "wt") as f:
        json.dump(c, f, indent=2, default=str)

    # Launch processes
    print("Launching processes...")
    torch.multiprocessing.set_start_method("spawn", force=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        if c["num_gpus"] == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(
                fn=subprocess_fn, args=(c, temp_dir), nprocs=c["num_gpus"]
            )


# ---------------------------------------------------------------------------
# Base configurations (selected via --cfg, like SAN-v2)
# ---------------------------------------------------------------------------

BASE_CONFIGS = {
    "diffit-256": dict(
        image_size=256,
        model_name="Diffit",
        lr=1e-4,
        total_kimg=400000,
        kimg_per_tick=4,
        snap=50,
        ema_rate=0.9999,
        use_fp16=False,
        schedule_sampler_name="uniform",
        num_fid_samples=2048,
    ),
    "diffit-512": dict(
        image_size=512,
        model_name="Diffit",
        lr=1e-4,
        total_kimg=400000,
        kimg_per_tick=4,
        snap=50,
        ema_rate=0.9999,
        use_fp16=True,
        schedule_sampler_name="uniform",
        num_fid_samples=2048,
    ),
}


# ---------------------------------------------------------------------------


@click.command()

# Required.
@click.option("--outdir",       help="Where to save training runs", metavar="DIR",             type=str, required=True)
@click.option("--cfg",          help="Base configuration",                                      type=click.Choice(list(BASE_CONFIGS.keys())), required=True)
@click.option("--data",         help="Training data path (directory or .zip)", metavar="PATH",  type=str, required=True)
@click.option("--gpus",         help="Number of GPUs", metavar="INT",                          type=click.IntRange(min=1), required=True)
@click.option("--batch-gpu",    help="Batch size per GPU (total batch = batch-gpu * gpus)", metavar="INT", type=click.IntRange(min=1), required=True)

# Optional overrides (cfg provides defaults).
@click.option("--image-size",   help="Image resolution [default: from cfg]",                    type=int, default=None)
@click.option("--model",        "model_name", help="Model constructor name [default: from cfg]", type=str, default=None)
@click.option("--kimg",         help="Total training duration [default: from cfg]", metavar="KIMG", type=click.IntRange(min=1), default=None)
@click.option("--tick",         help="How often to print progress [default: from cfg]", metavar="KIMG", type=click.IntRange(min=1), default=None)
@click.option("--snap",         help="How often to save snapshots [default: from cfg]", metavar="TICKS", type=click.IntRange(min=1), default=None)
@click.option("--seed",         help="Random seed", metavar="INT",                             type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--lr",           help="Learning rate [default: from cfg]", metavar="FLOAT",     type=float, default=None)
@click.option("--fp32",         help="Disable mixed-precision", metavar="BOOL",                type=bool, default=None)
@click.option("--ema-rate",     help="EMA decay rate [default: from cfg]", metavar="FLOAT",    type=float, default=None)
@click.option("--resume",       help="Resume from checkpoint path", metavar="PATH",            type=str, default=None)
@click.option("--schedule-sampler", "schedule_sampler_name", help="Timestep sampler [default: from cfg]", type=str, default=None)
@click.option("--num-fid-samples", help="Samples for FID eval during training (0=disable)", metavar="INT", type=click.IntRange(min=0), default=None)

# Misc settings.
@click.option("--desc",         help="String to include in result dir name", metavar="STR",    type=str, default=None)
@click.option("--metrics",      help="Quality metrics", metavar="NAME",                        type=str, default="fid50k", show_default=True)
@click.option("--tf32/--no-tf32", "allow_tf32", help="Enable TF32 for matmul/conv",            default=True, show_default=True)
@click.option("--workers",      help="DataLoader worker processes", metavar="INT",             type=click.IntRange(min=1), default=4, show_default=True)
@click.option("-n", "--dry-run", help="Print training options and exit",                        is_flag=True)

def main(**kwargs):
    """Train DiffiT on class-conditional ImageNet."""
    opts = kwargs

    # Start from base configuration.
    cfg = dict(BASE_CONFIGS[opts["cfg"]])

    # CLI overrides: any explicitly provided option takes precedence.
    if opts["image_size"] is not None:
        cfg["image_size"] = opts["image_size"]
    if opts["model_name"] is not None:
        cfg["model_name"] = opts["model_name"]
    if opts["kimg"] is not None:
        cfg["total_kimg"] = opts["kimg"]
    if opts["tick"] is not None:
        cfg["kimg_per_tick"] = opts["tick"]
    if opts["snap"] is not None:
        cfg["snap"] = opts["snap"]
    if opts["lr"] is not None:
        cfg["lr"] = opts["lr"]
    if opts["fp32"] is not None:
        cfg["use_fp16"] = not opts["fp32"]
    if opts["ema_rate"] is not None:
        cfg["ema_rate"] = opts["ema_rate"]
    if opts["schedule_sampler_name"] is not None:
        cfg["schedule_sampler_name"] = opts["schedule_sampler_name"]
    if opts["num_fid_samples"] is not None:
        cfg["num_fid_samples"] = opts["num_fid_samples"]

    # Build full config dict.
    c = dict(
        data=opts["data"],
        image_size=cfg["image_size"],
        num_gpus=opts["gpus"],
        batch_size=opts["batch_gpu"] * opts["gpus"],
        batch_gpu=opts["batch_gpu"],
        total_kimg=cfg["total_kimg"],
        kimg_per_tick=cfg["kimg_per_tick"],
        snap=cfg["snap"],
        seed=opts["seed"],
        lr=cfg["lr"],
        use_fp16=cfg["use_fp16"],
        ema_rate=cfg["ema_rate"],
        resume=opts["resume"],
        model_name=cfg["model_name"],
        schedule_sampler_name=cfg["schedule_sampler_name"],
        allow_tf32=opts["allow_tf32"],
        workers=opts["workers"],
        num_fid_samples=cfg["num_fid_samples"],
        log_interval=10,
        save_interval=10000,
    )

    # Description string.
    desc = f"{opts['cfg']}-gpus{c['num_gpus']}-batch{c['batch_size']}"
    if opts["desc"] is not None:
        desc += f"-{opts['desc']}"

    launch_training(c=c, desc=desc, outdir=opts["outdir"], dry_run=opts["dry_run"])


if __name__ == "__main__":
    main()
