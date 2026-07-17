"""
Train DiffiT: Diffusion Vision Transformers for Image Generation.

Uses PyTorch DDP for multi-GPU training with the same interface style as
SAN-v2 / the v2 convention. The --cfg flag selects a base preset (model,
resolution, learning rate, diffusion settings); individual options override it.

Usage (multi-GPU):
    diffit-train --outdir ./training-runs \
        --cfg diffit-256 \
        --data ./datasets/imagenet_256x256.zip \
        --gpus 4 --batch-gpu 64

Single GPU:
    diffit-train --outdir ./training-runs --cfg diffit-256 \
        --data ./datasets/imagenet_256x256.zip --gpus 1 --batch-gpu 64

Progressive higher-resolution finetuning warm-starts from a previous stage's
EMA snapshot (weights only, fresh optimizer):
    diffit-train ... --cfg diffit-512 --init-weights <prev>/diffit-snapshot-000400-inference.pt
"""

import copy
import dataclasses
import datetime
import glob
import json
import os
import re
import tempfile
import time
import warnings
from contextlib import nullcontext

import click
import numpy as np
import PIL.Image
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from torch.nn.parallel import DistributedDataParallel as DDP

import diffit.diffit as diffit_module
from diffit import create_diffusion, diffusion_defaults, logger
from diffit.constants import PIXEL_NORM_HALF, UINT8_MAX, VAE_SCALE_FACTOR
from diffit.dist_util import extract_inference_state_dict, load_state_dict
from diffit.image_datasets import count_data, load_data, read_class_meta
from diffit.inception import InceptionFeatureExtractor
from diffit.metrics import (
    HAS_COMBRA,
    compute_activations,
    evaluate_metrics,
    precompute_combra_reference,
    sample_latents,
)
from diffit.nn import update_ema
from diffit.timestep_sampler import create_named_schedule_sampler

# ---------------------------------------------------------------------------
# Image grid utilities (matching SAN-v2 style)
# ---------------------------------------------------------------------------


def save_image_grid(img, fname, drange, grid_size):
    """Save a grid of images as a single PNG (SAN-v2 style)."""
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (UINT8_MAX / (hi - lo))
    img = np.rint(img).clip(0, UINT8_MAX).astype(np.uint8)

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
        gw = int(max(np.clip(2560 // image_size, 7, 32), 1))
    if gh is None:
        gh = int(max(np.clip(1440 // image_size, 4, 32), 1))
    return gw, gh


@torch.inference_mode()
def generate_snapshot_images(
    ema_model, vae, diffusion, grid_z, grid_classes, batch_gpu, device,
    *,
    cfg_scale, null_class_idx, num_sampling_steps=25, scale_pow=4.0, sampler="dpm++",
):
    """Generate a batch of images from the EMA model for snapshot grids."""
    all_samples = []
    for z_chunk, c_chunk in zip(grid_z.split(batch_gpu), grid_classes.split(batch_gpu)):
        bs = z_chunk.shape[0]
        z_cfg = torch.cat([z_chunk, z_chunk], 0)
        classes_null = torch.full((bs,), null_class_idx, device=device, dtype=torch.long)
        model_kwargs = {
            "y": torch.cat([c_chunk, classes_null], 0),
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
        all_samples.append(decoded.permute(0, 2, 3, 1).cpu().numpy())

    # Return as NCHW uint8 for save_image_grid
    images = np.concatenate(all_samples, axis=0)
    images = images.transpose(0, 3, 1, 2)
    return images


def atomic_save(obj, path):
    """Write ``obj`` to ``path`` atomically (temp file + ``os.replace``).

    A snapshot that exists under its final name is therefore always complete: a
    crash or walltime kill mid-write leaves only a ``.tmp`` file, never a
    truncated ``.pt`` (§3 atomic-write MUST).
    """
    tmp = f"{path}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_inference_snapshot(ema_model, run_dir, cur_nimg, meta):
    """Atomically write the EMA-only inference snapshot for ``cur_nimg`` kimg.

    The saved object is a ``.pt`` state dict plus self-describing metadata
    (``n_classes`` / ``resolution`` / ``class_names`` / ``cur_nimg``, §3) — no
    optimizer state, no raw (non-EMA) weights, no pickled modules.
    """
    path = os.path.join(run_dir, f"diffit-snapshot-{cur_nimg // 1000:06d}-inference.pt")
    obj = {"ema": ema_model.state_dict(), "cur_nimg": int(cur_nimg), **meta}
    atomic_save(obj, path)
    return path


def prune_inference_snapshots(run_dir, keep_last):
    """Delete all but the ``keep_last`` newest inference snapshots.

    ``keep_last <= 0`` keeps everything. Matches only
    ``diffit-snapshot-<kimg>-inference.pt``.
    """
    if keep_last <= 0:
        return
    snaps = glob.glob(os.path.join(run_dir, "diffit-snapshot-*-inference.pt"))

    def _kimg(path):
        m = re.search(r"diffit-snapshot-(\d+)-inference\.pt$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    snaps.sort(key=_kimg)
    for old in snaps[:-keep_last]:
        try:
            os.remove(old)
        except OSError:
            pass


def subprocess_fn(rank, c, temp_dir):
    """Entry point for each DDP worker."""
    logger.configure(c["run_dir"], run_name=os.path.basename(c["run_dir"]), is_main=(rank == 0))

    # Pin this process to its GPU *before* initializing the NCCL process group.
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = c.get("allow_tf32", True)
    torch.backends.cudnn.allow_tf32 = c.get("allow_tf32", True)
    torch.backends.cudnn.benchmark = c.get("bench", True)

    if c["num_gpus"] > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, ".torch_distributed_init"))
        init_method = f"file://{init_file}"
        dist.init_process_group(
            backend="nccl", init_method=init_method, rank=rank, world_size=c["num_gpus"]
        )

    training_loop(rank=rank, **c)

    if c["num_gpus"] > 1 and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------


def combra_smoke_test(ref_images, device, log_fn):
    """Verify combra metrics actually *compute* (not just import) before training.

    ``HAS_COMBRA`` only proves the package imports. The image-feature metrics
    depend on optional backends that combra records as ``nan`` rather than
    raising when missing, so a broken install silently produces useless ``nan``
    metrics. Run the real pipeline once on a tiny slice and fail fast.
    """
    from combra.metrics import compute_all_metrics

    sample = ref_images[: min(4, len(ref_images))]
    try:
        metrics = compute_all_metrics(
            sample, sample, device=device, image_metrics=True, reference_cache={},
        )
    except Exception as e:
        raise RuntimeError(f"combra metrics smoke test failed to run: {e}") from e
    bad = sorted(k for k, v in metrics.items() if not np.isfinite(v))
    if bad:
        raise RuntimeError(
            f"combra metrics smoke test produced non-finite values for {bad} -- "
            "a metric backend or optional dependency is missing/broken. Fix the "
            "install or pass --combra-metrics=False."
        )
    log_fn(f"combra metrics smoke test passed ({len(metrics)} metrics computed).")


def load_reference_shard(data, ref_count, total_count, image_size, workers, rank, world_size, seed, num_classes):
    """Load this rank's slice of a ``ref_count``-image combra reference (uint8 NCHW).

    When ``ref_count < total_count`` the reference is a **seeded random subset**
    of the whole dataset (never the first N — dataset zips are class-sorted, so a
    first-N slice is class-biased, §6). The subset is chosen from one seeded
    permutation shared by every rank; each rank keeps a disjoint stripe of the
    selected global indices, so the union of the per-rank shards is exactly the
    reference set and the expensive feature/angle extraction is sharded.
    """
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.permutation(total_count)[:ref_count])
    own = set(int(i) for i in selected[rank::world_size])
    max_gidx = int(selected.max()) if selected.size else -1

    ref_iter = load_data(
        data_dir=data, batch_size=64, image_size=image_size, num_classes=num_classes,
        class_cond=True, mirror=False, num_workers=workers,
        distributed=False, deterministic=True, drop_last=False,
    )
    shard = []
    seen = 0
    while seen <= max_gidx:
        batch_ref, _ = next(ref_iter)
        batch_np = batch_ref.numpy()  # uint8 NCHW
        for j in range(batch_np.shape[0]):
            if (seen + j) in own:
                shard.append(batch_np[j])
        seen += batch_np.shape[0]
    if shard:
        return np.stack(shard, axis=0)
    return np.empty((0, 3, image_size, image_size), dtype=np.uint8)


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
    precision,
    ema_rate,
    init_weights,
    model_name,
    schedule_sampler_name,
    cfg_scale,
    mirror=False,
    grad_accum_steps=1,
    gradient_checkpointing=False,
    lr_warmup_kimg=0,
    workers=3,
    cache_in_ram=False,
    num_fid_samples=10000,
    combra_ref_count=0,
    eval_sampler="ddim",
    eval_sampling_steps=100,
    combra_metrics=True,
    snapshot_keep_last=3,
    **_extra,
):
    """Main training loop for DiffiT."""
    device = torch.device("cuda", rank)
    is_main = rank == 0

    # Seeding — data shuffling (incl. the DistributedSampler), weight init and
    # eval/grid latent draws all derive from --seed (§2).
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    # Startup header, written first thing so the .log is self-sufficient (§7).
    if is_main:
        logger.log(f"Run: {os.path.basename(run_dir)}")
        logger.log(f"torch {torch.__version__}, CUDA {torch.version.cuda}, "
                   f"GPUs={num_gpus} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "CUDA_VISIBLE_DEVICES"):
            if var in os.environ:
                logger.log(f"env {var}={os.environ[var]}")

    # Resolve precision: fp32 = no autocast; fp16 = autocast + GradScaler;
    # bf16 = autocast, no scaler (§2). GradScaler is used only for fp16.
    if precision == "bf16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        if is_main:
            logger.log("bf16 requested but unsupported on this GPU; falling back to fp16.")
        precision = "fp16"
    amp_enabled = precision in ("fp16", "bf16")
    amp_dtype_torch = torch.bfloat16 if precision == "bf16" else torch.float16
    use_grad_scaler = precision == "fp16"
    if is_main:
        logger.log(f"Precision: {precision} (amp={amp_enabled}, grad_scaler={use_grad_scaler})")

    # Class count & names come from the dataset's dataset.json (§5/§12): the
    # integer label is the class-folder index in alphabetical order, and
    # class_names travels index-aligned. No startup probe, no label remap.
    class_names, num_dataset_classes = read_class_meta(data)
    if is_main:
        logger.log(f"Dataset classes: {num_dataset_classes}"
                   + (f" {class_names}" if class_names else " (no class_names in dataset.json)"))
    ckpt_meta = {
        "n_classes": num_dataset_classes,
        "resolution": image_size,
        "class_names": class_names,
    }

    # Build model (sized to the dataset's actual class count).
    latent_size = image_size // 8
    model = diffit_module.__dict__[model_name](
        input_size=latent_size, num_classes=num_dataset_classes,
    )
    model.to(device)

    if gradient_checkpointing:
        model.gradient_checkpointing = True
        if is_main:
            logger.log("Gradient checkpointing enabled.")

    # Weights-only warm start for progressive higher-resolution finetuning (§2):
    # load a previous stage's EMA weights into both the trainable model and the
    # EMA copy; fresh optimizer, cur_nimg resets to 0. This replaces --resume.
    if init_weights is not None:
        logger.log(f"Warm-starting from EMA weights in {init_weights}...")
        warm_sd = extract_inference_state_dict(load_state_dict(init_weights, map_location="cpu"))
        model.load_state_dict(warm_sd)

    # EMA model.
    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)
    ema_model.eval()
    ema_model.forward_with_cfg = torch.compile(ema_model.forward_with_cfg)

    # Diffusion.
    diff_config = diffusion_defaults()
    diffusion = create_diffusion(**diff_config)
    schedule_sampler = create_named_schedule_sampler(schedule_sampler_name, diffusion)

    if eval_sampler in ("dpm++", "unipc"):
        eval_diffusion = diffusion
    else:
        eval_diffusion = create_diffusion(**{**diff_config, "timestep_respacing": str(eval_sampling_steps)})

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, fused=True)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_grad_scaler)

    if num_gpus > 1:
        ddp_raw = DDP(model, device_ids=[rank], gradient_as_bucket_view=True)
    else:
        ddp_raw = model

    compile_mode = "max-autotune-no-cudagraphs" if gradient_checkpointing else "max-autotune"
    ddp_model = torch.compile(ddp_raw, mode=compile_mode)

    # VAE encoder (for latent diffusion).
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()
    vae.requires_grad_(False)
    vae.enable_slicing()
    if image_size >= 1024:
        vae.enable_tiling()

    cur_nimg = 0

    # Data loader — mirror is the loader-level per-item horizontal flip (§2).
    logger.log("Loading data...")
    data_iter = load_data(
        data_dir=data,
        batch_size=batch_gpu,
        image_size=image_size,
        num_classes=num_dataset_classes,
        class_cond=True,
        mirror=mirror,
        num_workers=workers,
        distributed=(num_gpus > 1),
        cache_in_ram=cache_in_ram,
        seed=seed,
    )

    # --- Logs: stats.jsonl (scalar rows only, §7) + TensorBoard ---
    stats_jsonl = None
    stats_tfevents = None
    run_name = os.path.basename(run_dir)
    if is_main:
        stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "at")
        try:
            import torch.utils.tensorboard as tensorboard
            # The event file carries the run name as a filename_suffix so a copied
            # tfevents file stays self-identifying (§7).
            stats_tfevents = tensorboard.SummaryWriter(run_dir, filename_suffix=f".{run_name}")
        except ImportError as err:
            logger.log(f"Skipping TensorBoard export: {err}")

    start_time = time.time()
    gw, gh = setup_snapshot_image_grid(image_size)
    grid_size = (gw, gh)
    n_grid = gw * gh

    if combra_metrics and is_main and not HAS_COMBRA:
        warnings.warn(
            "--combra-metrics=True but the `combra` package is not installed -- "
            "combra metrics will be skipped. Install it or pass --combra-metrics=False."
        )
    use_combra = combra_metrics and HAS_COMBRA

    # --- Pre-load reference (combra: sharded reference; Inception: rank-0) ---
    inception_extractor = None
    ref_acts = None
    combra_ref = None
    if num_fid_samples > 0:
        if use_combra:
            total_count = count_data(data)
            ref_count = min(combra_ref_count, total_count) if combra_ref_count > 0 else total_count
            if is_main:
                logger.log(
                    f"combra metrics: {num_fid_samples} fakes scored against "
                    f"{ref_count}/{total_count} reals"
                    + (" (seeded random subset)" if ref_count < total_count else " (whole dataset)")
                )
            local_ref = load_reference_shard(
                data, ref_count, total_count, image_size, workers, rank, num_gpus, seed, num_dataset_classes,
            )
            if is_main:
                logger.log("Running combra metrics smoke test...")
                combra_smoke_test(local_ref, device, logger.log)
                logger.log("combra metrics enabled → DiffiT Inception metrics disabled.")
            combra_ref = precompute_combra_reference(local_ref, device, rank, num_gpus)
            if is_main:
                logger.log("Reference features computed.")
        elif is_main:
            ref_count = num_fid_samples
            logger.log(f"Pre-loading {ref_count} reference images (Inception path)...")
            ref_iter = load_data(
                data_dir=data, batch_size=min(ref_count, 64), image_size=image_size,
                num_classes=num_dataset_classes, class_cond=True, mirror=False,
                num_workers=workers, distributed=False, deterministic=True, drop_last=False,
            )
            ref_images = []
            n_collected = 0
            while n_collected < ref_count:
                batch_ref, _ = next(ref_iter)
                batch_np = batch_ref.numpy()  # uint8 NCHW
                ref_images.append(batch_np)
                n_collected += batch_np.shape[0]
            ref_images = np.concatenate(ref_images, axis=0)[:ref_count]

            logger.log("Loading InceptionV3 for metric evaluation...")
            from torchvision.models import Inception_V3_Weights, inception_v3
            inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT).eval().to(device)
            inception_extractor = InceptionFeatureExtractor(inception_model)
            logger.log(f"Pre-computing reference Inception features ({ref_count} images)...")
            ref_acts = compute_activations(ref_images, inception_extractor, batch_size=64, device=device)
            del ref_images
            logger.log("Reference features computed.")

    # Class-sorted grid: each row cycles through classes 0..K-1.
    grid_row_classes = [r % num_dataset_classes for r in range(gh)]

    # Save real training images grid (class-sorted).
    if is_main:
        logger.log("Exporting sample images (reals.png, fakes_init.png)...")
        class_images = {c: [] for c in range(num_dataset_classes)}
        collect_iter = load_data(
            data_dir=data, batch_size=64, image_size=image_size,
            num_classes=num_dataset_classes, class_cond=True, mirror=False,
            num_workers=2, distributed=False,
        )
        class_count_needed = {}
        for c in grid_row_classes:
            class_count_needed[c] = class_count_needed.get(c, 0) + gw

        while any(len(class_images[c]) < class_count_needed.get(c, 0) for c in range(num_dataset_classes)):
            img_batch, cond_batch = next(collect_iter)
            if "y" not in cond_batch:
                break
            labels = cond_batch["y"].argmax(dim=1)
            for i in range(img_batch.shape[0]):
                c = int(labels[i].item())
                if c in class_images and len(class_images[c]) < class_count_needed.get(c, 0):
                    class_images[c].append(img_batch[i:i + 1].numpy())  # uint8 NCHW

        grid_images = []
        class_cursors = {c: 0 for c in range(num_dataset_classes)}
        for r in range(gh):
            c = grid_row_classes[r]
            for _col in range(gw):
                idx = class_cursors[c]
                if idx < len(class_images[c]):
                    grid_images.append(class_images[c][idx])
                    class_cursors[c] += 1
                else:
                    grid_images.append(np.zeros((1, 3, image_size, image_size), dtype=np.uint8))
        real_np = np.concatenate(grid_images, axis=0)
        save_image_grid(real_np, os.path.join(run_dir, "reals.png"), drange=[0, 255], grid_size=grid_size)
        del class_images, collect_iter

    # Fixed latents + class-sorted classes for consistent snapshot generation.
    # Seeded from --seed alone (never scaled by GPU count, §2).
    grid_z = torch.randn(
        n_grid, 4, latent_size, latent_size, device=device,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    grid_classes_list = []
    for r in range(gh):
        grid_classes_list.extend([r % num_dataset_classes] * gw)
    grid_classes = torch.tensor(grid_classes_list, device=device, dtype=torch.long)

    if is_main:
        fakes_init = generate_snapshot_images(
            ema_model, vae, eval_diffusion, grid_z, grid_classes,
            batch_gpu=batch_gpu, device=device,
            cfg_scale=cfg_scale, null_class_idx=num_dataset_classes,
            sampler=eval_sampler, num_sampling_steps=eval_sampling_steps,
        )
        save_image_grid(fakes_init, os.path.join(run_dir, "fakes_init.png"), drange=[0, 255], grid_size=grid_size)

    if num_gpus > 1:
        dist.barrier()

    effective_batch = batch_gpu * num_gpus * grad_accum_steps
    if is_main:
        logger.log(
            f"Training for {total_kimg} kimg "
            f"(batch_gpu={batch_gpu} × {num_gpus} GPUs × {grad_accum_steps} accum "
            f"= {effective_batch} effective batch)"
        )

    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    cur_tick = 0

    def _format_time(seconds):
        s = int(seconds)
        if s < 60:
            return f"{s}s"
        elif s < 3600:
            return f"{s // 60}m {s % 60:02d}s"
        else:
            return f"{s // 3600}h {(s % 3600) // 60:02d}m {s % 60:02d}s"

    while cur_nimg < total_kimg * 1000:
        opt.zero_grad(set_to_none=True)

        for micro_step in range(grad_accum_steps):
            batch, cond = next(data_iter)
            # Normalize uint8 → [-1, 1] in the loop (the dataset yields uint8, §5).
            batch = batch.to(device, non_blocking=True).float().div_(PIXEL_NORM_HALF).sub_(1.0)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype_torch, enabled=amp_enabled):
                latent = vae.encode(batch).latent_dist.sample() * VAE_SCALE_FACTOR

            t, weights = schedule_sampler.sample(latent.shape[0], device)

            model_kwargs = {}
            if "y" in cond:
                # Dataset labels are one-hot float32 over the canonical class set;
                # the model's embedding wants the integer index.
                model_kwargs["y"] = cond["y"].argmax(dim=1).to(device, non_blocking=True)

            is_last_micro = (micro_step == grad_accum_steps - 1)
            sync_ctx = nullcontext() if is_last_micro or num_gpus <= 1 else ddp_raw.no_sync()

            with sync_ctx:
                with torch.amp.autocast("cuda", dtype=amp_dtype_torch, enabled=amp_enabled):
                    losses = diffusion.training_losses(ddp_model, latent, t, model_kwargs=model_kwargs)
                    loss = (losses["loss"] * weights).mean() / grad_accum_steps
                scaler.scale(loss).backward()

            cur_nimg += batch_gpu * num_gpus
            logger.logkv_mean("Loss/train", loss.item() * grad_accum_steps)
            if "vb" in losses:
                logger.logkv_mean("Loss/vb", losses["vb"].mean().item())

        scaler.step(opt)
        scaler.update()

        if lr_warmup_kimg > 0:
            warmup_frac = min(1.0, cur_nimg / (lr_warmup_kimg * 1000))
            for pg in opt.param_groups:
                pg["lr"] = lr * warmup_frac

        update_ema(ema_model.parameters(), model.parameters(), rate=ema_rate)
        if "mse" in losses:
            logger.logkv_mean("Loss/mse", losses["mse"].mean().item())

        # Tick.
        done_kimg = (cur_nimg - tick_start_nimg) / 1000
        is_last_tick = cur_nimg >= total_kimg * 1000
        if done_kimg >= kimg_per_tick or is_last_tick:
            tick_end_time = time.time()
            tick_elapsed = tick_end_time - tick_start_time
            total_elapsed = tick_end_time - start_time
            sec_per_tick = tick_elapsed
            sec_per_kimg = tick_elapsed / done_kimg if done_kimg > 0 else 0.0
            kimg_done = cur_nimg / 1000

            cpumem = 0.0
            try:
                import psutil
                cpumem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
            except ImportError:
                pass
            gpumem = reserved = 0.0
            if torch.cuda.is_available():
                gpumem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

            logger.log(
                f"tick {cur_tick:<6d} kimg {kimg_done:<10.1f} "
                f"time {_format_time(total_elapsed):<14s} "
                f"sec/tick {sec_per_tick:<9.1f} sec/kimg {sec_per_kimg:<9.2f} "
                f"cpumem {cpumem:<7.2f} gpumem {gpumem:<8.2f} reserved {reserved:<8.2f}"
            )

            # Per-tick scalar row (§7 namespaces; step = cur_nimg everywhere).
            loss_kvs = logger.dumpkvs()
            row = dict(loss_kvs)
            row["Progress/kimg"] = kimg_done
            row["Progress/tick"] = cur_tick
            row["Timing/sec_per_tick"] = sec_per_tick
            row["Timing/sec_per_kimg"] = sec_per_kimg
            row["Resources/cpu_mem_gb"] = cpumem
            row["Resources/gpu_mem_gb"] = gpumem
            row["Resources/gpu_reserved_gb"] = reserved
            row["LearningRate/lr"] = float(opt.param_groups[0]["lr"])

            cur_tick += 1
            tick_start_nimg = cur_nimg
            tick_start_time = time.time()

            # Snapshot at every `snap` ticks and ALWAYS at the last tick (§3), so
            # the newest snapshot is always the final model.
            do_snap = (cur_tick % snap == 0) or is_last_tick
            if do_snap:
                snap_start = time.time()

                if is_main:
                    logger.log(f"Saving image snapshot (kimg={cur_nimg / 1e3:.1f})...")
                    fakes = generate_snapshot_images(
                        ema_model, vae, eval_diffusion, grid_z, grid_classes,
                        batch_gpu=batch_gpu, device=device,
                        cfg_scale=cfg_scale, null_class_idx=num_dataset_classes,
                        sampler=eval_sampler, num_sampling_steps=eval_sampling_steps,
                    )
                    save_image_grid(
                        fakes, os.path.join(run_dir, f"fakes{cur_nimg // 1000:06d}.png"),
                        drange=[0, 255], grid_size=grid_size,
                    )
                    if stats_tfevents is not None:
                        grid_u8 = save_image_grid_to_array(fakes, grid_size)
                        stats_tfevents.add_image("Fakes", grid_u8, global_step=cur_nimg, dataformats="HWC")

                if num_fid_samples > 0:
                    stats_metrics = evaluate_metrics(
                        ema_model, vae, eval_diffusion, ref_acts, inception_extractor,
                        num_fid_samples, batch_gpu, latent_size, device,
                        cfg_scale=cfg_scale,
                        sampler=eval_sampler, num_sampling_steps=eval_sampling_steps,
                        rank=rank, world_size=num_gpus,
                        log_fn=logger.log,
                        class_list=list(range(num_dataset_classes)),
                        null_class_idx=num_dataset_classes,
                        combra_ref=combra_ref,
                        inception_metrics=not use_combra,
                    )
                    if is_main and stats_metrics is not None:
                        for name, value in stats_metrics.items():
                            row[f"Metrics/{name}"] = float(value)

                if is_main:
                    row["Timing/eval_sec"] = time.time() - snap_start
                    path = save_inference_snapshot(ema_model, run_dir, cur_nimg, ckpt_meta)
                    prune_inference_snapshots(run_dir, snapshot_keep_last)
                    logger.log(f"Snapshot saved: {os.path.basename(path)}")

            # Write the one scalar row for this tick (§7).
            if is_main:
                if stats_jsonl is not None:
                    now = time.time()
                    fields = dict(row, wall_time=now - start_time,
                                  datetime=datetime.datetime.now().isoformat(timespec="seconds"))
                    stats_jsonl.write(json.dumps(fields) + "\n")
                    stats_jsonl.flush()
                if stats_tfevents is not None:
                    for name, value in row.items():
                        if isinstance(value, (int, float)):
                            stats_tfevents.add_scalar(name, value, global_step=cur_nimg)
                    stats_tfevents.flush()

    if stats_jsonl is not None:
        stats_jsonl.close()
    if stats_tfevents is not None:
        stats_tfevents.close()
    logger.close()
    logger.log("Training complete.")


def save_image_grid_to_array(img, grid_size):
    """Assemble an NCHW uint8 batch into one HWC uint8 grid array (for TB)."""
    img = np.asarray(img, dtype=np.uint8)
    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W]).transpose(0, 3, 1, 4, 2).reshape([gh * H, gw * W, C])
    return img


# ---------------------------------------------------------------------------


def launch_training(c, desc, outdir, dry_run):
    """Set up a fresh run directory and launch training processes.

    A fresh run id is always allocated — existing directories are never reused
    (§2/§3: with resume gone there is nothing to re-enter).
    """
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r"^\d+", x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c["run_dir"] = os.path.join(outdir, f"{cur_run_id:05d}-{desc}")

    print()
    print("Training options:")
    print(json.dumps(c, indent=2, default=str))
    print()
    print(f"Output directory:    {c['run_dir']}")
    print(f"Number of GPUs:      {c['num_gpus']}")
    print(f"Batch size:          {c['batch_size']} images")
    print(f"Training duration:   {c['total_kimg']} kimg")
    print(f"Image size:          {c['image_size']}")
    print(f"Precision:           {c['precision']}")
    print()

    if dry_run:
        print("Dry run; exiting.")
        return

    print("Creating output directory...")
    os.makedirs(c["run_dir"], exist_ok=True)
    with open(os.path.join(c["run_dir"], "training_options.json"), "wt") as f:
        json.dump(c, f, indent=2, default=str)

    print("Launching processes...")
    torch.multiprocessing.set_start_method("spawn", force=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        if c["num_gpus"] == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c["num_gpus"])


# ---------------------------------------------------------------------------
# Base configurations (selected via --cfg)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Preset training hyperparameters selected via ``--cfg``."""

    image_size: int
    cfg_scale: float
    lr: float
    model_name: str = "Diffit"
    total_kimg: int = 400000
    kimg_per_tick: int = 4
    snap: int = 50
    ema_rate: float = 0.9999
    precision: str = "bf16"
    schedule_sampler_name: str = "uniform"
    num_fid_samples: int = 10000
    eval_sampler: str = "ddim"
    eval_sampling_steps: int = 100
    grad_accum_steps: int = 1
    gradient_checkpointing: bool = False
    lr_warmup_kimg: int = 0


BASE_CONFIGS = {
    "diffit-256": TrainConfig(image_size=256, lr=3e-4, cfg_scale=4.4),
    "diffit-512": TrainConfig(image_size=512, lr=1e-4, cfg_scale=1.49),
    "diffit-1024": TrainConfig(
        image_size=1024, lr=1e-4, cfg_scale=1.49,
        grad_accum_steps=2, gradient_checkpointing=True, lr_warmup_kimg=1000,
    ),
}


# ---------------------------------------------------------------------------


@click.command()
# Required.
@click.option("--outdir",       help="Where to save training runs", metavar="DIR", type=str, required=True)
@click.option("--cfg",          help="Base configuration", type=click.Choice(list(BASE_CONFIGS.keys())), required=True)
@click.option("--data",         help="Training data path (directory or .zip)", metavar="PATH", type=str, required=True)
@click.option("--gpus",         help="Number of GPUs", metavar="INT", type=click.IntRange(min=1), required=True)
@click.option("--batch-gpu",    help="Batch size per GPU (total = batch-gpu × gpus × grad-accum)", metavar="INT", type=click.IntRange(min=1), required=True)
# Optional overrides (cfg provides defaults).
@click.option("--image-size",   help="Image resolution [default: from cfg]", type=int, default=None)
@click.option("--model",        "model_name", help="Model constructor name [default: from cfg]", type=str, default=None)
@click.option("--kimg",         help="Total training duration [default: from cfg]", metavar="KIMG", type=click.IntRange(min=1), default=None)
@click.option("--tick",         help="How often to print progress [default: from cfg]", metavar="KIMG", type=click.IntRange(min=1), default=None)
@click.option("--snap",         help="How often to snapshot [default: from cfg]", metavar="TICKS", type=click.IntRange(min=1), default=None)
@click.option("--seed",         help="Random seed", metavar="INT", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--lr",           help="Learning rate [default: from cfg]", metavar="FLOAT", type=float, default=None)
@click.option("--precision",    help="Compute precision", type=click.Choice(["fp32", "fp16", "bf16"]), default=None)
@click.option("--ema-rate",     help="EMA decay rate [default: from cfg]", metavar="FLOAT", type=float, default=None)
@click.option("--init-weights", help="Warm-start from a previous stage's EMA snapshot (weights only, fresh optimizer)", metavar="PATH", type=str, default=None)
@click.option("--schedule-sampler", "schedule_sampler_name", help="Timestep sampler [default: from cfg]", type=str, default=None)
@click.option("--num-fid-samples", help="Fakes for eval / combra each snapshot tick (0=disable)", metavar="INT", type=click.IntRange(min=0), default=None)
@click.option("--combra-ref-count", help="Cap the combra reference to a seeded random subset of N reals (0 = whole dataset)", metavar="INT", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--cfg-scale",    help="Classifier-free guidance scale for eval [default: from cfg]", metavar="FLOAT", type=float, default=None)
@click.option("--eval-sampler", help="Sampler for training-time eval/snapshots [default: from cfg]", type=click.Choice(["dpm++", "unipc", "ddim", "ddpm"]), default=None)
@click.option("--eval-sampling-steps", help="Eval sampler steps [default: per-sampler]", metavar="INT", type=click.IntRange(min=1), default=None)
# Performance tuning.
@click.option("--grad-accum",  help="Gradient accumulation steps", metavar="INT", type=click.IntRange(min=1), default=None)
@click.option("--grad-ckpt",   "gradient_checkpointing", help="Gradient checkpointing (trades compute for memory)", metavar="BOOL", type=bool, default=None)
@click.option("--lr-warmup",   help="Linear LR warmup duration in kimg (0 = disabled) [default: from cfg]", metavar="KIMG", type=click.IntRange(min=0), default=None)
@click.option("--tf32",        "allow_tf32", help="Enable TF32 for matmul/conv", metavar="BOOL", type=bool, default=True, show_default=True)
@click.option("--bench",       help="Enable cuDNN autotune (benchmark)", metavar="BOOL", type=bool, default=True, show_default=True)
@click.option("--mirror",      help="Stochastic per-item horizontal flip in the training loader", metavar="BOOL", type=bool, default=False, show_default=True)
@click.option("--workers",     help="DataLoader worker processes", metavar="INT", type=click.IntRange(min=1), default=3, show_default=True)
@click.option("--cache-in-ram", help="Cache the entire dataset in RAM", metavar="BOOL", type=bool, default=True, show_default=True)
# Misc.
@click.option("--desc",        help="String to include in result dir name", metavar="STR", type=str, default=None)
@click.option("--combra-metrics", help="Compute combra generative-quality metrics each snapshot tick", metavar="BOOL", type=bool, default=True, show_default=True)
@click.option("--snapshot-keep-last", help="Keep only the N newest inference snapshots (0 = keep all)", metavar="INT", type=click.IntRange(min=0), default=3, show_default=True)
@click.option("-n", "--dry-run", help="Print training options and exit", is_flag=True)
def main(**kwargs):
    """Train DiffiT on class-conditional data."""
    launch_from_opts(kwargs)


def launch_from_opts(opts):
    """Build the run config from a CLI-style opts dict and launch training."""
    overrides = {}
    if opts["image_size"] is not None:
        overrides["image_size"] = opts["image_size"]
    if opts["model_name"] is not None:
        overrides["model_name"] = opts["model_name"]
    if opts["kimg"] is not None:
        overrides["total_kimg"] = opts["kimg"]
    if opts["tick"] is not None:
        overrides["kimg_per_tick"] = opts["tick"]
    if opts["snap"] is not None:
        overrides["snap"] = opts["snap"]
    if opts["lr"] is not None:
        overrides["lr"] = opts["lr"]
    if opts["precision"] is not None:
        overrides["precision"] = opts["precision"]
    if opts["ema_rate"] is not None:
        overrides["ema_rate"] = opts["ema_rate"]
    if opts["schedule_sampler_name"] is not None:
        overrides["schedule_sampler_name"] = opts["schedule_sampler_name"]
    if opts["num_fid_samples"] is not None:
        overrides["num_fid_samples"] = opts["num_fid_samples"]
    if opts["cfg_scale"] is not None:
        overrides["cfg_scale"] = opts["cfg_scale"]
    if opts["eval_sampler"] is not None:
        overrides["eval_sampler"] = opts["eval_sampler"]
    resolved_eval_sampler = opts["eval_sampler"] or BASE_CONFIGS[opts["cfg"]].eval_sampler
    if opts["eval_sampling_steps"] is not None:
        overrides["eval_sampling_steps"] = opts["eval_sampling_steps"]
    else:
        overrides["eval_sampling_steps"] = {"dpm++": 25, "unipc": 20, "ddim": 100, "ddpm": 250}[resolved_eval_sampler]
    if opts["grad_accum"] is not None:
        overrides["grad_accum_steps"] = opts["grad_accum"]
    if opts["gradient_checkpointing"] is not None:
        overrides["gradient_checkpointing"] = opts["gradient_checkpointing"]
    if opts["lr_warmup"] is not None:
        overrides["lr_warmup_kimg"] = opts["lr_warmup"]

    cfg = dataclasses.replace(BASE_CONFIGS[opts["cfg"]], **overrides)

    c = dict(
        data=opts["data"],
        image_size=cfg.image_size,
        num_gpus=opts["gpus"],
        batch_size=opts["batch_gpu"] * opts["gpus"] * cfg.grad_accum_steps,
        batch_gpu=opts["batch_gpu"],
        total_kimg=cfg.total_kimg,
        kimg_per_tick=cfg.kimg_per_tick,
        snap=cfg.snap,
        seed=opts["seed"],
        lr=cfg.lr,
        precision=cfg.precision,
        ema_rate=cfg.ema_rate,
        init_weights=opts["init_weights"],
        model_name=cfg.model_name,
        schedule_sampler_name=cfg.schedule_sampler_name,
        allow_tf32=opts["allow_tf32"],
        bench=opts["bench"],
        mirror=opts["mirror"],
        workers=opts["workers"],
        cache_in_ram=opts["cache_in_ram"],
        num_fid_samples=cfg.num_fid_samples,
        combra_ref_count=opts["combra_ref_count"],
        cfg_scale=cfg.cfg_scale,
        eval_sampler=cfg.eval_sampler,
        eval_sampling_steps=cfg.eval_sampling_steps,
        grad_accum_steps=cfg.grad_accum_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        lr_warmup_kimg=cfg.lr_warmup_kimg,
        combra_metrics=opts["combra_metrics"],
        snapshot_keep_last=opts["snapshot_keep_last"],
    )

    # Run directory: <id>-<cfg>-gpus<G>-batch<B>[-desc], B = total batch.
    desc = f"{opts['cfg']}-gpus{c['num_gpus']}-batch{c['batch_size']}"
    if opts["desc"] is not None:
        desc += f"-{opts['desc']}"

    launch_training(c=c, desc=desc, outdir=opts["outdir"], dry_run=opts["dry_run"])


if __name__ == "__main__":
    main()
