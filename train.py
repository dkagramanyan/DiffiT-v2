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
from contextlib import nullcontext

import click
import numpy as np
import PIL.Image
import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers.models import AutoencoderKL
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import diffit.diffit as diffit_module
from diffit import create_diffusion, diffusion_defaults, NUM_CLASSES
from diffit.dist_util import setup_dist, dev, get_rank, get_world_size, synchronize
from diffit.image_datasets import load_data
from diffit.nn import update_ema
from diffit import logger
from diffit.timestep_sampler import create_named_schedule_sampler
from diffit.dpm_solver import dpm_solver_sample
from diffit.metrics import (
    InceptionFeatureExtractor,
    compute_activations,
    evaluate_metrics,
)


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
        gw = int(max(np.clip(2560 // image_size, 7, 32), 1))
    if gh is None:
        gh = int(max(np.clip(1440 // image_size, 4, 32), 1))
    return gw, gh


@torch.inference_mode()
def generate_snapshot_images(
    ema_model, vae, diffusion, grid_z, grid_classes, batch_gpu, device,
    *,
    cfg_scale, num_sampling_steps=25, scale_pow=4.0,
):
    """Generate a batch of images from the EMA model for snapshot grids."""
    all_samples = []
    for z_chunk, c_chunk in zip(grid_z.split(batch_gpu), grid_classes.split(batch_gpu)):
        bs = z_chunk.shape[0]
        z_cfg = torch.cat([z_chunk, z_chunk], 0)
        classes_null = torch.full((bs,), NUM_CLASSES, device=device, dtype=torch.long)
        model_kwargs = {
            "y": torch.cat([c_chunk, classes_null], 0),
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
        all_samples.append(decoded.permute(0, 2, 3, 1).cpu().numpy())

    # Return as NCHW uint8 for save_image_grid
    images = np.concatenate(all_samples, axis=0)
    # Convert NHWC -> NCHW
    images = images.transpose(0, 3, 1, 2)
    return images


def subprocess_fn(rank, c, temp_dir):
    """Entry point for each DDP worker."""
    logger.configure(log_dir=c["run_dir"])

    # Pin this process to its GPU *before* initializing the NCCL process
    # group. NCCL inspects the current CUDA device at init time to build
    # its communicator; flipping devices afterwards causes silent
    # contention and (sometimes) hangs on the first collective.
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = c.get("allow_tf32", True)
    torch.backends.cudnn.allow_tf32 = c.get("allow_tf32", True)
    torch.backends.cudnn.benchmark = True

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
    cfg_scale,
    grad_accum_steps=1,
    gradient_checkpointing=False,
    lr_warmup_kimg=0,
    amp_dtype="fp16",
    workers=4,
    cache_in_ram=False,
    num_fid_samples=1024,
    **_extra,
):
    """Main training loop for DiffiT."""
    device = torch.device("cuda", rank)
    is_main = rank == 0

    # Seeding
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)

    # Resolve AMP dtype: bf16 preferred on Ampere+ (A100, H100), fp16 fallback
    if amp_dtype == "bf16" and torch.cuda.is_bf16_supported():
        amp_dtype_torch = torch.bfloat16
    else:
        amp_dtype_torch = torch.float16
    amp_enabled = use_fp16
    use_grad_scaler = amp_enabled and (amp_dtype_torch == torch.float16)
    if is_main:
        logger.log(f"AMP: enabled={amp_enabled}, dtype={amp_dtype_torch}, grad_scaler={use_grad_scaler}")

    # Discover dataset class count before building model so embedding table
    # matches the dataset (avoids wasted untrained-class params).
    probe_iter = load_data(
        data_dir=data, batch_size=64, image_size=image_size,
        class_cond=True, random_flip=False, num_workers=2,
        distributed=False,
    )
    discovered_classes = set()
    for _ in range(50):  # probe up to 3200 samples
        _, cond_probe = next(probe_iter)
        if "y" in cond_probe:
            discovered_classes.update(cond_probe["y"].numpy().tolist())
    num_dataset_classes = max(len(discovered_classes), 1)
    sorted_class_list = sorted(discovered_classes)
    del probe_iter
    if is_main:
        logger.log(f"Discovered {num_dataset_classes} classes in dataset.")

    # Build model (sized to the dataset's actual class count).
    latent_size = image_size // 8
    model = diffit_module.__dict__[model_name](
        input_size=latent_size, num_classes=num_dataset_classes,
    )
    model.to(device)

    if gradient_checkpointing:
        model.gradient_checkpointing = True
        if is_main:
            logger.log("Gradient checkpointing enabled — "
                       "block activations will be recomputed during backward.")

    # EMA model (deep-copied *after* setting checkpointing flag so the
    # flag is present, but eval never triggers it because of the
    # `and self.training` guard in forward).
    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)
    ema_model.eval()

    # Create diffusion
    diff_config = diffusion_defaults()
    diffusion = create_diffusion(**diff_config)
    schedule_sampler = create_named_schedule_sampler(schedule_sampler_name, diffusion)

    # Optimizer — fused=True keeps param updates in a single CUDA kernel
    # and is ~15-20 % faster than the default loop on Ampere / Hopper.
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, fused=True)

    # GradScaler for fp16 (not needed for bf16)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_grad_scaler)

    # Wrap with DDP first, then torch.compile (recommended order: compile
    # traces through DDP's forward/backward hooks so allreduce is part of
    # the compiled graph).  We keep a reference to the raw DDP wrapper for
    # the no_sync() context needed during gradient accumulation.
    if num_gpus > 1:
        ddp_raw = DDP(
            model,
            device_ids=[rank],
            gradient_as_bucket_view=True,
        )
    else:
        ddp_raw = model

    # max-autotune enables CUDA-graph capture, which conflicts with
    # gradient checkpointing (the replayed graph overwrites tensors the
    # backward recomputation still needs).  Use the no-cudagraphs
    # variant when checkpointing is active.
    compile_mode = "max-autotune-no-cudagraphs" if gradient_checkpointing else "max-autotune"
    ddp_model = torch.compile(ddp_raw, mode=compile_mode)

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
        if "scaler" in ckpt and use_grad_scaler:
            scaler.load_state_dict(ckpt["scaler"])
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
        cache_in_ram=cache_in_ram,
    )

    # --- Initialize logs: stats.jsonl + TensorBoard (SAN-v2 style) ---
    stats_jsonl = None
    stats_tfevents = None
    if is_main:
        stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "at")
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
            logger.register_tb_writer(stats_tfevents, stats_jsonl)
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
        n_collected = 0
        while n_collected < num_fid_samples:
            batch_ref, _ = next(ref_iter)
            batch_np = ((batch_ref + 1) * 127.5).clamp(0, 255).to(torch.uint8).numpy()
            ref_images.append(batch_np)
            n_collected += batch_np.shape[0]
        ref_images = np.concatenate(ref_images, axis=0)[:num_fid_samples]
        ref_acts = compute_activations(ref_images, inception_extractor, batch_size=64, device=device)
        del ref_images
        logger.log("Reference features computed.")

    # Build class-sorted grid: each row cycles through classes
    # Row 0 → class 0, row 1 → class 1, ..., row K → class K % num_classes
    grid_row_classes = [sorted_class_list[r % num_dataset_classes] for r in range(gh)]

    # Save real training images grid (sorted by class)
    if is_main:
        logger.log("Exporting sample images (reals.png, fakes_init.png)...")
        # Collect images per class
        class_images = {c: [] for c in sorted_class_list}
        images_per_class_needed = gw
        collect_iter = load_data(
            data_dir=data, batch_size=64, image_size=image_size,
            class_cond=True, random_flip=False, num_workers=2,
            distributed=False,
        )
        # How many images we need per class (some classes appear in multiple rows)
        class_count_needed = {}
        for c in grid_row_classes:
            class_count_needed[c] = class_count_needed.get(c, 0) + gw

        while any(len(class_images[c]) < class_count_needed.get(c, 0) for c in sorted_class_list):
            img_batch, cond_batch = next(collect_iter)
            if "y" not in cond_batch:
                break
            for i in range(img_batch.shape[0]):
                c = int(cond_batch["y"][i].item())
                if c in class_images and len(class_images[c]) < class_count_needed.get(c, 0):
                    img_np = ((img_batch[i:i+1] + 1) * 127.5).clamp(0, 255).to(torch.uint8).numpy()
                    class_images[c].append(img_np)

        # Assemble grid: row by row
        grid_images = []
        class_cursors = {c: 0 for c in sorted_class_list}
        for r in range(gh):
            c = grid_row_classes[r]
            for col in range(gw):
                idx = class_cursors[c]
                if idx < len(class_images[c]):
                    grid_images.append(class_images[c][idx])
                    class_cursors[c] += 1
                else:
                    # Fallback: black image
                    grid_images.append(np.zeros((1, 3, image_size, image_size), dtype=np.uint8))
        real_np = np.concatenate(grid_images, axis=0)
        save_image_grid(real_np, os.path.join(run_dir, "reals.png"), drange=[0, 255], grid_size=grid_size)
        del class_images, collect_iter

    # Fixed latents + class-sorted classes for consistent snapshot generation
    grid_z = torch.randn(
        n_grid, 4, latent_size, latent_size, device=device,
        generator=torch.Generator(device=device).manual_seed(seed * max(num_gpus, 1)),
    )
    # Each row gets a single class, cycling through discovered classes
    grid_classes_list = []
    for r in range(gh):
        c = grid_row_classes[r]
        grid_classes_list.extend([c] * gw)
    grid_classes = torch.tensor(grid_classes_list, device=device, dtype=torch.long)

    # Save initial fakes
    if is_main:
        fakes_init = generate_snapshot_images(
            ema_model, vae, diffusion, grid_z, grid_classes,
            batch_gpu=batch_gpu, device=device,
            cfg_scale=cfg_scale,
        )
        save_image_grid(fakes_init, os.path.join(run_dir, "fakes_init.png"), drange=[0, 255], grid_size=grid_size)

    # Synchronize all ranks before entering the training loop (rank 0 may
    # still be generating snapshot images / computing reference features).
    if num_gpus > 1:
        dist.barrier()

    # Training
    effective_batch = batch_gpu * num_gpus * grad_accum_steps
    if is_main:
        logger.log(
            f"Training for {total_kimg} kimg  "
            f"(batch_gpu={batch_gpu} × {num_gpus} GPUs × {grad_accum_steps} accum "
            f"= {effective_batch} effective batch)"
        )
        if lr_warmup_kimg > 0:
            logger.log(f"LR warmup: 0 → {lr} over {lr_warmup_kimg} kimg")

    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = start_time - time.time()  # negative initially
    cur_tick = 0
    opt_steps = 0

    def _format_time(seconds):
        """Format seconds into human-readable string like SAN-v2."""
        s = int(seconds)
        if s < 60:
            return f"{s}s"
        elif s < 3600:
            return f"{s // 60}m {s % 60:02d}s"
        else:
            h = s // 3600
            m = (s % 3600) // 60
            sec = s % 60
            return f"{h}h {m:02d}m {sec:02d}s"

    while cur_nimg < total_kimg * 1000:
        opt.zero_grad(set_to_none=True)

        # --- Gradient accumulation over micro-batches -----------------------
        for micro_step in range(grad_accum_steps):
            batch, cond = next(data_iter)
            batch = batch.to(device, non_blocking=True)

            # Encode to latent space
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype_torch, enabled=amp_enabled):
                latent = vae.encode(batch).latent_dist.sample() * 0.18215

            # Sample timesteps
            t, weights = schedule_sampler.sample(latent.shape[0], device)

            # Model kwargs
            model_kwargs = {}
            if "y" in cond:
                model_kwargs["y"] = cond["y"].to(device, non_blocking=True)

            # Skip DDP allreduce on all micro-steps except the last to
            # avoid redundant gradient synchronisation.
            is_last_micro = (micro_step == grad_accum_steps - 1)
            sync_ctx = nullcontext() if is_last_micro or num_gpus <= 1 else ddp_raw.no_sync()

            with sync_ctx:
                with torch.amp.autocast("cuda", dtype=amp_dtype_torch, enabled=amp_enabled):
                    losses = diffusion.training_losses(ddp_model, latent, t, model_kwargs=model_kwargs)
                    loss = (losses["loss"] * weights).mean() / grad_accum_steps

                scaler.scale(loss).backward()

            cur_nimg += batch_gpu * num_gpus

            # Accumulate loss for tick-level reporting (undo the 1/accum
            # scaling so the logged value reflects per-sample magnitude).
            logger.logkv_mean("Loss/train", loss.item() * grad_accum_steps)
            if "vb" in losses:
                logger.logkv_mean("Loss/vb", losses["vb"].mean().item())

        # --- Optimizer step (once per effective batch) ----------------------
        scaler.step(opt)
        scaler.update()
        opt_steps += 1

        # LR warmup (linear ramp from 0 → lr)
        if lr_warmup_kimg > 0:
            warmup_nimg = lr_warmup_kimg * 1000
            warmup_frac = min(1.0, cur_nimg / warmup_nimg)
            for pg in opt.param_groups:
                pg["lr"] = lr * warmup_frac

        # Update EMA
        update_ema(ema_model.parameters(), model.parameters(), rate=ema_rate)

        if "mse" in losses:
            logger.logkv_mean("Loss/mse", losses["mse"].mean().item())

        # Tick
        done_kimg = (cur_nimg - tick_start_nimg) / 1000
        if done_kimg >= kimg_per_tick or cur_nimg >= total_kimg * 1000:
            tick_end_time = time.time()
            tick_elapsed = tick_end_time - tick_start_time
            total_elapsed = tick_end_time - start_time
            sec_per_tick = tick_elapsed
            sec_per_kimg = tick_elapsed / done_kimg if done_kimg > 0 else 0

            # Remaining time estimate
            kimg_done = cur_nimg / 1000
            kimg_left = total_kimg - kimg_done
            eta_sec = sec_per_kimg * kimg_left if sec_per_kimg > 0 else 0

            # Memory stats
            cpumem = 0.0
            gpumem = 0.0
            reserved = 0.0
            try:
                import psutil
                cpumem = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
            except ImportError:
                pass
            if torch.cuda.is_available():
                gpumem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

            # SAN-v2 style tick line
            tick_line = (
                f"tick {cur_tick:<6d} "
                f"kimg {kimg_done:<10.1f} "
                f"time {_format_time(total_elapsed):<14s} "
                f"sec/tick {sec_per_tick:<9.1f} "
                f"sec/kimg {sec_per_kimg:<9.2f} "
                f"eta {_format_time(eta_sec):<14s} "
                f"maintenance {abs(maintenance_time):<7.1f} "
                f"cpumem {cpumem:<7.2f} "
                f"gpumem {gpumem:<8.2f} "
                f"reserved {reserved:<8.2f}"
            )
            logger.log(tick_line)

            # Collect logged loss means
            logged_kvs = logger.dumpkvs()
            logged_kvs["kimg"] = kimg_done
            logged_kvs["sec/tick"] = sec_per_tick
            logged_kvs["sec/kimg"] = sec_per_kimg

            # Write stats.jsonl + TensorBoard
            if is_main:
                timestamp = time.time()
                global_step = int(kimg_done)
                walltime = timestamp - start_time

                if stats_jsonl is not None:
                    fields = dict(logged_kvs, timestamp=timestamp)
                    stats_jsonl.write(json.dumps(fields) + "\n")
                    stats_jsonl.flush()

                if stats_tfevents is not None:
                    for name, value in logged_kvs.items():
                        if isinstance(value, (int, float, float)):
                            stats_tfevents.add_scalar(f"Train/{name}", value, global_step=global_step, walltime=walltime)
                    stats_tfevents.flush()

            cur_tick += 1
            tick_start_nimg = cur_nimg
            tick_start_time = time.time()
            maintenance_time = 0.0

            # Save image snapshot + evaluate metrics + checkpoint (every `snap` ticks)
            do_snap = cur_tick % snap == 0
            if do_snap:
                snap_start = time.time()

                # Snapshot grid (rank 0 only, fast)
                if is_main:
                    logger.log(f"Saving image snapshot (kimg={cur_nimg / 1e3:.1f})...")
                    fakes = generate_snapshot_images(
                        ema_model, vae, diffusion, grid_z, grid_classes,
                        batch_gpu=batch_gpu, device=device,
                        cfg_scale=cfg_scale,
                    )
                    save_image_grid(
                        fakes, os.path.join(run_dir, f"fakes{cur_nimg // 1000:06d}.png"),
                        drange=[0, 255], grid_size=grid_size,
                    )
                    logger.log(f"Image snapshot saved (kimg={cur_nimg / 1e3:.1f})")

                # Evaluate quality metrics (ALL ranks generate samples)
                if num_fid_samples > 0:
                    stats_metrics = evaluate_metrics(
                        ema_model, vae, diffusion, ref_acts, inception_extractor,
                        num_fid_samples, batch_gpu, latent_size, device,
                        cfg_scale=cfg_scale,
                        rank=rank, world_size=num_gpus,
                        log_fn=logger.log,
                        class_list=sorted_class_list,
                        null_class_idx=num_dataset_classes,
                    )
                    # Only rank 0 logs and saves metrics
                    if is_main and stats_metrics is not None:
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

                # Save checkpoint (rank 0 only) — inference-only: EMA weights as a raw state_dict
                if is_main:
                    save_path = os.path.join(run_dir, f"network-snapshot-{cur_nimg // 1000:06d}.pt")
                    logger.log(f"Saving checkpoint to {save_path}...")
                    torch.save(ema_model.state_dict(), save_path)
                    logger.log(f"Checkpoint saved (kimg={cur_nimg / 1e3:.1f})")

                    maintenance_time = time.time() - snap_start

    # Final save — inference-only: EMA weights as a raw state_dict
    if is_main:
        save_path = os.path.join(run_dir, "network-final.pt")
        logger.log(f"Saving final model to {save_path}...")
        torch.save(ema_model.state_dict(), save_path)

        # Final image snapshot
        logger.log("Saving final image snapshot...")
        fakes = generate_snapshot_images(
            ema_model, vae, diffusion, grid_z, grid_classes,
            batch_gpu=batch_gpu, device=device,
            cfg_scale=cfg_scale,
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
    print(f"Mixed precision:     {c['use_fp16']} (dtype={c['amp_dtype']})")
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
        lr=3e-4,
        total_kimg=400000,
        kimg_per_tick=4,
        snap=50,
        ema_rate=0.9999,
        use_fp16=True,
        schedule_sampler_name="uniform",
        num_fid_samples=10000,
        cfg_scale=4.4,
        grad_accum_steps=1,
        gradient_checkpointing=False,
        lr_warmup_kimg=0,
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
        num_fid_samples=10000,
        cfg_scale=1.49,
        grad_accum_steps=1,
        gradient_checkpointing=False,
        lr_warmup_kimg=0,
    ),
    "diffit-1024": dict(
        image_size=1024,
        model_name="Diffit",
        lr=1e-4,
        total_kimg=400000,
        kimg_per_tick=4,
        snap=50,
        ema_rate=0.9999,
        use_fp16=True,
        schedule_sampler_name="uniform",
        num_fid_samples=512,
        cfg_scale=1.49,
        # 1024² is memory-bound: enable checkpointing by default and
        # use 2-step accumulation to double the effective batch.
        grad_accum_steps=2,
        gradient_checkpointing=True,
        lr_warmup_kimg=1000,
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
@click.option("--amp-dtype",   help="AMP dtype: fp16 or bf16 (bf16 preferred on A100/H100)",   type=click.Choice(["fp16", "bf16"]), default="bf16", show_default=True)
@click.option("--ema-rate",     help="EMA decay rate [default: from cfg]", metavar="FLOAT",    type=float, default=None)
@click.option("--resume",       help="Resume from checkpoint path", metavar="PATH",            type=str, default=None)
@click.option("--schedule-sampler", "schedule_sampler_name", help="Timestep sampler [default: from cfg]", type=str, default=None)
@click.option("--num-fid-samples", help="Samples for FID eval during training (0=disable)", metavar="INT", type=click.IntRange(min=0), default=None)
@click.option("--cfg-scale",    help="Classifier-free guidance scale used during training-time eval [default: from cfg]", metavar="FLOAT", type=float, default=None)

# Performance tuning.
@click.option("--grad-accum",  help="Gradient accumulation steps (effective batch = batch-gpu × gpus × accum)", metavar="INT", type=click.IntRange(min=1), default=None)
@click.option("--grad-ckpt/--no-grad-ckpt", "gradient_checkpointing", help="Enable gradient checkpointing (trades compute for memory)", default=None)
@click.option("--lr-warmup",   help="Linear LR warmup duration in kimg (0 = disabled) [default: from cfg]", metavar="KIMG", type=click.IntRange(min=0), default=None)

# Misc settings.
@click.option("--desc",         help="String to include in result dir name", metavar="STR",    type=str, default=None)
@click.option("--metrics",      help="Quality metrics", metavar="NAME",                        type=str, default="fid50k", show_default=True)
@click.option("--tf32/--no-tf32", "allow_tf32", help="Enable TF32 for matmul/conv",            default=True, show_default=True)
@click.option("--workers",      help="DataLoader worker processes", metavar="INT",             type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--cache-in-ram/--no-cache-in-ram", help="Cache entire dataset in RAM to reduce disk I/O", default=True, show_default=True)
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
    if opts["cfg_scale"] is not None:
        cfg["cfg_scale"] = opts["cfg_scale"]
    if opts["grad_accum"] is not None:
        cfg["grad_accum_steps"] = opts["grad_accum"]
    if opts["gradient_checkpointing"] is not None:
        cfg["gradient_checkpointing"] = opts["gradient_checkpointing"]
    if opts["lr_warmup"] is not None:
        cfg["lr_warmup_kimg"] = opts["lr_warmup"]

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
        amp_dtype=opts["amp_dtype"],
        ema_rate=cfg["ema_rate"],
        resume=opts["resume"],
        model_name=cfg["model_name"],
        schedule_sampler_name=cfg["schedule_sampler_name"],
        allow_tf32=opts["allow_tf32"],
        workers=opts["workers"],
        cache_in_ram=opts["cache_in_ram"],
        num_fid_samples=cfg["num_fid_samples"],
        cfg_scale=cfg["cfg_scale"],
        grad_accum_steps=cfg["grad_accum_steps"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        lr_warmup_kimg=cfg["lr_warmup_kimg"],
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
