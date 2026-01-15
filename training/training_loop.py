# Copyright (c) 2024, DiffiT authors.
# Main training loop for DiffiT diffusion model.

from __future__ import annotations

import copy
import json
import os
import pickle
import time
from datetime import datetime

import numpy as np
import PIL.Image
import psutil
import torch
import torch.nn.functional as F

import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix


def setup_snapshot_image_grid(training_set, random_seed: int = 0, gw: int | None = None, gh: int | None = None):
    """Setup grid for snapshot images."""
    rnd = np.random.RandomState(random_seed)
    if gw is None:
        gw = max(np.clip(2560 // training_set.image_shape[2], 7, 32), 1)
    if gh is None:
        gh = max(np.clip(1440 // training_set.image_shape[1], 4, 32), 1)

    all_indices = list(range(len(training_set)))
    rnd.shuffle(all_indices)
    grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)


def save_image_grid(img, fname, drange, grid_size):
    """Save a grid of images to a file."""
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


@torch.no_grad()
def generate_snapshot_images(diffusion, grid_size, device, batch_gpu: int):
    """Generate snapshot images using the diffusion model."""
    gw, gh = grid_size
    n_samples = gw * gh
    all_images = []

    for start_idx in range(0, n_samples, batch_gpu):
        end_idx = min(start_idx + batch_gpu, n_samples)
        batch_size = end_idx - start_idx
        images = diffusion.sample(batch_size)
        all_images.append(images.cpu().numpy())

    return np.concatenate(all_images, axis=0)


def training_loop(
    run_dir: str = ".",
    training_set_kwargs: dict = {},
    data_loader_kwargs: dict = {},
    model_kwargs: dict = {},
    diffusion_kwargs: dict = {},
    opt_kwargs: dict = {},
    random_seed: int = 0,
    num_gpus: int = 1,
    rank: int = 0,
    batch_size: int = 4,
    batch_gpu: int = 4,
    ema_kimg: float = 10.0,
    ema_rampup: float | None = 0.05,
    total_kimg: int = 25000,
    kimg_per_tick: int = 4,
    image_snapshot_ticks: int | None = 50,
    network_snapshot_ticks: int | None = 50,
    resume_pkl: str | None = None,
    resume_kimg: int = 0,
    cudnn_benchmark: bool = True,
    fp32: bool = False,
    abort_fn=None,
    progress_fn=None,
    restart_every: int = -1,
):
    """Main training loop for DiffiT diffusion model."""
    start_time = time.time()
    device = torch.device("cuda", rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    conv2d_gradfix.enabled = True

    __RESTART__ = torch.tensor(0.0, device=device)
    __CUR_NIMG__ = torch.tensor(resume_kimg * 1000, dtype=torch.long, device=device)
    __CUR_TICK__ = torch.tensor(0, dtype=torch.long, device=device)
    __BATCH_IDX__ = torch.tensor(0, dtype=torch.long, device=device)

    # Helper for stage logging
    def stage(msg):
        if rank == 0:
            dt = time.time() - start_time
            dt_min = dt / 60.0
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[Stage {now}] {dt_min:7.1f}m | {msg}", flush=True)

    # Load training set
    stage("Loading training set")
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=training_set, sampler=training_set_sampler, batch_size=batch_size // num_gpus, **data_loader_kwargs
        )
    )
    if rank == 0:
        print()
        print("Num images: ", len(training_set))
        print("Image shape:", training_set.image_shape)
        print()

    # Construct model
    stage("Constructing model")
    model = dnnlib.util.construct_class_by_name(**model_kwargs).train().requires_grad_(False).to(device)
    model_ema = copy.deepcopy(model).eval()

    # Construct diffusion
    stage("Constructing diffusion")
    diffusion = dnnlib.util.construct_class_by_name(model=model, **diffusion_kwargs)
    diffusion_ema = dnnlib.util.construct_class_by_name(model=model_ema, **diffusion_kwargs)

    # Check for existing checkpoint
    ckpt_pkl = None
    if restart_every > 0 and os.path.isfile(misc.get_ckpt_path(run_dir)):
        ckpt_pkl = resume_pkl = misc.get_ckpt_path(run_dir)

    if resume_pkl is not None and rank == 0:
        stage(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = pickle.load(f)
        misc.copy_params_and_buffers(resume_data["model"], model, require_all=False)
        misc.copy_params_and_buffers(resume_data["model_ema"], model_ema, require_all=False)

        if ckpt_pkl is not None:
            __CUR_NIMG__ = resume_data["progress"]["cur_nimg"].to(device)
            __CUR_TICK__ = resume_data["progress"]["cur_tick"].to(device)
            __BATCH_IDX__ = resume_data["progress"]["batch_idx"].to(device)

    # Print network summary
    if rank == 0:
        x = torch.empty([batch_gpu, *training_set.image_shape], device=device)
        t = torch.zeros([batch_gpu], dtype=torch.long, device=device)
        misc.print_module_summary(model, [x, t])

    # Distribute across GPUs
    stage(f"Distributing across {num_gpus} GPUs")
    for module in [model, model_ema]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)

    # Setup optimizer
    stage("Setting up optimizer")
    optimizer = dnnlib.util.construct_class_by_name(params=model.parameters(), **opt_kwargs)

    # Export sample images
    grid_size = None
    stage("Exporting sample images")
    grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
    if rank == 0:
        save_image_grid(images, os.path.join(run_dir, "reals.png"), drange=[0, 255], grid_size=grid_size)

    # Generate initial samples
    if rank == 0:
        initial_samples = generate_snapshot_images(diffusion_ema, grid_size, device, batch_gpu)
        save_image_grid(initial_samples, os.path.join(run_dir, "fakes_init.png"), drange=[0, 1], grid_size=grid_size)

    # Initialize logs
    stage("Initializing logs")
    stats_collector = training_stats.Collector(regex=".*")
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "wt")
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print("Skipping tfevents export:", err)

    # Train
    stage(f"Training start (total_kimg={total_kimg}, batch_size={batch_size}, batch_gpu={batch_gpu})")
    if num_gpus > 1:
        torch.distributed.broadcast(__CUR_NIMG__, 0)
        torch.distributed.broadcast(__CUR_TICK__, 0)
        torch.distributed.broadcast(__BATCH_IDX__, 0)
        torch.distributed.barrier()

    cur_nimg = __CUR_NIMG__.item()
    cur_tick = __CUR_TICK__.item()
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = __BATCH_IDX__.item()
    if progress_fn is not None:
        progress_fn(cur_nimg // 1000, total_kimg)

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=not fp32)

    while True:
        # Fetch training batch
        with torch.autograd.profiler.record_function("data_fetch"):
            batch_images, batch_labels = next(training_set_iterator)
            batch_images = (batch_images.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)

        # Training step
        model.requires_grad_(True)
        optimizer.zero_grad(set_to_none=True)

        for real_img in batch_images:
            # Scale from [-1, 1] to [0, 1] for diffusion (it will rescale internally)
            real_img_01 = (real_img + 1) / 2

            with torch.cuda.amp.autocast(enabled=not fp32):
                # Forward pass: perturb and predict
                _, epsilon, pred_epsilon = diffusion.perturb_and_predict(real_img_01)
                loss = F.mse_loss(epsilon, pred_epsilon)

            # Backward pass
            scaler.scale(loss).backward()
            training_stats.report("Loss/train", loss)

        # Update weights
        model.requires_grad_(False)
        with torch.autograd.profiler.record_function("opt_step"):
            # Gradient clipping
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            training_stats.report("Loss/grad_norm", grad_norm)

            # All-reduce gradients if distributed
            if num_gpus > 1:
                params = [p for p in model.parameters() if p.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([p.grad.flatten() for p in params])
                    torch.distributed.all_reduce(flat)
                    flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([p.numel() for p in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)

            scaler.step(optimizer)
            scaler.update()

        # Update EMA
        with torch.autograd.profiler.record_function("ema_update"):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(model_ema.parameters(), model.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(model_ema.buffers(), model.buffers()):
                b_ema.copy_(b)

        # Update state
        cur_nimg += batch_size
        batch_idx += 1

        # Perform maintenance tasks once per tick
        done = cur_nimg >= total_kimg * 1000
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        training_stats.report0("Timing/total_hours", (tick_end_time - start_time) / (60 * 60))
        training_stats.report0("Timing/total_days", (tick_end_time - start_time) / (24 * 60 * 60))

        if rank == 0:
            print(" ".join(fields))

        # Check for abort
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print("Aborting...")

        # Check for restart
        if (rank == 0) and (restart_every > 0) and (time.time() - start_time > restart_every):
            print("Restart job...")
            __RESTART__ = torch.tensor(1.0, device=device)
        if num_gpus > 1:
            torch.distributed.broadcast(__RESTART__, 0)
        if __RESTART__:
            done = True
            print(f"Process {rank} leaving...")
            if num_gpus > 1:
                torch.distributed.barrier()

        # Save image snapshot
        if (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            stage(f"Saving image snapshot (kimg={cur_nimg/1e3:.1f})")
            if rank == 0:
                images = generate_snapshot_images(diffusion_ema, grid_size, device, batch_gpu)
                save_image_grid(images, os.path.join(run_dir, f"fakes{cur_nimg//1000:06d}.png"), drange=[0, 1], grid_size=grid_size)

        # Save network snapshot
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            stage(f"Preparing network snapshot (kimg={cur_nimg/1e3:.1f})")
            snapshot_data = dict(model=model, model_ema=model_ema, training_set_kwargs=dict(training_set_kwargs))

        # Save checkpoint
        if (rank == 0) and (restart_every > 0) and (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_pkl = misc.get_ckpt_path(run_dir)
            stage(f'Saving checkpoint "{snapshot_pkl}"')
            snapshot_data["progress"] = {
                "cur_nimg": torch.tensor(cur_nimg, dtype=torch.long),
                "cur_tick": torch.tensor(cur_tick, dtype=torch.long),
                "batch_idx": torch.tensor(batch_idx, dtype=torch.long),
            }
            with open(snapshot_pkl, "wb") as f:
                pickle.dump(snapshot_data, f)

        del snapshot_data

        # Collect statistics
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + "\n")
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done
    if rank == 0:
        stage("Exiting")
