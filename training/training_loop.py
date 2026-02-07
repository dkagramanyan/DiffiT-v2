# Copyright (c) 2024, DiffiT authors.
# Main training loop for DiffiT diffusion model.
#
# Implements the training objective from the DiffiT paper (Eq. 1):
#   L = E[λ(t) * ||ε - ε_θ(z_0 + σ_t * ε, t)||^2]
#
# Multi-GPU training uses DistributedDataParallel for efficient gradient sync.
# Includes TensorBoard logging and FID evaluation.

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
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from metrics import metric_main


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
def generate_snapshot_images(
    diffusion, 
    grid_size, 
    device, 
    batch_gpu: int,
    labels: np.ndarray | None = None,
    cfg_scale: float = 1.0,
):
    """Generate snapshot images using the diffusion model.
    
    Args:
        diffusion: Diffusion model wrapper.
        grid_size: Tuple (width, height) of the grid.
        device: Device to generate on.
        batch_gpu: Batch size per GPU.
        labels: Optional class labels for each image in the grid.
        cfg_scale: Classifier-free guidance scale (1.0 = no guidance).
    """
    gw, gh = grid_size
    n_samples = gw * gh
    all_images = []

    for start_idx in range(0, n_samples, batch_gpu):
        end_idx = min(start_idx + batch_gpu, n_samples)
        batch_size = end_idx - start_idx
        
        # Get labels for this batch if provided
        batch_labels = None
        if labels is not None:
            batch_labels = torch.from_numpy(labels[start_idx:end_idx]).long().to(device)
        
        images = diffusion.sample_ddim(
            batch_size, 
            labels=batch_labels, 
            cfg_scale=cfg_scale,
            num_inference_steps=50,
        )
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
    # Metrics parameters
    metrics: list = [],
    metrics_ticks: int | None = None,
    fid_num_samples: int = 10000,
    fid_inference_steps: int = 50,
    # Class conditioning parameters
    use_labels: bool = False,
    label_dim: int = 0,
    cfg_scale: float = 1.5,  # CFG scale for snapshot generation
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
        print("Use labels: ", use_labels)
        if use_labels:
            print("Label dim:  ", label_dim)
            print("CFG scale:  ", cfg_scale)
        print()

    # Construct model
    stage("Constructing model")
    model = dnnlib.util.construct_class_by_name(**model_kwargs).train().to(device)
    model_ema = copy.deepcopy(model).eval().requires_grad_(False)

    # Construct diffusion for EMA model (used for sampling/snapshots)
    stage("Constructing diffusion")
    diffusion_ema = dnnlib.util.construct_class_by_name(model=model_ema, **diffusion_kwargs)
    # Note: For training, we'll create diffusion with ddp_model after DDP wrapping

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

    # Print network summary (use base model before DDP wrapping)
    if rank == 0:
        x = torch.empty([batch_gpu, *training_set.image_shape], device=device)
        t = torch.zeros([batch_gpu], dtype=torch.long, device=device)
        with torch.no_grad():
            misc.print_module_summary(model, [x, t])

    # Distribute across GPUs using DistributedDataParallel
    stage(f"Distributing across {num_gpus} GPUs")
    ddp_model = None
    if num_gpus > 1:
        # Sync initial model weights across all GPUs
        for module in [model, model_ema]:
            if module is not None:
                for param in misc.params_and_buffers(module):
                    dist.broadcast(param, src=0)
        
        # Wrap model in DDP for automatic gradient synchronization
        ddp_model = DDP(
            model, 
            device_ids=[rank],
            output_device=rank,
            broadcast_buffers=True,
            find_unused_parameters=False,
        )
    else:
        ddp_model = model

    # Setup optimizer (use ddp_model.module for DDP to get the underlying model)
    stage("Setting up optimizer")
    base_model = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model
    optimizer = dnnlib.util.construct_class_by_name(params=base_model.parameters(), **opt_kwargs)
    
    # Create diffusion wrapper for training (uses DDP-wrapped model)
    diffusion = dnnlib.util.construct_class_by_name(model=ddp_model, **diffusion_kwargs)

    # Export sample images
    grid_size = None
    stage("Exporting sample images")
    grid_size, images, grid_labels = setup_snapshot_image_grid(training_set=training_set)
    
    # Prepare labels for snapshot generation
    snapshot_labels = None
    if use_labels and grid_labels is not None:
        if grid_labels.ndim == 2:
            # One-hot encoded, convert to indices
            snapshot_labels = np.argmax(grid_labels, axis=1)
        else:
            snapshot_labels = grid_labels.astype(np.int64)
    
    if rank == 0:
        save_image_grid(images, os.path.join(run_dir, "reals.png"), drange=[0, 255], grid_size=grid_size)

    # Generate initial samples
    if rank == 0:
        initial_samples = generate_snapshot_images(
            diffusion_ema, grid_size, device, batch_gpu,
            labels=snapshot_labels,
            cfg_scale=cfg_scale if use_labels else 1.0,
        )
        save_image_grid(initial_samples, os.path.join(run_dir, "fakes_init.png"), drange=[0, 1], grid_size=grid_size)

    # Initialize logs and TensorBoard
    stage("Initializing logs")
    stats_collector = training_stats.Collector(regex=".*")
    stats_jsonl = None
    stats_tfevents = None
    stats_metrics = dict()  # Store metric results for logging
    best_fid = float('inf')  # Track best FID score
    
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "wt")
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
            
            # Log training hyperparameters to TensorBoard
            hparams = {
                'batch_size': batch_size,
                'batch_gpu': batch_gpu,
                'learning_rate': opt_kwargs.get('lr', 0),
                'total_kimg': total_kimg,
                'ema_kimg': ema_kimg,
                'num_gpus': num_gpus,
                'resolution': training_set.image_shape[1],
                'base_dim': model_kwargs.get('base_dim', 0),
                'hidden_dim': model_kwargs.get('hidden_dim', 0),
                'num_heads': model_kwargs.get('num_heads', 0),
                'timesteps': diffusion_kwargs.get('n_times', 1000),
                'fp32': fp32,
                'use_labels': use_labels,
                'label_dim': label_dim,
                'cfg_scale': cfg_scale,
            }
            # Log hyperparameters as text
            hparams_text = '\n'.join([f'{k}: {v}' for k, v in hparams.items()])
            stats_tfevents.add_text('Hyperparameters', hparams_text, global_step=0)
            
            # Log model architecture info
            num_params = sum(p.numel() for p in base_model.parameters())
            num_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
            stats_tfevents.add_text('Model/info', 
                f'Total parameters: {num_params:,}\nTrainable: {num_trainable:,}', 
                global_step=0)
            
        except ImportError as err:
            print("Skipping TensorBoard export:", err)

    # Train
    stage(f"Training start (total_kimg={total_kimg}, batch_size={batch_size}, batch_gpu={batch_gpu})")
    if num_gpus > 1:
        dist.broadcast(__CUR_NIMG__, 0)
        dist.broadcast(__CUR_TICK__, 0)
        dist.broadcast(__BATCH_IDX__, 0)
        dist.barrier()

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
            
            # Prepare labels if using conditional training
            if use_labels:
                # Convert labels to class indices
                if batch_labels.ndim == 2:
                    # One-hot encoded, convert to indices
                    batch_labels = batch_labels.argmax(dim=1)
                batch_labels = batch_labels.long().to(device).split(batch_gpu)
            else:
                batch_labels = [None] * len(batch_images)

        # Training step
        # DDP handles gradient sync automatically during backward pass
        optimizer.zero_grad(set_to_none=True)
        
        # Gradient accumulation: split batch and accumulate
        num_accumulation_steps = len(batch_images)
        
        for accum_idx, (real_img, labels_batch) in enumerate(zip(batch_images, batch_labels)):
            # Scale from [-1, 1] to [0, 1] for diffusion (it will rescale internally)
            real_img_01 = (real_img + 1) / 2
            
            # Only sync gradients on the last accumulation step (DDP optimization)
            sync_grads = (accum_idx == num_accumulation_steps - 1)
            
            with misc.ddp_sync(ddp_model, sync_grads):
                with torch.cuda.amp.autocast(enabled=not fp32):
                    # Forward pass: perturb and predict (paper Eq. 1)
                    # The diffusion uses the model to predict noise ε_θ
                    # Note: labels are passed for conditional training; model handles dropout internally
                    _, epsilon, pred_epsilon = diffusion.perturb_and_predict(real_img_01, labels=labels_batch)
                    # MSE loss: ||ε - ε_θ||^2 (paper Eq. 1, with λ(t) = 1)
                    loss = F.mse_loss(epsilon, pred_epsilon)
                    # Scale loss for gradient accumulation
                    loss = loss / num_accumulation_steps

                # Backward pass
                scaler.scale(loss).backward()
            
            training_stats.report("Loss/train", loss * num_accumulation_steps)

        # Update weights
        with torch.autograd.profiler.record_function("opt_step"):
            # Gradient clipping (before optimizer step)
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
            training_stats.report("Loss/grad_norm", grad_norm)

            # Optimizer step (DDP already synchronized gradients)
            scaler.step(optimizer)
            scaler.update()

        # Update EMA (Exponential Moving Average)
        # This creates a smoother model for sampling
        with torch.autograd.profiler.record_function("ema_update"):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(model_ema.parameters(), base_model.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(model_ema.buffers(), base_model.buffers()):
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
            dist.broadcast(__RESTART__, 0)
        if __RESTART__:
            done = True
            print(f"Process {rank} leaving...")
            if num_gpus > 1:
                dist.barrier()

        # Save image snapshot
        if (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            stage(f"Saving image snapshot (kimg={cur_nimg/1e3:.1f})")
            if rank == 0:
                images = generate_snapshot_images(
                    diffusion_ema, grid_size, device, batch_gpu,
                    labels=snapshot_labels,
                    cfg_scale=cfg_scale if use_labels else 1.0,
                )
                save_image_grid(images, os.path.join(run_dir, f"fakes{cur_nimg//1000:06d}.png"), drange=[0, 1], grid_size=grid_size)

        # Save network snapshot (save base_model, not the DDP wrapper)
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            stage(f"Preparing network snapshot (kimg={cur_nimg/1e3:.1f})")
            snapshot_data = dict(model=base_model, model_ema=model_ema, training_set_kwargs=dict(training_set_kwargs))

        # Save checkpoint
        snapshot_pkl_path = None
        if (rank == 0) and (restart_every > 0) and (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_pkl_path = misc.get_ckpt_path(run_dir)
            stage(f'Saving checkpoint "{snapshot_pkl_path}"')
            snapshot_data["progress"] = {
                "cur_nimg": torch.tensor(cur_nimg, dtype=torch.long),
                "cur_tick": torch.tensor(cur_tick, dtype=torch.long),
                "batch_idx": torch.tensor(batch_idx, dtype=torch.long),
                "best_fid": best_fid,
            }
            with open(snapshot_pkl_path, "wb") as f:
                pickle.dump(snapshot_data, f)

        # Evaluate metrics (FID, etc.)
        # Skip tick 0: model is untrained, FID is meaningless, and generating
        # 50k images takes ~1 hour. Start evaluating from metrics_ticks onward.
        should_eval_metrics = (
            (len(metrics) > 0) and
            (snapshot_data is not None) and
            (cur_tick > 0) and
            (metrics_ticks is None or cur_tick % metrics_ticks == 0 or done)
        )
        
        if should_eval_metrics:
            stage(f'Evaluating metrics (kimg={cur_nimg/1e3:.1f})')
            for metric in metrics:
                try:
                    result_dict = metric_main.calc_metric(
                        metric=metric,
                        diffusion=diffusion_ema,
                        model=model_ema,
                        dataset_kwargs=training_set_kwargs,
                        num_gpus=num_gpus,
                        rank=rank,
                        device=device,
                        batch_size=batch_gpu,
                        num_inference_steps=fid_inference_steps,
                    )
                    if rank == 0:
                        metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl_path)
                    stats_metrics.update(result_dict.results)
                    
                    # Track best FID and save best model
                    for key, value in result_dict.results.items():
                        if 'fid' in key.lower() and value < best_fid:
                            best_fid = value
                            if rank == 0:
                                best_pkl = os.path.join(run_dir, 'best_model.pkl')
                                stage(f'New best FID: {best_fid:.2f}, saving to {best_pkl}')
                                with open(best_pkl, 'wb') as f:
                                    pickle.dump(dict(
                                        model=base_model,
                                        model_ema=model_ema,
                                        training_set_kwargs=dict(training_set_kwargs),
                                        best_fid=best_fid,
                                        cur_nimg=cur_nimg,
                                    ), f)
                                # Also save the current nimg for reference
                                with open(os.path.join(run_dir, 'best_nimg.txt'), 'w') as f:
                                    f.write(f'{cur_nimg}\n')
                except Exception as e:
                    if rank == 0:
                        print(f'Warning: Failed to compute metric {metric}: {e}')

        del snapshot_data

        # Collect statistics
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs with training stats and metrics
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            # Add metrics to jsonl
            for name, value in stats_metrics.items():
                fields[f'Metrics/{name}'] = value
            stats_jsonl.write(json.dumps(fields) + "\n")
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            # Log training statistics
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            # Log metrics (FID, etc.)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            # Log best FID
            if best_fid < float('inf'):
                stats_tfevents.add_scalar('Metrics/best_fid', best_fid, global_step=global_step, walltime=walltime)
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

    # Cleanup
    if stats_jsonl is not None:
        stats_jsonl.close()
    if stats_tfevents is not None:
        stats_tfevents.close()
    
    # Synchronize before exiting (for clean distributed shutdown)
    if num_gpus > 1:
        dist.barrier()
    
    # Done
    if rank == 0:
        stage("Exiting")
