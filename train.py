#!/usr/bin/env python3
# Copyright (c) 2024, DiffiT authors.
# Train a DiffiT diffusion model.
#
# Multi-GPU Training:
#   Single node:  torchrun --nproc_per_node=N train.py --outdir=... --data=... --batch-gpu=...
#   Multi-node:   torchrun --nnodes=M --nproc_per_node=N --rdzv_backend=c10d train.py ...
#   Legacy:       python train.py --outdir=... --data=... --gpus=N --batch-gpu=...

from __future__ import annotations

import json
import os
import re
import tempfile

import click
import torch
import torch.distributed as dist

import dnnlib
from torch_utils import custom_ops
from torch_utils import misc
from torch_utils import training_stats
from training import training_loop


def init_distributed():
    """Initialize distributed training from environment variables.
    
    Supports both torchrun and legacy multiprocessing spawn.
    Returns (rank, world_size, local_rank).
    """
    # Check if launched via torchrun/torch.distributed.launch
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        if not dist.is_initialized():
            torch.cuda.set_device(local_rank)
            backend = "gloo" if os.name == "nt" else "nccl"
            dist.init_process_group(backend=backend, init_method="env://")
        
        return rank, world_size, local_rank
    
    # Check SLURM environment
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        
        # SLURM provides master addr/port
        master_addr = os.environ.get("MASTER_ADDR", os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "127.0.0.1"))
        master_port = os.environ.get("MASTER_PORT", "29500")
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        
        if not dist.is_initialized():
            torch.cuda.set_device(local_rank)
            backend = "gloo" if os.name == "nt" else "nccl"
            dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
        
        return rank, world_size, local_rank
    
    return None, None, None


def subprocess_fn(rank: int, c: dnnlib.EasyDict, temp_dir: str):
    """Subprocess entry point for distributed training (legacy spawn mode)."""
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, "log.txt"), file_mode="a", should_flush=True)

    # Init torch.distributed
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, ".torch_distributed_init"))
        if os.name == "nt":
            init_method = "file:///" + init_file.replace("\\", "/")
            torch.distributed.init_process_group(backend="gloo", init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f"file://{init_file}"
            torch.cuda.set_device(rank)
            torch.distributed.init_process_group(backend="nccl", init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils
    sync_device = torch.device("cuda", rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = "none"

    # Execute training loop
    training_loop.training_loop(rank=rank, **c)


def launch_training(c: dnnlib.EasyDict, desc: str, outdir: str, dry_run: bool):
    """Launch training with the given configuration."""
    # Check if already in distributed mode (torchrun/SLURM)
    rank, world_size, local_rank = init_distributed()
    is_distributed_launch = rank is not None
    
    if is_distributed_launch:
        # Update config for distributed launch
        c.num_gpus = world_size
        c.batch_size = c.batch_gpu * world_size
    
    # Only rank 0 should do setup logging and directory creation
    is_main = (rank == 0) if is_distributed_launch else True
    
    if is_main:
        dnnlib.util.Logger(should_flush=True)

    # Pick output directory (only on main process)
    if is_main:
        prev_run_dirs = []
        if os.path.isdir(outdir):
            prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]

        matching_dirs = [
            re.fullmatch(r"\d{5}" + f"-{desc}", x)
            for x in prev_run_dirs
            if re.fullmatch(r"\d{5}" + f"-{desc}", x) is not None
        ]
        if c.restart_every > 0 and len(matching_dirs) > 0:
            assert len(matching_dirs) == 1, f"Multiple directories found for resuming: {matching_dirs}"
            c.run_dir = os.path.join(outdir, matching_dirs[0].group())
        else:
            prev_run_ids = [re.match(r"^\d+", x) for x in prev_run_dirs]
            prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
            cur_run_id = max(prev_run_ids, default=-1) + 1
            c.run_dir = os.path.join(outdir, f"{cur_run_id:05d}-{desc}")
            if not (c.restart_every > 0):
                assert not os.path.exists(c.run_dir)

        # Print options
        print()
        print("Training options:")
        print(json.dumps(c, indent=2))
        print()
        print(f"Output directory:    {c.run_dir}")
        print(f"Number of GPUs:      {c.num_gpus}")
        print(f"Batch size:          {c.batch_size} images")
        print(f"Training duration:   {c.total_kimg} kimg")
        print(f"Dataset path:        {c.training_set_kwargs.path}")
        print(f"Dataset resolution:  {c.training_set_kwargs.resolution}")
        print()

        # Dry run?
        if dry_run:
            print("Dry run; exiting.")
            return

        # Create output directory
        print("Creating output directory...")
        os.makedirs(c.run_dir, exist_ok=c.restart_every > 0)
        with open(os.path.join(c.run_dir, "training_options.json"), "wt+") as f:
            json.dump(c, f, indent=2)
    
    # Synchronize run_dir across all processes for distributed launch
    if is_distributed_launch:
        dist.barrier()
        # Broadcast run_dir from rank 0
        if rank == 0:
            run_dir_bytes = c.run_dir.encode("utf-8")
            run_dir_len = torch.tensor([len(run_dir_bytes)], dtype=torch.long, device="cuda")
        else:
            run_dir_len = torch.tensor([0], dtype=torch.long, device="cuda")
        
        dist.broadcast(run_dir_len, src=0)
        
        if rank == 0:
            run_dir_tensor = torch.tensor(list(run_dir_bytes), dtype=torch.uint8, device="cuda")
        else:
            run_dir_tensor = torch.zeros(run_dir_len.item(), dtype=torch.uint8, device="cuda")
        
        dist.broadcast(run_dir_tensor, src=0)
        
        if rank != 0:
            c.run_dir = bytes(run_dir_tensor.cpu().tolist()).decode("utf-8")
        
        # Init torch_utils for distributed
        sync_device = torch.device("cuda", local_rank)
        training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
        if rank != 0:
            custom_ops.verbosity = "none"
        
        # Setup logger for this process
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, "log.txt"), file_mode="a", should_flush=True)
        
        # Execute training loop directly
        print(f"Process {rank}/{world_size} starting training...")
        training_loop.training_loop(rank=rank, **c)
    else:
        # Legacy spawn mode
        print("Launching processes...")
        try:
            torch.multiprocessing.set_start_method("spawn")
        except RuntimeError:
            pass  # Already set
        
        with tempfile.TemporaryDirectory() as temp_dir:
            if c.num_gpus == 1:
                subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
            else:
                torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)


def init_dataset_kwargs(data: str, resolution: int | None = None):
    """Initialize dataset configuration."""
    try:
        dataset_kwargs = dnnlib.EasyDict(
            class_name="training.dataset.ImageFolderDataset",
            path=data,
            use_labels=False,
            max_size=None,
            xflip=False,
            resolution=resolution,
        )
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
        dataset_kwargs.resolution = dataset_obj.resolution
        dataset_kwargs.max_size = len(dataset_obj)
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f"--data: {err}")


def parse_comma_separated_list(s):
    """Parse comma-separated list from string."""
    if isinstance(s, list):
        return s
    if s is None or s.lower() == "none" or s == "":
        return []
    return s.split(",")


@click.command()
# Required
@click.option("--outdir", help="Where to save the results", metavar="DIR", required=True)
@click.option("--data", help="Training data", metavar="[ZIP|DIR]", type=str, required=True)
@click.option("--gpus", help="Number of GPUs (ignored when using torchrun)", metavar="INT", type=click.IntRange(min=1), default=1)
@click.option("--batch-gpu", help="Batch size per GPU", metavar="INT", type=click.IntRange(min=1), required=True)
# Model configuration
@click.option("--resolution", help="Image resolution", metavar="INT", type=click.IntRange(min=4), default=64)
@click.option("--base-dim", help="Base channel dimension", metavar="INT", type=click.IntRange(min=1), default=128)
@click.option("--timesteps", help="Number of diffusion timesteps", metavar="INT", type=click.IntRange(min=1), default=1000)
@click.option("--hidden-dim", help="Hidden dimension for ViT blocks", metavar="INT", type=click.IntRange(min=1), default=64)
@click.option("--num-heads", help="Number of attention heads", metavar="INT", type=click.IntRange(min=1), default=4)
@click.option("--num-blocks", help="Number of transformer blocks", metavar="INT", type=click.IntRange(min=1), default=1)
# Optional features
@click.option("--mirror", help="Enable dataset x-flips", metavar="BOOL", type=bool, default=False, show_default=True)
@click.option("--resume", help="Resume from given network pickle", metavar="[PATH|URL]", type=str)
# Training parameters
@click.option("--lr", help="Learning rate", metavar="FLOAT", type=click.FloatRange(min=0), default=1e-4, show_default=True)
@click.option("--ema-kimg", help="EMA half-life in kimg", metavar="FLOAT", type=float, default=10.0, show_default=True)
# Misc settings
@click.option("--desc", help="String to include in result dir name", metavar="STR", type=str)
@click.option("--kimg", help="Total training duration", metavar="KIMG", type=click.IntRange(min=1), default=25000, show_default=True)
@click.option("--tick", help="How often to print progress", metavar="KIMG", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--snap", help="How often to save snapshots", metavar="TICKS", type=click.IntRange(min=1), default=50, show_default=True)
@click.option("--seed", help="Random seed", metavar="INT", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--fp32", help="Disable mixed-precision", metavar="BOOL", type=bool, default=False, show_default=True)
@click.option("--nobench", help="Disable cuDNN benchmarking", metavar="BOOL", type=bool, default=False, show_default=True)
@click.option("--workers", help="DataLoader worker processes", metavar="INT", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("-n", "--dry-run", help="Print training options and exit", is_flag=True)
@click.option("--restart_every", help="Time interval in seconds to restart code", metavar="INT", type=int, default=999999999, show_default=True)
def main(**kwargs):
    """Train a DiffiT diffusion model.
    
    Multi-GPU training can be launched in two ways:
    
    \b
    1. Using torchrun (recommended):
       torchrun --nproc_per_node=4 train.py --outdir=./runs --data=./data --batch-gpu=32
    
    \b
    2. Using legacy spawn (with --gpus):
       python train.py --outdir=./runs --data=./data --gpus=4 --batch-gpu=32
    """
    opts = dnnlib.EasyDict(kwargs)
    c = dnnlib.EasyDict()

    # Model configuration (matches DiffiT paper architecture)
    c.model_kwargs = dnnlib.EasyDict(
        class_name="models.diffit.DiffiT",
        image_shape=[3, opts.resolution, opts.resolution],
        base_dim=opts.base_dim,
        hidden_dim=opts.hidden_dim,
        num_heads=opts.num_heads,
        num_res_blocks=opts.num_blocks,
    )

    # Diffusion configuration (variance-exploding as per paper Sec. 3.1)
    c.diffusion_kwargs = dnnlib.EasyDict(
        class_name="diffusion.diffusion.Diffusion",
        n_times=opts.timesteps,
    )

    # Optimizer configuration (AdamW as commonly used)
    c.opt_kwargs = dnnlib.EasyDict(class_name="torch.optim.AdamW", lr=opts.lr, betas=[0.9, 0.999], weight_decay=0.01)

    # Data loader configuration
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2, persistent_workers=True)

    # Training set
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data, resolution=opts.resolution)
    c.training_set_kwargs.xflip = opts.mirror

    # Hyperparameters & settings
    # Note: num_gpus and batch_size may be overridden by launch_training for torchrun
    c.num_gpus = opts.gpus
    c.batch_gpu = opts.batch_gpu
    c.batch_size = c.batch_gpu * c.num_gpus
    c.opt_kwargs.lr = opts.lr
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers
    c.ema_kimg = opts.ema_kimg
    c.cudnn_benchmark = not opts.nobench
    c.fp32 = opts.fp32

    # Resume
    if opts.resume is not None:
        c.resume_pkl = opts.resume

    # Restart
    c.restart_every = opts.restart_every

    # Description string (will be updated by launch_training if using torchrun)
    desc = f"diffit-{dataset_name:s}-{opts.resolution}x{opts.resolution}-gpus{c.num_gpus:d}-batch{c.batch_size:d}"
    if opts.desc is not None:
        desc += f"-{opts.desc}"

    try:
        # Launch training
        launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)
    finally:
        # Cleanup distributed process group
        if dist.is_initialized():
            dist.destroy_process_group()

    # Check for restart (only on main process)
    rank = int(os.environ.get("RANK", 0))
    if rank == 0 and c.restart_every > 0 and hasattr(c, 'run_dir') and os.path.isfile(misc.get_ckpt_path(c.run_dir)):
        with dnnlib.util.open_url(misc.get_ckpt_path(c.run_dir)) as f:
            import pickle
            resume_data = pickle.load(f)
            cur_nimg = resume_data["progress"]["cur_nimg"].item()
        if (cur_nimg // 1000) < c.total_kimg:
            print("Restart: exit with code 3")
            exit(3)


if __name__ == "__main__":
    main()
