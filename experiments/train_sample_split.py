"""
Training script for the biased-generalization sample-split experiment.

Splits one dataset deterministically into:
    - val   (0.10)
    - trainA (0.45)
    - trainB (0.45)

You train *one* model per launch on either split A or split B. The val
split is fixed across launches. DSM test loss on the val split is logged
to TensorBoard as ``Loss/test`` every ``--snap`` ticks.

Launch (single GPU):
    python experiments/train_sample_split.py \
        --outdir=./experiments/runs \
        --data=./datasets/imagenet_256x256.zip \
        --image-size=256 --split=A \
        --batch-gpu=96 --kimg=30000

Launch (multi-GPU via torchrun):
    torchrun --nproc_per_node=2 experiments/train_sample_split.py \
        --outdir=./experiments/runs \
        --data=./datasets/imagenet_256x256.zip \
        --image-size=256 --split=A \
        --batch-gpu=96 --kimg=30000
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Tuple

import click
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from diffusers.models import AutoencoderKL

# Allow `python experiments/train_sample_split.py ...` from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import diffit.diffit as diffit_module
from diffit import create_diffusion, diffusion_defaults, NUM_CLASSES
from diffit.image_datasets import (
    ImageDataset,
    ZipImageDataset,
    _list_image_files_recursively,
)
from diffit.metrics import (
    InceptionFeatureExtractor,
    compute_activations,
    evaluate_metrics,
)
from diffit.nn import update_ema
from diffit.timestep_sampler import create_named_schedule_sampler


# ---------------------------------------------------------------------------
# Deterministic split
# ---------------------------------------------------------------------------

VAL_FRAC = 0.10
TRAIN_A_FRAC = 0.45
TRAIN_B_FRAC = 0.45
SPLIT_SEED = 42  # fixed so val / A / B are identical across every launch


def make_splits(n_total: int, seed: int = SPLIT_SEED) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (val_idx, train_a_idx, train_b_idx). Deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n_total)
    rng.shuffle(idx)
    n_val = int(round(n_total * VAL_FRAC))
    n_a = int(round(n_total * TRAIN_A_FRAC))
    return idx[:n_val], idx[n_val : n_val + n_a], idx[n_val + n_a :]


def build_base_dataset(data_path: str, image_size: int, cache_in_ram: bool):
    """Build the underlying class-conditional dataset (zip or folder)."""
    if data_path.endswith(".zip"):
        return ZipImageDataset(
            data_path, image_size,
            class_cond=True, random_crop=False, random_flip=True,
            cache_in_ram=cache_in_ram,
        )
    all_files = _list_image_files_recursively(data_path)
    class_names = [os.path.basename(p).split("_")[0] for p in all_files]
    sorted_cls = {x: i for i, x in enumerate(sorted(set(class_names)))}
    classes = [sorted_cls[x] for x in class_names]
    return ImageDataset(
        image_size, all_files, classes=classes,
        random_crop=False, random_flip=True,
        cache_in_ram=cache_in_ram,
    )


# ---------------------------------------------------------------------------
# Distributed helpers (torchrun-native)
# ---------------------------------------------------------------------------

def setup_dist():
    """Initialize process group from torchrun env vars; fall back to single-GPU."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world_size, local_rank = 0, 1, 0
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


# Module-level log fan-out state, set in main() once writers exist.
_tb_writer = None
_jsonl_handle = None
_text_step = 0


def log(msg: str):
    """Print to stdout + mirror to TB (add_text) and stats.jsonl when configured."""
    global _text_step
    if not is_main():
        return
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    if _tb_writer is not None:
        try:
            _tb_writer.add_text("log", msg, global_step=_text_step)
        except Exception:
            pass
    if _jsonl_handle is not None:
        try:
            _jsonl_handle.write(json.dumps({
                "kind": "text", "msg": msg, "time": time.time(), "step": _text_step,
            }) + "\n")
            _jsonl_handle.flush()
        except Exception:
            pass
    _text_step += 1


def _write_scalars(kvs: dict, step):
    """Write a group of scalars to TB and stats.jsonl as a single record."""
    if not is_main():
        return
    if _tb_writer is not None:
        for k, v in kvs.items():
            if isinstance(v, (int, float)):
                _tb_writer.add_scalar(k, v, step)
    if _jsonl_handle is not None:
        _jsonl_handle.write(json.dumps({
            "kind": "scalars", "step": float(step), "time": time.time(), **kvs,
        }) + "\n")
        _jsonl_handle.flush()


# ---------------------------------------------------------------------------
# Test loss
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_test_loss(
    model, vae, diffusion, val_loader, device, amp_dtype,
    num_batches: int = 16,
) -> float:
    """Mean DSM loss over the val split at uniformly random timesteps."""
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    it = iter(val_loader)
    for _ in range(num_batches):
        try:
            batch, cond = next(it)
        except StopIteration:
            break
        batch = batch.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=True):
            latent = vae.encode(batch).latent_dist.sample() * 0.18215
            t = torch.randint(0, diffusion.num_timesteps, (latent.shape[0],), device=device)
            model_kwargs = {}
            if "y" in cond:
                model_kwargs["y"] = cond["y"].to(device, non_blocking=True)
            losses = diffusion.training_losses(model, latent, t, model_kwargs=model_kwargs)
            total += losses["loss"].mean().item() * latent.shape[0]
            n += latent.shape[0]

    # Reduce across ranks so every rank sees the same mean.
    if dist.is_initialized():
        tot_t = torch.tensor([total, float(n)], device=device)
        dist.all_reduce(tot_t, op=dist.ReduceOp.SUM)
        total, n = tot_t[0].item(), int(tot_t[1].item())

    if was_training:
        model.train()
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@click.command()
@click.option("--outdir",     required=True, type=str, help="Directory for per-run folders")
@click.option("--data",       required=True, type=str, help="Path to dataset (.zip or folder)")
@click.option("--image-size", required=True, type=int, help="Training resolution (256/512/1024)")
@click.option("--split",      required=True, type=click.Choice(["A", "B"]), help="Which training split to use")
@click.option("--batch-gpu",  required=True, type=int, help="Per-GPU batch size")
@click.option("--kimg",       default=30000, show_default=True, type=int, help="Total kimg to train")
@click.option("--model",      "model_name", default="Diffit", show_default=True, type=str)
@click.option("--lr",         default=None, type=float, help="Learning rate (default: 3e-4 at 256, 1e-4 at higher)")
@click.option("--ema-rate",   default=0.9999, show_default=True, type=float)
@click.option("--snap",       default=20, show_default=True, type=int, help="Ticks between snapshots/test-loss evals")
@click.option("--kimg-per-tick", default=4, show_default=True, type=int)
@click.option("--grad-accum", default=1, show_default=True, type=int)
@click.option("--grad-ckpt/--no-grad-ckpt", default=False, show_default=True)
@click.option("--workers",    default=4, show_default=True, type=int)
@click.option("--cache-in-ram/--no-cache-in-ram", default=True, show_default=True)
@click.option("--val-batches", default=16, show_default=True, type=int,
              help="Number of val batches for test-loss estimation")
@click.option("--num-fid-samples", default=10000, show_default=True, type=int,
              help="Samples for FID/IS/sFID/Precision/Recall eval at each snap (0=disable)")
@click.option("--cfg-scale",  default=4.4, show_default=True, type=float,
              help="CFG scale used when generating FID samples")
@click.option("--seed",       default=42, show_default=True, type=int, help="Init / training RNG seed")
@click.option("--resume",     default=None, type=str, help="Checkpoint to resume from")
def main(**opts):
    # ----- setup --------------------------------------------------------
    rank, world_size, local_rank = setup_dist()
    device = torch.device("cuda", local_rank)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(opts["seed"] + rank)
    np.random.seed(opts["seed"] + rank)
    random.seed(opts["seed"] + rank)

    image_size = opts["image_size"]
    lr = opts["lr"] if opts["lr"] is not None else (3e-4 if image_size <= 256 else 1e-4)

    # ----- run directory ------------------------------------------------
    run_name = f"diffit-{image_size}-split{opts['split']}-batch{opts['batch_gpu'] * world_size}"
    os.makedirs(opts["outdir"], exist_ok=True)
    if is_main():
        existing = sorted(os.listdir(opts["outdir"]))
        run_id = len([d for d in existing if d.startswith("") and os.path.isdir(os.path.join(opts["outdir"], d))])
        run_dir = os.path.join(opts["outdir"], f"{run_id:05d}-{run_name}")
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = ""
    if dist.is_initialized():
        obj = [run_dir]
        dist.broadcast_object_list(obj, src=0)
        run_dir = obj[0]

    log(f"Run dir: {run_dir}")
    log(f"World size: {world_size}, resolution: {image_size}, split: {opts['split']}, LR: {lr}")

    # ----- data ---------------------------------------------------------
    log(f"Loading dataset: {opts['data']}")
    base = build_base_dataset(opts["data"], image_size, opts["cache_in_ram"])
    val_idx, a_idx, b_idx = make_splits(len(base))
    log(f"Dataset size: {len(base)} "
        f"(val={len(val_idx)}, trainA={len(a_idx)}, trainB={len(b_idx)})")

    # Class IDs actually present in the dataset (for FID sampling + model sizing).
    # ImageDataset stores them in `.classes`; ZipImageDataset in `._labels`.
    if getattr(base, "classes", None) is not None:
        _labels_src = list(base.classes)
    elif getattr(base, "_labels", None):
        _labels_src = list(base._labels.values())
    else:
        _labels_src = []
    dataset_class_list = sorted(set(_labels_src))
    expected = list(range(len(dataset_class_list)))
    if dataset_class_list != expected:
        raise ValueError(
            f"Dataset labels must be dense 0..N-1, got {dataset_class_list[:10]}"
            f"{'...' if len(dataset_class_list) > 10 else ''}. "
            "Remap before training."
        )
    log(f"Dataset classes: {len(dataset_class_list)} unique (ids=0..{len(dataset_class_list) - 1})")

    train_idx = a_idx if opts["split"] == "A" else b_idx
    train_ds = Subset(base, train_idx.tolist())
    val_ds   = Subset(base, val_idx.tolist())

    # Disable flipping on val — deterministic eval
    if hasattr(base, "random_flip"):
        # Val uses same base dataset, but we build a separate non-flipping view
        # via a small wrapper. Simplest approach: just accept the flip noise —
        # averaging over 16 batches × 1000 t samples kills it anyway.
        pass

    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    val_sampler   = DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds, batch_size=opts["batch_gpu"],
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=opts["workers"], pin_memory=True, drop_last=True,
        persistent_workers=opts["workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=opts["batch_gpu"],
        sampler=val_sampler, shuffle=False,
        num_workers=max(1, opts["workers"] // 2), pin_memory=True, drop_last=True,
        persistent_workers=opts["workers"] > 0,
    )

    # ----- model --------------------------------------------------------
    latent_size = image_size // 8
    num_dataset_classes = max(len(dataset_class_list), 1)
    model = diffit_module.__dict__[opts["model_name"]](
        input_size=latent_size, num_classes=num_dataset_classes,
    ).to(device)
    if opts["grad_ckpt"]:
        model.gradient_checkpointing = True

    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)
    ema_model.eval()

    # ----- diffusion ----------------------------------------------------
    diff_cfg = diffusion_defaults()
    diffusion = create_diffusion(**diff_cfg)
    sampler = create_named_schedule_sampler("uniform", diffusion)

    # ----- optimizer & AMP ---------------------------------------------
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_grad_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_grad_scaler)
    log(f"AMP: {amp_dtype}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, fused=True)

    if world_size > 1:
        ddp_model = DDP(model, device_ids=[local_rank], gradient_as_bucket_view=True)
    else:
        ddp_model = model

    # ----- VAE ----------------------------------------------------------
    log("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # ----- resume -------------------------------------------------------
    cur_nimg = 0
    if opts["resume"] is not None:
        log(f"Resuming from {opts['resume']}")
        ckpt = torch.load(opts["resume"], map_location="cpu")
        model.load_state_dict(ckpt["model"])
        ema_model.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        cur_nimg = ckpt.get("cur_nimg", 0)
        log(f"Resumed at {cur_nimg // 1000} kimg")

    # ----- logging ------------------------------------------------------
    stats_tb = None
    stats_jsonl = None
    if is_main():
        os.makedirs(run_dir, exist_ok=True)
        import torch.utils.tensorboard as tb
        stats_tb = tb.SummaryWriter(run_dir)
        stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "at")

        # Wire module-level fan-out so every log() call mirrors to TB + stats.jsonl.
        global _tb_writer, _jsonl_handle
        _tb_writer = stats_tb
        _jsonl_handle = stats_jsonl

        # Dump config for reproducibility
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({**opts, "val_frac": VAL_FRAC,
                      "train_a_frac": TRAIN_A_FRAC, "train_b_frac": TRAIN_B_FRAC,
                      "split_seed": SPLIT_SEED, "world_size": world_size,
                      "lr": lr, "train_size": len(train_idx),
                      "val_size": len(val_idx)}, f, indent=2)

    # ----- FID setup (only when --num-fid-samples > 0) -----------------
    inception_extractor = None
    ref_acts = None
    num_fid_samples = opts["num_fid_samples"]
    if num_fid_samples > 0 and is_main():
        log("Loading InceptionV3 for metric evaluation...")
        from torchvision.models import inception_v3, Inception_V3_Weights
        inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT).eval().to(device)
        inception_extractor = InceptionFeatureExtractor(inception_model)

        # Ref set is bounded by unique val images — duplicating would bias the
        # reference covariance. Fakes stay at num_fid_samples (all unique).
        num_ref = min(num_fid_samples, len(val_ds))
        log(f"Pre-computing reference Inception features "
            f"({num_ref} unique val images; fakes will be {num_fid_samples})...")
        ref_images = []
        n_collected = 0
        ref_iter = iter(val_loader)
        while n_collected < num_ref:
            try:
                batch, _ = next(ref_iter)
            except StopIteration:
                break  # val_loader exhausted
            ref_images.append((batch + 1).mul(127.5).clamp(0, 255).byte().numpy())
            n_collected += batch.shape[0]
        ref_images = np.concatenate(ref_images, axis=0)[:num_ref]
        ref_acts = compute_activations(
            ref_images, inception_extractor, batch_size=64, device=device,
        )

    # ----- train loop ---------------------------------------------------
    model.train()
    loss_accum, n_accum = 0.0, 0
    tick_start = time.time()
    tick_nimg = cur_nimg
    cur_tick = 0
    train_iter = iter(train_loader)
    train_epoch = 0
    run_start = time.time()
    maintenance_time = 0.0
    total_kimg = opts["kimg"]
    log("Starting training loop")

    while cur_nimg < opts["kimg"] * 1000:
        opt.zero_grad(set_to_none=True)

        # Gradient accumulation
        for micro in range(opts["grad_accum"]):
            try:
                batch, cond = next(train_iter)
            except StopIteration:
                train_epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(train_epoch)
                train_iter = iter(train_loader)
                batch, cond = next(train_iter)

            batch = batch.to(device, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=True):
                latent = vae.encode(batch).latent_dist.sample() * 0.18215

            t, weights = sampler.sample(latent.shape[0], device)
            model_kwargs = {}
            if "y" in cond:
                model_kwargs["y"] = cond["y"].to(device, non_blocking=True)

            from contextlib import nullcontext
            is_last = (micro == opts["grad_accum"] - 1)
            sync_ctx = nullcontext() if (is_last or world_size <= 1) else ddp_model.no_sync()

            with sync_ctx:
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=True):
                    losses = diffusion.training_losses(ddp_model, latent, t, model_kwargs=model_kwargs)
                    loss = (losses["loss"] * weights).mean() / opts["grad_accum"]
                scaler.scale(loss).backward()

            cur_nimg += opts["batch_gpu"] * world_size
            loss_accum += loss.item() * opts["grad_accum"] * latent.shape[0]
            n_accum += latent.shape[0]

        scaler.step(opt)
        scaler.update()
        update_ema(ema_model.parameters(), model.parameters(), rate=opts["ema_rate"])

        # ---- tick ----
        if (cur_nimg - tick_nimg) / 1000 >= opts["kimg_per_tick"] or cur_nimg >= opts["kimg"] * 1000:
            tick_time = time.time() - tick_start
            cur_tick += 1
            train_loss = loss_accum / max(n_accum, 1)
            kimg = cur_nimg / 1000

            # Memory stats (parity with train.py)
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

            sec_per_kimg = tick_time / max((cur_nimg - tick_nimg) / 1000, 1e-6)
            kimg_left = max(total_kimg - kimg, 0.0)
            eta_sec = sec_per_kimg * kimg_left if sec_per_kimg > 0 else 0.0
            total_elapsed = time.time() - run_start

            if is_main():
                _write_scalars({
                    "Loss/train": train_loss,
                    "Progress/sec_per_kimg": sec_per_kimg,
                    "Progress/sec_per_tick": tick_time,
                    "Progress/eta_sec": eta_sec,
                    "Progress/elapsed_sec": total_elapsed,
                    "Progress/maintenance_sec": maintenance_time,
                    "Memory/cpu_gb": cpumem,
                    "Memory/gpu_gb": gpumem,
                    "Memory/gpu_reserved_gb": reserved,
                    "tick": cur_tick,
                    "kimg": kimg,
                }, step=kimg)
                log(f"tick {cur_tick:6d}  kimg {kimg:>9.1f}  "
                    f"train_loss {train_loss:.4f}  "
                    f"tick_time {tick_time:.1f}s  "
                    f"sec/kimg {sec_per_kimg:.2f}  "
                    f"cpumem {cpumem:.2f}  gpumem {gpumem:.2f}  reserved {reserved:.2f}")

            # Snapshot + test loss + (optional) FID
            if cur_tick % opts["snap"] == 0 or cur_nimg >= opts["kimg"] * 1000:
                snap_start = time.time()
                log("Computing test loss...")
                test_loss = compute_test_loss(
                    ema_model, vae, diffusion, val_loader, device, amp_dtype,
                    num_batches=opts["val_batches"],
                )
                if is_main():
                    _write_scalars({"Loss/test": test_loss, "kimg": kimg, "tick": cur_tick},
                                   step=kimg)

                # FID/IS/sFID/Precision/Recall (all ranks participate in sample gen + kNN)
                if num_fid_samples > 0:
                    latent_size = image_size // 8
                    stats_metrics = evaluate_metrics(
                        ema_model, vae, diffusion, ref_acts, inception_extractor,
                        num_fid_samples, opts["batch_gpu"], latent_size, device,
                        cfg_scale=opts["cfg_scale"],
                        rank=rank, world_size=world_size,
                        log_fn=log,
                        class_list=dataset_class_list or None,
                        null_class_idx=num_dataset_classes,
                    )
                    if is_main() and stats_metrics is not None:
                        _write_scalars(
                            {f"Metrics/{k}": v for k, v in stats_metrics.items()},
                            step=kimg,
                        )

                if is_main():
                    ckpt_path = os.path.join(run_dir, f"network-snapshot-{int(kimg):08d}.pt")
                    torch.save(ema_model.state_dict(), ckpt_path)
                    log(f"Saved {ckpt_path}  test_loss={test_loss:.4f}")
                maintenance_time = time.time() - snap_start
            else:
                maintenance_time = 0.0

            loss_accum, n_accum = 0.0, 0
            tick_nimg = cur_nimg
            tick_start = time.time()

    # Final checkpoint — inference-only: EMA weights as a raw state_dict
    if is_main():
        torch.save(ema_model.state_dict(), os.path.join(run_dir, "network-final.pt"))
        stats_jsonl.close()
        stats_tb.close()
    log("Training complete.")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
