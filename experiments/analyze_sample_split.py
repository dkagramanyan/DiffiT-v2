"""
Sample-split analysis for the biased-generalization experiment.

For each pair of checkpoints (A_i, B_i) at the same kimg from two
separately-trained DiffiT models, this script:

    1. Loads both EMA models to a single GPU.
    2. Samples N images from each using IDENTICAL noise trajectories
       (same z_T, same per-step noise, same class labels).
    3. Decodes via the SD-VAE and measures mean cosine distance between
       the two decoded images.
    4. Reads the already-logged ``Loss/test`` scalar from each run's
       TensorBoard events, if available.
    5. Writes combined curves to a new TensorBoard dir and saves a
       dual-axis matplotlib plot (Figure 1(a) style).

Usage:
    python experiments/analyze_sample_split.py \
        --run-a=./experiments/runs/00000-diffit-256-splitA-batch192 \
        --run-b=./experiments/runs/00001-diffit-256-splitB-batch192 \
        --outdir=./experiments/analysis/256 \
        --num-samples=256 --num-steps=50

Run on a single GPU. Both models are loaded into the same device; the
EMA weights are what the plot is measured on.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import diffit.diffit as diffit_module
from diffit import create_diffusion, diffusion_defaults


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

CKPT_RE = re.compile(r"network-snapshot-(\d+)\.pt$")


def list_checkpoints(run_dir: str) -> Dict[int, str]:
    """Map kimg (int) -> checkpoint path, for all snapshot files in a run dir."""
    out = {}
    for path in glob(os.path.join(run_dir, "network-snapshot-*.pt")):
        m = CKPT_RE.search(os.path.basename(path))
        if m:
            out[int(m.group(1))] = path
    return out


def match_checkpoints(run_a: str, run_b: str) -> List[Tuple[int, str, str]]:
    """Return sorted list of (kimg, path_a, path_b) for matching checkpoints."""
    a, b = list_checkpoints(run_a), list_checkpoints(run_b)
    common = sorted(set(a.keys()) & set(b.keys()))
    return [(k, a[k], b[k]) for k in common]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: torch.device, model_name: str = "Diffit"):
    """Load EMA weights into a fresh DiffiT model.

    Supports both the raw state_dict format saved by train_sample_split.py
    (torch.save(ema_model.state_dict(), ...)) and the wrapped dict format
    ({"ema": ..., "model": ..., "image_size": ...}).

    Infers num_classes from the class-embedding table shape so the model
    is built with the same size as the checkpoint — otherwise
    load_state_dict raises a shape-mismatch error, and y indices sampled
    for the fresh model would be out-of-bounds for the embedding table.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "ema" in ckpt:
        state = ckpt["ema"]
        image_size = ckpt.get("image_size", 256)
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
        image_size = ckpt.get("image_size", 256)
    else:
        state = ckpt  # raw state_dict; image_size not stored
        image_size = 256

    # LabelEmbedder stores nn.Embedding(num_classes + 1, hidden): +1 slot
    # reserved for the CFG null token.
    y_weight = state["y_embedder.embedding_table.weight"]
    num_classes = y_weight.shape[0] - 1

    latent_size = image_size // 8
    model = diffit_module.__dict__[model_name](
        input_size=latent_size, num_classes=num_classes,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()
    return model, image_size, num_classes


# ---------------------------------------------------------------------------
# Sampling with matched noise
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_with_fixed_noise(
    model, diffusion, z_T, y, device, num_steps: int,
    amp_dtype=torch.bfloat16, noise_seed: int = 0,
):
    """
    Run DDPM-respaced sampling. Seeds torch RNG identically before the loop
    so that when called twice (with different models) both runs consume the
    SAME sequence of per-step ``randn_like`` samples.
    """
    # Respace the diffusion to `num_steps` (keeps math identical to training
    # diffusion, just strided).
    diff_cfg = diffusion_defaults()
    diff_cfg["timestep_respacing"] = str(num_steps)
    spaced = create_diffusion(**diff_cfg)

    torch.manual_seed(noise_seed)
    torch.cuda.manual_seed(noise_seed)

    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=True):
        sample = spaced.p_sample_loop(
            model.forward,
            shape=z_T.shape,
            noise=z_T.clone(),  # z_T is reused across both calls
            clip_denoised=False,
            model_kwargs={"y": y},
            device=device,
            progress=False,
        )
    return sample


# ---------------------------------------------------------------------------
# Cosine distance on decoded images
# ---------------------------------------------------------------------------

@torch.no_grad()
def cosine_distance(
    model_a, model_b, vae, diffusion, device,
    num_samples: int, latent_size: int, batch_size: int,
    num_steps: int, num_classes: int, base_seed: int = 0,
):
    """Mean cosine distance between matched-noise decoded samples from A and B."""
    dists = []
    n_done = 0
    batch_idx = 0
    while n_done < num_samples:
        bs = min(batch_size, num_samples - n_done)

        # Shared noise/class for this batch (identical for A and B)
        torch.manual_seed(base_seed + batch_idx)
        torch.cuda.manual_seed(base_seed + batch_idx)
        z_T = torch.randn(bs, 4, latent_size, latent_size, device=device)
        y = torch.randint(0, num_classes, (bs,), device=device)

        img_a_lat = sample_with_fixed_noise(
            model_a, diffusion, z_T, y, device,
            num_steps=num_steps, noise_seed=base_seed + batch_idx + 10000,
        )
        img_b_lat = sample_with_fixed_noise(
            model_b, diffusion, z_T, y, device,
            num_steps=num_steps, noise_seed=base_seed + batch_idx + 10000,
        )

        img_a = vae.decode(img_a_lat / 0.18215).sample
        img_b = vae.decode(img_b_lat / 0.18215).sample

        # Cosine distance on flattened decoded pixels
        a_flat = img_a.reshape(bs, -1).float()
        b_flat = img_b.reshape(bs, -1).float()
        sim = F.cosine_similarity(a_flat, b_flat, dim=1)
        dist = (1 - sim).cpu().numpy()
        dists.extend(dist.tolist())

        n_done += bs
        batch_idx += 1

    dists = np.array(dists)
    return float(dists.mean()), float(dists.std(ddof=1) / np.sqrt(len(dists)))


# ---------------------------------------------------------------------------
# Read Loss/test from existing TensorBoard events
# ---------------------------------------------------------------------------

def read_test_loss(run_dir: str) -> Dict[int, float]:
    """Return kimg -> test_loss by reading stats.jsonl (preferred) or tfevents."""
    jsonl_path = os.path.join(run_dir, "stats.jsonl")
    if os.path.exists(jsonl_path):
        out = {}
        with open(jsonl_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "test_loss" in rec and "kimg" in rec:
                    out[int(round(rec["kimg"]))] = float(rec["test_loss"])
        if out:
            return out

    # Fall back to tfevents
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    try:
        ea.Reload()
    except Exception:
        return {}
    if "Loss/test" not in ea.Tags().get("scalars", []):
        return {}
    return {int(round(ev.step)): ev.value for ev in ea.Scalars("Loss/test")}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_figure1a(results: List[dict], outdir: str, title: str = ""):
    """Dual-axis plot mirroring Figure 1(a) from Garnier-Brun et al. 2026."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    kimgs   = np.array([r["kimg"] for r in results])
    cos     = np.array([r["cosine_distance"] for r in results])
    cos_err = np.array([r["cosine_distance_sem"] for r in results])
    tl_a    = np.array([r.get("test_loss_a", np.nan) for r in results], dtype=float)
    tl_b    = np.array([r.get("test_loss_b", np.nan) for r in results], dtype=float)

    fig, ax1 = plt.subplots(figsize=(6, 4.5))
    ax1.errorbar(kimgs, cos, yerr=cos_err, color="tab:blue", marker="o",
                 linewidth=2, label="Sample-split cosine distance")
    ax1.set_xlabel("kimg")
    ax1.set_ylabel("Sample-split cosine distance", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    if not np.all(np.isnan(tl_a)):
        ax2.plot(kimgs, tl_a, color="tab:red", linestyle="--", marker="x",
                 alpha=0.7, label="Test loss (A)")
    if not np.all(np.isnan(tl_b)):
        ax2.plot(kimgs, tl_b, color="tab:orange", linestyle="--", marker="x",
                 alpha=0.7, label="Test loss (B)")
    ax2.set_ylabel("DSM test loss", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Indicate minima
    if len(cos) > 1:
        kmin_cos = kimgs[np.nanargmin(cos)]
        ax1.axvline(kmin_cos, color="tab:blue", linestyle=":", alpha=0.5)
    tl_mean = np.nanmean(np.stack([tl_a, tl_b]), axis=0)
    if not np.all(np.isnan(tl_mean)):
        kmin_tl = kimgs[np.nanargmin(tl_mean)]
        ax2.axvline(kmin_tl, color="tab:red", linestyle=":", alpha=0.5)

    fig.suptitle(title or "Biased generalization in DiffiT")
    fig.legend(loc="upper right", bbox_to_anchor=(0.98, 0.95))
    fig.tight_layout()

    png_path = os.path.join(outdir, "figure1a.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-a",       required=True, type=str, help="Run dir for model A")
    p.add_argument("--run-b",       required=True, type=str, help="Run dir for model B")
    p.add_argument("--outdir",      required=True, type=str, help="Directory for plot + new TB logs")
    p.add_argument("--num-samples", default=256,   type=int, help="Images per checkpoint for cosine distance")
    p.add_argument("--batch-size",  default=16,    type=int)
    p.add_argument("--num-steps",   default=50,    type=int, help="Sampling steps (DDPM respaced)")
    p.add_argument("--base-seed",   default=0,     type=int)
    p.add_argument("--model",       default="Diffit", type=str)
    p.add_argument("--title",       default="",    type=str)
    p.add_argument("--only-kimgs",  default=None,  type=str,
                   help="Comma-separated list of kimgs to analyze (default: all common)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.outdir, exist_ok=True)

    pairs = match_checkpoints(args.run_a, args.run_b)
    if args.only_kimgs:
        wanted = set(int(x) for x in args.only_kimgs.split(","))
        pairs = [p for p in pairs if p[0] in wanted]

    if not pairs:
        raise SystemExit("No matching checkpoints found. Check --run-a / --run-b.")

    print(f"Found {len(pairs)} matching checkpoint pairs")
    print(f"kimgs: {[k for k, _, _ in pairs]}")

    # Read test loss from stats.jsonl of each run (logged by train script)
    test_a = read_test_loss(args.run_a)
    test_b = read_test_loss(args.run_b)

    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # Diffusion schedule (unrespaced; we respace inside sample_with_fixed_noise)
    diffusion = create_diffusion(**diffusion_defaults())

    import torch.utils.tensorboard as tb
    writer = tb.SummaryWriter(args.outdir)

    results = []
    for kimg, path_a, path_b in pairs:
        t0 = time.time()
        print(f"\n== kimg={kimg} ==")
        model_a, image_size, num_classes_a = load_model(path_a, device, args.model)
        model_b, _,          num_classes_b = load_model(path_b, device, args.model)
        if num_classes_a != num_classes_b:
            raise ValueError(
                f"num_classes mismatch between runs: A={num_classes_a}, B={num_classes_b}"
            )
        latent_size = image_size // 8

        mean, sem = cosine_distance(
            model_a, model_b, vae, diffusion, device,
            num_samples=args.num_samples,
            latent_size=latent_size,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            num_classes=num_classes_a,
            base_seed=args.base_seed,
        )
        tl_a = test_a.get(kimg, float("nan"))
        tl_b = test_b.get(kimg, float("nan"))
        print(f"  cosine_distance={mean:.4f} ± {sem:.4f}  "
              f"test_loss_A={tl_a:.4f}  test_loss_B={tl_b:.4f}  "
              f"({time.time() - t0:.1f}s)")

        writer.add_scalar("BiasedGen/cosine_distance", mean, kimg)
        writer.add_scalar("BiasedGen/cosine_distance_sem", sem, kimg)
        if not np.isnan(tl_a):
            writer.add_scalar("BiasedGen/test_loss_A", tl_a, kimg)
        if not np.isnan(tl_b):
            writer.add_scalar("BiasedGen/test_loss_B", tl_b, kimg)

        results.append({
            "kimg": kimg,
            "cosine_distance": mean,
            "cosine_distance_sem": sem,
            "test_loss_a": tl_a,
            "test_loss_b": tl_b,
        })

        # Free GPU memory
        del model_a, model_b
        torch.cuda.empty_cache()

    writer.close()

    # Save raw results
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    png = plot_figure1a(results, args.outdir, title=args.title)
    print(f"\nPlot saved: {png}")
    print(f"TensorBoard logs: {args.outdir}")


if __name__ == "__main__":
    main()
