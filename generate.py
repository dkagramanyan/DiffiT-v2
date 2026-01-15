#!/usr/bin/env python3
# Copyright (c) 2024, DiffiT authors.
# Generate images from a trained DiffiT model.
#
# Usage:
#   Unconditional:
#     python generate.py --network=./runs/00000-diffit/best_model.pkl --outdir=./samples
#   
#   Conditional (with CFG):
#     python generate.py --network=./runs/00000-diffit/best_model.pkl --outdir=./samples \
#         --class=207 --cfg-scale=2.0
#   
#   Generate grid of all classes:
#     python generate.py --network=./runs/00000-diffit/best_model.pkl --outdir=./samples \
#         --class-grid --cfg-scale=1.5

from __future__ import annotations

import os
import pickle
import re

import click
import numpy as np
import PIL.Image
import torch

import dnnlib
from diffusion.diffusion import Diffusion


def save_images(images: np.ndarray, outdir: str, prefix: str = "sample"):
    """Save individual images to disk."""
    os.makedirs(outdir, exist_ok=True)
    for i, img in enumerate(images):
        # Convert from [0, 1] to [0, 255]
        img = (img * 255).clip(0, 255).astype(np.uint8)
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)  # CHW -> HWC
        PIL.Image.fromarray(img).save(os.path.join(outdir, f"{prefix}_{i:05d}.png"))


def save_grid(images: np.ndarray, fname: str, grid_size: tuple[int, int] | None = None):
    """Save images as a grid."""
    n_images = len(images)
    
    if grid_size is None:
        # Compute grid size
        gw = int(np.ceil(np.sqrt(n_images)))
        gh = int(np.ceil(n_images / gw))
    else:
        gw, gh = grid_size
    
    # Convert from [0, 1] to [0, 255]
    images = (images * 255).clip(0, 255).astype(np.uint8)
    
    _, C, H, W = images.shape
    
    # Pad if needed
    if len(images) < gw * gh:
        padding = np.zeros((gw * gh - len(images), C, H, W), dtype=np.uint8)
        images = np.concatenate([images, padding], axis=0)
    
    # Create grid
    images = images.reshape(gh, gw, C, H, W)
    images = images.transpose(0, 3, 1, 4, 2)  # (gh, H, gw, W, C)
    images = images.reshape(gh * H, gw * W, C)
    
    if C == 1:
        PIL.Image.fromarray(images[:, :, 0], "L").save(fname)
    else:
        PIL.Image.fromarray(images, "RGB").save(fname)


@click.command()
@click.option("--network", "network_pkl", help="Network pickle file", required=True, metavar="PATH")
@click.option("--outdir", help="Output directory", required=True, metavar="DIR")
@click.option("--seeds", help="Random seeds (e.g., 0-31)", metavar="RANGE", default="0-15")
@click.option("--batch", "batch_size", help="Batch size", metavar="INT", type=int, default=16)
@click.option("--steps", "num_steps", help="Number of DDIM steps", metavar="INT", type=int, default=50)
@click.option("--class", "class_idx", help="Class label for conditional generation", metavar="INT", type=int, default=None)
@click.option("--cfg-scale", help="Classifier-free guidance scale", metavar="FLOAT", type=float, default=1.5)
@click.option("--class-grid", "class_grid", help="Generate a grid with one sample per class", is_flag=True)
@click.option("--samples-per-class", help="Samples per class in class grid mode", metavar="INT", type=int, default=1)
@click.option("--grid", help="Save as grid instead of individual images", is_flag=True)
@click.option("--device", help="Device to use", metavar="STR", default="cuda")
def main(
    network_pkl: str,
    outdir: str,
    seeds: str,
    batch_size: int,
    num_steps: int,
    class_idx: int | None,
    cfg_scale: float,
    class_grid: bool,
    samples_per_class: int,
    grid: bool,
    device: str,
):
    """Generate images from a trained DiffiT model.
    
    Examples:
    
    \b
    # Unconditional generation:
    python generate.py --network=model.pkl --outdir=./samples --seeds=0-63
    
    \b
    # Conditional generation (specific class):
    python generate.py --network=model.pkl --outdir=./samples --class=207 --cfg-scale=2.0
    
    \b
    # Generate grid of all classes:
    python generate.py --network=model.pkl --outdir=./samples --class-grid --cfg-scale=1.5
    """
    print(f"Loading network from {network_pkl}...")
    device = torch.device(device)
    
    with open(network_pkl, "rb") as f:
        data = pickle.load(f)
    
    # Get model (prefer EMA model)
    model = data.get("model_ema", data.get("model"))
    model = model.eval().to(device)
    
    # Get training set kwargs for resolution
    training_set_kwargs = data.get("training_set_kwargs", {})
    resolution = training_set_kwargs.get("resolution", 64)
    
    # Check if model is conditional
    label_dim = getattr(model, "label_dim", 0)
    is_conditional = label_dim > 0
    
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Conditional: {is_conditional}")
    if is_conditional:
        print(f"  Label dim: {label_dim}")
        print(f"  CFG scale: {cfg_scale}")
    
    # Create diffusion wrapper
    diffusion = Diffusion(
        model=model,
        image_resolution=(3, resolution, resolution),
        n_times=1000,
        device=device,
    )
    
    os.makedirs(outdir, exist_ok=True)
    
    # Parse seeds
    seed_ranges = seeds.replace(" ", "").split(",")
    all_seeds = []
    for seed_range in seed_ranges:
        if "-" in seed_range:
            start, end = map(int, seed_range.split("-"))
            all_seeds.extend(range(start, end + 1))
        else:
            all_seeds.append(int(seed_range))
    
    if class_grid and is_conditional:
        # Generate grid with one row per class
        print(f"Generating class grid ({label_dim} classes, {samples_per_class} samples each)...")
        
        all_images = []
        for c in range(label_dim):
            labels = torch.full((samples_per_class,), c, dtype=torch.long, device=device)
            
            # Use consistent random seed for reproducibility
            torch.manual_seed(c)
            
            images = diffusion.sample_ddim(
                n_samples=samples_per_class,
                labels=labels,
                cfg_scale=cfg_scale,
                num_inference_steps=num_steps,
            )
            all_images.append(images.cpu().numpy())
        
        all_images = np.concatenate(all_images, axis=0)
        
        # Save grid (classes as rows)
        grid_w = samples_per_class
        grid_h = label_dim
        save_grid(all_images, os.path.join(outdir, "class_grid.png"), grid_size=(grid_w, grid_h))
        print(f"Saved class grid to {os.path.join(outdir, 'class_grid.png')}")
        
    else:
        # Generate samples
        n_samples = len(all_seeds)
        print(f"Generating {n_samples} samples...")
        
        all_images = []
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_seeds = all_seeds[start_idx:end_idx]
            batch_n = len(batch_seeds)
            
            # Set seeds (use first seed for reproducibility in batch)
            torch.manual_seed(batch_seeds[0])
            
            # Prepare labels
            labels = None
            if is_conditional and class_idx is not None:
                labels = torch.full((batch_n,), class_idx, dtype=torch.long, device=device)
            
            # Generate
            images = diffusion.sample_ddim(
                n_samples=batch_n,
                labels=labels,
                cfg_scale=cfg_scale if labels is not None else 1.0,
                num_inference_steps=num_steps,
            )
            all_images.append(images.cpu().numpy())
            
            print(f"  Generated {end_idx}/{n_samples}")
        
        all_images = np.concatenate(all_images, axis=0)
        
        if grid:
            save_grid(all_images, os.path.join(outdir, "samples.png"))
            print(f"Saved grid to {os.path.join(outdir, 'samples.png')}")
        else:
            prefix = f"class{class_idx}" if class_idx is not None else "sample"
            save_images(all_images, outdir, prefix=prefix)
            print(f"Saved {len(all_images)} images to {outdir}")
    
    print("Done!")


if __name__ == "__main__":
    main()
