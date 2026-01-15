#!/usr/bin/env python3
# Copyright (c) 2024, DiffiT authors.
# Generate images using a trained DiffiT model.

from __future__ import annotations

import os
import pickle
import re

import click
import numpy as np
import PIL.Image
import torch

import dnnlib


def save_image_grid(images: np.ndarray, fname: str, grid_size: tuple[int, int]):
    """Save a grid of images to a file."""
    gw, gh = grid_size
    N, C, H, W = images.shape
    images = np.clip(images * 255, 0, 255).astype(np.uint8)

    images = images.reshape([gh, gw, C, H, W])
    images = images.transpose(0, 3, 1, 4, 2)
    images = images.reshape([gh * H, gw * W, C])

    if C == 1:
        PIL.Image.fromarray(images[:, :, 0], "L").save(fname)
    else:
        PIL.Image.fromarray(images, "RGB").save(fname)


def num_range(s: str) -> list[int]:
    """Parse a comma-separated list of numbers or ranges."""
    range_re = re.compile(r"^(\d+)-(\d+)$")
    result = []
    for part in s.split(","):
        m = range_re.match(part)
        if m:
            result.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            result.append(int(part))
    return result


@click.command()
@click.option("--network", help="Network pickle filename", required=True)
@click.option("--seeds", help="List of random seeds", type=num_range, default="0-63", show_default=True)
@click.option("--outdir", help="Where to save the output images", type=str, required=True)
@click.option("--batch", help="Maximum batch size", type=int, default=32, show_default=True)
@click.option("--grid", help="Save as grid (WxH)", type=str, default=None)
@click.option("--steps", help="Number of sampling steps (for DDIM)", type=int, default=None)
@click.option("--ddim", help="Use DDIM sampling", is_flag=True)
@click.option("--eta", help="DDIM eta parameter", type=float, default=0.0, show_default=True)
def main(
    network: str,
    seeds: list[int],
    outdir: str,
    batch: int,
    grid: str | None,
    steps: int | None,
    ddim: bool,
    eta: float,
):
    """Generate images using a trained DiffiT model.

    Examples:

    \b
    # Generate 64 images with default settings
    python gen_images.py --network=model.pkl --outdir=out --seeds=0-63

    \b
    # Generate a 8x8 grid
    python gen_images.py --network=model.pkl --outdir=out --seeds=0-63 --grid=8x8

    \b
    # Use DDIM sampling with 50 steps
    python gen_images.py --network=model.pkl --outdir=out --seeds=0-15 --ddim --steps=50
    """
    print(f'Loading network from "{network}"...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with dnnlib.util.open_url(network) as f:
        data = pickle.load(f)

    # Get model and diffusion from checkpoint
    model = data.get("model_ema", data.get("model"))
    model = model.to(device).eval()

    # Get training set kwargs to determine resolution
    training_set_kwargs = data.get("training_set_kwargs", {})
    resolution = training_set_kwargs.get("resolution", 64)

    # Create diffusion object
    from diffusion import Diffusion
    diffusion = Diffusion(
        model=model,
        image_resolution=(3, resolution, resolution),
        n_times=1000,
        device=device,
    )

    os.makedirs(outdir, exist_ok=True)

    # Generate images
    print(f"Generating {len(seeds)} images...")
    all_images = []

    for batch_start in range(0, len(seeds), batch):
        batch_seeds = seeds[batch_start : batch_start + batch]
        batch_size = len(batch_seeds)

        # Set random seed for reproducibility
        torch.manual_seed(batch_seeds[0])

        print(f"  Generating batch {batch_start // batch + 1}...")

        if ddim and steps is not None:
            images = diffusion.sample_ddim(batch_size, ddim_steps=steps, eta=eta)
        else:
            images = diffusion.sample(batch_size)

        all_images.append(images.cpu().numpy())

    all_images = np.concatenate(all_images, axis=0)

    # Save images
    if grid is not None:
        gw, gh = [int(x) for x in grid.split("x")]
        print(f"Saving {gw}x{gh} grid...")
        save_image_grid(all_images[: gw * gh], os.path.join(outdir, "grid.png"), (gw, gh))
    else:
        print(f"Saving individual images...")
        for idx, (seed, image) in enumerate(zip(seeds, all_images)):
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
            image = image.transpose(1, 2, 0)  # CHW -> HWC
            if image.shape[2] == 1:
                image = image[:, :, 0]
            PIL.Image.fromarray(image).save(os.path.join(outdir, f"seed{seed:04d}.png"))

    print("Done!")


if __name__ == "__main__":
    main()
