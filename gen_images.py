"""
Generate individual PNG images using a pretrained DiffiT model.

Similar interface to StyleGAN-XL gen_images.py — generates individual PNGs
for visual inspection rather than bulk .npz for FID evaluation.

Usage:
    python gen_images.py \
        --model-path ckpts/diffit_256.safetensors \
        --seeds 0-49 --image-size 256 --cfg-scale 4.4 --outdir ./out
"""

import os
import re
from typing import List, Union

import click
import numpy as np
import PIL.Image
import torch
from diffusers.models import AutoencoderKL

import diffit.diffit as diffit_module
from diffit import create_diffusion, diffusion_defaults, NUM_CLASSES
from diffit.dist_util import load_state_dict


def parse_range(s: Union[str, List]) -> List[int]:
    """Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]
    """
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


@click.command()
@click.option("--model-path", required=True, type=str, help="Path to model checkpoint")
@click.option("--seeds", type=parse_range, required=True, help="List of random seeds (e.g., '0,1,4-6')")
@click.option("--outdir", required=True, type=str, help="Where to save the output images", metavar="DIR")
@click.option("--image-size", type=int, default=256, show_default=True, help="Image resolution (256 or 512)")
@click.option("--model", "model_name", type=str, default="Diffit", show_default=True, help="Model constructor name")
@click.option("--class-idx", type=int, default=None, help="Class label (random if not specified)")
@click.option("--cfg-scale", type=float, default=4.4, show_default=True, help="Classifier-free guidance scale")
@click.option("--num-sampling-steps", type=int, default=250, show_default=True, help="Number of diffusion sampling steps")
@click.option("--use-ddim/--no-use-ddim", default=False, show_default=True, help="Use DDIM sampling")
@click.option("--scale-pow", type=float, default=4.0, show_default=True, help="Power for cosine CFG schedule (256)")
@click.option("--vae-decoder", type=click.Choice(["ema", "mse"]), default="ema", show_default=True, help="VAE decoder variant")
@click.option("--decode-layer", type=int, default=None, help="Decode layer override")
@click.option("--batch-sz", type=int, default=1, show_default=True, help="Batch size per sample")
def generate_images(
    model_path,
    seeds,
    outdir,
    image_size,
    model_name,
    class_idx,
    cfg_scale,
    num_sampling_steps,
    use_ddim,
    scale_pow,
    vae_decoder,
    decode_layer,
    batch_sz,
):
    """Generate individual PNG images using a pretrained DiffiT model."""
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f'Loading model from "{model_path}"...')
    latent_size = image_size // 8
    model = diffit_module.__dict__[model_name](
        input_size=latent_size, decode_layer=decode_layer
    )
    msg = model.load_state_dict(load_state_dict(model_path, map_location="cpu"))
    print(f"Model loaded: {msg}")
    model.to(device).eval()
    torch.set_grad_enabled(False)

    # Create diffusion
    diff_config = diffusion_defaults()
    diff_config["timestep_respacing"] = str(num_sampling_steps)
    diffusion = create_diffusion(**diff_config)

    # VAE decoder
    vae = AutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-{vae_decoder}"
    ).to(device)

    os.makedirs(outdir, exist_ok=True)

    # Generate images
    for seed_idx, seed in enumerate(seeds):
        torch.manual_seed(seed)
        print(f"Generating image for seed {seed} ({seed_idx + 1}/{len(seeds)}) ...")

        z = torch.randn(batch_sz, 4, latent_size, latent_size, device=device)

        # Class label
        if class_idx is not None:
            classes = torch.full((batch_sz,), class_idx, device=device, dtype=torch.long)
        else:
            classes = torch.randint(0, NUM_CLASSES, (batch_sz,), device=device)

        # Classifier-free guidance setup
        z_cfg = torch.cat([z, z], 0)
        classes_null = torch.full((batch_sz,), NUM_CLASSES, device=device, dtype=torch.long)
        model_kwargs = {
            "y": torch.cat([classes, classes_null], 0),
            "cfg_scale": cfg_scale,
            "diffusion_steps": diff_config["diffusion_steps"],
            "scale_pow": scale_pow,
        }

        sample_fn = diffusion.ddim_sample_loop if use_ddim else diffusion.p_sample_loop
        sample = sample_fn(
            model.forward_with_cfg,
            z_cfg.shape,
            z_cfg,
            clip_denoised=False,
            progress=True,
            model_kwargs=model_kwargs,
            device=device,
        )

        sample, _ = sample.chunk(2, dim=0)

        # Decode latent → image
        sample = vae.decode(sample / 0.18215).sample
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1).cpu().numpy()

        # Save each image in batch
        for i, img in enumerate(sample):
            fname = f"seed{seed:04d}"
            if batch_sz > 1:
                fname += f"_b{i:02d}"
            PIL.Image.fromarray(img, "RGB").save(os.path.join(outdir, f"{fname}.png"))

    print("Done.")


if __name__ == "__main__":
    generate_images()
