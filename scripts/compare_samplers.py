"""
Compare reverse-diffusion samplers by combra metrics vs. sampling steps.

Answers the practical question *how many diffusion steps ``k`` does each sampler
need to generate good-quality images?* For every sampler and every ``k``, this
generates a batch of samples from a trained DiffiT checkpoint, then scores the
batch against a fixed batch of real reference images with combra's
``compare_samplers`` (FID / CMMD / FD-DINOv2 + angle-Wasserstein / Gaussian-fit
metrics). The result is a tidy table plus a metric-vs-k plot.

This is the DiffiT-side wiring for :func:`combra.metrics.compare_samplers`; the
generic sweep/plot lives in combra so it stays sampler- and codebase-agnostic.

Usage (see also sbatch/h200_compare_samplers_256x256.sbatch):

    diffit-compare-samplers \\
        --model-path ./runs/.../network-final.pt \\
        --data ./datasets/imagenet_9to4_1024x1024_256x256.zip \\
        --image-size 256 --cfg-scale 4.4 --num-samples 512 \\
        --samplers dpm++,unipc,ddim,ddpm \\
        --k-values 5,10,20,50,100,250 \\
        --outdir ./sampler-comparison/256
"""

from pathlib import Path

import click
import numpy as np
import torch
from diffusers.models import AutoencoderKL

import diffit.diffit as diffit_module
from diffit import create_diffusion, diffusion_defaults
from diffit.constants import PIXEL_NORM_HALF, UINT8_MAX, VAE_SCALE_FACTOR
from diffit.dist_util import extract_inference_state_dict, load_state_dict
from diffit.image_datasets import load_data
from diffit.metrics import sample_latents

try:
    from combra.metrics import compare_samplers, plot_sampler_comparison

    HAS_COMBRA = True
except ImportError:
    HAS_COMBRA = False


def _parse_csv(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def _load_reference_images(data, image_size, num_samples):
    """Pull ``num_samples`` real reference images as an NCHW float batch in [-1, 1]."""
    loader = load_data(
        data_dir=data, batch_size=min(num_samples, 64), image_size=image_size,
        class_cond=False, deterministic=True, random_flip=False, num_workers=2,
        drop_last=False,
    )
    imgs = []
    got = 0
    for batch, _ in loader:
        imgs.append(batch.numpy())
        got += batch.shape[0]
        if got >= num_samples:
            break
    return np.concatenate(imgs, axis=0)[:num_samples]


@torch.inference_mode()
def _generate_batch(model, vae, dev, latent_size, num_classes, num_samples,
                    batch_size, *, sampler, k, cfg_scale, scale_pow, diffusion_steps, gen):
    """Generate ``num_samples`` decoded uint8 NHWC images with ``sampler`` at ``k`` steps."""
    # dpm++/unipc subsample the full 1000-step schedule; ddim/ddpm use a spaced
    # schedule whose num_timesteps == k (matches scripts/train.py).
    diff_config = diffusion_defaults()
    diff_config["timestep_respacing"] = "" if sampler in ("dpm++", "unipc") else str(k)
    diffusion = create_diffusion(**diff_config)

    out = []
    remaining = num_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        z = torch.randn(bs, 4, latent_size, latent_size, device=dev, generator=gen)
        class_labels = torch.randint(0, num_classes, (bs,), device=dev, generator=gen)
        classes_null = torch.full((bs,), num_classes, device=dev, dtype=torch.long)

        z_cfg = torch.cat([z, z], 0)
        model_kwargs = {
            "y": torch.cat([class_labels, classes_null], 0),
            "cfg_scale": cfg_scale,
            "diffusion_steps": diffusion_steps,
            "scale_pow": scale_pow,
        }
        sample = sample_latents(
            model.forward_with_cfg, diffusion, z_cfg.shape, dev,
            sampler=sampler, num_steps=k, model_kwargs=model_kwargs, noise=z_cfg,
        )
        sample, _ = sample.chunk(2, dim=0)
        sample = vae.decode(sample / VAE_SCALE_FACTOR).sample
        sample = ((sample + 1) * PIXEL_NORM_HALF).clamp(0, UINT8_MAX).to(torch.uint8)
        out.append(sample.permute(0, 2, 3, 1).cpu().numpy())
        remaining -= bs
    return np.concatenate(out, axis=0)


@click.command()
@click.option("--model-path", required=True, type=str, help="Path to model checkpoint")
@click.option("--data", required=True, type=str, help="Real reference dataset (zip or dir), same as training --data")
@click.option("--image-size", type=int, default=256, show_default=True)
@click.option("--cfg-scale", type=float, default=4.4, show_default=True, help="Classifier-free guidance scale")
@click.option("--scale-pow", type=float, default=4.0, show_default=True, help="Power for cosine CFG schedule (256)")
@click.option("--num-samples", type=int, default=512, show_default=True, help="Samples generated (and real refs) per (sampler, k)")
@click.option("--batch-size", type=int, default=32, show_default=True, help="Per-forward batch size")
@click.option("--samplers", type=str, default="dpm++,unipc,ddim,ddpm", show_default=True, help="Comma-separated sampler names")
@click.option("--k-values", type=str, default="5,10,20,50,100,250", show_default=True, help="Comma-separated step counts to sweep")
@click.option("--vae-decoder", type=click.Choice(["ema", "mse"]), default="ema", show_default=True)
@click.option("--model", "model_name", type=str, default="Diffit", show_default=True, help="Model constructor name")
@click.option("--num-classes", type=int, default=None, help="Override num_classes (default: auto-detect from checkpoint)")
@click.option("--decode-layer", type=int, default=None, help="Decode layer override")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--outdir", required=True, type=str, help="Output directory for the table + plot")
def main(model_path, data, image_size, cfg_scale, scale_pow, num_samples, batch_size,
         samplers, k_values, vae_decoder, model_name, num_classes, decode_layer, seed, outdir):
    """Compare samplers by combra metrics as a function of sampling steps."""
    if not HAS_COMBRA:
        raise click.UsageError("combra is not installed; `pip install combra` to run this comparison.")

    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler_names = _parse_csv(samplers)
    ks = [int(x) for x in _parse_csv(k_values)]
    latent_size = image_size // 8

    # --- Load model + VAE (mirrors scripts/gen_images.py) --------------------
    state = extract_inference_state_dict(load_state_dict(model_path, map_location="cpu"))
    ckpt_num_classes = int(state["y_embedder.embedding_table.weight"].shape[0]) - 1
    if num_classes is None:
        num_classes = ckpt_num_classes
    elif num_classes != ckpt_num_classes:
        raise click.UsageError(
            f"--num-classes={num_classes} conflicts with checkpoint (implies {ckpt_num_classes})"
        )

    model = diffit_module.__dict__[model_name](
        input_size=latent_size, decode_layer=decode_layer, num_classes=num_classes,
    )
    model.load_state_dict(state)
    model.to(dev).eval()
    del state

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_decoder}").to(dev).eval()
    for prm in vae.parameters():
        prm.requires_grad_(False)

    diffusion_steps = diffusion_defaults()["diffusion_steps"]

    # --- Reference real images -----------------------------------------------
    print(f"Loading {num_samples} real reference images from {data} ...")
    reference = _load_reference_images(data, image_size, num_samples)

    # --- One generator per sampler; gen(k) returns a uint8 NHWC batch --------
    def make_sampler_fn(name):
        def sampler_fn(k):
            gen = torch.Generator(device=dev).manual_seed(seed)
            print(f"Generating {num_samples} samples: sampler={name} k={k} ...")
            return _generate_batch(
                model, vae, dev, latent_size, num_classes, num_samples, batch_size,
                sampler=name, k=k, cfg_scale=cfg_scale, scale_pow=scale_pow,
                diffusion_steps=diffusion_steps, gen=gen,
            )
        return sampler_fn

    samplers_map = {name: make_sampler_fn(name) for name in sampler_names}

    # --- Sweep + score with combra ------------------------------------------
    df = compare_samplers(
        reference, samplers_map, ks, device=str(dev), image_metrics=True,
    )

    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)
    table_path = outdir_p / "sampler_comparison.parquet"
    plot_path = outdir_p / "sampler_comparison.png"
    df.to_parquet(table_path)
    plot_sampler_comparison(df, save_path=str(plot_path))
    print(f"Wrote {table_path}\nWrote {plot_path}")


if __name__ == "__main__":
    main()
