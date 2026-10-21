"""
Compare reverse-diffusion samplers by combra metrics vs. sampling steps.

Answers the practical question *how many diffusion steps ``k`` does each sampler
need to generate good-quality images?* For every sampler and every ``k``, this
generates a batch of samples from a trained DiffiT checkpoint, then scores the
batch against a fixed batch of real reference images with combra's
``compare_samplers`` (FID / CMMD / FD-DINOv2 + angle-Wasserstein / Gaussian-fit
metrics; the angle columns are combra's default angle version, v1 = P0, under
unprefixed names). The result is a tidy table saved as JSON.

The reference is built as ``scripts/train.py`` builds its combra reference: a
seeded random subset of ``--num-samples`` reals (all of them if the dataset is not
larger), expanded to the 8 dihedral transforms of each with ``--augment True``
(the default, as in train.py) -- pass the value the run was trained with.

This is the DiffiT-side wiring for :func:`combra.metrics.compare_samplers`; the
generic sweep lives in combra so it stays sampler- and codebase-agnostic.

Usage:

    diffit-compare-samplers \\
        --model-path ./training-runs/.../diffit-snapshot-000400-inference.pt \\
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

import diffit.diffit as diffit_module
from diffit import create_diffusion, diffusion_defaults
from diffit.constants import VAE_SCALE_FACTOR
from diffit.dist_util import extract_inference_state_dict, load_state_dict
from diffit.image_datasets import count_data, read_class_meta
from diffit.metrics import sample_latents, to_uint8
from scripts.gen_images import _load_vae
from scripts.train import load_reference_shard

try:
    from combra.metrics import compare_samplers, dihedral_expand

    HAS_COMBRA = True
except ImportError:
    HAS_COMBRA = False


def _parse_csv(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def _load_reference_images(data, image_size, num_samples, seed, augment):
    """The real reference batch, built the way ``scripts/train.py`` builds its combra reference.

    A seeded random subset of ``num_samples`` reals (the whole dataset when it holds
    no more), never the first N of the class-sorted zip, so the reference keeps the
    dataset's class mix like the uniform-class fakes; with ``augment`` it is expanded
    to all 8 dihedral transforms of each real, as train.py's ``--augment`` does.
    Returned as NCHW ``uint8``.
    """
    num_classes = read_class_meta(data)[1]
    total_count = count_data(data)
    ref_count = min(num_samples, total_count)
    reference = load_reference_shard(data, ref_count, total_count, image_size, 2, 0, 1, seed, num_classes)
    return dihedral_expand(reference) if augment else reference


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
        sample = to_uint8(sample)
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
@click.option("--augment", type=bool, default=True, show_default=True, metavar="BOOL",
              help="Expand the reference to its 8 dihedral transforms; pass the run's train.py --augment")
@click.option("--outdir", required=True, type=str, help="Output directory for the JSON table")
def main(model_path, data, image_size, cfg_scale, scale_pow, num_samples, batch_size,
         samplers, k_values, vae_decoder, model_name, num_classes, decode_layer, seed, augment, outdir):
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

    # Sliced, and tiled at >= 1024 px, as train.py decodes its eval samples.
    vae = _load_vae(vae_decoder, dev, image_size)

    diffusion_steps = diffusion_defaults()["diffusion_steps"]

    # --- Reference real images -----------------------------------------------
    print(f"Loading {num_samples} real reference images from {data}"
          f"{' (x8 dihedral)' if augment else ''} ...")
    reference = _load_reference_images(data, image_size, num_samples, seed, augment)

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
    table_path = outdir_p / "sampler_comparison.json"
    df.to_json(table_path, orient="records", indent=2)
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
