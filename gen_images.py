"""
Generate individual PNG images using a pretrained DiffiT model.

Two modes:
  seed mode      — pass --seeds, one image per seed (random or fixed class)
  per-class mode — pass --samples-per-class, generates N images for every
                   class (or for --classes subset). Mirrors the per-class
                   knob in experiments/notebooks/generate_class_samples.py.

Multi-GPU: one model + VAE copy is loaded per device, jobs are pre-sharded
round-robin across GPUs, and each GPU is driven by a single worker thread
(so two threads never touch the same model).

Usage:
    # seed mode
    python gen_images.py --model-path ckpt.pt --seeds 0-49 \
        --image-size 256 --cfg-scale 4.4 --outdir ./out --gpus 0,1

    # per-class mode (1000 images per class, all 1000 classes)
    python gen_images.py --model-path ckpt.pt --samples-per-class 1000 \
        --batch-size 32 --image-size 256 --outdir ./out --gpus 0,1

    # per-class mode, subset
    python gen_images.py --model-path ckpt.pt --samples-per-class 100 \
        --classes 0,1,207,999 --batch-size 32 --outdir ./out --gpus 0,1
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import List, Union

import click
import PIL.Image
import torch
from diffusers.models import AutoencoderKL
from tqdm.auto import tqdm

import diffit.diffit as diffit_module
from diffit import create_diffusion, diffusion_defaults, NUM_CLASSES
from diffit.dist_util import load_state_dict


def parse_range(s: Union[str, List, None]) -> List[int]:
    """Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]
    """
    if s is None or s == "":
        return []
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


def parse_gpus(s: Union[str, List, None]) -> List[int]:
    """Parse '0,1,2' or '0-3' into a list of GPU ids. None → all available."""
    if s is None or s == "":
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return []
    if isinstance(s, list):
        return [int(x) for x in s]
    return parse_range(s)


def _run_sampling(model, vae, diffusion, dev, latent_size, num_classes,
                  bs, class_labels, gen, *, cfg_scale, scale_pow,
                  diffusion_steps, use_ddim):
    """Single forward pass: bs latents → bs decoded uint8 NHWC images."""
    z = torch.randn(bs, 4, latent_size, latent_size, device=dev, generator=gen)
    classes_null = torch.full((bs,), num_classes, device=dev, dtype=torch.long)

    z_cfg = torch.cat([z, z], 0)
    model_kwargs = {
        "y": torch.cat([class_labels, classes_null], 0),
        "cfg_scale": cfg_scale,
        "diffusion_steps": diffusion_steps,
        "scale_pow": scale_pow,
    }

    sample_fn = diffusion.ddim_sample_loop if use_ddim else diffusion.p_sample_loop
    sample = sample_fn(
        model.forward_with_cfg,
        z_cfg.shape, z_cfg,
        clip_denoised=False, progress=False,
        model_kwargs=model_kwargs, device=dev,
    )
    sample, _ = sample.chunk(2, dim=0)

    sample = vae.decode(sample / 0.18215).sample
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    return sample.permute(0, 2, 3, 1).cpu().numpy()


@click.command()
@click.option("--model-path", required=True, type=str, help="Path to model checkpoint")
@click.option("--seeds", type=parse_range, default=None, help="Seed mode: list of seeds (e.g., '0,1,4-6')")
@click.option("--samples-per-class", type=int, default=None, help="Per-class mode: N images per class")
@click.option("--classes", type=parse_range, default=None, help="Per-class mode: subset (default: all 0..num_classes-1)")
@click.option("--batch-size", type=int, default=32, show_default=True, help="Per-class mode: forward-pass batch size per GPU")
@click.option("--base-seed", type=int, default=0, show_default=True, help="Per-class mode: seed = base_seed + cls*1e6 + first_sample_idx")
@click.option("--outdir", required=True, type=str, help="Where to save the output images", metavar="DIR")
@click.option("--image-size", type=int, default=256, show_default=True, help="Image resolution (256 or 512)")
@click.option("--model", "model_name", type=str, default="Diffit", show_default=True, help="Model constructor name")
@click.option("--class-idx", type=int, default=None, help="Seed mode: fixed class label (random if not specified)")
@click.option("--cfg-scale", type=float, default=4.4, show_default=True, help="Classifier-free guidance scale")
@click.option("--num-sampling-steps", type=int, default=250, show_default=True, help="Number of diffusion sampling steps")
@click.option("--use-ddim/--no-use-ddim", default=False, show_default=True, help="Use DDIM sampling")
@click.option("--scale-pow", type=float, default=4.0, show_default=True, help="Power for cosine CFG schedule (256)")
@click.option("--vae-decoder", type=click.Choice(["ema", "mse"]), default="ema", show_default=True, help="VAE decoder variant")
@click.option("--decode-layer", type=int, default=None, help="Decode layer override")
@click.option("--batch-sz", type=int, default=1, show_default=True, help="Seed mode: images per seed (sharing the seed)")
@click.option("--num-classes", type=int, default=1000, show_default=True, help="Must match num_classes used at train time")
@click.option("--gpus", type=parse_gpus, default=None, help="GPU ids to use, e.g. '0,1' or '0-3' (default: all available)")
def generate_images(
    model_path,
    seeds,
    samples_per_class,
    classes,
    batch_size,
    base_seed,
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
    num_classes,
    gpus,
):
    """Generate individual PNG images using a pretrained DiffiT model."""
    torch.backends.cuda.matmul.allow_tf32 = True

    # Mode validation: exactly one of --seeds / --samples-per-class.
    if (seeds and samples_per_class is not None) or (not seeds and samples_per_class is None):
        raise click.UsageError(
            "Provide exactly one of --seeds (seed mode) or --samples-per-class (per-class mode)."
        )
    per_class_mode = samples_per_class is not None

    gpu_ids = gpus if gpus else parse_gpus(None)
    devices = (
        [torch.device(f"cuda:{i}") for i in gpu_ids]
        if gpu_ids else [torch.device("cpu")]
    )
    print(f"Devices: {devices}")

    # One model + VAE per device. Each thread is pinned to one device, so
    # threads never touch the same model.
    print(f'Loading model from "{model_path}" onto {len(devices)} device(s)...')
    latent_size = image_size // 8
    state = load_state_dict(model_path, map_location="cpu")
    models = []
    vaes = {}
    for dev in devices:
        m = diffit_module.__dict__[model_name](
            input_size=latent_size, decode_layer=decode_layer, num_classes=num_classes,
        )
        msg = m.load_state_dict(state)
        m.to(dev).eval()
        models.append(m)

        v = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_decoder}").to(dev).eval()
        for prm in v.parameters():
            prm.requires_grad_(False)
        vaes[dev] = v
    print(f"Model loaded: {msg}")
    del state

    diff_config = diffusion_defaults()
    diff_config["timestep_respacing"] = str(num_sampling_steps)
    diffusion = create_diffusion(**diff_config)
    diffusion_steps = diff_config["diffusion_steps"]

    os.makedirs(outdir, exist_ok=True)

    # ------------------------------------------------------------------
    # Build job list
    # ------------------------------------------------------------------
    if per_class_mode:
        cls_list = classes if classes else list(range(num_classes))
        # Each job = (cls, sample_start, batch_size_actual, seed)
        jobs = []
        for cls in cls_list:
            for start in range(0, samples_per_class, batch_size):
                bs = min(batch_size, samples_per_class - start)
                seed = base_seed + cls * 10**6 + start
                jobs.append((cls, start, bs, seed))
        total_images = len(cls_list) * samples_per_class
        unit = "img"
        print(f"Per-class mode: {len(cls_list)} classes × {samples_per_class} samples "
              f"= {total_images} images in {len(jobs)} batches")
    else:
        # Seed mode: each job = (seed,)
        jobs = [(int(s),) for s in seeds]
        total_images = len(seeds) * batch_sz
        unit = "seed"
        print(f"Seed mode: {len(seeds)} seeds × batch_sz={batch_sz} = {total_images} images")

    # Pre-shard jobs round-robin across GPUs.
    jobs_per_gpu = [[] for _ in devices]
    for i, job in enumerate(jobs):
        jobs_per_gpu[i % len(devices)].append(job)

    pbar = tqdm(total=(total_images if per_class_mode else len(seeds)),
                desc="generating", unit=unit, smoothing=0.05)
    pbar_lock = Lock()

    def gpu_worker(gpu_idx):
        dev = devices[gpu_idx]
        model = models[gpu_idx]
        vae = vaes[dev]

        with torch.inference_mode():
            for job in jobs_per_gpu[gpu_idx]:
                if per_class_mode:
                    cls, start, bs, seed = job
                    gen = torch.Generator(device=dev).manual_seed(int(seed))
                    class_labels = torch.full((bs,), cls, device=dev, dtype=torch.long)

                    imgs = _run_sampling(
                        model, vae, diffusion, dev, latent_size, num_classes,
                        bs, class_labels, gen,
                        cfg_scale=cfg_scale, scale_pow=scale_pow,
                        diffusion_steps=diffusion_steps, use_ddim=use_ddim,
                    )

                    for i, img in enumerate(imgs):
                        fname = f"class{cls:04d}_sample{start + i:04d}.png"
                        PIL.Image.fromarray(img, "RGB").save(os.path.join(outdir, fname))

                    with pbar_lock:
                        pbar.update(bs)
                else:
                    (seed,) = job
                    gen = torch.Generator(device=dev).manual_seed(seed)

                    if class_idx is not None:
                        class_labels = torch.full((batch_sz,), class_idx, device=dev, dtype=torch.long)
                    else:
                        class_labels = torch.randint(0, NUM_CLASSES, (batch_sz,),
                                                     device=dev, generator=gen, dtype=torch.long)

                    imgs = _run_sampling(
                        model, vae, diffusion, dev, latent_size, num_classes,
                        batch_sz, class_labels, gen,
                        cfg_scale=cfg_scale, scale_pow=scale_pow,
                        diffusion_steps=diffusion_steps, use_ddim=use_ddim,
                    )

                    for i, img in enumerate(imgs):
                        fname = f"seed{seed:04d}"
                        if batch_sz > 1:
                            fname += f"_b{i:02d}"
                        PIL.Image.fromarray(img, "RGB").save(os.path.join(outdir, f"{fname}.png"))

                    with pbar_lock:
                        pbar.update(1)

    try:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futs = [pool.submit(gpu_worker, i) for i in range(len(devices))]
            for f in futs:
                f.result()  # surface exceptions
    finally:
        pbar.close()

    print("Done.")


if __name__ == "__main__":
    generate_images()
