"""
Generate a large batch of image samples from a DiffiT model and save them
as a .npz file for FID evaluation.

Usage with torchrun (multi-GPU):
    torchrun --nproc_per_node=4 sample.py \
        --model-path ckpts/diffit_256.safetensors \
        --image-size 256 --cfg-scale 4.4 --num-samples 50000 --outdir ./samples

Single GPU:
    python sample.py \
        --model-path ckpts/diffit_256.safetensors \
        --image-size 256 --cfg-scale 4.4 --num-samples 50000 --outdir ./samples
"""

import os

import click
import numpy as np
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL

import diffit.diffit as diffit_module
from diffit import create_diffusion, diffusion_defaults, NUM_CLASSES
from diffit.dist_util import setup_dist, dev, load_state_dict, get_rank, get_world_size


@click.command()
@click.option("--model-path", required=True, type=str, help="Path to model checkpoint (.safetensors or .pt)")
@click.option("--outdir", required=True, type=str, help="Output directory for .npz samples")
@click.option("--image-size", type=int, default=256, show_default=True, help="Image resolution (256 or 512)")
@click.option("--model", "model_name", type=str, default="Diffit", show_default=True, help="Model constructor name")
@click.option("--num-samples", type=int, default=50000, show_default=True, help="Total number of samples to generate")
@click.option("--batch-size", type=int, default=16, show_default=True, help="Batch size per GPU")
@click.option("--num-sampling-steps", type=int, default=250, show_default=True, help="Number of diffusion sampling steps")
@click.option("--cfg-scale", type=float, default=4.4, show_default=True, help="Classifier-free guidance scale")
@click.option("--cfg-cond/--no-cfg-cond", default=True, show_default=True, help="Use classifier-free guidance")
@click.option("--class-cond/--no-class-cond", default=True, show_default=True, help="Use class conditioning")
@click.option("--use-ddim/--no-use-ddim", default=False, show_default=True, help="Use DDIM sampling")
@click.option("--use-fp16/--no-use-fp16", default=False, show_default=True, help="Use FP16 inference")
@click.option("--scale-pow", type=float, default=4.0, show_default=True, help="Power for cosine CFG schedule (256 only)")
@click.option("--vae-decoder", type=click.Choice(["ema", "mse"]), default="ema", show_default=True, help="VAE decoder variant")
@click.option("--decode-layer", type=int, default=None, help="Decode layer override")
@click.option("--seed", type=int, default=None, help="Global random seed")
def generate_samples(
    model_path,
    outdir,
    image_size,
    model_name,
    num_samples,
    batch_size,
    num_sampling_steps,
    cfg_scale,
    cfg_cond,
    class_cond,
    use_ddim,
    use_fp16,
    scale_pow,
    vae_decoder,
    decode_layer,
    seed,
):
    """Generate samples for FID evaluation."""
    torch.backends.cuda.matmul.allow_tf32 = True

    # Distributed setup
    setup_dist()

    if seed is not None:
        torch.manual_seed(seed + get_rank())

    os.makedirs(outdir, exist_ok=True)

    # Create model
    latent_size = image_size // 8
    model = diffit_module.__dict__[model_name](
        input_size=latent_size, decode_layer=decode_layer
    )
    msg = model.load_state_dict(load_state_dict(model_path, map_location="cpu"))
    print(f"Model loaded: {msg}")

    # Create diffusion
    diff_config = diffusion_defaults()
    diff_config["timestep_respacing"] = str(num_sampling_steps)
    diffusion = create_diffusion(**diff_config)

    model.to(dev())
    if use_fp16:
        model.convert_to_fp16()
    model.eval()
    torch.set_grad_enabled(False)

    # VAE decoder
    vae = AutoencoderKL.from_pretrained(
        f"stabilityai/sd-vae-ft-{vae_decoder}"
    ).to(dev())

    # Sampling loop
    all_images = []
    all_labels = []
    world_size = get_world_size()

    print(f"Generating {num_samples} samples with batch_size={batch_size} "
          f"on {world_size} GPU(s)...")

    while len(all_images) * batch_size < num_samples:
        model_kwargs = {}
        z = torch.randn(batch_size, 4, latent_size, latent_size, device=dev())
        classes = torch.randint(
            low=0, high=NUM_CLASSES, size=(batch_size,), device=dev()
        )

        if cfg_cond:
            z = torch.cat([z, z], 0)
            classes_null = torch.full((batch_size,), NUM_CLASSES, device=dev())
            model_kwargs["y"] = torch.cat([classes, classes_null], 0)
            model_kwargs["cfg_scale"] = cfg_scale
            model_kwargs["diffusion_steps"] = diff_config["diffusion_steps"]
            model_kwargs["scale_pow"] = scale_pow
        else:
            model_kwargs["y"] = classes

        sample_fn = diffusion.ddim_sample_loop if use_ddim else diffusion.p_sample_loop
        sample = sample_fn(
            model.forward_with_cfg,
            z.shape,
            z,
            clip_denoised=False,
            progress=(get_rank() == 0),
            model_kwargs=model_kwargs,
            device=dev(),
        )

        if cfg_cond:
            sample, _ = sample.chunk(2, dim=0)

        # Decode latent → image
        sample = vae.decode(sample / 0.18215).sample
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1).contiguous()

        # Gather across GPUs
        if world_size > 1:
            gathered = [torch.zeros_like(sample) for _ in range(world_size)]
            dist.all_gather(gathered, sample)
            all_images.extend([s.cpu().numpy() for s in gathered])
        else:
            all_images.append(sample.cpu().numpy())

        if class_cond:
            if world_size > 1:
                gathered_labels = [torch.zeros_like(classes) for _ in range(world_size)]
                dist.all_gather(gathered_labels, classes)
                all_labels.extend([l.cpu().numpy() for l in gathered_labels])
            else:
                all_labels.append(classes.cpu().numpy())

        total_so_far = len(all_images) * batch_size
        print(f"Created {total_so_far} / {num_samples} samples")

    # Save results
    arr = np.concatenate(all_images, axis=0)[:num_samples]

    if get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(outdir, f"samples_{shape_str}.npz")
        print(f"Saving to {out_path}")
        if class_cond:
            label_arr = np.concatenate(all_labels, axis=0)[:num_samples]
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    if world_size > 1:
        dist.barrier()
    print("Sampling complete.")


if __name__ == "__main__":
    generate_samples()
