"""
Pre-download all models required by train.py, gen_images.py, and sample.py.

Run this on a machine with internet access BEFORE launching training on
an offline compute node.  The models are cached in the standard locations
(~/.cache/huggingface/ and ~/.cache/torch/hub/) and will be picked up
automatically by the training / sampling scripts.

Usage:
    python download_models.py
"""

import os


def main():
    # ------------------------------------------------------------------
    # 1. Stable Diffusion VAE  (used by train.py, gen_images.py, sample.py)
    # ------------------------------------------------------------------
    print("=" * 60)
    print("[1/3] Downloading stabilityai/sd-vae-ft-ema ...")
    print("=" * 60)
    from diffusers.models import AutoencoderKL

    AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    print("  -> cached.")

    # The MSE variant is used by gen_images.py --vae-decoder mse
    print()
    print("=" * 60)
    print("[2/3] Downloading stabilityai/sd-vae-ft-mse ...")
    print("=" * 60)
    AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    print("  -> cached.")

    # ------------------------------------------------------------------
    # 2. InceptionV3  (used by train.py for inline FID/IS evaluation)
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("[3/3] Downloading InceptionV3 (torchvision) ...")
    print("=" * 60)
    from torchvision.models import inception_v3, Inception_V3_Weights

    inception_v3(weights=Inception_V3_Weights.DEFAULT)
    print("  -> cached.")

    # ------------------------------------------------------------------
    print()
    print("All models downloaded successfully.")
    print(f"  HuggingFace cache: {os.path.expanduser('~/.cache/huggingface/')}")
    print(f"  Torch Hub cache:   {os.path.expanduser('~/.cache/torch/hub/')}")


if __name__ == "__main__":
    main()
