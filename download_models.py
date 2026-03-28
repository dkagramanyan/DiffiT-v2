"""
Pre-flight check for DiffiT: verify all dependencies and download models.

Run this on a machine with internet access BEFORE launching training on
an offline compute node.  The script will:

  1. Check that all required Python packages are installed.
  2. Check that CUDA is available and report GPU info.
  3. Download and cache all external models used by train.py,
     gen_images.py, and sample.py.

Usage:
    python download_models.py
"""

import importlib
import sys


# -----------------------------------------------------------------------
# 1. Dependency check
# -----------------------------------------------------------------------

REQUIRED_PACKAGES = {
    # package_name: import_name (if different from package_name)
    "torch": "torch",
    "torchvision": "torchvision",
    "timm": "timm",
    "diffusers": "diffusers",
    "tqdm": "tqdm",
    "click": "click",
    "numpy": "numpy",
    "Pillow": "PIL",
    "safetensors": "safetensors",
    "scipy": "scipy",
    "accelerate": "accelerate",
    "tensorboard": "tensorboard",
    "psutil": "psutil",
}


def check_dependencies():
    print("=" * 60)
    print("Checking installed packages")
    print("=" * 60)

    missing = []
    for pkg_name, import_name in REQUIRED_PACKAGES.items():
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  {pkg_name:<20s} {version}")
        except ImportError:
            print(f"  {pkg_name:<20s} *** NOT FOUND ***")
            missing.append(pkg_name)

    print()
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install with:  pip install {' '.join(missing)}")
        print()
        return False

    print("All packages OK.")
    print()
    return True


# -----------------------------------------------------------------------
# 2. CUDA check
# -----------------------------------------------------------------------

def check_cuda():
    print("=" * 60)
    print("Checking CUDA")
    print("=" * 60)

    import torch

    if not torch.cuda.is_available():
        print("  CUDA is NOT available. Training will not work.")
        print()
        return False

    print(f"  PyTorch:      {torch.__version__}")
    print(f"  CUDA runtime: {torch.version.cuda}")
    print(f"  cuDNN:        {torch.backends.cudnn.version()}")
    print(f"  GPUs:         {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        print(f"    [{i}] {name}  ({mem:.1f} GiB)")

    print()
    return True


# -----------------------------------------------------------------------
# 3. Download models
# -----------------------------------------------------------------------

def download_models():
    import os

    # --- VAE (EMA variant) ---
    print("=" * 60)
    print("[1/3] Downloading stabilityai/sd-vae-ft-ema ...")
    print("=" * 60)
    from diffusers.models import AutoencoderKL

    AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    print("  -> cached.\n")

    # --- VAE (MSE variant, used by gen_images.py --vae-decoder mse) ---
    print("=" * 60)
    print("[2/3] Downloading stabilityai/sd-vae-ft-mse ...")
    print("=" * 60)
    AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    print("  -> cached.\n")

    # --- InceptionV3 (used by train.py for inline FID/IS evaluation) ---
    print("=" * 60)
    print("[3/3] Downloading InceptionV3 (torchvision) ...")
    print("=" * 60)
    from torchvision.models import inception_v3, Inception_V3_Weights

    inception_v3(weights=Inception_V3_Weights.DEFAULT)
    print("  -> cached.\n")

    print("All models downloaded successfully.")
    print(f"  HuggingFace cache: {os.path.expanduser('~/.cache/huggingface/')}")
    print(f"  Torch Hub cache:   {os.path.expanduser('~/.cache/torch/hub/')}")
    print()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    print()
    deps_ok = check_dependencies()
    if not deps_ok:
        sys.exit(1)

    cuda_ok = check_cuda()
    if not cuda_ok:
        print("Continuing with model downloads anyway...\n")

    download_models()

    print("=" * 60)
    print("Pre-flight check complete. Ready to train.")
    print("=" * 60)


if __name__ == "__main__":
    main()
