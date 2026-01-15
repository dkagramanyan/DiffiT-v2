# DiffiT: Diffusion Vision Transformers for Image Generation

A PyTorch implementation of DiffiT - a U-Net style diffusion model with Vision Transformer blocks for image generation.

## Project Structure

```
DiffiT-v2/
├── train.py                    # Main training entry point
├── gen_images.py               # Image generation script
├── requirements.txt            # Python dependencies
│
├── models/                     # Model architectures
│   ├── __init__.py
│   ├── diffit.py               # Main DiffiT model
│   ├── attention.py            # Attention mechanisms (TDMHSA)
│   └── vit.py                  # Vision Transformer components
│
├── diffusion/                  # Diffusion model components
│   ├── __init__.py
│   ├── diffusion.py            # Diffusion process (forward/reverse)
│   └── schedule.py             # Noise schedules
│
├── training/                   # Training utilities
│   ├── __init__.py
│   ├── training_loop.py        # Main training loop
│   ├── dataset.py              # Dataset loaders
│   └── loss.py                 # Loss functions
│
├── torch_utils/                # PyTorch utilities
│   ├── __init__.py
│   ├── custom_ops.py           # CUDA kernel compilation
│   ├── misc.py                 # Miscellaneous utilities
│   ├── persistence.py          # Model pickling
│   ├── training_stats.py       # Training statistics
│   └── ops/                    # Custom CUDA operations
│       ├── __init__.py
│       ├── conv2d_gradfix.py   # Gradient-fixed convolutions
│       └── fma.py              # Fused multiply-add
│
├── dnnlib/                     # Deep learning utilities
│   ├── __init__.py
│   └── util.py                 # Utility functions
│
└── plugins/                    # Optional plugins
    └── __init__.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

Train a DiffiT model on your dataset:

```bash
python train.py \
    --outdir=./training-runs \
    --data=/path/to/dataset \
    --gpus=4 \
    --batch-gpu=16 \
    --resolution=64 \
    --kimg=25000
```

### Key Arguments

- `--outdir`: Output directory for checkpoints and logs
- `--data`: Path to training dataset (folder or zip)
- `--gpus`: Number of GPUs to use
- `--batch-gpu`: Batch size per GPU
- `--resolution`: Image resolution (default: 64)
- `--timesteps`: Number of diffusion timesteps (default: 1000)
- `--base-dim`: Base channel dimension (default: 128)
- `--lr`: Learning rate (default: 1e-4)
- `--kimg`: Total training duration in thousands of images

## Generation

Generate images using a trained model:

```bash
# Generate 64 random images
python gen_images.py \
    --network=training-runs/00000-diffit/network-snapshot.pkl \
    --outdir=./generated \
    --seeds=0-63

# Generate an 8x8 grid
python gen_images.py \
    --network=model.pkl \
    --outdir=./generated \
    --seeds=0-63 \
    --grid=8x8

# Use DDIM sampling (faster)
python gen_images.py \
    --network=model.pkl \
    --outdir=./generated \
    --seeds=0-15 \
    --ddim \
    --steps=50
```

## Model Architecture

DiffiT uses a U-Net architecture with Vision Transformer blocks:

1. **Time-Dependent Multi-Head Self-Attention (TDMHSA)**: Attention mechanism that incorporates diffusion timestep information
2. **Vision Transformer Blocks**: Process image patches with time-conditioned attention
3. **Residual Blocks**: Combine convolutions with transformer processing
4. **U-Net Structure**: Encoder-decoder with skip connections

## Dataset Format

The dataset should be organized as:
- A folder containing image files (PNG, JPEG, etc.)
- Or a ZIP file containing images
- Optional: `dataset.json` with labels for conditional generation

Example:
```
dataset/
├── img_001.png
├── img_002.png
├── ...
└── dataset.json  # Optional: {"labels": [["img_001.png", 0], ...]}
```

## License

See LICENSE file for details.

## Citation

If you use this code, please cite the original DiffiT paper:

```bibtex
@article{diffit2023,
  title={DiffiT: Diffusion Vision Transformers for Image Generation},
  author={...},
  journal={...},
  year={2023}
}
```
