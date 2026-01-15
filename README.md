# DiffiT: Diffusion Vision Transformers for Image Generation

A PyTorch implementation of DiffiT - a U-Net style diffusion model with Vision Transformer blocks for image generation.

## Features

- **Time-Dependent Multi-Head Self-Attention (TDMHSA)**: Attention mechanism that incorporates diffusion timestep information
- **Vision Transformer Blocks**: Process image patches with time-conditioned attention
- **Efficient Sampling**: Uses `diffusers` schedulers for DDPM and DDIM sampling
- **Multi-GPU Training**: Distributed training support with gradient accumulation
- **Mixed Precision**: FP16/BF16 training for faster training

## Project Structure

```
DiffiT-v2/
├── train.py                    # Main training entry point
├── gen_images.py               # Image generation script
├── requirements.txt            # Python dependencies
│
├── models/                     # Model architectures
│   ├── diffit.py               # Main DiffiT U-Net model
│   ├── attention.py            # TDMHSA and transformer blocks
│   └── vit.py                  # Vision Transformer components
│
├── diffusion/                  # Diffusion process
│   └── diffusion.py            # DDPM/DDIM using diffusers
│
├── training/                   # Training utilities
│   ├── training_loop.py        # Main training loop
│   └── dataset.py              # Dataset loaders
│
├── torch_utils/                # PyTorch utilities
└── dnnlib/                     # Deep learning utilities
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

| Argument | Description | Default |
|----------|-------------|---------|
| `--outdir` | Output directory | Required |
| `--data` | Path to dataset (folder or zip) | Required |
| `--gpus` | Number of GPUs | Required |
| `--batch-gpu` | Batch size per GPU | Required |
| `--resolution` | Image resolution | 64 |
| `--timesteps` | Diffusion timesteps | 1000 |
| `--base-dim` | Base channel dimension | 128 |
| `--hidden-dim` | ViT hidden dimension | 64 |
| `--num-heads` | Attention heads | 4 |
| `--lr` | Learning rate | 1e-4 |
| `--kimg` | Training duration (kimg) | 25000 |

## Generation

Generate images using a trained model:

```bash
# Generate 64 images with DDIM (fast, ~50 steps)
python gen_images.py \
    --network=training-runs/00000-diffit/network-snapshot.pkl \
    --outdir=./generated \
    --seeds=0-63

# Generate a grid
python gen_images.py \
    --network=model.pkl \
    --outdir=./generated \
    --seeds=0-63 \
    --grid=8x8

# Use DDPM sampling (slower, higher quality)
python gen_images.py \
    --network=model.pkl \
    --outdir=./generated \
    --seeds=0-15 \
    --ddpm
```

## Model Architecture

DiffiT uses a U-Net architecture with Vision Transformer blocks:

```
Input Image → [Encoder Path] → [Bottleneck] → [Decoder Path] → Output
                  ↓                               ↑
              Skip Connections ──────────────────┘

Each block contains:
- GroupNorm + SiLU + Conv
- Vision Transformer with TDMHSA
- Residual connection
```

### Key Components

1. **TDMHSA (Time-Dependent Multi-Head Self-Attention)**:
   - Combines spatial attention with temporal conditioning
   - Uses relative positional embeddings
   - Efficient implementation using `scaled_dot_product_attention`

2. **Timestep Embedding**:
   - Sinusoidal position embeddings for timesteps
   - MLP projection to model dimension

3. **Diffusion Process**:
   - Uses `diffusers` DDPMScheduler/DDIMScheduler when available
   - Fallback to custom implementation

## Dataset Format

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

```bibtex
@article{diffit2023,
  title={DiffiT: Diffusion Vision Transformers for Image Generation},
  author={...},
  journal={...},
  year={2023}
}
```
