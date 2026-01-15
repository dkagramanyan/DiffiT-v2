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

Train a DiffiT model on your dataset.

### Multi-GPU Training (Recommended: torchrun)

For multi-GPU training, use `torchrun` which provides robust distributed training:

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 train.py \
    --outdir=./training-runs \
    --data=/path/to/dataset \
    --batch-gpu=16 \
    --resolution=64 \
    --kimg=25000

# Multi-node training (2 nodes, 4 GPUs each)
torchrun --nnodes=2 --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py --outdir=./runs --data=/path/to/dataset --batch-gpu=16
```

### SLURM Cluster

For SLURM-managed clusters:

```bash
# The script automatically detects SLURM environment variables
srun --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 \
    python train.py --outdir=./runs --data=/path/to/dataset --batch-gpu=16
```

### Legacy Mode (Single Node)

For simpler single-node training:

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
| `--gpus` | Number of GPUs (legacy mode, ignored with torchrun) | 1 |
| `--batch-gpu` | Batch size per GPU | Required |
| `--resolution` | Image resolution | 64 |
| `--timesteps` | Diffusion timesteps | 1000 |
| `--base-dim` | Base channel dimension | 128 |
| `--hidden-dim` | ViT hidden dimension | 64 |
| `--num-heads` | Attention heads | 4 |
| `--lr` | Learning rate | 1e-4 |
| `--kimg` | Training duration (kimg) | 25000 |
| `--fp32` | Disable mixed precision | False |
| `--resume` | Resume from checkpoint | None |

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

DiffiT uses a U-Net architecture with Vision Transformer blocks, as described in the paper:

```
Input Image → [Encoder Path] → [Bottleneck] → [Decoder Path] → Output
                  ↓                               ↑
              Skip Connections ──────────────────┘

DiffiT ResBlock (Paper Eq. 9-10):
  x̂ = Conv3×3(Swish(GN(x)))      # Convolutional layer
  x = DiffiT-Block(x̂, t) + x     # Transformer + residual

DiffiT Transformer Block (Paper Eq. 7-8):
  x̂ = TMSA(LN(x), t) + x         # Time-dependent attention
  x = MLP(LN(x̂)) + x̂             # Feed-forward + residual
```

### Key Components (Paper Reference)

1. **Time-Dependent Multi-Head Self-Attention (TMSA)** - Paper Eq. 3-6:
   - Computes time-dependent queries, keys, values:
     - q = x·W_qs + t·W_qt
     - k = x·W_ks + t·W_kt  
     - v = x·W_vs + t·W_vt
   - Attention with relative position bias B:
     - Attention(Q,K,V) = Softmax(QK^T/√d + B)V

2. **Timestep Embedding** - Paper Section 3.2:
   - Sinusoidal position embeddings for timesteps
   - MLP with Swish activation for projection

3. **Diffusion Process** - Paper Section 3.1:
   - Training: denoising score matching (Eq. 1)
   - Sampling: SDE/ODE solvers (Eq. 2)
   - Uses `diffusers` DDPMScheduler/DDIMScheduler when available

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
