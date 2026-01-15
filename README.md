# DiffiT: Diffusion Vision Transformers for Image Generation

A PyTorch implementation of DiffiT - a U-Net style diffusion model with Vision Transformer blocks for image generation.

## Features

- **Time-Dependent Multi-Head Self-Attention (TDMHSA)**: Attention mechanism that incorporates diffusion timestep information
- **Vision Transformer Blocks**: Process image patches with time-conditioned attention
- **Class-Conditional Generation**: Support for class labels with Classifier-Free Guidance (CFG)
- **Efficient Sampling**: Uses `diffusers` schedulers for DDPM and DDIM sampling
- **Multi-GPU Training**: Distributed training support with gradient accumulation
- **Mixed Precision**: FP16/BF16 training for faster training

## Project Structure

```
DiffiT-v2/
├── train.py                    # Main training entry point
├── generate.py                 # Image generation script with CFG support
├── dataset_tool.py             # Dataset preparation tool
├── requirements.txt            # Python dependencies
│
├── models/                     # Model architectures
│   ├── diffit.py               # Main DiffiT U-Net model (supports class conditioning)
│   ├── attention.py            # TDMHSA and transformer blocks
│   └── vit.py                  # Vision Transformer components
│
├── diffusion/                  # Diffusion process
│   └── diffusion.py            # DDPM/DDIM with CFG support
│
├── training/                   # Training utilities
│   ├── training_loop.py        # Main training loop
│   └── dataset.py              # Dataset loaders
│
├── metrics/                    # Evaluation metrics
│   ├── metric_main.py          # Main metric interface
│   ├── metric_utils.py         # Utilities for metric computation
│   └── frechet_inception_distance.py  # FID implementation
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
| `--cond` | Enable class-conditional training | False |
| `--label-drop` | Label dropout probability for CFG | 0.1 |
| `--cfg-scale` | CFG scale for snapshot generation | 1.5 |
| `--metrics` | Quality metrics to compute | fid10k_full |
| `--metrics-ticks` | How often to evaluate metrics (ticks) | None (end only) |
| `--fid-samples` | Number of samples for FID | 10000 |
| `--fid-steps` | DDIM steps for FID sampling | 50 |

## Generation

Generate images using a trained model.

### Unconditional Generation

```bash
# Generate 64 images with DDIM (fast, ~50 steps)
python generate.py \
    --network=training-runs/00000-diffit/best_model.pkl \
    --outdir=./generated \
    --seeds=0-63

# Generate as a grid
python generate.py \
    --network=model.pkl \
    --outdir=./generated \
    --seeds=0-63 \
    --grid
```

### Conditional Generation (Class-Conditional with CFG)

For models trained with `--cond`, you can generate images conditioned on class labels:

```bash
# Generate images of a specific class (e.g., class 207 = "golden retriever" in ImageNet)
python generate.py \
    --network=model.pkl \
    --outdir=./generated \
    --class=207 \
    --cfg-scale=2.0 \
    --seeds=0-15

# Generate a grid showing all classes
python generate.py \
    --network=model.pkl \
    --outdir=./generated \
    --class-grid \
    --cfg-scale=1.5

# Generate multiple samples per class
python generate.py \
    --network=model.pkl \
    --outdir=./generated \
    --class-grid \
    --samples-per-class=4 \
    --cfg-scale=1.5
```

### CFG Scale Guidelines

The CFG (Classifier-Free Guidance) scale controls the trade-off between sample quality and diversity:

| CFG Scale | Effect |
|-----------|--------|
| 1.0 | No guidance (unconditional) |
| 1.5 | Mild guidance (recommended starting point) |
| 2.0-3.0 | Strong guidance (higher quality, less diversity) |
| 4.0+ | Very strong guidance (may cause artifacts) |

### Generation Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--network` | Path to model pickle | Required |
| `--outdir` | Output directory | Required |
| `--seeds` | Random seeds (e.g., 0-31) | 0-15 |
| `--steps` | DDIM sampling steps | 50 |
| `--class` | Class label for conditional generation | None |
| `--cfg-scale` | Classifier-free guidance scale | 1.5 |
| `--class-grid` | Generate grid with all classes | False |
| `--samples-per-class` | Samples per class in grid mode | 1 |
| `--grid` | Save as grid instead of individual files | False |

## Metrics and Monitoring

### TensorBoard

Training automatically logs to TensorBoard:

```bash
tensorboard --logdir=./training-runs
```

Logged metrics include:
- **Loss/train**: Training loss (MSE between predicted and actual noise)
- **Loss/grad_norm**: Gradient norm (for monitoring training stability)
- **Progress/kimg**: Training progress in thousands of images
- **Metrics/fid***: FID scores when evaluated
- **Metrics/best_fid**: Best FID achieved so far
- **Timing/***: Training speed statistics
- **Resources/***: CPU/GPU memory usage

### FID Evaluation

FID (Frechet Inception Distance) measures the quality of generated images:

```bash
# Evaluate FID every 10 ticks during training
python train.py --outdir=./runs --data=./data --batch-gpu=32 \
    --metrics=fid10k_full --metrics-ticks=10

# Use faster FID variants for validation
python train.py ... --metrics=fid5k  # 5k samples (faster)
python train.py ... --metrics=fid2k  # 2k samples (very fast)
```

Available metrics:
- `fid50k_full`: 50k samples vs full dataset (paper standard)
- `fid10k_full`: 10k samples vs full dataset (default)
- `fid5k`: 5k samples (fast validation)
- `fid2k`: 2k samples (quick check)
- `fid1k`: 1k samples (very quick)

The best model (lowest FID) is automatically saved to `best_model.pkl`.

## Class-Conditional Training

DiffiT supports class-conditional image generation using the approach from the latent DiffiT paper.

### How It Works

1. **Label Embedding**: Class labels are embedded using a learnable embedding table (similar to word embeddings)
2. **Conditioning**: Label embeddings are added to time embeddings and used throughout the network
3. **Classifier-Free Guidance (CFG)**: During training, labels are randomly dropped with probability `--label-drop` (default 0.1), enabling CFG at inference time

### Training

```bash
# Train class-conditional model on ImageNet
torchrun --nproc_per_node=4 train.py \
    --outdir=./runs \
    --data=./imagenet64.zip \
    --batch-gpu=32 \
    --resolution=64 \
    --cond \
    --label-drop=0.1 \
    --cfg-scale=1.5
```

### Inference with CFG

During generation, CFG combines conditional and unconditional predictions:

```
ε̂ = ε_uncond + cfg_scale × (ε_cond - ε_uncond)
```

Higher CFG scale values produce images that better match the class but with less diversity.

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

3. **Class Conditioning** (based on latent DiffiT approach):
   - Learnable embedding table for class labels
   - Combined with time embedding: cond_emb = time_emb + label_emb
   - Null embedding for unconditional generation (CFG support)
   - Random label dropout during training (default 10%)

4. **Diffusion Process** - Paper Section 3.1:
   - Training: denoising score matching (Eq. 1)
   - Sampling: SDE/ODE solvers (Eq. 2)
   - Classifier-free guidance for conditional generation
   - Uses `diffusers` DDPMScheduler/DDIMScheduler when available

## Dataset Format

DiffiT supports datasets in the StyleGAN/SAN format, which can be created using `dataset_tool.py`.

### Preparing a Dataset

Use `dataset_tool.py` to convert your images to the required format:

```bash
# From a folder of images (creates a ZIP archive)
python dataset_tool.py --source=/path/to/images --dest=./dataset.zip \
    --transform=center-crop --resolution=64x64

# From ImageNet or other datasets
python dataset_tool.py --source=/path/to/imagenet/train --dest=./imagenet64.zip \
    --transform=center-crop --resolution=64x64

# Keep as folder (useful for debugging)
python dataset_tool.py --source=/path/to/images --dest=./dataset/ \
    --transform=center-crop --resolution=256x256
```

### Dataset Structure

The tool creates datasets in this format:

```
dataset.zip (or folder)
├── 00000/
│   ├── img00000000.png
│   ├── img00000001.png
│   └── ...
├── 00001/
│   └── ...
└── dataset.json  # Optional class labels
```

The `dataset.json` contains class labels in this format:
```json
{
  "labels": [
    ["00000/img00000000.png", 6],
    ["00000/img00000001.png", 3],
    ...
  ]
}
```

### Training with Your Dataset

```bash
# Unconditional training
torchrun --nproc_per_node=4 train.py \
    --outdir=./runs --data=./dataset.zip --batch-gpu=32 --resolution=64

# Class-conditional training (if dataset has labels)
torchrun --nproc_per_node=4 train.py \
    --outdir=./runs --data=./imagenet64.zip --batch-gpu=32 --resolution=64 --cond
```

### Requirements

- Images must be square (NxN)
- Resolution must be a power of 2 (32, 64, 128, 256, etc.)
- Supports RGB (3-channel) and grayscale (1-channel) images

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
