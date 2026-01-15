# DiffiT SLURM Job Scripts

This directory contains SLURM batch scripts for training and generating images with DiffiT on different GPU configurations.

## Directory Structure

```
sbatch/
├── h200/          # Scripts for H200 GPUs (4 GPUs)
│   ├── train_4gpu_64x64.sbatch
│   ├── train_4gpu_128x128.sbatch
│   ├── train_4gpu_256x256.sbatch
│   ├── train_4gpu_256x256_uncond.sbatch
│   ├── train_4gpu_512x512.sbatch
│   ├── generate_256x256.sbatch
│   └── generate_specific_class.sbatch
│
└── a100/          # Scripts for A100 GPUs (2 GPUs)
    ├── train_2gpu_64x64.sbatch
    ├── train_2gpu_128x128.sbatch
    ├── train_2gpu_256x256.sbatch
    ├── train_2gpu_512x512.sbatch
    ├── generate_256x256.sbatch
    └── generate_specific_class.sbatch
```

## Training Scripts

### H200 (4 GPUs)

Training scripts for different resolutions using 4 H200 GPUs:

```bash
# 64x64 - Fast training, small images
sbatch sbatch/h200/train_4gpu_64x64.sbatch

# 128x128 - Balanced quality/speed
sbatch sbatch/h200/train_4gpu_128x128.sbatch

# 256x256 - High quality (recommended)
sbatch sbatch/h200/train_4gpu_256x256.sbatch

# 256x256 - Unconditional (no class labels)
sbatch sbatch/h200/train_4gpu_256x256_uncond.sbatch

# 512x512 - Very high quality, memory intensive
sbatch sbatch/h200/train_4gpu_512x512.sbatch
```

### A100 (2 GPUs)

Training scripts for 2 A100 GPUs:

```bash
# Use similar commands but with a100 directory
sbatch sbatch/a100/train_2gpu_256x256.sbatch
```

## Generation Scripts

### Generate Grid of All Classes

Generate samples for all classes (requires conditional model):

```bash
# H200
sbatch sbatch/h200/generate_256x256.sbatch

# A100
sbatch sbatch/a100/generate_256x256.sbatch
```

### Generate Specific Class

Generate many samples from a specific class:

```bash
# H200
sbatch sbatch/h200/generate_specific_class.sbatch

# A100
sbatch sbatch/a100/generate_specific_class.sbatch
```

Edit the `--class` parameter in the script to change which class to generate.

## Customization

### Before Submitting

1. **Check dataset path**: Update `--data` to point to your dataset
2. **Adjust batch size**: Modify `--batch-gpu` based on GPU memory
3. **Change output directory**: Update `--outdir` for your runs
4. **Set SLURM parameters**: Update account, partition, reservation, nodelist

### Key Parameters

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `--batch-gpu` | Batch size per GPU | 64 (64x64), 32 (128x128), 16 (256x256), 8 (512x512) |
| `--resolution` | Image resolution | 64, 128, 256, 512 |
| `--base-dim` | Base channel dimension | 128-192 |
| `--hidden-dim` | ViT hidden dimension | 64-128 |
| `--num-heads` | Attention heads | 4-8 |
| `--kimg` | Training duration (thousands of images) | 50000 |
| `--cfg-scale` | CFG scale for conditional generation | 1.5-2.0 |

### SLURM Parameters

| Parameter | Description |
|-----------|-------------|
| `--job-name` | Job name in queue |
| `--gpus` | Number of GPUs |
| `--cpus-per-task` | CPU cores per GPU |
| `--time` | Maximum runtime (D-HH:MM) |
| `--partition` | SLURM partition |
| `--account` | SLURM account |
| `--nodelist` | Specific node (optional) |

## Monitoring

### Check Job Status

```bash
squeue -u $USER
```

### View Job Output

```bash
# Real-time monitoring
tail -f slurm-JOBID.out

# TensorBoard monitoring
tensorboard --logdir=./runs
```

### Cancel Job

```bash
scancel JOBID
```

## Training Tips

1. **Start small**: Begin with 64x64 or 128x128 to verify everything works
2. **Monitor memory**: Watch GPU memory usage and adjust batch size if needed
3. **Check snapshots**: Generated samples are saved periodically
4. **Use TensorBoard**: Monitor training curves and FID scores
5. **Conditional vs Unconditional**:
   - Conditional: Use `--cond` flag, requires `dataset.json` with labels
   - Unconditional: Omit `--cond` flag, works with any image folder

## Requirements

- Conda environment with PyTorch, diffusers, tensorboard
- Dataset in ZIP or folder format (use `dataset_tool.py` to prepare)
- For conditional training: `dataset.json` with class labels

## Troubleshooting

### Out of Memory

Reduce `--batch-gpu`:
- 64x64: try 32 or 48
- 128x128: try 16 or 24
- 256x256: try 8 or 12
- 512x512: try 4 or 6

### Slow Training

- Increase `--workers` for faster data loading
- Ensure dataset is on fast storage (not network drive)
- Use `--nobench=False` to enable cuDNN benchmarking

### Poor Quality

- Increase `--kimg` (training duration)
- Try different `--cfg-scale` values (1.5-3.0)
- Adjust learning rate `--lr` (try 5e-5 or 2e-4)
- Increase model capacity (`--base-dim`, `--hidden-dim`, `--num-heads`)

## Examples

### Quick Test Run

```bash
# Small 64x64 model, short training
torchrun --nproc_per_node=2 train.py \
    --outdir=./test_run \
    --data=./datasets/test.zip \
    --batch-gpu=32 \
    --resolution=64 \
    --kimg=100
```

### Production Training

```bash
# High-quality 256x256 conditional model
sbatch sbatch/h200/train_4gpu_256x256.sbatch
```

### Generate Samples

```bash
# After training, generate class grid
python generate.py \
    --network=./runs/diffit_h200/best_model.pkl \
    --outdir=./samples \
    --class-grid \
    --cfg-scale=2.0
```
