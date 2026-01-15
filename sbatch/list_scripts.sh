#!/bin/bash
# Quick reference for available SLURM scripts

echo "=========================================="
echo "DiffiT SLURM Job Scripts"
echo "=========================================="
echo ""

echo "H200 Training Scripts (4 GPUs):"
echo "  sbatch sbatch/h200/train_4gpu_64x64.sbatch"
echo "  sbatch sbatch/h200/train_4gpu_128x128.sbatch"
echo "  sbatch sbatch/h200/train_4gpu_256x256.sbatch"
echo "  sbatch sbatch/h200/train_4gpu_256x256_uncond.sbatch"
echo "  sbatch sbatch/h200/train_4gpu_512x512.sbatch"
echo ""

echo "A100 Training Scripts (2 GPUs):"
echo "  sbatch sbatch/a100/train_2gpu_64x64.sbatch"
echo "  sbatch sbatch/a100/train_2gpu_128x128.sbatch"
echo "  sbatch sbatch/a100/train_2gpu_256x256.sbatch"
echo "  sbatch sbatch/a100/train_2gpu_512x512.sbatch"
echo ""

echo "H200 Generation Scripts:"
echo "  sbatch sbatch/h200/generate_256x256.sbatch"
echo "  sbatch sbatch/h200/generate_specific_class.sbatch"
echo ""

echo "A100 Generation Scripts:"
echo "  sbatch sbatch/a100/generate_256x256.sbatch"
echo "  sbatch sbatch/a100/generate_specific_class.sbatch"
echo ""

echo "=========================================="
echo "Before submitting:"
echo "  1. Update dataset path (--data)"
echo "  2. Adjust batch size (--batch-gpu)"
echo "  3. Set SLURM account/partition"
echo "  4. Choose output directory (--outdir)"
echo "=========================================="
