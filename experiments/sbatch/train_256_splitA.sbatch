#!/bin/bash
#SBATCH --job-name=DiffiT-256-A
#SBATCH --partition=rocky
#SBATCH --account=proj_1793
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --time=2-0:0
#SBATCH --output=/home/dgkagramanyan/DiffiT-v2/experiments/logs/slurm_%j.out
#SBATCH --error=/home/dgkagramanyan/DiffiT-v2/experiments/logs/slurm_%j.out
#SBATCH --constraint="type_f|type_h"
#SBATCH --gres-flags=enforce-binding
#SBATCH --gpu-bind=closest

# ── Environment ────────────────────────────────────────────────────────────
module purge
module load Python/Anaconda
module load CUDA/13.1

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export TORCH_CUDA_ARCH_LIST="9.0"

source /opt/software/python/miniconda/latest/etc/profile.d/conda.sh
conda activate diffit_rocky

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR=/home/dgkagramanyan/DiffiT-v2
DATASET=$PROJECT_DIR/datasets/imagenet_9to4_1024x1024_256x256.zip
OUTDIR=$PROJECT_DIR/experiments/runs/256

mkdir -p "$OUTDIR" "$PROJECT_DIR/experiments/logs"

echo "Job ID     : $SLURM_JOB_ID"
echo "Split      : A"
echo "Start time : $(date)"
echo "────────────────────────────────────"

# ── Train ──────────────────────────────────────────────────────────────────
# 2 GPUs × batch-gpu 96 = global batch 192.
# val=0.10, trainA=0.45, trainB=0.45. Split seed is fixed inside the script.
# Dense snapshots early: --snap=10 means a checkpoint + test-loss eval
# every 10 × 4 kimg = 40 kimg.
torchrun --standalone --nproc_per_node=2 \
    "$PROJECT_DIR/experiments/train_sample_split.py" \
    --outdir="$OUTDIR" \
    --data="$DATASET" \
    --image-size=256 \
    --split=A \
    --seed=42 \
    --batch-gpu=96 \
    --kimg=30000 \
    --snap=100

EXIT_CODE=$?
echo "────────────────────────────────────"
echo "End time   : $(date)"
echo "Exit code  : $EXIT_CODE"
exit $EXIT_CODE
