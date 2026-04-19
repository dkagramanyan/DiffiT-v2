#!/bin/bash
#SBATCH --job-name=DiffiT-256-B
#SBATCH --partition=rocky
#SBATCH --account=proj_1793
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=2-0:0
#SBATCH --output=/home/dgkagramanyan/DiffiT-v2/experiments/logs/slurm_%j.out
#SBATCH --error=/home/dgkagramanyan/DiffiT-v2/experiments/logs/slurm_%j.out
#SBATCH --constraint="type_f|type_h"
#SBATCH --gres-flags=enforce-binding

module purge
module load Python/Anaconda
module load CUDA/13.1

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export TORCH_CUDA_ARCH_LIST="9.0"

conda activate diffit

PROJECT_DIR=/home/dgkagramanyan/DiffiT-v2
DATASET=$PROJECT_DIR/datasets/imagenet_9to4_1024x1024_256x256.zip
OUTDIR=$PROJECT_DIR/experiments/runs/256

mkdir -p "$OUTDIR" "$PROJECT_DIR/experiments/logs"

echo "Job ID     : $SLURM_JOB_ID"
echo "Split      : B"
echo "Start time : $(date)"
echo "────────────────────────────────────"

torchrun --standalone --nproc_per_node=2 \
    "$PROJECT_DIR/experiments/train_sample_split.py" \
    --outdir="$OUTDIR" \
    --data="$DATASET" \
    --image-size=256 \
    --split=B \
    --batch-gpu=96 \
    --kimg=30000 \
    --snap=10

EXIT_CODE=$?
echo "────────────────────────────────────"
echo "End time   : $(date)"
echo "Exit code  : $EXIT_CODE"
exit $EXIT_CODE
