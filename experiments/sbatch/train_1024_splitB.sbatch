#!/bin/bash
#SBATCH --job-name=DiffiT-1024-B
#SBATCH --partition=rocky
#SBATCH --account=proj_1793
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --time=4-0:0
#SBATCH --output=/home/dgkagramanyan/DiffiT-v2/experiments/logs/slurm_%j.out
#SBATCH --error=/home/dgkagramanyan/DiffiT-v2/experiments/logs/slurm_%j.out
#SBATCH --constraint="type_h"
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TORCH_EXTENSIONS_DIR="${HOME}/.cache/torch_extensions"
export CUDA_CACHE_PATH="${HOME}/.cache/cuda_cache"

source /opt/software/python/miniconda/latest/etc/profile.d/conda.sh
conda activate diffit_rocky

PROJECT_DIR=/home/dgkagramanyan/DiffiT-v2
DATASET=$PROJECT_DIR/datasets/imagenet_9to4_1024x1024_1024x1024.zip
OUTDIR=$PROJECT_DIR/experiments/runs/1024

mkdir -p "$OUTDIR" "$PROJECT_DIR/experiments/logs"

echo "Job ID     : $SLURM_JOB_ID"
echo "Split      : B"
echo "Start time : $(date)"
echo "────────────────────────────────────"

torchrun --standalone --nproc_per_node=2 \
    "$PROJECT_DIR/experiments/train_sample_split.py" \
    --outdir="$OUTDIR" \
    --data="$DATASET" \
    --image-size=1024 \
    --split=B \
    --batch-gpu=8 \
    --grad-accum=2 \
    --grad-ckpt \
    --kimg=10000 \
    --snap=100

EXIT_CODE=$?
echo "────────────────────────────────────"
echo "End time   : $(date)"
echo "Exit code  : $EXIT_CODE"
exit $EXIT_CODE
