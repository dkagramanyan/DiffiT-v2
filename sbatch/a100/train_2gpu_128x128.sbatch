#!/bin/bash
#SBATCH --job-name=DiffiT-128
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --time=5-0:0
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --account=proj_1631
#SBATCH --constraint="type_e"

module purge
module load Python/Anaconda
module load CUDA/12.4
module load gnu13/13.3

source activate san

nvidia-smi
gcc --version
conda list | grep -E "torch|cuda|cudnn"

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

export TORCH_EXTENSIONS_DIR="${HOME}/.cache/torch_extensions"
export CUDA_CACHE_PATH="${HOME}/.cache/cuda_cache"

# DiffiT training with torchrun (recommended for multi-GPU)
torchrun --nproc_per_node=2 train.py \
    --outdir=./runs/diffit_a100 \
    --data=./datasets/imagenet128.zip \
    --batch-gpu=32 \
    --resolution=128 \
    --base-dim=128 \
    --hidden-dim=64 \
    --num-heads=8 \
    --num-blocks=2 \
    --timesteps=1000 \
    --lr=1e-4 \
    --kimg=50000 \
    --tick=4 \
    --snap=50 \
    --workers=4 \
    --cond=True \
    --label-drop=0.1 \
    --cfg-scale=1.5 \
    --metrics=fid10k_full \
    --metrics-ticks=50 \
    --fid-samples=10000 \
    --fid-steps=50
