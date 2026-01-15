#!/bin/bash
#SBATCH --job-name=DiffiT-64
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --time=3-0:0
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --account=proj_1661
#SBATCH --reservation=rocky
#SBATCH --nodelist=cn-049
#SBATCH --gres-flags=enforce-binding

module purge
module load Python
module load CUDA/12.9

conda activate san_rocky_51

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

nvidia-smi

which nvcc
nvcc --version

which gcc
gcc --version

which g++
g++ --version

conda list | grep -E "torch|cuda|cudnn|ninja"

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export TORCH_CUDA_ARCH_LIST="9.0"

export CUDA_VISIBLE_DEVICES=0,1,2,3

export TORCH_EXTENSIONS_DIR="${HOME}/.cache/torch_extensions"
export CUDA_CACHE_PATH="${HOME}/.cache/cuda_cache"

# DiffiT training with torchrun (recommended for multi-GPU)
torchrun --nproc_per_node=4 train.py \
    --outdir=./runs/diffit_h200 \
    --data=./datasets/imagenet_9to4_1024x1024_64x64.zip \
    --batch-gpu=64 \
    --resolution=64 \
    --base-dim=128 \
    --hidden-dim=64 \
    --num-heads=4 \
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
