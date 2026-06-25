#!/bin/bash
#SBATCH --job-name=DiffiT-512
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=3-0:0
#SBATCH --nodes=1
#SBATCH --partition=rocky

# Self-contained: submittable from anywhere (e.g. the sbatch/ folder). Resolve
# the repo root (the dir containing pyproject.toml) and cd into it so the
# relative --data / --outdir paths below resolve consistently.
SCRIPT_DIR="$(cd "$(dirname "${SLURM_JOB_SCRIPT:-${BASH_SOURCE[0]}}")" && pwd)"
REPO_DIR="${SLURM_SUBMIT_DIR:-$SCRIPT_DIR}"
while [[ ! -f "$REPO_DIR/pyproject.toml" && "$REPO_DIR" != / ]]; do
    REPO_DIR="$(dirname "$REPO_DIR")"
done
cd "$REPO_DIR" || exit 1

module purge
module load Python
conda activate diffit_rocky_51

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_EXTENSIONS_DIR="${HOME}/.cache/torch_extensions"
export CUDA_CACHE_PATH="${HOME}/.cache/cuda_cache"

nvidia-smi

# DiffiT is pure PyTorch (no custom CUDA-op compilation), so no system CUDA
# module / CUDA_HOME is required -- the conda env's torch ships its own runtime.

diffit-train --outdir=./runs/diffit_h200_51 \
        --cfg=diffit-512 \
        --data=./datasets/imagenet_512x512.zip \
        --gpus=2 \
        --batch-gpu 25 \
        --combra-metrics True \
        --save-inference-only True \
        --snap 100
