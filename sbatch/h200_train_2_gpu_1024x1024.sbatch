#!/bin/bash
#SBATCH --job-name=DiffiT-1024
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

conda activate diffit_rocky

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

diffit-train --outdir=./runs/wc_cv_combra \
        --cfg=diffit-1024 \
        --data=./datasets/imagenet_9to4_1024x1024_1024x1024.zip \
        --gpus=2 \
        --batch-gpu 8 \
        --combra-metrics True \
        --save-inference-only True \
        --snapshot-keep-last 3 \
        --snap 100

# Checkpoints written to the run dir:
#   network-snapshot-<kimg>-inference.pt  small G_ema snapshots (newest 3 kept)
#   best_model.pt                         full checkpoint of the lowest-FID tick
# --save-inference-only skips the rolling network-snapshot-latest.pt, so resume
# (and progressive higher-res stages) from best_model.pt or the final network-final.pt:
#   diffit-train ... --resume ./runs/wc_cv_combra/<run>/best_model.pt
