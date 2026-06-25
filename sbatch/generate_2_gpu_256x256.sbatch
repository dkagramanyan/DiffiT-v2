#!/bin/bash
#SBATCH --job-name=DiffiT-gen-256
#SBATCH --partition=rocky
#SBATCH --account=proj_1793
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --time=2-0:0
#SBATCH --gres-flags=enforce-binding

# Self-contained: submittable from anywhere (e.g. the sbatch/ folder). Resolve
# the repo root (the dir containing pyproject.toml) and cd into it so the
# relative --outdir / --model-path paths below resolve consistently.
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
conda list | grep -E "torch|cuda|cudnn|ninja"

# DiffiT is pure PyTorch (no custom CUDA-op compilation), so no system CUDA
# module / CUDA_HOME is required -- the conda env's torch ships its own runtime.

diffit-gen-images \
  --model-path ./training-runs/00017-diffit-256-gpus2-batch192/network-snapshot-021772.pt \
  --samples-per-class 10000 \
  --batch-size 32 \
  --outdir ./generated/00017-diffit-256-gpus2-batch192 \
  --image-size 256 \
  --cfg-scale 4.4 \
  --num-sampling-steps 250 \
  --use-ddim \
  --gpus 0,1
