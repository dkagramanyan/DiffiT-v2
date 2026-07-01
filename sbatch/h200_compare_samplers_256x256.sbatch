#!/bin/bash
#SBATCH --job-name=DiffiT-cmp-256
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-0:0
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
export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR="${HOME}/.cache/torch_extensions"
export CUDA_CACHE_PATH="${HOME}/.cache/cuda_cache"

nvidia-smi

# Sweep every sampler over a range of step counts and plot the combra metrics
# (FID / CMMD / FD-DINOv2 + angle metrics) vs. k. Same --data / model config
# knobs as the training sbatch, so the comparison scores the trained model on
# the same dataset it was trained on.
diffit-compare-samplers \
        --model-path=./runs/wc_cv_combra/network-final.pt \
        --data=./datasets/imagenet_9to4_1024x1024_256x256.zip \
        --image-size 256 \
        --cfg-scale 4.4 \
        --num-samples 512 \
        --batch-size 32 \
        --samplers dpm++,unipc,ddim,ddpm \
        --k-values 5,10,20,50,100,250 \
        --outdir=./sampler-comparison/256
