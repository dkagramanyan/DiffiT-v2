#!/bin/bash
#SBATCH --job-name=DiffiT-1024
#SBATCH --partition=rocky
#SBATCH --account=proj_1793
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --time=4-0:0
#SBATCH --output=/home/dgkagramanyan/DiffiT-v2/logs/slurm_%j.out
#SBATCH --error=/home/dgkagramanyan/DiffiT-v2/logs/slurm_%j.out
#SBATCH --constraint="type_h"
#SBATCH --gres-flags=enforce-binding


# ── Environment ────────────────────────────────────────────────────────────────
module purge
module load Python/Anaconda
module load CUDA/13.1

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++


echo "SLURM allocated CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv


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

# 1024² latents push attention sequence length to 4096 → cache extensions
# go through their own workdir to avoid clashes with 256/512 runs on the
# same node.
export TORCH_EXTENSIONS_DIR="${HOME}/.cache/torch_extensions"
export CUDA_CACHE_PATH="${HOME}/.cache/cuda_cache"


source /opt/software/python/miniconda/latest/etc/profile.d/conda.sh
conda activate diffit_rocky

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR=/home/dgkagramanyan/DiffiT-v2
DATASET=$PROJECT_DIR/datasets/imagenet_9to4_1024x1024.zip
OUTDIR=$PROJECT_DIR/training-runs
LOGDIR=$PROJECT_DIR/logs

mkdir -p "$OUTDIR" "$LOGDIR"

# ── Info ───────────────────────────────────────────────────────────────────────
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPUs       : $CUDA_VISIBLE_DEVICES"
echo "Start time : $(date)"
echo "────────────────────────────────────"

# ── Train ──────────────────────────────────────────────────────────────────────
# cfg=diffit-1024 pulls image_size=1024, lr=1e-4, cfg_scale=1.49 (constant)
# from BASE_CONFIGS in train.py. Global batch = 2 GPUs × 8 = 16.
# batch-gpu is intentionally small: at 1024² the latent grid is 128×128
# so each DiffiTBlock sees a 4096-token sequence — roughly 16× the 512²
# activation cost. If an H200 still OOMs, drop to --batch-gpu=4.
python "$PROJECT_DIR/train.py" \
    --outdir="$OUTDIR" \
    --cfg=diffit-1024 \
    --data="$DATASET" \
    --gpus=2 \
    --batch-gpu=8 \
    --snap=100

EXIT_CODE=$?

# ── Done ───────────────────────────────────────────────────────────────────────
echo "────────────────────────────────────"
echo "End time   : $(date)"
echo "Exit code  : $EXIT_CODE"

exit $EXIT_CODE
