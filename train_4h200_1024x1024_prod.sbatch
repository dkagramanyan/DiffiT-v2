#!/bin/bash
#SBATCH --job-name=DiffiT-1024
#SBATCH --partition=rocky
#SBATCH --account=proj_1793
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --time=4-0:0
#SBATCH --output=/home/dgkagramanyan/DiffiT-v2/logs/slurm_%j.out
#SBATCH --error=/home/dgkagramanyan/DiffiT-v2/logs/slurm_%j.out
#SBATCH --constraint="type_h|type_f"
#SBATCH --gres-flags=enforce-binding


# ── Environment ────────────────────────────────────────────────────────────────
module purge
module load Python/Anaconda
module load CUDA/13.1

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++


echo "SLURM allocated CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv


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

# ── Auto-detect GPU VRAM and set batch-gpu ─────────────────────────────────────
# At 1024² the latent grid is 128×128 → 4096-token attention sequence.
# The relative-position bias passed as attn_mask to SDPA disables
# FlashAttention; the memory-efficient backend (PyTorch ≥ 2.2) handles
# it but still uses more VRAM than flash.
#
# Memory budget per GPU (bf16 training, depth=28, hidden=1152):
#   Model + AdamW + buffers ≈ 16 GB fixed
#   Per-sample activations  ≈ 1.0–1.5 GB (mem-efficient SDPA)
#
#   H100  80 GB → ~64 GB free → batch-gpu 16  (global 64)
#   H200 141 GB → ~125 GB free → batch-gpu 32  (global 128)
#
# If the SDPA math-backend is selected instead (PyTorch < 2.2 or
# compile misses the bias pattern), it stores the full N²×H attention
# matrix per block → batch-gpu must drop to 4.  The script detects this
# at runtime: if the first attempt OOMs, re-run with --batch-gpu=4.

GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "Detected GPU VRAM: ${GPU_MEM_MIB} MiB"

if [ "$GPU_MEM_MIB" -ge 100000 ]; then
    # H200 (141 GB)
    BATCH_GPU=32
elif [ "$GPU_MEM_MIB" -ge 70000 ]; then
    # H100 (80 GB)
    BATCH_GPU=16
else
    # Unknown / smaller GPU — play it safe
    BATCH_GPU=4
fi

echo "Selected batch-gpu=$BATCH_GPU (4 GPUs → global batch $((BATCH_GPU * 4)))"

# ── Info ───────────────────────────────────────────────────────────────────────
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPUs       : $CUDA_VISIBLE_DEVICES"
echo "Start time : $(date)"
echo "────────────────────────────────────"

# ── Train ──────────────────────────────────────────────────────────────────────
# cfg=diffit-1024 pulls from BASE_CONFIGS:
#   image_size=1024, lr=1e-4, cfg_scale=1.49, gradient_checkpointing=True,
#   grad_accum_steps=2, lr_warmup_kimg=1000.
#
# With checkpointing + 2-step accumulation the effective batch per step is:
#   H100: 16 × 4 × 2 = 128    H200: 32 × 4 × 2 = 256
python "$PROJECT_DIR/train.py" \
    --outdir="$OUTDIR" \
    --cfg=diffit-1024 \
    --data="$DATASET" \
    --gpus=4 \
    --batch-gpu="$BATCH_GPU" \
    --snap=100

EXIT_CODE=$?

# ── Done ───────────────────────────────────────────────────────────────────────
echo "────────────────────────────────────"
echo "End time   : $(date)"
echo "Exit code  : $EXIT_CODE"

exit $EXIT_CODE
