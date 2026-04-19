#!/usr/bin/env bash
# Local (no-SLURM) version of sbatch/train_256_splitB.sbatch.
# See train_256_splitA.sh for env var knobs.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${NPROC:=1}"
: "${BATCH_GPU:=32}"
: "${KIMG:=1000}"
: "${SNAP:=5}"
: "${DATASET:=${PROJECT_DIR}/datasets/imagenet_9to4_1024x1024_256x256.zip}"
: "${OUTDIR:=${PROJECT_DIR}/experiments/runs/256}"
: "${CONDA_ENV:=diffit}"

echo "──────────────────────────────────────────"
echo "  PROJECT_DIR : $PROJECT_DIR"
echo "  SPLIT       : B"
echo "  IMAGE_SIZE  : 256"
echo "  NPROC       : $NPROC"
echo "  BATCH_GPU   : $BATCH_GPU"
echo "  KIMG        : $KIMG"
echo "  SNAP        : $SNAP"
echo "  DATASET     : $DATASET"
echo "  OUTDIR      : $OUTDIR"
echo "──────────────────────────────────────────"

mkdir -p "$OUTDIR"

if [[ -n "${CONDA_ENV}" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

torchrun --standalone --nproc_per_node="$NPROC" \
    "${PROJECT_DIR}/experiments/train_sample_split.py" \
    --outdir="$OUTDIR" \
    --data="$DATASET" \
    --image-size=256 \
    --split=B \
    --batch-gpu="$BATCH_GPU" \
    --kimg="$KIMG" \
    --snap="$SNAP"
