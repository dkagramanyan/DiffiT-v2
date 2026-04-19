#!/usr/bin/env bash
# Local (no-SLURM) version of sbatch/train_1024_splitB.sbatch.
# See train_1024_splitA.sh for env var knobs.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${NPROC:=1}"
: "${BATCH_GPU:=2}"
: "${GRAD_ACCUM:=2}"
: "${KIMG:=500}"
: "${SNAP:=5}"
: "${DATASET:=${PROJECT_DIR}/datasets/imagenet_9to4_1024x1024_1024x1024.zip}"
: "${OUTDIR:=${PROJECT_DIR}/experiments/runs/1024}"
: "${CONDA_ENV:=diffit}"
: "${GRAD_CKPT:=1}"
: "${FOREGROUND:=0}"
: "${LOG_DIR:=${PROJECT_DIR}/experiments/logs}"
SCRIPT_NAME=$(basename "${BASH_SOURCE[0]}" .sh)
: "${LOG_FILE:=${LOG_DIR}/${SCRIPT_NAME}_$(date +%Y%m%d_%H%M%S).log}"

echo "──────────────────────────────────────────"
echo "  PROJECT_DIR : $PROJECT_DIR"
echo "  SPLIT       : B"
echo "  IMAGE_SIZE  : 1024"
echo "  NPROC       : $NPROC"
echo "  BATCH_GPU   : $BATCH_GPU"
echo "  GRAD_ACCUM  : $GRAD_ACCUM"
echo "  GRAD_CKPT   : $GRAD_CKPT"
echo "  KIMG        : $KIMG"
echo "  SNAP        : $SNAP"
echo "  DATASET     : $DATASET"
echo "  OUTDIR      : $OUTDIR"
echo "  LOG_FILE    : $LOG_FILE"
echo "──────────────────────────────────────────"

mkdir -p "$OUTDIR" "$LOG_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -n "${CONDA_ENV}" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

CKPT_FLAG="--no-grad-ckpt"
[[ "$GRAD_CKPT" == "1" ]] && CKPT_FLAG="--grad-ckpt"

CMD=(
    torchrun --standalone --nproc_per_node="$NPROC"
    "${PROJECT_DIR}/experiments/train_sample_split.py"
    --outdir="$OUTDIR"
    --data="$DATASET"
    --image-size=1024
    --split=B
    --batch-gpu="$BATCH_GPU"
    --grad-accum="$GRAD_ACCUM"
    "$CKPT_FLAG"
    --kimg="$KIMG"
    --snap="$SNAP"
)

if [[ "$FOREGROUND" == "1" ]]; then
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
else
    nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
    PID=$!
    disown
    echo "Started in background. PID: $PID"
    echo "Monitor: tail -f $LOG_FILE"
    echo "Kill:    kill $PID   # or: pkill -P $PID"
fi
