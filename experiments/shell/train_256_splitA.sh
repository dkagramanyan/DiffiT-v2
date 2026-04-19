#!/usr/bin/env bash
# Local (no-SLURM) version of sbatch/train_256_splitA.sbatch.
#
# Defaults are tuned for a modest local GPU. Override any of these via env vars:
#   NPROC=2            # GPUs to use
#   BATCH_GPU=32       # per-GPU batch size
#   KIMG=1000          # total kimg (small for a smoke test)
#   SNAP=5             # ticks between test-loss evals / checkpoints
#   DATASET=...        # path to dataset zip / folder
#   OUTDIR=...         # where to write runs
#   CONDA_ENV=diffit   # conda env to activate (set to '' to skip)
#   FOREGROUND=1       # keep attached (tee to terminal + log). Default: detach via nohup.
#   LOG_FILE=...       # override log path
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

: "${NPROC:=1}"
: "${BATCH_GPU:=16}"
: "${KIMG:=1000}"
: "${SNAP:=1}"
: "${DATASET:=${PROJECT_DIR}/datasets/imagenet_9to4_1024x1024_256x256.zip}"
: "${OUTDIR:=${PROJECT_DIR}/experiments/runs/256}"
: "${CONDA_ENV:=diffit}"
: "${FOREGROUND:=0}"
: "${LOG_DIR:=${PROJECT_DIR}/experiments/logs}"
SCRIPT_NAME=$(basename "${BASH_SOURCE[0]}" .sh)
: "${LOG_FILE:=${LOG_DIR}/${SCRIPT_NAME}_$(date +%Y%m%d_%H%M%S).log}"

echo "──────────────────────────────────────────"
echo "  PROJECT_DIR : $PROJECT_DIR"
echo "  SPLIT       : A"
echo "  IMAGE_SIZE  : 256"
echo "  NPROC       : $NPROC"
echo "  BATCH_GPU   : $BATCH_GPU"
echo "  KIMG        : $KIMG"
echo "  SNAP        : $SNAP"
echo "  DATASET     : $DATASET"
echo "  OUTDIR      : $OUTDIR"
echo "  LOG_FILE    : $LOG_FILE"
echo "──────────────────────────────────────────"

mkdir -p "$OUTDIR" "$LOG_DIR"

# Optional conda activation — leave CONDA_ENV empty to skip.
if [[ -n "${CONDA_ENV}" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

CMD=(
    torchrun --standalone --nproc_per_node="$NPROC"
    "${PROJECT_DIR}/experiments/train_sample_split.py"
    --outdir="$OUTDIR"
    --data="$DATASET"
    --image-size=256
    --split=A
    --batch-gpu="$BATCH_GPU"
    --kimg="$KIMG"
    --snap="$SNAP"
)

if [[ "$FOREGROUND" == "1" ]]; then
    # Attached: stream to terminal AND tee to log file.
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
else
    # Detached: nohup survives SSH disconnect; redirect all output to log.
    nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
    PID=$!
    disown
    echo "Started in background. PID: $PID"
    echo "Monitor: tail -f $LOG_FILE"
    echo "Kill:    kill $PID   # or: pkill -P $PID"
fi
