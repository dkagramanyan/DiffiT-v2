#!/usr/bin/env bash
# Local analog of sbatch/analyze.sbatch.
#
# Required env vars (or positional args):
#   RESOLUTION  (256 | 512 | 1024)
#   RUN_A       path to split-A run directory
#   RUN_B       path to split-B run directory
#
# Usage:
#   RESOLUTION=256 RUN_A=./experiments/runs/256/00000-... RUN_B=./experiments/runs/256/00001-... \
#       bash experiments/shell/analyze.sh
#
# Or positional:
#   bash experiments/shell/analyze.sh 256 ./runs/256/00000-... ./runs/256/00001-...
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)

# Positional args override env vars.
RESOLUTION="${1:-${RESOLUTION:-}}"
RUN_A="${2:-${RUN_A:-}}"
RUN_B="${3:-${RUN_B:-}}"

if [[ -z "$RESOLUTION" || -z "$RUN_A" || -z "$RUN_B" ]]; then
    echo "Usage: $0 RESOLUTION RUN_A RUN_B   (or set env vars)" >&2
    exit 1
fi

: "${OUTDIR:=${PROJECT_DIR}/experiments/analysis/${RESOLUTION}}"
: "${CONDA_ENV:=diffit}"

# Resolution-dependent sampling budget.
case "$RESOLUTION" in
    256)  : "${NUM_SAMPLES:=256}"; : "${BATCH_SIZE:=16}"; : "${NUM_STEPS:=50}" ;;
    512)  : "${NUM_SAMPLES:=128}"; : "${BATCH_SIZE:=8}";  : "${NUM_STEPS:=50}" ;;
    1024) : "${NUM_SAMPLES:=64}";  : "${BATCH_SIZE:=4}";  : "${NUM_STEPS:=50}" ;;
    *)    : "${NUM_SAMPLES:=256}"; : "${BATCH_SIZE:=16}"; : "${NUM_STEPS:=50}" ;;
esac

echo "──────────────────────────────────────────"
echo "  RESOLUTION  : $RESOLUTION"
echo "  RUN_A       : $RUN_A"
echo "  RUN_B       : $RUN_B"
echo "  OUTDIR      : $OUTDIR"
echo "  NUM_SAMPLES : $NUM_SAMPLES"
echo "  BATCH_SIZE  : $BATCH_SIZE"
echo "  NUM_STEPS   : $NUM_STEPS"
echo "──────────────────────────────────────────"

mkdir -p "$OUTDIR"

if [[ -n "${CONDA_ENV}" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python "${PROJECT_DIR}/experiments/analyze_sample_split.py" \
    --run-a="$RUN_A" \
    --run-b="$RUN_B" \
    --outdir="$OUTDIR" \
    --num-samples="$NUM_SAMPLES" \
    --num-steps="$NUM_STEPS" \
    --batch-size="$BATCH_SIZE" \
    --title="Biased generalization in DiffiT (${RESOLUTION}²)"
