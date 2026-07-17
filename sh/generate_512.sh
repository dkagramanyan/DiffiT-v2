#!/usr/bin/env bash
# Generate DiffiT samples at 512x512 into the RankH5Writer HDF5 layout the
# wc_cv angle pipeline consumes. Runs on a workstation or under sbatch (SLURM
# specifics supplied at submission time).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [ "$REPO_ROOT" != "/" ] && [ ! -f "$REPO_ROOT/pyproject.toml" ]; do
  REPO_ROOT="$(dirname "$REPO_ROOT")"
done
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-diffit}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

diffit-gen-images \
  --network "${NETWORK:?set NETWORK=/path/to/diffit-snapshot-*-inference.pt}" \
  --outdir "${OUTDIR:-./generated}" \
  --image-size 512 \
  --samples-per-class "${SAMPLES_PER_CLASS:-1000}" \
  --batch-gpu "${BATCH_GPU:-32}" \
  --gpus "${GPUS:-2}" \
  --save-mode hdf5 \
  "$@"
