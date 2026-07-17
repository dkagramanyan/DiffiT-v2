#!/usr/bin/env bash
# Train DiffiT at 1024x1024. Runs unmodified on a workstation
# (`bash sh/train_1024.sh`) or a cluster
# (`sbatch --account=<proj> --partition=<part> --gpus=<N> sh/train_1024.sh`).
# SLURM specifics (account, partition, nodelist, gpu count) are supplied at
# submission time -- never hardcoded here.
set -euo pipefail

# --- Environment ---------------------------------------------------------
# Repo root is self-located (walk up to pyproject.toml), so no hardcoded homes.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
while [ "$REPO_ROOT" != "/" ] && [ ! -f "$REPO_ROOT/pyproject.toml" ]; do
  REPO_ROOT="$(dirname "$REPO_ROOT")"
done
cd "$REPO_ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-diffit}"

# Offline-cluster contract: backbones are prefetched once on a login node via
# `diffit-download-models`; compute nodes never reach the network.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- One console-command call --------------------------------------------
diffit-train \
  --outdir "${OUTDIR:-./training-runs}" \
  --cfg diffit-1024 \
  --data "${DATA:?set DATA=/path/to/1024x1024.zip}" \
  --gpus "${GPUS:-2}" \
  --batch-gpu "${BATCH_GPU:-32}" \
  "$@"
