#!/usr/bin/env bash
# DiffiT -- generate 512x512 samples per class into the merged <desc>.h5 the wc_cv angle pipeline consumes.
#
# Workstation:  NETWORK=<run_dir>/<snapshot>-inference.pt bash sh/generate_512.sh
# SLURM:        sbatch --account=<proj> --partition=<part> --gpus=2 --cpus-per-task=16 --time=3-0:0 sh/generate_512.sh
#
# Every knob is an env var with a default; anything after the script name is appended to
# the command (e.g. `... --kimg 200 --snap 2` for a smoke run). No user homes, --nodelist
# or account IDs live here -- SLURM specifics come from the sbatch line (spec §9).
set -euo pipefail

# --- Environment -------------------------------------------------------------
# Repo root: under SLURM the script runs from a spool copy, so walk up from the submit
# dir there and from this file's own location on a workstation.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
while [[ ! -f "$REPO_DIR/pyproject.toml" && "$REPO_DIR" != / ]]; do REPO_DIR="$(dirname "$REPO_DIR")"; done
[[ -f "$REPO_DIR/pyproject.toml" ]] || { echo "cannot find the repo root -- submit from inside the repo" >&2; exit 1; }
cd "$REPO_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-diffit-v2}"   # env name = repo name
# Pure PyTorch: no custom CUDA ops, so no toolkit or arch list is needed.
# Offline-cluster contract: backbones are prefetched once on a login node
# (diffit-download-models); compute nodes never reach the network.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- One console-command call ------------------------------------------------
diffit-gen-images \
    --network "${NETWORK:?set NETWORK=<run_dir>/diffit-snapshot-<kimg>-inference.pt}" \
    --desc "${DESC:-$(basename "${NETWORK%.pt}")}" \
    --image-size 512 \
    --outdir "${OUTDIR:-./generated/512}" \
    --classes "${CLASSES:-0,1,2}" \
    --samples-per-class "${SAMPLES_PER_CLASS:-1000}" \
    --gpus "${GPUS:-2}" --batch-gpu "${BATCH_GPU:-32}" \
    --seed "${SEED:-42}" \
    --save-mode hdf5 \
    --sampler "${SAMPLER:-ddim}" --steps "${STEPS:-250}" --cfg-scale "${CFG_SCALE:-1.49}" \
    "$@"
