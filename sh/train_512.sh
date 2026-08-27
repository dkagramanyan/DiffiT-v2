#!/usr/bin/env bash
# DiffiT -- train at 512x512.
#
# Workstation:  bash sh/train_512.sh    (DATA=<zip> GPUS=<n> ... to override)
# SLURM:        sbatch --account=<proj> --partition=<part> --nodes=1 --gpus=2 --cpus-per-task=8 --time=3-0:0 sh/train_512.sh
#
# Defaults target the production allocation: 2x H200 (sm_90), 8 CPUs, fixed seed 42.
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
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"      # CLIP (CMMD) weights
export TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"      # torch.hub DINOv2 + Inception weights

# GPUs / CPUs: 2x H200 and 8 CPUs. SLURM sets CUDA_VISIBLE_DEVICES itself; the default
# only applies on a workstation. 8 CPUs / 2 ranks -> 4 threads per rank, 3 loader
# workers per rank (WORKERS) so the two main processes keep a core each.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

# Determinism / logging: PYTHONHASHSEED pins Python hashing alongside --seed; NCCL
# surfaces a dead rank as an error instead of a hang; Python output is unbuffered so
# the SLURM log follows the run.
export PYTHONHASHSEED=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1

# --- One console-command call ------------------------------------------------
# Higher-resolution stages: set INIT_WEIGHTS to the previous resolution's newest
# diffit-snapshot-<kimg>-inference.pt for a weights-only warm start (fresh optimizer).
INIT_ARGS=()
if [[ -n "${INIT_WEIGHTS:-}" ]]; then
    INIT_ARGS=(--init-weights "$INIT_WEIGHTS")
fi

diffit-train \
    --outdir "${OUTDIR:-./training-runs}" \
    --cfg "${CFG:-diffit-512}" \
    --data "${DATA:-./datasets/imagenet_9to4_1024x1024_512x512.zip}" \
    --gpus "${GPUS:-2}" \
    --batch-gpu "${BATCH_GPU:-64}" \
    --mirror False \
    --snapshot-keep-last "${KEEP_LAST:-1}" \
    --combra-metrics True --num-fid-samples "${NUM_FID_SAMPLES:-10000}" \
    --seed "${SEED:-42}" --workers "${WORKERS:-3}" \
    ${INIT_ARGS[@]+"${INIT_ARGS[@]}"} \
    "$@"
