#!/usr/bin/env bash
# DiffiT -- train at 512x512.
#
# Workstation:  bash sh/train_512.sh               (detaches; output goes to logs/)
#               FOREGROUND=1 bash sh/train_512.sh  (stays attached; output also goes to logs/)
# SLURM:        sbatch --account=<proj> --partition=<part> --nodes=1 --gpus=2 --cpus-per-task=8 --time=3-0:0 sh/train_512.sh
#
# Defaults target the production allocation: 2x H200 (sm_90), 8 CPUs, fixed seed 42.
#
# Every knob is in the run-settings block below; each can be overridden from the
# environment (DATA=<zip> GPUS=<n> bash sh/train_512.sh). Anything after the script name
# is appended to the command (e.g. `... --kimg 200 --snap 2` for a smoke run). No user
# homes, --nodelist or account IDs live here -- SLURM specifics come from the sbatch
# line (spec §9).
set -euo pipefail

# --- Run settings (env overrides) -------------------------------------------
# AUGMENT=True: each training image gets a random dihedral transform (rot90 x hflip);
# the zip holds the 1080 originals, one per crop, so an epoch is 1080 images.
# AUGMENT=False trains on the originals as stored.
# A run keeps the KEEP_LAST newest snapshots plus the best by combra_fid /
# combra_fd_dinov2 / combra_cmmd (KEEP_LAST=0 keeps every one).
# INIT_WEIGHTS (empty = from scratch): a previous stage's best-by-combra_fid snapshot --
# the file the last "Best snapshots:" line of its .log names for combra_fid (the last
# snapshot if that run had no eval) -- for a weights-only warm start (fresh optimizer).
# A warm start also gets a LR_WARMUP-kimg linear LR warmup (default 1000, as the
# diffit-1024 preset); the diffit-512 preset itself has none, so a from-scratch 512 run
# keeps the paper recipe (LR 1e-4, constant).
# CUDA_VISIBLE_DEVICES: SLURM sets it itself; the default only applies on a workstation.
# 8 CPUs / 2 ranks -> 4 threads per rank (OMP/MKL), 3 loader workers per rank (WORKERS)
# so the two main processes keep a core each.
CONDA_ENV="${CONDA_ENV:-diffit-v2}"   # env name = repo name
OUTDIR="${OUTDIR:-./training-runs}"
CFG="${CFG:-diffit-512}"
DATA="${DATA:-./datasets/imagenet_9to4_1024x1024_512x512.zip}"
GPUS="${GPUS:-2}"
BATCH_GPU="${BATCH_GPU:-64}"
KEEP_LAST="${KEEP_LAST:-1}"
AUGMENT="${AUGMENT:-True}"
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-10000}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-3}"
INIT_WEIGHTS="${INIT_WEIGHTS:-}"
LR_WARMUP="${LR_WARMUP:-1000}"   # kimg; used only with INIT_WEIGHTS
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"   # CLIP (CMMD) weights
TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"   # torch.hub DINOv2 + Inception weights
RUN_VARS=(CONDA_ENV OUTDIR CFG DATA GPUS BATCH_GPU KEEP_LAST AUGMENT NUM_FID_SAMPLES SEED
          WORKERS INIT_WEIGHTS LR_WARMUP CUDA_VISIBLE_DEVICES OMP_NUM_THREADS MKL_NUM_THREADS
          NCCL_DEBUG HF_HOME TORCH_HOME)

# --- Environment -------------------------------------------------------------
# Repo root: under SLURM the script runs from a spool copy, so walk up from the submit
# dir there and from this file's own location on a workstation.
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
while [[ ! -f "$REPO_DIR/pyproject.toml" && "$REPO_DIR" != / ]]; do REPO_DIR="$(dirname "$REPO_DIR")"; done
[[ -f "$REPO_DIR/pyproject.toml" ]] || { echo "cannot find the repo root -- submit from inside the repo" >&2; exit 1; }
cd "$REPO_DIR"

# --- Launch: detach and log ----------------------------------------------------
# On a workstation the script re-launches itself in its own session (setsid nohup) and
# returns at once: the run survives closing the terminal, and everything it prints
# goes to logs/<name>-<date>.log (with a .pid file beside it). FOREGROUND=1 keeps it
# attached; the output is still copied to the log. Under SLURM it never detaches (the
# job already runs unattended); the output goes both to the slurm .out and to the log.
RUN_NAME=diffit-train_512
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs}"
if [[ -z "${RUN_LOG:-}" ]]; then
  mkdir -p "$LOG_DIR"
  export RUN_LOG="$LOG_DIR/$RUN_NAME-$(date +%Y%m%d-%H%M%S).log"
  if [[ -z "${SLURM_JOB_ID:-}" && "${FOREGROUND:-0}" != 1 ]]; then
    setsid nohup bash "${BASH_SOURCE[0]}" "$@" > "$RUN_LOG" 2>&1 < /dev/null &
    pid=$!
    echo "$pid" > "${RUN_LOG%.log}.pid"
    echo "Started $RUN_NAME in the background (pid $pid, its own process group)."
    echo "  log:    $RUN_LOG"
    echo "  follow: tail -f $RUN_LOG"
    echo "  stop:   kill -- -$pid"
    exit 0
  fi
  exec > >(tee -a "$RUN_LOG") 2>&1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
# Pure PyTorch: no custom CUDA ops, so no toolkit or arch list is needed.
# Offline-cluster contract: backbones are prefetched once on a login node
# (diffit-download-models); compute nodes never reach the network.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME TORCH_HOME

# GPUs / CPUs (values in the run-settings block).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES OMP_NUM_THREADS MKL_NUM_THREADS

# Determinism / logging: PYTHONHASHSEED pins Python hashing alongside --seed; NCCL
# surfaces a dead rank as an error instead of a hang; Python output is unbuffered so
# the log follows the run.
export PYTHONHASHSEED=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG
export PYTHONUNBUFFERED=1

# --- One console-command call ------------------------------------------------
INIT_ARGS=()
if [[ -n "$INIT_WEIGHTS" ]]; then
    INIT_ARGS=(--init-weights "$INIT_WEIGHTS" --lr-warmup "$LR_WARMUP")
fi

CMD=(diffit-train
    --outdir "$OUTDIR"
    --cfg "$CFG"
    --data "$DATA"
    --gpus "$GPUS"
    --batch-gpu "$BATCH_GPU"
    --snapshot-keep-last "$KEEP_LAST"
    --augment "$AUGMENT"
    --combra-metrics True --num-fid-samples "$NUM_FID_SAMPLES"
    --seed "$SEED" --workers "$WORKERS"
    ${INIT_ARGS[@]+"${INIT_ARGS[@]}"}
    "$@")

# --- Run settings, into the log -------------------------------------------------
COMMIT="$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [[ "$COMMIT" != unknown && -n "$(git -C "$REPO_DIR" status --porcelain 2>/dev/null)" ]]; then
    COMMIT="$COMMIT-dirty"
fi
echo "Run settings:"
for v in "${RUN_VARS[@]}"; do echo "  $v=${!v}"; done   # includes CUDA_VISIBLE_DEVICES
echo "  commit=$COMMIT"
echo "  host=$(hostname)"
echo "  date=$(date '+%Y-%m-%d %H:%M:%S %z')"
printf -v CMDLINE '%q ' "${CMD[@]}"
echo "  command=$CMDLINE"

"${CMD[@]}"
