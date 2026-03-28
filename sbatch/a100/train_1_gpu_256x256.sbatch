#!/bin/bash
#SBATCH --job-name=DiffiT-256
#SBATCH --partition=gpu-ef-quick
#SBATCH --account=proj_1793
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-00:30:00
#SBATCH --output=/home/dgkagramanyan/DiffiT-v2/logs/slurm_%j.out
#SBATCH --error=/home/dgkagramanyan/DiffiT-v2/logs/slurm_%j.err

# ── Environment ────────────────────────────────────────────────────────────────
module purge
module load Python/Anaconda
module load CUDA/12.4
module load gnu13/13.3

source activate diffit

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR=/home/dgkagramanyan/DiffiT-v2
DATASET=$PROJECT_DIR/datasets/imagenet_9to4_1024x1024_256x256.zip
OUTDIR=$PROJECT_DIR/training-runs
LOGDIR=$PROJECT_DIR/logs

mkdir -p "$OUTDIR" "$LOGDIR"

# ── Info ───────────────────────────────────────────────────────────────────────
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPUs       : $CUDA_VISIBLE_DEVICES"
echo "Start time : $(date)"
echo "────────────────────────────────────"

# ── Train ──────────────────────────────────────────────────────────────────────
python "$PROJECT_DIR/train.py" \
    --outdir="$OUTDIR" \
    --cfg=diffit-256 \
    --data="$DATASET" \
    --gpus=1 \
    --batch-gpu=96 \
    --amp-dtype bf16 \
    --workers=8 \
    --snap=100

EXIT_CODE=$?

# ── Done ───────────────────────────────────────────────────────────────────────
echo "────────────────────────────────────"
echo "End time   : $(date)"
echo "Exit code  : $EXIT_CODE"

exit $EXIT_CODE