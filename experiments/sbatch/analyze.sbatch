#!/bin/bash
#SBATCH --job-name=DiffiT-analyze
#SBATCH --partition=rocky
#SBATCH --account=proj_1793
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-8:0
#SBATCH --output=/home/dgkagramanyan/DiffiT-v2/experiments/logs/slurm_%j.out
#SBATCH --error=/home/dgkagramanyan/DiffiT-v2/experiments/logs/slurm_%j.out
#SBATCH --constraint="type_f|type_h"

# Pass RUN_A, RUN_B, RESOLUTION as env vars:
#   sbatch --export=ALL,RUN_A=./experiments/runs/256/00000-...,RUN_B=...,RESOLUTION=256 analyze.sbatch

module purge
module load Python/Anaconda
module load CUDA/13.1

source /opt/software/python/miniconda/latest/etc/profile.d/conda.sh
conda activate diffit_rocky

PROJECT_DIR=/home/dgkagramanyan/DiffiT-v2
: "${RESOLUTION:?RESOLUTION env var required (256/512/1024)}"
: "${RUN_A:?RUN_A env var required}"
: "${RUN_B:?RUN_B env var required}"

OUTDIR=$PROJECT_DIR/experiments/analysis/${RESOLUTION}
mkdir -p "$OUTDIR"

# Fewer samples/steps at higher resolution to keep walltime reasonable.
case $RESOLUTION in
    256)  N_SAMPLES=256; N_STEPS=50; BATCH=16 ;;
    512)  N_SAMPLES=128; N_STEPS=50; BATCH=8  ;;
    1024) N_SAMPLES=64;  N_STEPS=50; BATCH=4  ;;
    *)    N_SAMPLES=256; N_STEPS=50; BATCH=16 ;;
esac

python "$PROJECT_DIR/experiments/analyze_sample_split.py" \
    --run-a="$RUN_A" \
    --run-b="$RUN_B" \
    --outdir="$OUTDIR" \
    --num-samples=$N_SAMPLES \
    --num-steps=$N_STEPS \
    --batch-size=$BATCH \
    --title="Biased generalization in DiffiT (${RESOLUTION}²)"
