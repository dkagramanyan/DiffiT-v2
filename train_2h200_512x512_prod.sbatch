#!/bin/bash
#SBATCH --job-name=DiffiT-512
#SBATCH --partition=rocky
#SBATCH --account=proj_1793
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --time=4-0:0
#SBATCH --output=/home/dgkagramanyan/DiffiT-v2/logs/slurm_%j.out
#SBATCH --error=/home/dgkagramanyan/DiffiT-v2/logs/slurm_%j.out
#SBATCH --constraint="type_h"
#SBATCH --gres-flags=enforce-binding


# ── Environment ────────────────────────────────────────────────────────────────
module purge
module load Python/Anaconda
module load CUDA/13.1

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++


echo "SLURM allocated CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv


which nvcc
nvcc --version

which gcc
gcc --version

which g++
g++ --version

conda list | grep -E "torch|cuda|cudnn|ninja"

export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export TORCH_CUDA_ARCH_LIST="9.0"


source /opt/software/python/miniconda/latest/etc/profile.d/conda.sh
conda activate diffit_rocky

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR=/home/dgkagramanyan/DiffiT-v2
DATASET=$PROJECT_DIR/datasets/imagenet_9to4_1024x1024_512x512.zip
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
# cfg=diffit-512 pulls image_size=512, lr=1e-4, cfg_scale=1.49 (constant,
# matches the paper's 512 recipe) from BASE_CONFIGS in train.py.
python "$PROJECT_DIR/train.py" \
    --outdir="$OUTDIR" \
    --cfg=diffit-512 \
    --data="$DATASET" \
    --gpus=2 \
    --batch-gpu=64 \
    --snap=100

EXIT_CODE=$?

# ── Done ───────────────────────────────────────────────────────────────────────
echo "────────────────────────────────────"
echo "End time   : $(date)"
echo "Exit code  : $EXIT_CODE"

exit $EXIT_CODE