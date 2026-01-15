#!/bin/bash
#SBATCH --job-name=DiffiT-Gen-256
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-4:0
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --account=proj_1631
#SBATCH --constraint="type_e"

module purge
module load Python/Anaconda
module load CUDA/12.4
module load gnu13/13.3

source activate san

nvidia-smi

export TORCH_EXTENSIONS_DIR="${HOME}/.cache/torch_extensions"
export CUDA_CACHE_PATH="${HOME}/.cache/cuda_cache"

# Generate class-conditional samples with CFG
python generate.py \
    --network=./runs/diffit_a100/best_model.pkl \
    --outdir=./generated/256x256 \
    --class-grid \
    --samples-per-class=10 \
    --cfg-scale=2.0 \
    --steps=50 \
    --batch=32
