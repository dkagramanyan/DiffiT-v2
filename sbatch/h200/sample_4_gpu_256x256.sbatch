#!/bin/bash
#SBATCH --job-name=DiffiT-sample-256
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --time=0-12:0
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --account=proj_1661
#SBATCH --reservation=rocky
#SBATCH --nodelist=cn-049
#SBATCH --gres-flags=enforce-binding

module purge
module load Python
module load CUDA/12.9

conda activate diffit_rocky_49

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

nvidia-smi
conda list | grep -E "torch|cuda|cudnn|ninja"

export TORCH_CUDA_ARCH_LIST="9.0"

MODEL="./runs/CHANGE_ME/network-final.pt"
OUTDIR="./samples/256"

torchrun --nproc_per_node=4 sample.py \
    --model-path $MODEL \
    --outdir $OUTDIR \
    --image-size 256 \
    --cfg-scale 4.4 \
    --num-samples 50000 \
    --batch-size 16 \
    --num-sampling-steps 250 \
    --cfg-cond

echo "Sampling complete at $(date)"
