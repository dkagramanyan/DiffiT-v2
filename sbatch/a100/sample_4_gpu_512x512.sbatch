#!/bin/bash
#SBATCH --job-name=DiffiT-sample-512
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --time=0-12:0
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --account=proj_1631
#SBATCH --constraint="type_e"

module purge
module load Python/Anaconda
module load CUDA/12.4
module load gnu13/13.3

source activate diffit

cd ..

MODEL="./training-runs/CHANGE_ME/network-final.pt"
OUTDIR="./samples/512"

torchrun --nproc_per_node=4 sample.py \
    --model-path $MODEL \
    --outdir $OUTDIR \
    --image-size 512 \
    --cfg-scale 1.49 \
    --num-samples 50000 \
    --batch-size 8 \
    --num-sampling-steps 250 \
    --cfg-cond

echo "Sampling complete at $(date)"
