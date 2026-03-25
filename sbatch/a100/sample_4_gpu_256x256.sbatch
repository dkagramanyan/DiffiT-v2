#!/bin/bash
#SBATCH --job-name=DiffiT-sample-256
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --time=0-6:0
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
echo "Running FID evaluation..."

python evaluator.py \
    --ref-batch ./VIRTUAL_imagenet256_labeled.npz \
    --sample-batch ${OUTDIR}/samples_50000x256x256x3.npz

echo "All tasks completed at $(date)"
