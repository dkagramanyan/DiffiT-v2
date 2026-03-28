#!/bin/bash
#SBATCH --job-name=DiffiT-256
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-10:0
#SBATCH --nodes=1
#SBATCH --partition=gpu-ef-quick
#SBATCH --account=proj_1793


module purge
module load Python/Anaconda
module load CUDA/12.4
module load gnu13/13.3

source activate diffit

cd ..

python train.py --outdir=./training-runs \
        --cfg=diffit-256 \
        --data=./datasets/imagenet_9to4_1024x1024_256x256.zip \
        --gpus 1 \
        --batch-gpu 48 \
        --snap 100 \
