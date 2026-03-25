#!/bin/bash
#SBATCH --job-name=DiffiT-512
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4
#SBATCH --time=4-0:0
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

python train.py --outdir=./training-runs \
        --data=./datasets/imagenet_512x512.zip \
        --image-size 512 \
        --gpus=2 \
        --batch=20 \
        --batch-gpu 10 \
        --kimg 400000 \
        --snap 75 \
        --seed 0 \
        --lr 1e-4
