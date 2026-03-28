#!/bin/bash
#SBATCH --job-name=DiffiT-256
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
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
        --cfg=diffit-256 \
        --data=./datasets/imagenet_256x256.zip \
        --gpus=2 \
        --batch-gpu 48 \
        --amp-dtype bf16 \
        --workers 8 \
        --snap 100
