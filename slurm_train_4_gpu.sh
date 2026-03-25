#!/bin/bash
#SBATCH --job-name=DiffiT-train
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --time=3-0:0
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --account=proj_1631
#SBATCH --constraint="type_e"

module purge
module load Python/Anaconda
module load CUDA/12.4
module load gnu13/13.3

source activate diffit

nvidia-smi
gcc --version
conda list | grep cuda

python train.py --outdir=./training-runs \
        --cfg=diffit-256 \
        --data=./datasets/imagenet_256x256.zip \
        --gpus=4 \
        --batch-gpu 64 \
        --snap 50
