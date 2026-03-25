#!/bin/bash
#SBATCH --job-name=DiffiT-gen-512
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0-4:0
#SBATCH --partition=normal
#SBATCH --account=proj_1631
#SBATCH --constraint="type_e"

module purge
module load Python/Anaconda
module load CUDA/12.4
module load gnu13/13.3

source activate diffit

cd ..

python gen_images.py \
  --model-path ./training-runs/CHANGE_ME/network-final.pt \
  --seeds 0-49 \
  --outdir ./generated/512 \
  --image-size 512 \
  --cfg-scale 1.49 \
  --num-sampling-steps 250 \
  --use-ddim
