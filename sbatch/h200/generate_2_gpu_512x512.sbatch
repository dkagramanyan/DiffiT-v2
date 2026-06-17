#!/bin/bash
#SBATCH --job-name=DiffiT-256
#SBATCH --partition=rocky
#SBATCH --account=proj_1793
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --time=2-0:0
#SBATCH --output=/home/dgkagramanyan/DiffiT-v2/logs/slurm_%j.out
#SBATCH --error=/home/dgkagramanyan/DiffiT-v2/logs/slurm_%j.out
#SBATCH --constraint="type_f|type_h"
#SBATCH --gres-flags=enforce-binding


module purge
module load Python
module load CUDA/13.1

conda activate diffit_rocky

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

nvidia-smi

which nvcc
nvcc --version

which gcc
gcc --version

which g++
g++ --version

conda list | grep -E "torch|cuda|cudnn|ninja"

export TORCH_CUDA_ARCH_LIST="9.0"

diffit-gen-images \
  --model-path ./training-runs/00018-diffit-512-gpus4-batch256/network-snapshot-007372.pt \
  --samples-per-class 10000 \
  --batch-size 16 \
  --outdir ./generated/00018-diffit-512-gpus4-batch256 \
  --image-size 512 \
  --cfg-scale 1.49 \
  --num-sampling-steps 250 \
  --use-ddim \
  --gpus 0,1
