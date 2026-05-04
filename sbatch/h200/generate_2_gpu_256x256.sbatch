#!/bin/bash
#SBATCH --job-name=DiffiT-gen-00017
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4
#SBATCH --time=0-4:0
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

which nvcc
nvcc --version

which gcc
gcc --version

which g++
g++ --version

conda list | grep -E "torch|cuda|cudnn|ninja"

export TORCH_CUDA_ARCH_LIST="9.0"

python gen_images.py \
  --model-path ./training-runs/00017-diffit-256-gpus2-batch192/network-snapshot-021772.pt \
  --seeds 0-49 \
  --outdir ./generated/00017_snap021772_256 \
  --image-size 256 \
  --cfg-scale 4.4 \
  --num-sampling-steps 250 \
  --use-ddim \
  --gpus 0,1
