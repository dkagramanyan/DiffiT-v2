#!/bin/bash
# DiffiT ImageNet-256 Training - Optimized for H200 (4x GPUs, 141GB each)
#
# Target: FID < 2.0 (paper achieves 1.73)
# Hardware: 4x H200 GPUs (141GB VRAM each)

# Dataset - UPDATE THIS PATH
DATA="./datasets/imagenet_9to4_1024x1024_128x128.zip"

# Output directory
OUTDIR="./runs/diffit_imagenet256_h200"

# Hardware setup
NGPUS=4
BATCH_PER_GPU=128  # H200 can handle this with 256x256 images
                    # Total batch: 512 (4 GPUs × 128)

# Model architecture
# Using larger model to leverage H200's memory
RESOLUTION=256
BASE_DIM=128
HIDDEN_DIM=1024     # Large model (paper uses 1152 for ImageNet-256)
NUM_HEADS=16        # More heads for better attention
NUM_BLOCKS=4        # 4 ResBlocks per stage
TIMESTEPS=1000

# Training (paper Section I.2)
LR=3e-4
KIMG=1000000        # 1M kimg = full training

# Class-conditional + CFG
LABEL_DROP=0.1      # 10% label dropout for CFG
CFG_SCALE=4.6       # Paper's best scale for ImageNet-256

# Evaluation with FID-50k
METRICS="fid50k_full"
METRICS_TICKS=100   # Evaluate every 100 ticks (~400k images)
FID_SAMPLES=50000
FID_STEPS=50        # DDIM steps for FID

# Launch
echo "╔════════════════════════════════════════════════╗"
echo "║   DiffiT ImageNet-256 Training (H200 × 4)     ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "Hardware:"
echo "  GPUs: ${NGPUS}x H200 (141GB)"
echo "  Batch per GPU: ${BATCH_PER_GPU}"
echo "  Total batch: $((NGPUS * BATCH_PER_GPU))"
echo ""
echo "Model:"
echo "  Resolution: ${RESOLUTION}×${RESOLUTION}"
echo "  Hidden dim: ${HIDDEN_DIM}"
echo "  Num heads: ${NUM_HEADS}"
echo "  Num blocks: ${NUM_BLOCKS}"
echo "  Parameters: ~500M (estimated)"
echo ""
echo "Training:"
echo "  Total kimg: ${KIMG}"
echo "  Learning rate: ${LR}"
echo "  CFG scale: ${CFG_SCALE}"
echo ""
echo "Target: FID < 2.0 (paper: 1.73)"
echo "════════════════════════════════════════════════"
echo ""

torchrun --nproc_per_node=${NGPUS} train.py \
    --outdir="${OUTDIR}" \
    --data="${DATA}" \
    --batch-gpu=${BATCH_PER_GPU} \
    --resolution=${RESOLUTION} \
    --base-dim=${BASE_DIM} \
    --hidden-dim=${HIDDEN_DIM} \
    --num-heads=${NUM_HEADS} \
    --num-blocks=${NUM_BLOCKS} \
    --timesteps=${TIMESTEPS} \
    --lr=${LR} \
    --kimg=${KIMG} \
    --tick=4 \
    --snap=50 \
    --workers=4 \
    --cond=True \
    --label-drop=${LABEL_DROP} \
    --cfg-scale=${CFG_SCALE} \
    --metrics="${METRICS}" \
    --metrics-ticks=${METRICS_TICKS} \
    --fid-samples=${FID_SAMPLES} \
    --fid-steps=${FID_STEPS}
