#!/bin/bash
# DiffiT ImageNet-256 Training - Optimized for H200 (4x GPUs, 141GB each)
#
# Target: FID < 2.0 (paper achieves 1.73)
# Hardware: 4x H200 GPUs (141GB VRAM each)
#
# KEY FIX: num_patches must be high enough so that patch_size stays small.
# With num_patches=2 (default), patch_size=128 at 256x256 → model is 136GB!
# With num_patches=16, patch_size=16 at 256x256 → model is ~769M params. ✅
#
# Verified model sizes (all with num_patches=16, base_dim=128):
#   hidden=256  heads=8  blocks=2  →  172M params (conservative)
#   hidden=512  heads=8  blocks=2  →  349M params (medium)
#   hidden=768  heads=12 blocks=2  →  548M params (close to paper's 561M)
#   hidden=1024 heads=16 blocks=2  →  769M params (recommended for H200)
#   hidden=1024 heads=16 blocks=4  → 1384M params (deep, needs more memory)

# Dataset - UPDATE THIS PATH
DATA="./datasets/imagenet_9to4_1024x1024_256x256.zip"

# Output directory
OUTDIR="./runs/diffit_imagenet256_h200"

# Hardware setup
NGPUS=4
BATCH_PER_GPU=64    # 64 per GPU × 4 GPUs = 256 total batch
                    # Model is ~769M params → ~3GB FP32 → fits easily in 141GB
                    # Reduce to 32 if OOM during training

# Model architecture
# CRITICAL: num_patches controls patch_size = resolution / num_patches
#   num_patches=2  → patch_size=128 → 2M features per patch → OOM!
#   num_patches=16 → patch_size=16  → manageable linear layers → OK
RESOLUTION=256
BASE_DIM=128
HIDDEN_DIM=1024     # Paper-like capacity (paper uses 1152 for ImageNet-256)
NUM_PATCHES=16      # patch_size=16 at full res → 769M params
NUM_HEADS=16        # 16 heads, head_dim=64 (matches paper)
NUM_BLOCKS=2        # 2 ResBlocks per stage
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
echo "=================================================="
echo "  DiffiT ImageNet-256 Training (H200 x 4)"
echo "=================================================="
echo ""
echo "Hardware:"
echo "  GPUs: ${NGPUS}x H200 (141GB)"
echo "  Batch per GPU: ${BATCH_PER_GPU}"
echo "  Total batch: $((NGPUS * BATCH_PER_GPU))"
echo ""
echo "Model:"
echo "  Resolution: ${RESOLUTION}x${RESOLUTION}"
echo "  Hidden dim: ${HIDDEN_DIM}"
echo "  Num patches: ${NUM_PATCHES} (patch_size=$((RESOLUTION / NUM_PATCHES)))"
echo "  Num heads: ${NUM_HEADS}"
echo "  Num blocks: ${NUM_BLOCKS}"
echo "  Estimated params: ~769M (with hidden=1024, patches=16, blocks=2)"
echo ""
echo "Training:"
echo "  Total kimg: ${KIMG}"
echo "  Learning rate: ${LR}"
echo "  CFG scale: ${CFG_SCALE}"
echo "  Metrics: ${METRICS}"
echo ""
echo "=================================================="
echo ""

torchrun --nproc_per_node=${NGPUS} train.py \
    --outdir="${OUTDIR}" \
    --data="${DATA}" \
    --batch-gpu=${BATCH_PER_GPU} \
    --resolution=${RESOLUTION} \
    --base-dim=${BASE_DIM} \
    --hidden-dim=${HIDDEN_DIM} \
    --num-patches=${NUM_PATCHES} \
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
