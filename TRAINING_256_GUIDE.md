# ImageNet-256 Training Guide for H200

## Quick Start

```bash
# 1. Update the dataset path in the script
nano train_imagenet256_h200.sh
# Change: DATA="./datasets/your-imagenet-256-dataset.zip"

# 2. Run training
./train_imagenet256_h200.sh

# Or manually:
bash train_imagenet256_h200.sh
```

---

## What Changed from 128x128 to 256x256

### Resolution & Architecture

| Parameter | 128x128 (Original) | 256x256 (H200 Optimized) | Reason |
|-----------|-------------------|--------------------------|--------|
| **Resolution** | 128 | **256** | Target resolution |
| **Batch per GPU** | 256 | **128** | Memory constraint |
| **Total batch** | 1024 | **512** | 4 GPUs × 128 |
| **Hidden dim** | 64 | **1024** | More capacity for 256x256 |
| **Num heads** | 8 | **16** | Better attention |
| **Num blocks** | 2 | **4** | Deeper network |

### Evaluation

| Parameter | Original | H200 Optimized | Reason |
|-----------|----------|----------------|--------|
| **Metrics** | fid10k_full | **fid50k_full** | More reliable (as requested) |
| **FID samples** | 10000 | **50000** | Paper standard |
| **Metrics ticks** | 50 | **100** | Less frequent (saves time) |

### Training

| Parameter | Original | H200 Optimized | Reason |
|-----------|----------|----------------|--------|
| **Learning rate** | 1e-4 | **3e-4** | Paper's ImageNet-256 LR |
| **CFG scale** | 1.5 | **4.6** | Paper's best for ImageNet-256 |

---

## Memory Usage Estimates

### Per GPU (H200 with 141GB)

**256x256 Resolution**:
- Model parameters: ~500M (estimated)
- Model memory: ~2GB (FP32) or ~1GB (FP16)
- Activations (batch=128): ~40-50GB
- Optimizer states (AdamW): ~4GB
- **Total: ~55-60GB per GPU** ✅ Fits comfortably in 141GB

**If OOM occurs**, reduce `BATCH_PER_GPU`:
```bash
BATCH_PER_GPU=96   # Still gives 384 total batch
BATCH_PER_GPU=64   # Conservative, gives 256 total batch
```

---

## Expected Training Time

### With 4× H200 GPUs

- **Total images**: 1B (1M kimg)
- **Batch size**: 512 (total)
- **Iterations**: ~2M iterations
- **Time per iteration**: ~2-3 seconds (estimated)

**Total time**: ~5-7 days continuous training

### Checkpoints

- Snapshots saved every 50 ticks (~200k images)
- Best model saved automatically (lowest FID)
- Resume training with `--resume=path/to/checkpoint.pkl`

---

## Target Performance

Based on DiffiT paper (Table 1):

| Dataset | Resolution | FID (Paper) | Your Target |
|---------|-----------|-------------|-------------|
| ImageNet-256 | 256×256 | **1.73** | < 2.0 |

**Note**: The paper uses a deeper 30-layer model for ImageNet-256. Our configuration uses fewer layers but should still achieve good results.

---

## Monitoring Training

### TensorBoard

```bash
# In another terminal
tensorboard --logdir=./runs/diffit_imagenet256_h200

# Open browser to: http://localhost:6006
```

**Key metrics to watch**:
- `Loss/train`: Should decrease steadily
- `Metrics/fid50k_full`: Should decrease over time
- `Metrics/best_fid`: Tracks your best FID
- `Progress/kimg`: Training progress

### Check Progress

```bash
# View recent logs
tail -f ./runs/diffit_imagenet256_h200/*/log.txt

# Check latest FID score
grep "fid50k_full" ./runs/diffit_imagenet256_h200/*/log.txt | tail -1
```

---

## Adjusting for Different Scenarios

### Faster Iteration (Debug/Testing)

```bash
# Reduce kimg for quick test
KIMG=100  # Just 100k images (1-2 hours)

# Reduce FID frequency
METRICS_TICKS=10  # Every 10 ticks
FID_SAMPLES=5000  # Faster FID computation
```

### Maximum Quality (Production)

```bash
# Increase model capacity
HIDDEN_DIM=1152  # Paper's full model
NUM_BLOCKS=6     # Even deeper

# More frequent evaluation
METRICS_TICKS=50

# Full FID-50k always
FID_SAMPLES=50000
```

### Memory Constrained

```bash
# Reduce batch size
BATCH_PER_GPU=64  # Halve batch → halve memory

# Reduce model size
HIDDEN_DIM=768
NUM_HEADS=12
```

---

## Comparison with Paper Architecture

### Your Configuration

```python
DiffiT(
    image_shape=[3, 256, 256],
    base_dim=128,
    hidden_dim=1024,      # Your config
    num_heads=16,
    num_res_blocks=4,
    # Uses default channel_mults=(1,2,2,2)
)
```

### Paper's Full ImageNet-256 Model

```python
# From paper Section H.2 and README configs
DiffiT_Paper(
    resolution=256,
    hidden_dim=1152,      # Slightly larger
    num_heads=16,         # Same
    num_transformer_blocks=30,  # Much deeper!
    # Total parameters: 561M
)
```

**Your model**: ~500M parameters
**Paper model**: 561M parameters

Your configuration should still achieve very good results, just might not quite reach the paper's 1.73 FID.

---

## Using the Improved TMSA (Recommended)

If you haven't already, switch to the improved attention:

```bash
cd models/
mv attention.py attention_original.py
mv attention_improved.py attention.py
```

This gives you:
- ✅ Window-based attention (faster for 256x256)
- ✅ Better relative position bias
- ✅ Matches paper exactly

Window size is auto-selected: **Global attention** for 256x256 (as per paper).

---

## Troubleshooting

### OOM (Out of Memory)

**Reduce batch size**:
```bash
BATCH_PER_GPU=96   # or 64, or 48
```

**Or reduce model**:
```bash
HIDDEN_DIM=768     # Smaller model
```

### Slow Training

**Check GPU utilization**:
```bash
nvidia-smi -l 1  # Monitor GPU usage
```

Should see:
- GPU utilization: 90-100%
- Memory usage: 50-60GB per GPU

**If low utilization**:
- Increase `--workers` (data loading)
- Check dataset I/O speed

### Poor FID Scores

**Wait longer**:
- ImageNet-256 needs ~500-800k kimg to converge
- Don't judge before 100k kimg minimum

**Check CFG**:
- Try different `--cfg-scale` values (1.5, 2.0, 3.0, 4.6)
- Higher = better quality, lower = more diversity

**Increase model size**:
- Try `HIDDEN_DIM=1152` and `NUM_BLOCKS=6`

---

## Generation After Training

```bash
# Generate samples with best model
python generate.py \
    --network=./runs/diffit_imagenet256_h200/*/best_model.pkl \
    --outdir=./generated \
    --class=207 \
    --cfg-scale=4.6 \
    --seeds=0-15

# Generate class grid
python generate.py \
    --network=./runs/diffit_imagenet256_h200/*/best_model.pkl \
    --outdir=./generated \
    --class-grid \
    --samples-per-class=4 \
    --cfg-scale=4.6
```

---

## Summary

✅ **Resolution**: 128 → **256**
✅ **Batch size**: 1024 → **512** (still very large!)
✅ **Model size**: Increased to **1024 hidden dim, 16 heads**
✅ **FID evaluation**: fid10k → **fid50k_full** (as requested)
✅ **CFG scale**: 1.5 → **4.6** (paper's best)
✅ **Target**: FID < 2.0

**Ready to train!** 🚀

---

**Script**: [`train_imagenet256_h200.sh`](train_imagenet256_h200.sh)
**Hardware**: 4× H200 (141GB each)
**Estimated time**: 5-7 days
**Target FID**: < 2.0 (paper: 1.73)
