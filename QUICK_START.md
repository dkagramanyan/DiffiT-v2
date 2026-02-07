# Quick Start - ImageNet-256 Training

## ✅ Fixed Script Ready!

The training script has been corrected and is ready to use.

## 🚀 How to Run

### Step 1: Update Dataset Path

```bash
nano train_imagenet256_h200.sh
```

Change line 7:
```bash
DATA="./datasets/your-imagenet-256-dataset.zip"
```

### Step 2: Run Training

```bash
./train_imagenet256_h200.sh
```

---

## 🔧 What Was Fixed

**Error**:
```
Error: Invalid value for '--cond': '--label-drop=0.1' is not a valid boolean.
```

**Fix**: Changed `--cond` to `--cond=True`

The `--cond` parameter requires an explicit value:
- ✅ `--cond=True` (correct)
- ✅ `--cond=1` (also works)
- ❌ `--cond` (wrong - needs value)

---

## 📋 Current Configuration

**Hardware**: 4× H200 (141GB each)

**Model**:
- Resolution: 256×256
- Hidden dim: 1024
- Num heads: 16
- Batch per GPU: 128
- Total batch: 512

**Training**:
- Class-conditional: ✅ Enabled
- CFG scale: 4.6
- Learning rate: 3e-4
- Total: 1M kimg

**Evaluation**:
- Metric: FID-50k (as requested)
- Frequency: Every 100 ticks
- Samples: 50,000

---

## 🎯 Expected Results

**Training time**: ~5-7 days on 4× H200

**Target**: FID < 2.0 (paper: 1.73)

---

## 📊 Monitor Training

```bash
# TensorBoard
tensorboard --logdir=./runs/diffit_imagenet256_h200

# Check logs
tail -f ./runs/diffit_imagenet256_h200/*/log.txt

# Check FID
grep "fid50k_full" ./runs/diffit_imagenet256_h200/*/log.txt | tail -1
```

---

## ⚙️ Adjust Settings (if needed)

### Reduce Memory Usage

If you get OOM errors:
```bash
# In train_imagenet256_h200.sh, change:
BATCH_PER_GPU=96   # Reduce from 128
# or
BATCH_PER_GPU=64   # Even more conservative
```

### Test Run (Quick)

For a quick test before full training:
```bash
# In train_imagenet256_h200.sh, change:
KIMG=100           # Just 100k images (~1-2 hours)
METRICS_TICKS=10   # More frequent evaluation
```

### Maximum Quality

For best possible results:
```bash
# In train_imagenet256_h200.sh, change:
HIDDEN_DIM=1152    # Paper's full size
NUM_BLOCKS=6       # Deeper network
```

---

## 🐛 Troubleshooting

### Still Get Boolean Error?

Make sure you're using the **fixed** script:
```bash
# Check the script has --cond=True
grep "cond=" train_imagenet256_h200.sh
# Should show: --cond=True
```

### Wrong Dataset Path?

Update line 7 in the script:
```bash
DATA="/path/to/your/imagenet-256-dataset.zip"
```

### Check Dataset Format

Your dataset should have:
- `dataset.json` with class labels
- Images in `00000/img00000000.png` format

Verify:
```bash
unzip -l your-dataset.zip | head -20
```

---

## 🎓 More Help

- [`TRAINING_256_GUIDE.md`](TRAINING_256_GUIDE.md) - Complete guide
- [`README_FIXES.md`](README_FIXES.md) - All project fixes
- [`train_imagenet256_h200.sh`](train_imagenet256_h200.sh) - Training script

---

**Ready to train!** 🚀
