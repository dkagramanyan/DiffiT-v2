# DiffiT: Diffusion Vision Transformers for Image Generation

Official PyTorch implementation of [**DiffiT: Diffusion Vision Transformers for Image Generation**](https://arxiv.org/abs/2312.02139).

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

[![Star on GitHub](https://img.shields.io/github/stars/NVlabs/DiffiT.svg?style=social)](https://github.com/NVlabs/DiffiT/stargazers)

**DiffiT** (Diffusion Vision Transformers) is a generative model that combines the expressive power of diffusion models with Vision Transformers (ViTs), introducing **Time-dependent Multihead Self Attention (TMSA)** for fine-grained control over the denoising at each timestep. DiffiT achieves SOTA performance on class-conditional ImageNet generation at multiple resolutions, notably an **FID score of 1.73** on ImageNet-256.

**DiffiT-v2** is a performance refresh of this codebase. TMSA is preserved exactly as defined in the paper (Section 3.2, Eqs 3–5). What changes is the attention plumbing around it — the learned relative-position bias is replaced with **RoPE-2D** so SDPA can dispatch to FlashAttention, **QK-norm** is added for bf16 stability, LayerNorm becomes RMSNorm, and the MLP switches to SwiGLU. Training and sampling are rebuilt around `torchrun`, `torch.compile`, and `click`. See [Differences from the original NVlabs/DiffiT](#differences-from-the-original-nvlabsdiffit) below.

![teaser](./assets/imagenet.png)

![teaser](./assets/latent_diffit.png)

## News
- **[04.19.2026]** DiffiT-v2 performance refresh: RoPE-2D + FlashAttention, QK-norm, RMSNorm, SwiGLU. 1024² now a first-class config.
- **[03.08.2026]** DiffiT code and pretrained model are released!
- **[07.01.2024]** DiffiT has been accepted to [ECCV 2024](https://eccv.ecva.net/)!
- **[04.02.2024]** Updated [manuscript](https://arxiv.org/abs/2312.02139) now available on arXiv!
- **[12.04.2023]** Paper is published on arXiv!

## Differences from the original NVlabs/DiffiT

This repo is an engineering refresh of the upstream [NVlabs/DiffiT](https://github.com/NVlabs/DiffiT). The model's core contribution — **Time-dependent Multihead Self-Attention (TMSA)** — is preserved exactly as in the paper (Eqs 3–5, Fig 7b), and the parameter count stays at **561M** (matches Table 9). The changes are to the *engineering* around it: the attention internals, the training/sampling infrastructure, and the latent pipeline.

**Architecture (per-block attention internals):**

| Change | What it does | Why |
|---|---|---|
| **RoPE-2D** replaces learned relative-position bias | Axial rotary embeddings on Q, K | Unlocks FlashAttention-2 (attn_mask is no longer needed) and enables progressive-resolution finetuning across 256/512/1024. Frees ~2 GB `relative_position_index` buffer at 1024². |
| **QK-norm** (RMSNorm on Q, K) | Stateful per-head RMSNorm before RoPE | Prevents bf16/fp16 attention-logit blow-up at depth-28, hidden-1152 scale. |
| **RMSNorm** replaces LayerNorm | Stateless RMS normalization | ~1–2% faster, no quality regression. |
| **SwiGLU** MLP | Matched-param `hidden × 8/3` inner width, rounded to 64 | Modern FFN; small consistent quality gain. |
| **TMSA preserved** | Additive time-token QKV projection | Paper's core contribution (Eqs 3–5, Fig 7b). Parameter count stays at **561M** — matches Table 9 of the paper. |
| **CFG split fix** | Split at `self.in_channels` (=4 for SD-VAE) | Original code hardcoded `:3` which was wrong for 4-channel latents. |

**Pipeline, training & sampling infrastructure:**

- **Latent diffusion** — operates in the Stable-Diffusion VAE latent space (`stabilityai/sd-vae-ft-ema`, scaled by `0.18215`), so the transformer denoises 4-channel latents rather than pixels.
- **`torchrun` + `torch.compile`** — modern distributed launch (replaces MPI); `max-autotune` mode (with CUDA graphs, or `no-cudagraphs` when gradient checkpointing is on) for both training and eval. Fused AdamW, DDP with `no_sync()` gradient accumulation.
- **kimg/tick training loop** with inline, **distributed** quality metrics (FID / IS / sFID / Precision / Recall) computed every `snap` ticks during training — no separate eval job needed.
- **DPM-Solver++** for fast training-time sample snapshots; DDPM/DDIM available for full FID-50K sampling.
- **CFG schedule** — power-cosine CFG schedule at 256² (`input_size ≤ 32`), constant CFG scale at 512²/1024².

**Expected performance** (vs. original DiffiT): ~1.1–1.3× at 256², ~1.8–2.5× at 512², **~3–5× at 1024²** — dominated by FlashAttention at high resolution. Quality impact: ±0.1–0.3 FID, directionally positive.

> **⚠️ Checkpoints from the original DiffiT are not compatible with v2** — parameter names and shapes changed (learned position bias removed, RoPE/QK-norm/SwiGLU added).

## Models

### ImageNet-256

| Model | Dataset | Resolution | FID-50K | Inception Score | Download |
|-------|---------|-----------|---------|-----------------|----------|
| **DiffiT** | ImageNet | 256x256 | **1.73** | **276.49** | [model](https://huggingface.co/nvidia/DiffiT/resolve/main/diffit_256.safetensors) |

### ImageNet-512

| Model | Dataset | Resolution | FID-50K | Inception Score | Download |
|-------|---------|-----------|---------|-----------------|----------|
| **DiffiT** | ImageNet | 512x512 | **2.67** | **252.12** | [model](https://huggingface.co/nvidia/DiffiT/resolve/main/diffit_512.safetensors) |

## Installation

```bash
conda create -n diffit python=3.11 -y
conda activate diffit
pip install -e .
```

This installs the package (editable) along with its dependencies and the
console entry-points used throughout this README: `diffit-train`,
`diffit-sample`, `diffit-gen-images`, `diffit-eval`, `diffit-prepare-data`,
and `diffit-download-models`. For development extras (tests + linter) use
`pip install -e ".[dev]"`. A plain `pip install -r requirements.txt` still
works if you prefer to invoke scripts as `python -m scripts.<name>`.


## Pre-download Models (Offline Nodes)

The training script downloads two external models on first run. If your compute nodes have no internet access, run this **on a login node** first:

```bash
diffit-download-models
```

This caches the following models locally:
- **stabilityai/sd-vae-ft-ema** (335 MB) — VAE for latent diffusion (`~/.cache/huggingface/`)
- **stabilityai/sd-vae-ft-mse** (335 MB) — VAE variant for `diffit-gen-images --vae-decoder mse`
- **InceptionV3** (104 MB) — for IS/FID metrics during training (`~/.cache/torch/hub/`)
- **combra backbones** (InceptionV3-FID / CLIP / DINOv2) — only when the optional `combra` package is installed, for `--combra-metrics`

Alternatively, for fully offline nodes without a Python environment, a pure
`wget`/`curl`/`git` variant fetches the torch-hub / CLIP weights directly into the
caches (and the VAEs via `huggingface-cli` when present):

```bash
bash download_models.sh                         # caches under ~/.cache
MODEL_CACHE=/shared/team/caches bash download_models.sh
```

> If your compute nodes use a shared filesystem with the login node, the cached files will be available automatically. Otherwise, ensure `~/.cache/huggingface/` and `~/.cache/torch/hub/` are synced.


## Data Preparation

Use `diffit-prepare-data` to convert an ImageNet-style directory into a ZIP archive with resized images and a `dataset.json` containing class labels.

```
diffit-prepare-data \
    --source /path/to/ILSVRC \
    --dest ./datasets/imagenet_256x256.zip \
    --resolution 256x256 \
    --transform center-crop

diffit-prepare-data \
    --source /path/to/ILSVRC \
    --dest ./datasets/imagenet_512x512.zip \
    --resolution 512x512 \
    --transform center-crop

diffit-prepare-data \
    --source /path/to/ILSVRC \
    --dest ./datasets/imagenet_1024x1024.zip \
    --resolution 1024x1024 \
    --transform center-crop
```

For custom datasets, point `--source` at a directory with the ImageNet folder structure (`train/<class_id>/image.JPEG`). The tool will create a ZIP with resized images and a JSON with class labels.


## Training

### Base configurations

The `--cfg` flag selects a base configuration that sets model architecture,
resolution, learning rate, diffusion settings, etc. Individual CLI options
can still override any preset value.

| Config | Resolution | Model | LR | AMP | kimg | CFG scale | Grad ckpt |
|--------|-----------|-------|------|------|------|-----------|-----------|
| `diffit-256` | 256 | DiffiT-XL/2 | 3e-4 | bf16 | 400000 | 4.4 (power-cosine) | off |
| `diffit-512` | 512 | DiffiT-XL/2 | 1e-4 | bf16 | 400000 | 1.49 (constant) | off |
| `diffit-1024` | 1024 | DiffiT-XL/2 | 1e-4 | bf16 | 400000 | 1.49 (constant) | on |

Paper's recipe (Appendix I.2, p.22): AdamW, EMA 0.9999, DDPM sampler 250 steps, ADM diffusion hyperparameters.

### Training strategies

You have two viable approaches. **We strongly recommend the progressive-finetune path** (B) — it is 3–5× cheaper and historically reaches better final FID than independent from-scratch runs at higher resolutions.

---

### Strategy A: From scratch at each resolution

Use this if you need apples-to-apples per-resolution baselines for a paper.

**Cost warning:** on 2× H200 with the default `total_kimg=400000`, each run takes roughly:

| Resolution | Est. walltime (2× H200, bf16) |
|---|---|
| 256² | ~10–20 days |
| 512² | ~25–45 days |
| 1024² | ~50–100 days |

Total sequential: 3–5 months. Shrink `--kimg` to something reachable (e.g. `--kimg=100000`) or plan for multi-job resume chains.

#### 256² from scratch

```bash
diffit-train --outdir=./training-runs \
    --cfg=diffit-256 \
    --data=./datasets/imagenet_256x256.zip \
    --gpus 2 \
    --batch-gpu 96
```
Global batch = 192. Per paper (Section I.2): LR 3e-4, batch 256, EMA 0.9999.

#### 512² from scratch

```bash
diffit-train --outdir=./training-runs \
    --cfg=diffit-512 \
    --data=./datasets/imagenet_512x512.zip \
    --gpus 2 \
    --batch-gpu 64 \
    --lr-warmup 1000
```
Global batch = 128. LR warmup is recommended for from-scratch high-res runs.

#### 1024² from scratch

```bash
diffit-train --outdir=./training-runs \
    --cfg=diffit-1024 \
    --data=./datasets/imagenet_1024x1024.zip \
    --gpus 2 \
    --batch-gpu 16 \
    --grad-accum 4
```
Global effective batch = 16 × 2 × 4 = 128. With RoPE+FlashAttention you may be able to drop `--grad-accum` and/or disable checkpointing (`--no-grad-ckpt`) — start conservative and raise `--batch-gpu` once you confirm it fits.

---

### Strategy B: Progressive finetuning (recommended)

RoPE-2D lets you re-use a 256² checkpoint at higher resolutions — something the original learned-bias DiffiT could not do. This cuts total compute by ~3–5× and typically yields better final FID.

**Step 1. Train 256² from scratch** (same as Strategy A):

```bash
diffit-train --outdir=./training-runs \
    --cfg=diffit-256 \
    --data=./datasets/imagenet_256x256.zip \
    --gpus 2 \
    --batch-gpu 96
```
Let it run until FID plateaus on the inline eval (check TensorBoard). For a strong base, aim for 100k–200k kimg.

**Step 2. Finetune 256² → 512²**:

```bash
diffit-train --outdir=./training-runs \
    --cfg=diffit-512 \
    --data=./datasets/imagenet_512x512.zip \
    --gpus 2 \
    --batch-gpu 64 \
    --resume ./training-runs/00000-diffit-256-*/network-final.pt \
    --lr 5e-5 \
    --lr-warmup 500 \
    --kimg 100000
```
Lower LR (5e-5 ≈ half of the `diffit-512` default) for finetuning, short warmup, and a smaller total-kimg budget — finetuning converges faster than from-scratch.

**Step 3. Finetune 512² → 1024²**:

```bash
diffit-train --outdir=./training-runs \
    --cfg=diffit-1024 \
    --data=./datasets/imagenet_1024x1024.zip \
    --gpus 2 \
    --batch-gpu 16 \
    --resume ./training-runs/00001-diffit-512-*/network-final.pt \
    --lr 2e-5 \
    --lr-warmup 500 \
    --kimg 50000
```

**Why this works:** RoPE encodes positions via rotation, not learned weights. The frequency table is regenerated at the target grid size at load time (non-persistent buffer), so the saved state dict transplants cleanly. The rest of the network (QKV, QK-norm scales, SwiGLU, final linear) sees the same per-token distribution at any resolution — it just processes more tokens per image.

---

### SLURM sbatch scripts

> The sbatch scripts invoke the `diffit-*` console entry points (and
> `torchrun -m scripts.sample`), so make sure the cluster conda env has the
> package installed once: `pip install -e .` after `conda activate diffit`.

Pre-configured sbatch files are provided for H200:

```bash
sbatch train_2h200_256x256_prod.sbatch
sbatch train_2h200_512x512_prod.sbatch
sbatch train_2h200_1024x1024_prod.sbatch
sbatch train_4h200_1024x1024_prod.sbatch
```

For progressive finetuning, add `--resume=$PATH_TO_PREV_FINAL` to the sbatch's `diffit-train ...` line and lower the LR as shown above.

### Chaining runs with SLURM dependencies

Walltime caps will not fit `total_kimg=400000` in a single job. Chain jobs:

```bash
JID=$(sbatch --parsable train_2h200_256x256_prod.sbatch)
for i in 1 2 3; do
    JID=$(sbatch --parsable --dependency=afterany:$JID train_2h200_256x256_prod.sbatch)
done
```

Or add self-requeue to the sbatch:

```bash
#SBATCH --signal=B:USR1@300
trap 'scontrol requeue $SLURM_JOB_ID' USR1
```

Make sure your sbatch picks up the latest checkpoint on restart:

```bash
LATEST_CKPT=$(ls -t "$OUTDIR"/*/network-snapshot-*.pt 2>/dev/null | head -1)
RESUME_FLAG=""
[ -n "$LATEST_CKPT" ] && RESUME_FLAG="--resume=$LATEST_CKPT"

diffit-train \
    --outdir="$OUTDIR" \
    --cfg=diffit-256 \
    --data="$DATASET" \
    --gpus 2 \
    --batch-gpu 96 \
    --snap 100 \
    $RESUME_FLAG
```

### Resume from checkpoint (manual)

```bash
diffit-train --outdir=./training-runs \
    --cfg=diffit-256 \
    --data=./datasets/imagenet_256x256.zip \
    --gpus 2 \
    --batch-gpu 96 \
    --resume ./training-runs/00000-diffit-256-gpus2-batch192/network-snapshot-001000.pt
```

### Hydra entry point (optional)

`train_hydra.py` is a thin wrapper over the same training code, for users who
prefer Hydra/YAML config management. It derives every default by introspecting
the `train.py` click options (single source of truth), so `configs/config.yaml`
only declares the required fields and any new flag propagates automatically:

```bash
python train_hydra.py outdir=./training-runs cfg=diffit-256 \
    data=./datasets/imagenet_256x256.zip gpus=2 batch_gpu=42 \
    combra_metrics=false save_inference_only=true snap=100
```

Override any `train.py` option by its Python name (dashes → underscores). Both
entry points call the same `launch_from_opts`, so runs are identical.

### Training options

| Option | Default | Description |
|--------|---------|-------------|
| `--outdir` | required | Output directory for training runs |
| `--cfg` | required | Base configuration (`diffit-256`, `diffit-512`, `diffit-1024`) |
| `--data` | required | Path to dataset directory or .zip |
| `--gpus` | required | Number of GPUs |
| `--batch-gpu` | required | Batch size per GPU (total batch = batch-gpu * gpus) |
| `--image-size` | from cfg | Image resolution override |
| `--model` | from cfg | Model constructor name override |
| `--kimg` | from cfg | Total training duration in kimg |
| `--tick` | from cfg | Progress print interval (kimg) |
| `--snap` | from cfg | Snapshot save interval (ticks) |
| `--seed` | 0 | Random seed |
| `--lr` | from cfg | Learning rate override |
| `--fp32` | from cfg | Disable mixed precision (sets `use_fp16=False`) |
| `--amp-dtype` | bf16 | AMP dtype: `fp16` or `bf16` (bf16 preferred on A100/H100) |
| `--ema-rate` | from cfg | EMA decay rate override |
| `--resume` | None | Path to checkpoint for resuming |
| `--schedule-sampler` | from cfg | Timestep sampler override |
| `--cfg-scale` | from cfg | CFG scale used during training-time eval |
| `--num-fid-samples` | from cfg (10000) | Samples for FID/IS eval during training (0=disable) |
| `--combra-metrics` | true | Compute combra generative-quality metrics each snapshot tick (independent of `--num-fid-samples`); warns if requested but combra is not installed |
| `--save-inference-only` | false | Also write a small `network-snapshot-<kimg>-inference.pt` (EMA weights only, no optimizer/resume state) each snapshot tick — the smallest artifact for `gen_images.py` / `sample.py` |
| `--grad-accum` | from cfg | Gradient accumulation steps (effective batch = batch-gpu × gpus × accum) |
| `--grad-ckpt / --no-grad-ckpt` | from cfg | Toggle gradient checkpointing (trades compute for memory) |
| `--lr-warmup` | from cfg | Linear LR warmup duration in kimg (0 = disabled) |
| `--tf32 / --no-tf32` | tf32 on | Enable TF32 for matmul/conv |
| `--workers` | 4 | DataLoader worker processes |
| `--cache-in-ram / --no-cache-in-ram` | cache on | Cache entire dataset in RAM to reduce disk I/O |
| `-n, --dry-run` | off | Print resolved training options and exit |

### Training output

Each run creates a directory with the following structure:

```
training-runs/00000-diffit-256-gpus4-batch256/
├── training_options.json         # All training hyperparameters
├── log.txt                       # Human-readable training log
├── progress.csv                  # CSV training metrics
├── progress.json                 # JSON training metrics
├── stats.jsonl                   # JSON Lines stats (SAN-v2 style)
├── events.out.tfevents.*         # TensorBoard event files
├── reals.png                     # Real training image grid
├── fakes_init.png                # Initial generated images (before training)
├── fakes000200.png               # Generated images at 200 kimg
├── fakes000400.png               # Generated images at 400 kimg
├── ...
├── network-snapshot-001000.pt    # Periodic checkpoint (full resumable state)
├── network-snapshot-001000-inference.pt  # (only with --save-inference-only) EMA weights only
├── network-snapshot-002000.pt
├── ...
└── network-final.pt              # Final trained model (full resumable state)
```

`network-snapshot-*.pt` / `network-final.pt` hold the **full** resumable state
(`model`, `ema`, `opt`, optional `scaler`, `cur_nimg`) so `--resume` can continue
training exactly. The inference loaders (`gen_images.py`, `sample.py`) transparently
extract the EMA weights from any of these — a full checkpoint, an older bare
EMA `state_dict`, or a `--save-inference-only` `*-inference.pt` file.

Quality metrics (**IS**, **FID**, **sFID**, **Precision**, **Recall**) are computed automatically every `snap` ticks during training using 10000 samples by default (configurable per `--cfg`). Results are logged to TensorBoard under `Metrics/` and to `stats.jsonl`. Adjust with `--num-fid-samples` (set to 0 to disable). When the optional `combra` package is installed (`pip install -e ".[combra]"`) and `--combra-metrics` is on (default), additional `combra_*` metrics (e.g. CMMD, FD-DINOv2) are logged each tick too — and the eval reference becomes the whole training set. Pre-fetch combra's CLIP/DINOv2 backbones for offline nodes with `python scripts/download_models.py` or `bash download_models.sh`.

Monitor training with TensorBoard:

```bash
tensorboard --logdir ./training-runs
```


## Generating Samples

### Individual image generation

Generate individual PNG images for visual inspection:

```bash
diffit-gen-images \
    --model-path ./training-runs/00000-diffit-256-gpus4-batch256/network-final.pt \
    --seeds 0-49 \
    --outdir ./generated/256 \
    --image-size 256 \
    --cfg-scale 4.4 \
    --num-sampling-steps 250
```

Options:
- `--seeds`: Comma-separated list or ranges (e.g., `0,1,4-6`)
- `--class-idx`: Specific class label (random if not specified)
- `--batch-sz`: Batch size per seed
- `--use-ddim`: Use DDIM sampling instead of DDPM
- `--cfg-scale`: Classifier-free guidance scale (4.4 for 256, 1.49 for 512)
- `--scale-pow`: Power for cosine CFG schedule

SLURM:
```bash
sbatch sbatch/a100/generate_1_gpu_256x256.sbatch
sbatch sbatch/a100/generate_1_gpu_512x512.sbatch
sbatch sbatch/h200/generate_4_gpu_256x256.sbatch
sbatch sbatch/h200/generate_4_gpu_512x512.sbatch
```

### Bulk sampling for FID evaluation

Generate 50K samples as `.npz` for FID evaluation:

**ImageNet-256:**
```bash
torchrun --nproc_per_node=4 -m scripts.sample \
    --model-path ./training-runs/00000-diffit-256-gpus4-batch256/network-final.pt \
    --outdir ./samples/256 \
    --image-size 256 \
    --cfg-scale 4.4 \
    --num-samples 50000 \
    --batch-size 16 \
    --num-sampling-steps 250 \
    --cfg-cond
```

**ImageNet-512:**
```bash
torchrun --nproc_per_node=4 -m scripts.sample \
    --model-path ./training-runs/00000-diffit-512-gpus4-batch100/network-final.pt \
    --outdir ./samples/512 \
    --image-size 512 \
    --cfg-scale 1.49 \
    --num-samples 50000 \
    --batch-size 8 \
    --num-sampling-steps 250 \
    --cfg-cond
```

SLURM:
```bash
sbatch sbatch/a100/sample_4_gpu_256x256.sbatch
sbatch sbatch/a100/sample_4_gpu_512x512.sbatch
sbatch sbatch/h200/sample_4_gpu_256x256.sbatch
sbatch sbatch/h200/sample_4_gpu_512x512.sbatch
```


## Quality Metrics

Quality metrics are computed **inline during training** every `snap` ticks. The following metrics are evaluated and logged to TensorBoard (`Metrics/`) and `stats.jsonl`:

- **Inception Score (IS)** — diversity and quality of generated classes
- **FID** — Frechet Inception Distance (pool features)
- **sFID** — spatial FID (captures spatial structure)
- **Precision** — fraction of generated samples in the real data manifold
- **Recall** — fraction of real samples covered by the generated manifold

By default, 10000 samples are generated for each evaluation (configurable via `--num-fid-samples`). For a full FID-50K evaluation, use the standalone evaluator:

```bash
diffit-eval \
    --ref-batch ./VIRTUAL_imagenet256_labeled.npz \
    --sample-batch ./samples/256/samples_50000x256x256x3.npz
```

### Expected Results

**ImageNet-256:**

| Inception Score | FID | sFID | Precision | Recall |
|:-:|:-:|:-:|:-:|:-:|
| 276.49 | 1.73 | 4.54 | 0.8024 | 0.6205 |

**ImageNet-512:**

| Inception Score | FID | sFID | Precision | Recall |
|:-:|:-:|:-:|:-:|:-:|
| 252.13 | 2.67 | 4.99 | 0.8277 | 0.5500 |

> **Note:** Small variations in the reported numbers are expected depending on the device used for sampling and due to numerical precision differences.


## Tests

A lightweight CPU smoke-test suite guards the model's forward contract and the
core diffusion math (no GPU, dataset, or external weights required):

```bash
pip install pytest
pytest tests/ -q
```

## Project Structure

```
DiffiT-v2/
├── diffit/                          # Core model architecture
│   ├── __init__.py                 # Diffusion creation & defaults
│   ├── diffit.py                   # DiffiT model (ViT + TMSA)
│   ├── constants.py                # Shared numeric constants (VAE scale, norm)
│   ├── gaussian_diffusion.py       # Diffusion process (DDPM/DDIM)
│   ├── dpm_solver.py               # DPM-Solver++ fast sampler
│   ├── respace.py                  # Timestep respacing
│   ├── dist_util.py                # Distributed training (PyTorch DDP)
│   ├── image_datasets.py           # Dataset loading (dir/zip + DistributedSampler)
│   ├── inception.py                # Shared InceptionV3 feature extractor (FID/IS)
│   ├── metrics.py                  # Inline FID/IS/sFID/Precision/Recall
│   ├── logger.py                   # Logging (stdout, JSON, CSV, TensorBoard)
│   ├── nn.py                       # Neural network utilities (EMA, etc.)
│   ├── timestep_sampler.py         # Timestep sampling strategies
│   ├── diffusion_utils.py          # KL divergence & likelihood
│   └── pos_emb.py                  # Positional embeddings (CoordConv, Swin)
├── scripts/                         # Command-line entry points (installed as diffit-*)
│   ├── train.py                    # Training (DDP, click CLI)      -> diffit-train
│   ├── sample.py                   # Bulk FID sampling (.npz)       -> diffit-sample
│   ├── gen_images.py               # Individual PNG generation      -> diffit-gen-images
│   ├── evaluator.py                # FID/IS evaluation (PyTorch)    -> diffit-eval
│   ├── dataset_tool_for_imagenet.py # ImageNet -> ZIP converter     -> diffit-prepare-data
│   └── download_models.py          # Pre-download VAE + InceptionV3 -> diffit-download-models
├── tests/                           # CPU smoke tests (forward, diffusion, RoPE)
├── eval_run.sh                      # Evaluation convenience script
├── pyproject.toml                   # Packaging, entry points, ruff/pytest config
├── .github/workflows/ci.yml         # CI: ruff lint + pytest smoke tests
├── sbatch/                          # SLURM job scripts
│   ├── a100/                       # A100 cluster (2 GPU)
│   │   ├── train_2_gpu_256x256.sbatch
│   │   ├── train_2_gpu_512x512.sbatch
│   │   ├── generate_1_gpu_256x256.sbatch
│   │   ├── generate_1_gpu_512x512.sbatch
│   │   ├── sample_4_gpu_256x256.sbatch
│   │   └── sample_4_gpu_512x512.sbatch
│   └── h200/                       # H200 cluster (4 GPU)
│       ├── h200_train_4_gpu_256x256.sbatch
│       ├── h200_train_4_gpu_512x512.sbatch
│       ├── h200_train_1_gpu_256x256.sbatch
│       ├── generate_4_gpu_256x256.sbatch
│       ├── generate_4_gpu_512x512.sbatch
│       ├── sample_4_gpu_256x256.sbatch
│       └── sample_4_gpu_512x512.sbatch
├── requirements.txt                 # Python dependencies (also declared in pyproject.toml)
└── README.md
```

## Citation

```
@inproceedings{hatamizadeh2025diffit,
  title={Diffit: Diffusion vision transformers for image generation},
  author={Hatamizadeh, Ali and Song, Jiaming and Liu, Guilin and Kautz, Jan and Vahdat, Arash},
  booktitle={European Conference on Computer Vision},
  pages={37--55},
  year={2025},
  organization={Springer}
}
```

## Licenses

Copyright 2026, NVIDIA Corporation. All rights reserved.

This work is made available under the NVIDIA Source Code License-NC. Click [here](LICENSE) to view a copy of this license.

The pre-trained models are shared under [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

## Acknowledgement
We gratefully acknowledge the authors of [Guided-Diffusion](https://github.com/openai/guided-diffusion/tree/main/), [DiT](https://github.com/facebookresearch/DiT/tree/main) and [MDT](https://github.com/sail-sg/MDT/tree/mdtv1) for making their excellent codebases publicly available.
