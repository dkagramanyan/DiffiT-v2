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

Create and activate a Python 3.12 conda env:

```bash
conda create -n diffit python=3.12 -y
conda activate diffit
```

Install the latest **PyTorch** first, from the CUDA 13.2 wheels (H200; the wheel
bundles the CUDA runtime), then install the package:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install -e .
```

Unlike SAN-v2, DiffiT-v2 has **no custom CUDA ops to JIT-compile** — attention
runs through PyTorch's built-in `scaled_dot_product_attention` (FlashAttention),
so no `nvcc`/`ninja` toolchain is required. Installing torch from the `cu132`
index first means the `torch>=2.0.0` lower bound below is already satisfied, so
`pip install -e .` won't pull a different CUDA build over it.

This installs the package (editable) along with its dependencies and the
console entry-points used throughout this README: `diffit-train`,
`diffit-sample`, `diffit-gen-images`, `diffit-eval`, `diffit-prepare-data`,
and `diffit-download-models`. For development extras (tests + linter) use
`pip install -e ".[dev]"`; for the optional combra metrics use
`pip install -e ".[combra]"` (combra is pulled over `git+https`). `pyproject.toml`
is the only dependency declaration — there is no `requirements.txt`.


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

`diffit-prepare-data` is a click group; its `convert` subcommand turns an
ImageNet-style directory into a ZIP with resized RGB images and a `dataset.json`
carrying both the integer `labels` and an index-aligned `class_names` list.
Transforms: `center-crop` / `center-crop-wide` / `center-crop-dhariwal`.

```
diffit-prepare-data convert \
    --source /path/to/ILSVRC \
    --dest ./datasets/imagenet_256x256.zip \
    --resolution 256x256 \
    --transform center-crop

diffit-prepare-data convert \
    --source /path/to/ILSVRC \
    --dest ./datasets/imagenet_512x512.zip \
    --resolution 512x512 \
    --transform center-crop

diffit-prepare-data convert \
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
Global effective batch = 16 × 2 × 4 = 128. With RoPE+FlashAttention you may be able to drop `--grad-accum` and/or disable checkpointing (`--grad-ckpt False`) — start conservative and raise `--batch-gpu` once you confirm it fits.

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
    --init-weights ./training-runs/00000-diffit-256-*/diffit-snapshot-*-inference.pt \
    --lr 5e-5 \
    --lr-warmup 500 \
    --kimg 100000
```
`--init-weights` is a weights-only warm start (loads the previous stage's EMA
weights, fresh optimizer) — not a resume. Lower LR (5e-5 ≈ half of the
`diffit-512` default) for finetuning, short warmup, and a smaller total-kimg
budget — finetuning converges faster than from-scratch.

**Step 3. Finetune 512² → 1024²**:

```bash
diffit-train --outdir=./training-runs \
    --cfg=diffit-1024 \
    --data=./datasets/imagenet_1024x1024.zip \
    --gpus 2 \
    --batch-gpu 16 \
    --init-weights ./training-runs/00001-diffit-512-*/diffit-snapshot-*-inference.pt \
    --lr 2e-5 \
    --lr-warmup 500 \
    --kimg 50000
```

**Why this works:** RoPE encodes positions via rotation, not learned weights. The frequency table is regenerated at the target grid size at load time (non-persistent buffer), so the saved state dict transplants cleanly. The rest of the network (QKV, QK-norm scales, SwiGLU, final linear) sees the same per-token distribution at any resolution — it just processes more tokens per image.

---

### Cluster launch (`sh/` scripts)

Cluster launches are plain shell scripts under `sh/` — no `.sbatch` files in the
repo. Each self-locates the repo root, activates the conda env (`CONDA_ENV`,
default `diffit`), sets the offline-cluster contract (`HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`), and makes one `diffit-*` console call. SLURM specifics
are supplied at submission time — never hardcoded:

```bash
# workstation
DATA=./datasets/imagenet_256x256.zip bash sh/train_256.sh

# cluster (account / partition / gpus at submit time)
DATA=./datasets/imagenet_256x256.zip \
  sbatch --account=<proj> --partition=<part> --gpus=2 sh/train_256.sh
```

Override `OUTDIR` / `GPUS` / `BATCH_GPU` (and, for generation, `NETWORK` /
`SAMPLES_PER_CLASS`) via env vars; extra `diffit-train` flags pass through after
the script name. Prefetch backbones once on a login node with
`diffit-download-models` before an offline run.

### No resume — size runs to the walltime

Runs go start-to-finish: **there is no `--resume`, no auto-restart, and no
rolling/best/final full checkpoint.** A crash or walltime kill cannot be
continued, so size `--kimg` (or split into progressive `--init-weights` stages)
to fit the job's time limit. Every snapshot is written atomically and the last
tick always snapshots, so a completed run always ends in a usable model. Pick the
best checkpoint post-hoc from `stats.jsonl` against the kept
`diffit-snapshot-<kimg>-inference.pt` history (raise `--snapshot-keep-last` to
keep more, `0` for all).

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
| `--seed` | 0 | Random seed (weight init, data shuffle incl. DistributedSampler, eval/grid latents) |
| `--lr` | from cfg | Learning rate override |
| `--precision` | from cfg (`bf16`) | Compute precision: `fp32` / `fp16` / `bf16` (GradScaler only for fp16) |
| `--ema-rate` | from cfg | EMA decay rate override |
| `--init-weights` | None | Warm-start from a previous stage's EMA snapshot (weights only, fresh optimizer) |
| `--schedule-sampler` | from cfg | Timestep sampler override |
| `--cfg-scale` | from cfg | CFG scale used during training-time eval |
| `--num-fid-samples` | from cfg (10000) | Fakes for eval / combra each tick (0=disable) |
| `--combra-ref-count` | 0 | Cap the combra reference to a seeded random subset of N reals (0 = whole dataset) |
| `--combra-metrics` | True | Compute combra generative-quality metrics each snapshot tick; warns if requested but combra is not installed |
| `--snapshot-keep-last` | 3 | Keep only the N newest `diffit-snapshot-<kimg>-inference.pt` snapshots (0 = keep all) |
| `--grad-accum` | from cfg | Gradient accumulation steps (effective batch = batch-gpu × gpus × accum) |
| `--grad-ckpt` | from cfg | Gradient checkpointing (`True`/`False`) |
| `--lr-warmup` | from cfg | Linear LR warmup duration in kimg (0 = disabled) |
| `--tf32` | True | Enable TF32 for matmul/conv (`True`/`False`) |
| `--bench` | True | Enable cuDNN autotune / benchmark (`True`/`False`) |
| `--mirror` | False | Stochastic per-item horizontal flip in the training loader (`True`/`False`) |
| `--workers` | 3 | DataLoader worker processes |
| `--cache-in-ram` | True | Cache entire dataset in RAM (`True`/`False`) |
| `-n, --dry-run` | off | Print resolved training options and exit |

### Training output

Each run creates a directory with the following structure:

```
training-runs/00000-diffit-256-gpus4-batch256/
├── training_options.json                  # Resolved launch config
├── 00000-diffit-256-gpus4-batch256.log    # Rank-0 console transcript
├── stats.jsonl                            # Machine-readable scalar rows (one per tick)
├── events.out.tfevents.*.<run-name>       # TensorBoard scalars/images (run-name suffix)
├── reals.png                              # Real training image grid
├── fakes_init.png                         # Initial generated images (before training)
├── fakes000200.png                        # Generated images at 200 kimg
├── fakes000400.png                        # Generated images at 400 kimg
├── ...
├── diffit-snapshot-000998-inference.pt    # EMA-only snapshot + metadata (newest --snapshot-keep-last kept)
├── diffit-snapshot-000999-inference.pt
└── diffit-snapshot-001000-inference.pt    # Last tick always snapshots → this IS the final model
```

There is exactly one checkpoint kind: `diffit-snapshot-<kimg>-inference.pt` — EMA
weights only, plus self-describing metadata (`n_classes`, `resolution`,
`class_names`, `cur_nimg`). It is written every `--snap` ticks **and always at the
last tick**, atomically (temp file + `os.replace`), and pruned to the newest
`--snapshot-keep-last`. No optimizer state, discriminators, or raw (non-EMA)
weights ever touch disk; there is no resume, best-model, rolling `latest`, or
final full checkpoint. The inference loaders (`gen_images.py`, `sample.py`)
extract the EMA weights from any of these (or an older bare EMA `state_dict`).

Quality metrics (**IS**, **FID**, **sFID**, **Precision**, **Recall**) are computed automatically every `snap` ticks during training using 10000 samples by default (configurable per `--cfg`), when combra is **not** used. Results are logged to TensorBoard under `Metrics/` and to `stats.jsonl`. Adjust with `--num-fid-samples` (set to 0 to disable).

`--combra-metrics` (on by default) is **mutually exclusive** with the Inception suite above: when it is on, the IS/FID/sFID/Precision/Recall metrics are disabled and only `combra_*` metrics are logged. combra generates `--num-fid-samples` fakes each tick, scored against the training set (capped to a seeded random subset by `--combra-ref-count`). The image-feature metrics are logged as `combra_fid10k`, `combra_cmmd10k`, `combra_fd_dinov2_10k` (the `10k` suffix is literal and does not change with `--num-fid-samples`); the angle-density metrics (`combra_w1`, `combra_mu1`, …) keep their bare names.

To enable combra metrics, install the optional extra:

```bash
pip install -e ".[combra]"      # pulls combra (all image metrics included)
```

All combra image metrics are covered by combra's base dependencies: `combra_fid10k` (pytorch-fid + InceptionV3 weights), `combra_cmmd10k` (**open-clip-torch** CLIP backbone) and `combra_fd_dinov2_10k` (a `torch.hub` DINOv2 download) — so a plain `combra` install enables them all, no separate extra. Pre-fetch combra's CLIP/DINOv2 backbones for offline nodes with `python scripts/download_models.py` or `bash download_models.sh`.

Monitor training with TensorBoard:

```bash
tensorboard --logdir ./training-runs
```


## Generating Samples

### Individual image generation

Generate individual PNG images for visual inspection:

```bash
diffit-gen-images \
    --network ./training-runs/00000-diffit-256-gpus4-batch256/diffit-snapshot-000400-inference.pt \
    --seeds 0-49 \
    --outdir ./generated/256 \
    --image-size 256 \
    --cfg-scale 4.4 \
    --steps 250
```

Options:
- `--seeds`: Comma-separated list or ranges (e.g., `0,1,4-6`) — seed mode, one image per seed
- `--class-idx`: Specific class label (random if not specified)
- `--sampler`: `dpm++` / `unipc` / `ddim` / `ddpm` (replaces the former `--use-ddim`)
- `--steps` (alias `--num-sampling-steps`): sampler steps
- `--cfg-scale`: Classifier-free guidance scale (4.4 for 256, 1.49 for 512)
- `--scale-pow`: Power for cosine CFG schedule

For bulk per-class generation into the RankH5Writer HDF5 layout the wc_cv angle
pipeline consumes, use `--samples-per-class` with self-spawning `--gpus N` and
`sh/generate_<res>.sh`.

### Bulk sampling for FID evaluation (legacy)

`scripts.sample` is a **legacy** bulk-`.npz` sampler for the upstream-paper FID
protocol — outside the v2 generation contract, no guarantees. Prefer
`diffit-gen-images` for WC-Co work.

**ImageNet-256:**
```bash
torchrun --nproc_per_node=4 -m scripts.sample \
    --model-path ./training-runs/00000-diffit-256-gpus4-batch256/diffit-snapshot-000400-inference.pt \
    --outdir ./samples/256 \
    --image-size 256 \
    --cfg-scale 4.4 \
    --num-samples 50000 \
    --batch-size 16 \
    --num-sampling-steps 250 \
    --cfg-cond
```


## Quality Metrics

Quality metrics are computed **inline during training** every `snap` ticks. The following metrics are evaluated and logged to TensorBoard (`Metrics/`) and `stats.jsonl`:

- **Inception Score (IS)** — diversity and quality of generated classes
- **FID** — Frechet Inception Distance (pool features)
- **sFID** — spatial FID (captures spatial structure)
- **Precision** — fraction of generated samples in the real data manifold
- **Recall** — fraction of real samples covered by the generated manifold

With `--combra-metrics` on (default) these Inception metrics are replaced by the combra suite instead — the angle-density metrics plus `combra_fid10k` / `combra_cmmd10k` / `combra_fd_dinov2_10k` (10k fakes vs the whole training set). See [Training output](#training-output) above for the install needed (CMMD requires `open-clip-torch`).

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
│   ├── metrics.py                  # Inline FID/IS/sFID/Precision/Recall + combra split APIs
│   ├── logger.py                   # Minimal rank-0 .log transcript + scalar accumulators
│   ├── nn.py                       # Neural network utilities (EMA, etc.)
│   ├── timestep_sampler.py         # Timestep sampling strategies
│   ├── diffusion_utils.py          # KL divergence & likelihood
│   └── pos_emb.py                  # Positional embeddings (CoordConv, Swin)
├── scripts/                         # Command-line entry points (installed as diffit-*)
│   ├── train.py                    # Training (DDP, click CLI)      -> diffit-train
│   ├── sample.py                   # Bulk FID sampling (.npz)       -> diffit-sample
│   ├── gen_images.py               # Individual PNG generation      -> diffit-gen-images
│   ├── evaluator.py                # FID/IS evaluation (PyTorch)    -> diffit-eval
│   ├── dataset_tool_for_imagenet.py # dir -> ZIP converter (click group) -> diffit-prepare-data
│   └── download_models.py          # Pre-download VAE + InceptionV3 -> diffit-download-models
├── tests/                           # CPU smoke tests (forward, diffusion, RoPE)
├── sh/                              # Launch scripts (workstation or sbatch)
│   ├── train_256.sh  train_512.sh  train_1024.sh
│   └── generate_256.sh  generate_512.sh  generate_1024.sh
├── pyproject.toml                   # Packaging, entry points, ruff/pytest config, deps
├── .github/workflows/ci.yml         # CI: ruff lint + pytest smoke tests
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
