---
name: run-diffit-v2
description: Build, run, and drive DiffiT-v2 (latent-diffusion vision transformer). Use when asked to run, start, smoke-test, or generate images with DiffiT, exercise the model/samplers, or confirm a DiffiT-v2 change works end-to-end on the GPU.
---

DiffiT-v2 is a class-conditional **latent-diffusion vision transformer** (561M
DiffiT-XL/2) with no GUI — it's driven by a `click` CLI (`diffit-*` console
scripts, also runnable as `python -m scripts.*`). Drive it with the committed
**`.claude/skills/run-diffit-v2/driver.py`**, which exposes the two layers PRs
here touch: `smoke` (direct-invocation — builds a real model on the GPU and runs
forward + every sampler, no checkpoint/dataset needed) and `generate`
(end-to-end — a real checkpoint → SD-VAE decode → verified PNG).

All paths below are relative to `DiffiT-v2/`. Everything here was run in this
container on an RTX 3090 (24 GB).

## Prerequisites

- **GPU**: an NVIDIA GPU (verified on RTX 3090, 24 GB). CUDA runtime ships with
  the torch wheel — no system CUDA/`nvcc` needed (no custom ops to compile).
- **No `apt-get` packages required** — DiffiT-v2 is pure-Python + PyTorch.
- **conda env `diffit` already exists on this box** with torch 2.11.0+cu130 and
  the repo importable. Activate it and verify:

```bash
source /home/david/anaconda3/etc/profile.d/conda.sh
conda activate diffit
cd /home/david/mnt/ssd_2_sata/phd/DiffiT-v2
python -c "import torch, diffit; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| diffit OK')"
# -> torch 2.11.0+cu130 | cuda True | diffit OK
```

The package is used **editable-from-source** (not `pip install`-ed), so the
`diffit-*` console entry-points are NOT on PATH — invoke via `python -m
scripts.<name>` or through the driver. From a clean machine instead, follow the
README: `conda create -n diffit python=3.12`, `pip3 install torch torchvision
--index-url https://download.pytorch.org/whl/cu132`, then `pip install -e .`
(not exercised here — the env was pre-built).

## Run (agent path) — the driver

**Smoke** (fast, ~4 s, no checkpoint/dataset/VAE). Builds a real DiffiT at a
tiny config on the GPU; checks forward for both `learn_sigma` modes, a training
loss step, and all four samplers through the real `sample_latents` dispatcher:

```bash
python .claude/skills/run-diffit-v2/driver.py smoke
```

Expected tail:

```
[driver] forward learn_sigma=True: out (2, 8, 8, 8) finite OK
[driver] sampler dpm++: latent (2, 4, 8, 8) finite OK
[driver] sampler unipc: latent (2, 4, 8, 8) finite OK
[driver] sampler  ddim: latent (2, 4, 8, 8) finite OK
[driver] sampler  ddpm: latent (2, 4, 8, 8) finite OK
[driver] SMOKE PASSED
```

**Generate** (end-to-end, ~18 s for 2 images at 512²). Runs the real
`scripts.gen_images` seed-mode pipeline from a checkpoint, then asserts each PNG
is a genuine image (correct shape, non-constant pixels — catches blank/NaN
frames):

```bash
python .claude/skills/run-diffit-v2/driver.py generate \
  --network ./training-runs/00018-diffit-512-gpus4-batch256/network-snapshot-019251.pt \
  --image-size 512 --seeds 2 --outdir /tmp/diffit_driver_gen
# [driver] seed0000.png: (512, 512, 3) pixel-std=30.3 -> real image OK
# [driver] GENERATE PASSED — 2 image(s) in /tmp/diffit_driver_gen
```

Then **open a PNG** (`/tmp/diffit_driver_gen/seed0000.png`) to eyeball it — the
checkpoints in `training-runs/00018-*` are the WC-Co microstructure model
(3 classes), so output is a grayscale grainy SEM-style texture, not ImageNet.

The driver is cwd-independent (it locates the repo root from its own path) and
exits non-zero on any failure, so it doubles as a CI gate.

## Run the CLI directly

The driver wraps these; run them raw when you need other modes. Set
`HF_HUB_OFFLINE=1` to use the cached SD-VAE without a network round-trip.

```bash
# individual PNGs (seed mode). --save-mode dir is REQUIRED for --seeds.
HF_HUB_OFFLINE=1 python -m scripts.gen_images \
  --network ./training-runs/00018-diffit-512-gpus4-batch256/network-snapshot-019251.pt \
  --seeds 0-1 --class-idx 0 --save-mode dir --outdir /tmp/gen512 \
  --image-size 512 --sampler ddim --steps 30 --cfg-scale 1.49

# bulk per-class HDF5 (the RankH5Writer layout the wc_cv angle pipeline consumes)
HF_HUB_OFFLINE=1 python -m scripts.gen_images \
  --network ./training-runs/00018-diffit-512-gpus4-batch256/network-snapshot-019251.pt \
  --samples-per-class 2 --classes 0 --save-mode hdf5 --desc demo \
  --outdir /tmp/gen_h5 --image-size 512 --sampler ddim --steps 20 \
  --cfg-scale 1.49 --batch-gpu 2
# -> /tmp/gen_h5/demo.h5  (+ shards/rank_000.h5)

# resolve a training config without launching (validates cfg + dataset path)
python -m scripts.train --outdir /tmp/tr --cfg diffit-512 \
  --data ./datasets/imagenet_9to4_1024x1024_512x512.zip \
  --gpus 2 --batch-gpu 64 -n        # -n / --dry-run: print options and exit
```

## Test

CPU smoke suite (forward contract, RoPE cache, diffusion math, UniPC) — no GPU,
dataset, or weights needed:

```bash
python -m pytest tests/ -q        # 8 passed
```

## Gotchas

- **Console entry-points (`diffit-gen-images`, `diffit-train`, …) are NOT on
  PATH** in the `diffit` env — the package is source-editable, not installed.
  Use `python -m scripts.<name>` (or the driver). The README shows the
  entry-point names assuming `pip install -e .`.
- **Seed mode (`--seeds`) forces `--save-mode dir`.** With the default
  `--save-mode hdf5` it errors: *"--save-mode=hdf5 requires per-class mode."*
  HDF5 output is only for per-class mode (`--samples-per-class`).
- **`num_classes` comes from the checkpoint**, not a flag — it's
  `y_embedder.embedding_table.weight.shape[0] - 1`. The `00018-*` snapshots are
  the 3-class WC-Co model; passing a conflicting `--num-classes` errors out.
- **`--image-size` must match the checkpoint's training resolution** (512 for
  `00018-*`). It sets `latent_size = image_size // 8`; a mismatch loads fine
  (RoPE is resolution-agnostic) but produces garbage. Use 512 for these.
- **These `training-runs/*.pt` are bare EMA `state_dict`s** (an `OrderedDict`,
  no metadata wrapper). The inference loader handles that via
  `extract_inference_state_dict`, but `class_names` is then `None`.
- **SD-VAE decode needs `stabilityai/sd-vae-ft-ema`** (already in
  `~/.cache/huggingface`). Offline nodes: prefetch with `python -m
  scripts.download_models` first.

## Troubleshooting

- `ModuleNotFoundError: No module named 'torch'` — you're in `base`, not the
  `diffit` env. `conda activate diffit`.
- `ModuleNotFoundError: No module named 'diffit'` when running a script by path
  from outside the repo — run from the repo root, use `python -m`, or use the
  driver (it injects the repo root onto `sys.path`).
- Generation hangs on first run fetching the VAE — export `HF_HUB_OFFLINE=1`
  (the driver already does) to force the local cache.
