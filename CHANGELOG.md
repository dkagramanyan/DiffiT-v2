# Changelog

All notable changes to this fork (`DiffiT-v2`) are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased] — 2026-07-16

### Added
- **`--snapshot-keep-last N` training flag** (default `3`) — keeps only the `N`
  newest per-tick `network-snapshot-<kimg>-inference.pt` snapshots, pruning older
  ones so inference-snapshot history stays bounded. `0` keeps everything. Never
  touches `best_model.pt`, `network-snapshot-latest.pt`, or `network-final*.pt`.
  (`scripts/train.py`)
- **`best_model.pt`** — a full resumable checkpoint refreshed only when FID
  improves (`combra_fid10k` in combra mode, else `FID`). Written in **both**
  checkpoint modes, so a full resume anchor always exists even under
  `--save-inference-only`. (`scripts/train.py`)

### Changed
- **Checkpointing reworked to "best of both".** Every snapshot tick now writes a
  small G_ema `network-snapshot-<kimg>-inference.pt` for history (pruned to
  `--snapshot-keep-last`) **plus** full checkpoints that never accumulate: a single
  rolling `network-snapshot-latest.pt` overwritten in place each tick (atomic
  temp-file + `os.replace`), and `best_model.pt`. Previously the full checkpoint was
  one file **per tick** (`network-snapshot-<kimg>.pt`), which accumulated
  unbounded. (`scripts/train.py`)
- **`--save-inference-only` semantics** — now means "skip the rolling
  `network-snapshot-latest.pt`". Per-tick inference snapshots and the full
  `best_model.pt` are still written, so the mode remains resumable (it no longer
  leaves a run without any full checkpoint). (`scripts/train.py`)
- **Final save** — always writes a full `network-final.pt` (plus
  `network-final-inference.pt`) regardless of `--save-inference-only`, so
  progressive higher-resolution stages can always `--resume` from the previous
  stage's final checkpoint.
- **Library pins bumped** — `timm==0.9.16` → `timm>=1.0.11` (only `PatchEmbed` is
  used, a stable import) and dropped the `scipy<=1.14.1` upper cap (only stable
  `scipy.linalg` / `scipy.special.softmax` APIs are used). (`requirements.txt`,
  `pyproject.toml`)
- **sbatch / docs** — the `sbatch/h200_train_2_gpu_*` scripts pass
  `--snapshot-keep-last 3` and document resuming from `best_model.pt`; the
  `train_*h200_*_prod.sbatch` scripts explicitly pass `--save-inference-only=0`
  so each tick refreshes the full rolling `network-snapshot-latest.pt` their
  SLURM-dependency chaining resumes from; the DiffiT example doc and README
  describe the new checkpoint layout.

## [Unreleased] — 2026-06-25

### Added
- **`--combra-metrics` training flag** (default `true`) — computes the combra
  generative-quality metrics (`cmmd`, `fd_dinov2`, …) each snapshot tick,
  **independent of `--num-fid-samples`** (the Inception FID/IS path). combra runs
  only when the flag is on **and** the package is installed; logged to TensorBoard
  under `Metrics/combra_*`. (`scripts/train.py`)
- **Startup warning** when `--combra-metrics=true` but the `combra` package is not
  installed (instead of silently skipping), with a hint to pass
  `--combra-metrics=false` to silence it.
- **`--save-inference-only` training flag** (default `false`) — additionally writes
  a tiny `network-snapshot-<kimg>-inference.pt` (and `network-final-inference.pt`)
  containing only the EMA weights, the smallest artifact for `gen_images.py` /
  `sample.py`. (`scripts/train.py`)
- **`extract_inference_state_dict()` helper** (`diffit/dist_util.py`) — normalises
  anything the loaders read (a full checkpoint dict, an older bare EMA
  `state_dict`, or a `*-inference.pt` file) down to the EMA weights, so
  `gen_images.py` and `sample.py` load every snapshot format transparently.
- **combra backbones in the pre-download step** — `scripts/download_models.py`
  pre-fetches combra's CLIP / DINOv2 / FID backbones when combra is installed
  (skips cleanly otherwise).
- **`download_models.sh`** — pure `wget`/`curl`/`git` prefetch of the torch-hub /
  CLIP weights (and the VAEs via `huggingface-cli` when present) into the standard
  caches, for offline compute nodes with no Python environment.
- **2× H200 sbatch train scripts** for 256×256 / 512×512 / 1024×1024
  (`sbatch/h200/h200_train_2_gpu_*.sbatch`) — self-contained (resolve the repo
  root, so submittable from `sbatch/`), queue on the **`rocky` partition**, and
  pass `--save-inference-only True` / `--combra-metrics True`.
- **Hydra entry point** — `train_hydra.py` + `configs/config.yaml`. Defaults are
  derived by introspecting the `train.py` click CLI (single source of truth), so
  `configs/config.yaml` only declares the required fields and new flags propagate
  automatically. Both entry points call the same `train.launch_from_opts`.

### Changed
- **Snapshots are now full resumable checkpoints by default.**
  `network-snapshot-<kimg>.pt` / `network-final.pt` now hold
  `{model, ema, opt, scaler?, cur_nimg}` to match the `--resume` loader — **this
  fixes resume**, which previously failed because snapshots saved only the EMA
  `state_dict` while `--resume` expected the full dict. (Trade-off: snapshots are
  ~4–5× larger; use `--save-inference-only` for the small inference artifact.)
- **combra evaluation gating** — the whole-dataset eval reference is now selected
  by `use_combra = --combra-metrics and combra-installed`, rather than purely by
  whether combra happens to be importable.
- **`train.py` refactor** — the body of the click `main()` moved into a reusable
  `launch_from_opts(opts)` so the click and Hydra entry points share one code path.
- **Install** — added `hydra-core>=1.3` to `requirements.txt` and
  `pyproject.toml`.
- **Docs** — README documents the two new flags, the full-checkpoint snapshot
  format, the Hydra entry point, and the `download_models.sh` offline path.

### Notes
- DiffiT-v2 is a latent-diffusion model, so several san-v2 changelog items do not
  apply and were intentionally **not** ported: `legacy.load_network_pkl` G_ema
  mirroring (no G/D/G_ema split), the `timm==0.4.12` pin (DiffiT requires
  `timm>=1.0.11`), the `imgui`/`glfw`/`pyopengl`/`imageio-ffmpeg`/`ninja`
  removals (never present), the `test.py`→`tests/test_cuda_ops.py` move (no custom
  CUDA ops), and the FFHQ-leftover removals (ImageNet-based).
