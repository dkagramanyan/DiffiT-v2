# Changelog

All notable changes to this fork (`DiffiT-v2`) are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [3.1.0] — 2026-08-18

Repairs the combra integration and finishes the click-CLI convergence.

### Fixed
- **combra metrics were silently disabled.** `diffit/metrics.py` imported
  `angle_density_metrics_from_pooled`, `fid_from_features` and
  `fd_dinov2_from_features`, all removed in combra 0.5.0. The module-level
  `except ImportError` set `HAS_COMBRA = False` and training then reported
  *"the `combra` package is not installed"* — a false diagnosis that sent anyone
  debugging it to reinstall a package that was already present. Now imports
  `frechet_from_features` (one helper for both Fréchet metrics); combra >= 0.7.0
  restores `angle_density_metrics_from_pooled`.
- **The startup warning now tells the truth.** An `ImportError` from a combra that
  *is* installed is a version incompatibility, not an absence, and it is now fatal:
  training refuses to start rather than burning a run that will log no metrics.
  A genuinely absent combra still warns and continues, as before.
- **`[combra]` installed a combra with no metric backends.** The extra pulled bare
  `combra`; since combra 0.5.0 the torch / `pytorch-fid` / `open-clip-torch` stack is
  behind `combra[metrics]`, so FID / CMMD / FD-DINOv2 would have returned `nan` even
  after the import fix. Now `combra[metrics] @ git+…`.
- **Angle extraction ran single-threaded.** `images_to_pooled_angles` was called
  without `workers`, leaving the most expensive part of an eval tick on one core.
  It now uses `cpu_count // gpus` (capped at 32), which parallelises without
  oversubscribing a multi-GPU node.

### Changed
- **Metric keys lost the literal `10k`.** `combra_fid10k` was emitted whatever
  `--num-fid-samples` said, so any chart built from it was mislabelled. Keys are now
  bare — `combra_fid`, `combra_cmmd`, `combra_fd_dinov2` — and the count is logged
  once as `combra_num_fid_samples`.
- **`diffit-eval` is a click command.** It was the last argparse entry point, against
  a convention that names click the single source of truth for options and defaults.
  Same flags, same behaviour; `--device` is now validated and missing-input errors are
  proper usage errors.
- **`diffit-download-models` is a click command** too, with a `--skip-cuda-check` flag
  for CPU boxes that only need the weights cached. It previously had no parser at all.
- `requires-python` raised to **3.12** to match combra.

### Added
- `Metrics/combra_fid_best`, the running best FID, in `stats.jsonl`.
- `tests/test_combra_contract.py` — asserts every combra symbol this repo imports
  actually exists. CPU-only, no GPU/dataset/network, so it runs in every CI job. The
  previous guard (`assert isinstance(HAS_COMBRA, bool)`) passed either way, which is
  why the breakage above survived a whole combra release.

### Notes
- **`--cond` is deliberately absent.** DiffiT is class-conditional by construction —
  classifier-free guidance trains against a null class — so an unconditional switch
  would be a fake flag rather than CLI alignment. This is a model-family difference,
  documented as such in the combra docs `models_api` page.


## [3.0.0] — 2026-07-17

Adopts the shared **v2 convention** (`wc_cv` `models_api_proposal` §12). This is
a **breaking** release: interrupted runs can no longer be resumed, and commands
using removed or renamed flags fail.

### Added
- **`--init-weights <snapshot>`** — weights-only warm start for progressive
  higher-resolution finetuning (loads a previous stage's EMA weights, fresh
  optimizer). Replaces the removed `--resume` flow. (`scripts/train.py`)
- **`--precision {fp32,fp16,bf16}`** replacing `--fp32` + `--amp-dtype`
  (GradScaler only for fp16). (`scripts/train.py`)
- **`--combra-ref-count N`** — cap the combra reference to a *seeded random
  subset* of `N` reals (0 = whole dataset). (`scripts/train.py`)
- **`--mirror` / `--bench` `True/False` flags**; boolean flags are now
  `--flag True/False` throughout (no `--x/--no-x` pairs). (`scripts/train.py`)
- **Self-describing checkpoints** — every snapshot embeds
  `{n_classes, resolution, class_names, cur_nimg}`. (`scripts/train.py`)
- **`class_names` in the label contract** — `diffit-prepare-data` writes an
  index-aligned `class_names` into `dataset.json`; it flows into checkpoints and
  generated `.h5`, and `diffit-gen-images --classes` accepts names.
- **Generation self-spawns** — `diffit-gen-images --gpus N` launches one worker
  per GPU via `torch.multiprocessing`; per-image seed
  `base + class·samples_per_class + idx`; `--network` / `--steps` aliases; merged
  `<desc>.h5`; the merge hard-fails on incomplete shards. (`scripts/gen_images.py`)
- **`center-crop-dhariwal`** transform and `diffit-prepare-data` as a click group
  with a `convert` subcommand. (`scripts/dataset_tool_for_imagenet.py`)
- **`sh/` launch scripts** (`train_{256,512,1024}.sh`,
  `generate_{256,512,1024}.sh`) — self-locating repo root, offline-cluster env
  (`HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`), no hardcoded homes/nodelists/accounts.

### Changed
- **Checkpoint scheme (§3)** — exactly one artifact kind:
  `diffit-snapshot-<kimg>-inference.pt` (EMA-only, atomic, written every `--snap`
  tick **and always at the last tick**, pruned to `--snapshot-keep-last`). No
  resume, no `best_model.pt`, no rolling `network-snapshot-latest.pt`, no
  `network-final*.pt`. A fresh run id is always allocated.
- **Dataset item contract (§5)** — the dataset yields uint8 CHW images and
  one-hot float32 labels; normalization moved into the training loop; grayscale→
  RGB conversion is a build-time step (the loader asserts 3 channels). Class count
  and names are read from `dataset.json` (no startup class probe, no label remap).
- **`DistributedSampler` seeded from `--seed`** (multi-GPU data order reproducible).
- **Logging (§7)** — `stats.jsonl` is scalar rows only; TensorBoard tags use the
  `Loss/*` / `LearningRate/*` / `Timing/*` / `Resources/*` / `Metrics/*` / `Fakes`
  namespaces with global step = `cur_nimg`; the event file carries the run-name
  `filename_suffix`; the rank-0 `.log` transcript replaces the vendored logger's
  `progress.csv` / `progress.json`.
- **combra is pulled over `git+https`** via the `[combra]` extra.

### Removed
- **`--resume`**, `--save-inference-only`, `best_model.pt`,
  `network-snapshot-latest.pt`, `network-final*.pt`, the unused `--metrics` flag,
  the `--fp32` / `--amp-dtype` / `--grad-ckpt/--no-grad-ckpt` / `--tf32/--no-tf32`
  / `--cache-in-ram/--no-cache-in-ram` flag forms, and the gen-images
  `--batch-size` / `--batch-sz` / `--output-hdf5` flags and its `ThreadPool`.
- **Hydra** (`train_hydra.py`, `configs/`, `hydra-core` dep), `requirements.txt`,
  the duplicate `diffit/resample.py`, the `sbatch/` collection and the root
  `train_*prod.sbatch` / `run_train.sh`.

### Note
- `scripts/sample.py` (bulk-`.npz` sampler) is retained as **legacy** — outside
  the v2 generation contract, no guarantees.

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
