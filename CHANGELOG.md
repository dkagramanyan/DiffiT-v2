# Changelog

All notable changes to this fork (`DiffiT-v2`) are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [3.2.0] — 2026-08-27

### Fixed
- **Eval-tick latents came from the ambient per-rank RNG.** `torch.randn` /
  `torch.randint` with no generator meant a specific fake batch was not reproducible
  from `--seed`, differed with `--gpus`, and changed every tick — unlike the GAN
  repos, which fix their eval latents at startup. Sample `i`'s noise and class now
  come from a CPU generator seeded by `seed + i` alone (`diffit.metrics._eval_draw`,
  threaded from the loop's `--seed`), so the eval set is identical at any world
  size and on every tick, and any subset reproduces in isolation (§2).

- **`tests/test_combra_contract.py` asserted combra symbols the training loop no
  longer imports.** Its `REQUIRED` list still named the eight feature / angle
  functions from before the sharded harness moved into combra, and never named
  `combra.metrics.distributed`'s `all_ranks_ok` / `distributed_metrics` /
  `gather_generated` / `precompute_reference` — the four symbols the loop actually
  depends on. That is the exact blind spot the test exists to close (combra 0.5.0
  removing three functions hid for a release the same way). It now pins
  `(module, name)` pairs for every combra import in the repo and the unguarded
  import block mirrors the loop's real imports.

- **A zero-sample rank crashed `RankH5Writer.close()` with a `KeyError`.** When
  `--gpus` exceeds `--samples-per-class`, a rank's index block is empty, so no
  batch is ever written and `_init` never runs — but `close()` indexed the
  per-class datasets unconditionally. Such a rank now deletes its group-less
  shard and returns cleanly (the same fix san-v2 shipped), and both the merge
  and the `--no-merge` validator tolerate the absent file: a missing shard for
  a rank that owned zero samples is legitimate, while one that owned samples
  still hard-fails as a crashed worker.

- **A `class_names` list shorter than the class set silently dropped
  `class_name` group attrs.** The writer and the merge stamped `class_name`
  only for classes inside the list, conditionally omitting the attr §4 says is
  always present. Both now raise up front when any selected class has no name.

- **Rank 0 raising before a collective hung every other rank.** Two startup paths
  aborted under `is_main`: the incompatible-combra `RuntimeError`, and the metrics
  smoke test. Both sit immediately before `precompute_combra_reference`, whose
  `all_reduce` the surviving ranks were already blocked in — so a misconfigured
  install surfaced as an NCCL watchdog timeout instead of the error that caused it.
  The incompatible-combra check no longer gates its raise on `is_main` (the condition
  is rank-uniform, so all ranks raise together, and only the warning stays rank-0),
  and the smoke test's failure is agreed through `all_ranks_ok` before every rank
  raises together.

- **`--no-merge` skipped the incomplete-shard check entirely.** The hard-fail on
  missing samples lived only inside the merge, so with `--no-merge` a crashed run
  left shards on disk with nonzero `missing_count` looking finished. Rank 0 now
  recomputes missing slots from the `written` masks after generation even when
  merging is skipped, and raises naming each incomplete shard and its count.
- **`--seeds` failed under the default `--save-mode hdf5`.** Seed mode can only write
  a directory, but the flag defaults to `hdf5`, so every first attempt errored naming a
  flag the caller never set. It now falls back to `dir` when the default was left
  alone, and still refuses an explicit `--save-mode hdf5`.
- **Pre-v2 snapshots lost their resolution.** Bare EMA `state_dict`s carry no
  `n_classes` / `resolution` / `class_names`, and DiffiT's axial RoPE loads at any
  size, so a wrong `--image-size` produced noise rather than an error. `gen_images`
  now reads `training_options.json` from the snapshot's own run directory -- written
  by `train.py` all along -- uses its `image_size` when the flag was left at its
  default, and errors on an explicit contradiction. No checkpoint was rewritten.
- **Generation blocked on a Hub call before consulting the cache.** `_load_vae` tries
  the local cache first, announces any download before starting it, and names the
  cache plus `diffit-download-models` if that fails.
- **combra is pinned to a tag (`@v0.10.0`) instead of tracking `main`.** Unpinned, every
  fresh env resolved whatever combra `main` was that day, so the FID / CMMD / FD-DINOv2 /
  angle numbers a run is judged on could change with no signal and no record. combra
  0.8.0 also stamps `combra/version` into this run's TensorBoard HPARAMS, so the metric
  code behind a run is now recoverable from its log. Local development is unaffected --
  the env's editable combra install shadows the URL.
- **Console scripts are now covered by a packaging test** (`tests/test_entry_points.py`).
  It launches every entry point declared in `[project.scripts]` with `--help` from a
  temp cwd, which is the only way to see this class of bug: pytest runs with the repo
  root on `sys.path`, so an in-repo test passes while the installed script is broken.
  Confirmed to fail against the pre-fix packaging before being kept.
- **`stats.jsonl` rows are built by a testable function**, and a new
  `tests/test_stats_contract.py` feeds a real row to `combra.metrics.load_fid_by_kimg`.
  The reader was only ever tested against a synthetic flat row, so nothing checked the
  producer.
- **The §7 logging contract is now asserted** (`tests/test_logging_contract.py`).
  Thirteen scalar keys had drifted across the four repos; nothing failed because
  nothing checked. See below for this repo's share.

### Changed
- **`class_names` is mandatory in the generated-images h5.** `RankH5Writer` and
  the shard merge now raise when the checkpoint carries no `class_names`, instead
  of silently writing an h5 with no class attribution.
- **The sharded eval harness moved into combra** (`combra.metrics.distributed`). This
  repo kept only what is model-specific: producing a shard of generated images and the
  float->uint8 denormalisation. The four private copies had drifted three ways --
  `all_gather` vs `gather`, a failure flag or none, and a different
  `precompute_reference` signature in each.
- **The combra startup check is `self_test(image_metrics=True, strict=True, images=...)`.**
  A missing CLIP download previously surfaced only as a whole run logging `nan`.
- **Hyperparameters reach TensorBoard.** The resolved config is read back from
  `training_options.json` at the end of training and written to the HPARAMS tab with
  the run's final `Metrics/combra_fid_best`, so runs are comparable by configuration
  and not only by curve shape. Nothing logged them before.
- **§7 keys:** `Resources/gpu_mem_gb` / `gpu_reserved_gb` renamed to
  `Resources/peak_gpu_mem_gb` / `peak_gpu_mem_reserved_gb` to match the other three;
  `Timing/total_sec` and `Timing/maintenance_sec` added; the `datetime` column now uses
  `%Y-%m-%d %H:%M:%S` rather than ISO-8601, so all four `stats.jsonl` files parse the
  same way.

- **The combra contract test fed a unimodal sample to a bimodal-fit metric.**
  `test_angle_metrics_run_on_pooled_angles` drew two near-identical normals
  (mu 120 and 126), so the second Gaussian had no mode to sit on. combra now
  reports that as `nan` rather than dividing by the phantom, which turned the
  assertion red. The fixture is now genuinely bimodal (a 70/30 mixture at
  100 deg and 240 deg), which is what a WC-Co vertex-angle distribution
  actually looks like.
- **`scipy.linalg.sqrtm(..., disp=False)` raises under SciPy >= 1.18**, which
  removed the `disp` parameter. Fixed in `diffit/metrics.py` and `scripts/evaluator.py`. Calling `sqrtm(X)` without `disp` returns
  the matrix alone on every SciPy version, so the fix is version-agnostic. This
  surfaced when the environment moved to SciPy 1.18 (see below); before that the
  call would have failed at runtime the moment anyone upgraded.

- Dropped the `scipy<=1.14.1` ceiling from `pyproject.toml`. The CHANGELOG
  recorded this cap as removed when `requirements.txt` went away, but it
  survived in `pyproject.toml` and directly contradicted combra's `scipy>=1.18`.
- `REQUIRED` in the contract test listed `self_test` (never called) and omitted
  `compute_all_metrics` (imported by `combra_smoke_test`), so the guard could not
  have caught a rename of the one symbol this repo actually depends on.

### Changed
- **The conda environment is now `diffit-v2`** (Python 3.12, torch 2.13+cu130,
  numpy 2.5, SciPy 1.18), rebuilt alongside the previous `diffit` env rather
  than replacing it. `requires-python` has said `>=3.12` since the v2 convention
  landed, but the working env was still 3.11 — so `pip install -e .` could not
  succeed, which is why the console scripts were missing and combra was absent.
  README and `sh/` launch scripts point at the new name.
- **CI installs combra and arms the contract test.** `tests/test_combra_contract.py`
  is entirely `skipif(not combra_installed)`, and no CI job installed combra, so the
  file could go green by doing nothing. CI now installs combra when a `COMBRA_TOKEN`
  secret is present and sets `COMBRA_REQUIRED=1`; a new always-on test fails if
  combra is missing under that flag.

### Removed
- **The legacy writer `experiments/notebooks/generate_class_samples.py`.** It
  stamped `format="diffit_generated_images"` with no `schema_version` /
  `class_names` and used a per-batch seed convention incompatible with §4. Use
  `diffit-gen-images --save-mode hdf5` instead.

- **The legacy generation notebook** (`experiments/notebooks/generate_class_samples.ipynb`).
  It still carried the pre-v2 writer — `format='diffit_generated_images'`, no
  `schema_version`/`class_names`, and a different seed convention — which the
  removal of `generate_class_samples.py` was supposed to retire; downstream
  readers emit `UnknownFormatWarning` on its output. `diffit-gen-images
  --save-mode hdf5` is the one generation path.

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

## Pre-release development — 2026-07-16

> Kept for history. This predates the 3.0.0 release below and was previously
> mislabelled `[Unreleased]`, which put shipped work under a heading implying it
> was pending. Much of it — `--resume`, `best_model.pkl`,
> `--save-inference-only`, `network-snapshot-latest.pt` — was **removed** by the
> v2 convention, so read it as a record of what changed then, not as current
> behaviour.

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

## Pre-release development — 2026-06-25

> Kept for history. This predates the 3.0.0 release below and was previously
> mislabelled `[Unreleased]`, which put shipped work under a heading implying it
> was pending. Much of it — `--resume`, `best_model.pkl`,
> `--save-inference-only`, `network-snapshot-latest.pt` — was **removed** by the
> v2 convention, so read it as a record of what changed then, not as current
> behaviour.

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
