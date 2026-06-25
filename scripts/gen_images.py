"""
Generate images using a pretrained DiffiT model.

Two run modes:
  seed mode      — pass --seeds, one image per seed (random or fixed class)
  per-class mode — pass --samples-per-class, generates N images for every
                   class (or for --classes subset)

Two save modes:
  hdf5  (default, per-class only) — san-v2-style per-rank shards merged into
                                    one big HDF5. Layout matches
                                    san-v2/gen_images.py exactly:
        <outdir>/shards/rank_NNN.h5    (one per GPU worker, no write lock)
        <outdir>/generated.h5          (merged; --no-merge to skip)
  dir                              — per-image PNG files

The san-v2 technique: each GPU worker writes its own shard with NO
cross-thread coordination on disk (h5py is not thread-safe with shared
handles). After workers finish, a single-threaded merge pass collates
shards into one HDF5, deterministically resolving overlap by "first shard
with written=True wins."

Multi-GPU: one model + VAE copy per device, sample indices block-split
per class across devices (matches san-v2's _split_indices_block), one
worker thread per device.

Usage:
    # per-class → HDF5 (default)
    python gen_images.py --model-path ckpt.pt \\
        --samples-per-class 1000 --batch-size 32 \\
        --image-size 256 --outdir ./out --gpus 0,1

    # per-class → PNGs
    python gen_images.py --model-path ckpt.pt --save-mode dir \\
        --samples-per-class 100 --classes 0,1,207,999 \\
        --batch-size 32 --outdir ./out

    # seed mode (always PNGs)
    python gen_images.py --model-path ckpt.pt --seeds 0-49 \\
        --image-size 256 --cfg-scale 4.4 --outdir ./out --gpus 0,1
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple, Union

import click
import h5py
import numpy as np
import PIL.Image
import torch
from diffusers.models import AutoencoderKL
from tqdm.auto import tqdm

import diffit.diffit as diffit_module
from diffit import create_diffusion, diffusion_defaults
from diffit.constants import PIXEL_NORM_HALF, UINT8_MAX, VAE_SCALE_FACTOR
from diffit.dist_util import extract_inference_state_dict, load_state_dict
from diffit.metrics import sample_latents


def parse_range(s: Union[str, List, None]) -> List[int]:
    """Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]
    """
    if s is None or s == "":
        return []
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


def parse_gpus(s: Union[str, List, None]) -> List[int]:
    """Parse '0,1,2' or '0-3' into a list of GPU ids. None → all available."""
    if s is None or s == "":
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return []
    if isinstance(s, list):
        return [int(x) for x in s]
    return parse_range(s)


def _split_indices_block(n: int, rank: int, world_size: int) -> np.ndarray:
    """Contiguous block split — every rank gets ~equal workload.
    Matches san-v2/gen_images.py exactly.
    """
    base = n // world_size
    rem = n % world_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    if start >= end:
        return np.empty((0,), dtype=np.int64)
    return np.arange(start, end, dtype=np.int64)


# ---------------------------------------------------------------------------
# Per-rank HDF5 shard writer (san-v2 technique: no shared file handle, no
# write lock — each worker thread owns its own shard)
# ---------------------------------------------------------------------------

class RankH5Writer:
    def __init__(
        self,
        shard_path: Path,
        classes: List[int],
        samples_per_class: int,
        compression: Optional[str],
        chunk_images: int,
    ):
        self.shard_path = Path(shard_path)
        self.classes = [int(c) for c in classes]
        self.samples_per_class = int(samples_per_class)
        self.compression = compression
        self.chunk_images = int(chunk_images)

        self.f: Optional[h5py.File] = None
        self.initialized = False
        self.img_shape: Optional[Tuple[int, int, int]] = None

        self.d_images: Dict[int, h5py.Dataset] = {}
        self.d_seeds: Dict[int, h5py.Dataset] = {}
        self.d_written: Dict[int, h5py.Dataset] = {}

    def open(self):
        self.shard_path.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(str(self.shard_path), "w")

    def _init(self, img_shape: Tuple[int, int, int]):
        assert self.f is not None
        H, W, C = (int(x) for x in img_shape)
        self.img_shape = (H, W, C)

        self.f.attrs["format"] = "diffit_generated_images_shard"
        self.f.attrs["image_shape_hwc"] = self.img_shape
        self.f.attrs["samples_per_class"] = int(self.samples_per_class)
        self.f.attrs["classes"] = np.array(self.classes, dtype=np.int32)

        chunks0 = max(1, min(self.chunk_images, self.samples_per_class))
        chunks_meta = max(1, min(chunks0 * 4, self.samples_per_class))

        for c in self.classes:
            g = self.f.create_group(f"class_{c}")
            g.attrs["class_idx"] = int(c)
            g.attrs["samples_per_class"] = int(self.samples_per_class)
            g.attrs["image_shape_hwc"] = self.img_shape

            self.d_images[c] = g.create_dataset(
                "images",
                shape=(self.samples_per_class, H, W, C),
                dtype=np.uint8,
                chunks=(chunks0, H, W, C),
                compression=self.compression if self.compression else None,
                shuffle=bool(self.compression),
            )
            self.d_seeds[c] = g.create_dataset(
                "seeds",
                shape=(self.samples_per_class,),
                dtype=np.int64,
                chunks=(chunks_meta,),
            )
            self.d_written[c] = g.create_dataset(
                "written",
                shape=(self.samples_per_class,),
                dtype=np.bool_,
                chunks=(chunks_meta,),
            )
            self.d_written[c][:] = False

        self.initialized = True
        self.f.flush()

    def write_batch(self, class_idx, sample_idxs, seeds, images):
        assert self.f is not None
        class_idx = int(class_idx)

        sample_idxs = np.asarray(sample_idxs, dtype=np.int64)
        seeds = np.asarray(seeds, dtype=np.int64)
        images = np.asarray(images, dtype=np.uint8)

        if not self.initialized:
            if images.ndim != 4:
                raise RuntimeError(f"Expected images (B,H,W,C), got {images.shape}")
            self._init(tuple(images.shape[1:]))

        if tuple(images.shape[1:]) != tuple(self.img_shape):
            raise RuntimeError(
                f"Shape mismatch: got {images.shape[1:]}, expected {self.img_shape}"
            )
        if sample_idxs.size != images.shape[0] or seeds.size != images.shape[0]:
            raise RuntimeError("idxs/seeds/images batch size mismatch")

        order = np.argsort(sample_idxs)
        sample_idxs = sample_idxs[order]
        seeds = seeds[order]
        images = images[order]

        self.d_images[class_idx][sample_idxs, :, :, :] = images
        self.d_seeds[class_idx][sample_idxs] = seeds
        self.d_written[class_idx][sample_idxs] = True
        self.f.flush()

    def close(self):
        if self.f is None:
            return
        for c in self.classes:
            written = int(np.count_nonzero(self.d_written[c][:]))
            grp = self.f[f"class_{c}"]
            grp.attrs["written_count"] = written
            grp.attrs["missing_count"] = int(self.samples_per_class - written)
        self.f.flush()
        self.f.close()
        self.f = None


# ---------------------------------------------------------------------------
# Merge per-rank shards into one big HDF5 (rank0 step in san-v2; here just
# runs after all workers finish since we're single-process)
# ---------------------------------------------------------------------------

def _merge_shards_to_one_h5(
    merged_path: Path,
    shards_dir: Path,
    classes: List[int],
    samples_per_class: int,
    compression: Optional[str],
    chunk_images: int,
    world_size: int,
    extra_attrs: Optional[Dict[str, object]] = None,
):
    """Combine shards into one HDF5. Deterministic: for each (class, idx),
    take the first shard whose written[idx]=True.
    """
    merged_path.parent.mkdir(parents=True, exist_ok=True)

    shard_files = [
        h5py.File(str(shards_dir / f"rank_{r:03d}.h5"), "r")
        for r in range(world_size)
    ]
    try:
        img_shape = None
        for sf in shard_files:
            if "image_shape_hwc" in sf.attrs:
                img_shape = tuple(sf.attrs["image_shape_hwc"])
                break
        if img_shape is None:
            raise RuntimeError("No shard contains image shape metadata.")

        H, W, C = (int(x) for x in img_shape)

        with h5py.File(str(merged_path), "w") as out:
            out.attrs["format"] = "diffit_generated_images"
            out.attrs["image_shape_hwc"] = (H, W, C)
            out.attrs["samples_per_class"] = int(samples_per_class)
            out.attrs["classes"] = np.array([int(c) for c in classes], dtype=np.int32)
            out.attrs["world_size"] = int(world_size)
            out.attrs["merged_from"] = str(shards_dir)
            if extra_attrs:
                for k, v in extra_attrs.items():
                    out.attrs[k] = v

            chunks0 = max(1, min(int(chunk_images), int(samples_per_class)))
            chunks_meta = max(1, min(chunks0 * 4, int(samples_per_class)))

            for c in classes:
                c = int(c)
                g = out.create_group(f"class_{c}")
                g.attrs["class_idx"] = c
                g.attrs["samples_per_class"] = int(samples_per_class)
                g.attrs["image_shape_hwc"] = (H, W, C)

                dimg = g.create_dataset(
                    "images",
                    shape=(samples_per_class, H, W, C),
                    dtype=np.uint8,
                    chunks=(chunks0, H, W, C),
                    compression=compression if compression else None,
                    shuffle=bool(compression),
                )
                dseed = g.create_dataset(
                    "seeds", shape=(samples_per_class,), dtype=np.int64,
                    chunks=(chunks_meta,),
                )
                dw = g.create_dataset(
                    "written", shape=(samples_per_class,), dtype=np.bool_,
                    chunks=(chunks_meta,),
                )
                dw[:] = False

                for r, sf in enumerate(shard_files):
                    grp = sf.get(f"class_{c}", None)
                    if grp is None:
                        continue
                    wmask = np.asarray(grp["written"][:], dtype=bool)
                    if not wmask.any():
                        continue
                    need = wmask & (~dw[:])
                    if not need.any():
                        continue
                    idxs = np.nonzero(need)[0]
                    dimg[idxs] = grp["images"][idxs]
                    dseed[idxs] = grp["seeds"][idxs]
                    dw[idxs] = True
                    out.flush()

                g.attrs["written_count"] = int(np.count_nonzero(dw[:]))
                g.attrs["missing_count"] = int(samples_per_class - g.attrs["written_count"])
                out.flush()
    finally:
        for sf in shard_files:
            sf.close()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _run_sampling(model, vae, diffusion, dev, latent_size, num_classes,
                  bs, class_labels, gen, *, cfg_scale, scale_pow,
                  diffusion_steps, sampler, num_sampling_steps):
    """Single forward pass: bs latents → bs decoded uint8 NHWC images."""
    z = torch.randn(bs, 4, latent_size, latent_size, device=dev, generator=gen)
    classes_null = torch.full((bs,), num_classes, device=dev, dtype=torch.long)

    z_cfg = torch.cat([z, z], 0)
    model_kwargs = {
        "y": torch.cat([class_labels, classes_null], 0),
        "cfg_scale": cfg_scale,
        "diffusion_steps": diffusion_steps,
        "scale_pow": scale_pow,
    }

    sample = sample_latents(
        model.forward_with_cfg,
        diffusion,
        z_cfg.shape,
        dev,
        sampler=sampler,
        num_steps=num_sampling_steps,
        model_kwargs=model_kwargs,
        noise=z_cfg,
    )
    sample, _ = sample.chunk(2, dim=0)

    sample = vae.decode(sample / VAE_SCALE_FACTOR).sample
    sample = ((sample + 1) * PIXEL_NORM_HALF).clamp(0, UINT8_MAX).to(torch.uint8)
    return sample.permute(0, 2, 3, 1).cpu().numpy()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--model-path", required=True, type=str, help="Path to model checkpoint")
@click.option("--seeds", type=parse_range, default=None, help="Seed mode: list of seeds (e.g., '0,1,4-6')")
@click.option("--samples-per-class", type=int, default=None, help="Per-class mode: N images per class")
@click.option("--classes", type=parse_range, default=None, help="Per-class mode: subset (default: all 0..num_classes-1)")
@click.option("--batch-size", type=int, default=32, show_default=True, help="Per-class mode: forward-pass batch size per GPU")
@click.option("--base-seed", type=int, default=0, show_default=True, help="Per-class mode: seed = base_seed + cls*1e6 + first_sample_idx")
@click.option("--outdir", required=True, type=str, help="Output directory (shards + merged HDF5 in hdf5 mode; PNGs in dir mode)", metavar="DIR")
@click.option("--save-mode", type=click.Choice(["hdf5", "dir"], case_sensitive=False), default="hdf5", show_default=True, help="Output format")
@click.option("--merge/--no-merge", default=True, show_default=True, help="hdf5 mode: merge per-rank shards into one HDF5")
@click.option("--output-hdf5", type=str, default=None, help="Merged HDF5 filename (default: <outdir>/generated.h5)")
@click.option("--hdf5-compression", type=click.Choice(["lzf", "gzip", "none"], case_sensitive=False), default="lzf", show_default=True)
@click.option("--hdf5-chunk-images", type=int, default=256, show_default=True)
@click.option("--image-size", type=int, default=256, show_default=True, help="Image resolution (256 or 512)")
@click.option("--model", "model_name", type=str, default="Diffit", show_default=True, help="Model constructor name")
@click.option("--class-idx", type=int, default=None, help="Seed mode: fixed class label (random if not specified)")
@click.option("--cfg-scale", type=float, default=4.4, show_default=True, help="Classifier-free guidance scale")
@click.option("--num-sampling-steps", type=int, default=250, show_default=True, help="Number of diffusion sampling steps")
@click.option("--sampler", type=click.Choice(["dpm++", "ddim", "ddpm"]), default="ddim", show_default=True, help="Reverse-diffusion sampler")
@click.option("--scale-pow", type=float, default=4.0, show_default=True, help="Power for cosine CFG schedule (256)")
@click.option("--vae-decoder", type=click.Choice(["ema", "mse"]), default="ema", show_default=True, help="VAE decoder variant")
@click.option("--decode-layer", type=int, default=None, help="Decode layer override")
@click.option("--batch-sz", type=int, default=1, show_default=True, help="Seed mode: images per seed (sharing the seed)")
@click.option("--num-classes", type=int, default=None, help="Override num_classes (default: auto-detect from checkpoint)")
@click.option("--gpus", type=parse_gpus, default=None, help="GPU ids to use, e.g. '0,1' or '0-3' (default: all available)")
def generate_images(
    model_path,
    seeds,
    samples_per_class,
    classes,
    batch_size,
    base_seed,
    outdir,
    save_mode,
    merge,
    output_hdf5,
    hdf5_compression,
    hdf5_chunk_images,
    image_size,
    model_name,
    class_idx,
    cfg_scale,
    num_sampling_steps,
    sampler,
    scale_pow,
    vae_decoder,
    decode_layer,
    batch_sz,
    num_classes,
    gpus,
):
    """Generate images using a pretrained DiffiT model."""
    torch.backends.cuda.matmul.allow_tf32 = True
    save_mode = save_mode.lower()
    compression_norm = None if hdf5_compression.lower() == "none" else hdf5_compression.lower()

    # Mode validation
    if (seeds and samples_per_class is not None) or (not seeds and samples_per_class is None):
        raise click.UsageError(
            "Provide exactly one of --seeds (seed mode) or --samples-per-class (per-class mode)."
        )
    per_class_mode = samples_per_class is not None

    if save_mode == "hdf5" and not per_class_mode:
        raise click.UsageError("--save-mode=hdf5 requires per-class mode (--samples-per-class).")

    gpu_ids = gpus if gpus else parse_gpus(None)
    devices = (
        [torch.device(f"cuda:{i}") for i in gpu_ids]
        if gpu_ids else [torch.device("cpu")]
    )
    world_size = len(devices)
    print(f"Devices: {devices}  (world_size={world_size})")

    # One model + VAE per device. Each thread is pinned to one device, so
    # threads never touch the same model.
    print(f'Loading model from "{model_path}" onto {world_size} device(s)...')
    latent_size = image_size // 8
    state = extract_inference_state_dict(load_state_dict(model_path, map_location="cpu"))

    # Auto-detect num_classes from the checkpoint's class embedding table.
    # Last row is the CFG null token, so subtract 1.
    ckpt_num_classes = int(state["y_embedder.embedding_table.weight"].shape[0]) - 1
    if num_classes is None:
        num_classes = ckpt_num_classes
        print(f"Auto-detected num_classes={num_classes} from checkpoint")
    elif num_classes != ckpt_num_classes:
        raise click.UsageError(
            f"--num-classes={num_classes} conflicts with checkpoint "
            f"(embedding table implies num_classes={ckpt_num_classes})"
        )

    models = []
    vaes = {}
    for dev in devices:
        m = diffit_module.__dict__[model_name](
            input_size=latent_size, decode_layer=decode_layer, num_classes=num_classes,
        )
        msg = m.load_state_dict(state)
        m.to(dev).eval()
        models.append(m)

        v = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_decoder}").to(dev).eval()
        for prm in v.parameters():
            prm.requires_grad_(False)
        vaes[dev] = v
    print(f"Model loaded: {msg}")
    del state

    # DPM-Solver++ subsamples the full 1000-step schedule itself; DDIM/DDPM use a
    # spaced schedule whose num_timesteps == num_sampling_steps.
    diff_config = diffusion_defaults()
    diff_config["timestep_respacing"] = "" if sampler == "dpm++" else str(num_sampling_steps)
    diffusion = create_diffusion(**diff_config)
    diffusion_steps = diff_config["diffusion_steps"]

    outdir_p = Path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build per-rank job lists (san-v2 block split per class)
    # ------------------------------------------------------------------
    if per_class_mode:
        cls_list = classes if classes else list(range(num_classes))

        # For each (rank, class), the rank owns a contiguous block of
        # sample indices [block_start, block_end). Then we chunk that
        # block by --batch-size.
        jobs_per_gpu: List[List[Tuple[int, np.ndarray, int]]] = [[] for _ in devices]
        for cls in cls_list:
            for rank in range(world_size):
                my_ids = _split_indices_block(samples_per_class, rank, world_size)
                if my_ids.size == 0:
                    continue
                for start in range(0, my_ids.size, batch_size):
                    batch_ids = my_ids[start:start + batch_size]
                    seed = base_seed + cls * 10**6 + int(batch_ids[0])
                    jobs_per_gpu[rank].append((cls, batch_ids, seed))

        total_images = len(cls_list) * samples_per_class
        unit = "img"
        total_units = total_images
        print(f"Per-class mode: {len(cls_list)} classes × {samples_per_class} samples "
              f"= {total_images} images (block-split across {world_size} workers)")
    else:
        # Seed mode (always dir): round-robin over GPUs.
        seed_jobs = [(int(s),) for s in seeds]
        jobs_per_gpu = [[] for _ in devices]
        for i, job in enumerate(seed_jobs):
            jobs_per_gpu[i % world_size].append(job)
        total_units = len(seeds)
        unit = "seed"
        print(f"Seed mode: {len(seeds)} seeds × batch_sz={batch_sz} = {len(seeds) * batch_sz} images")

    # ------------------------------------------------------------------
    # Per-rank HDF5 shard writers (only for hdf5 mode)
    # ------------------------------------------------------------------
    shards_dir = outdir_p / "shards"
    shard_writers: List[Optional[RankH5Writer]] = [None] * world_size
    if save_mode == "hdf5":
        for rank in range(world_size):
            sw = RankH5Writer(
                shard_path=shards_dir / f"rank_{rank:03d}.h5",
                classes=cls_list,
                samples_per_class=samples_per_class,
                compression=compression_norm,
                chunk_images=hdf5_chunk_images,
            )
            sw.open()
            shard_writers[rank] = sw

    pbar = tqdm(total=total_units, desc="generating", unit=unit, smoothing=0.05)
    pbar_lock = Lock()

    def gpu_worker(gpu_idx):
        dev = devices[gpu_idx]
        model = models[gpu_idx]
        vae = vaes[dev]
        sw = shard_writers[gpu_idx]

        with torch.inference_mode():
            for job in jobs_per_gpu[gpu_idx]:
                if per_class_mode:
                    cls, batch_ids, seed = job
                    bs = int(batch_ids.size)
                    gen = torch.Generator(device=dev).manual_seed(int(seed))
                    class_labels = torch.full((bs,), cls, device=dev, dtype=torch.long)

                    imgs = _run_sampling(
                        model, vae, diffusion, dev, latent_size, num_classes,
                        bs, class_labels, gen,
                        cfg_scale=cfg_scale, scale_pow=scale_pow,
                        diffusion_steps=diffusion_steps, sampler=sampler,
                        num_sampling_steps=num_sampling_steps,
                    )

                    if save_mode == "hdf5":
                        # Each rank owns its own shard handle → no lock.
                        seeds_arr = np.full(bs, seed, dtype=np.int64)
                        sw.write_batch(cls, batch_ids, seeds_arr, imgs)
                    else:
                        cls_dir = outdir_p / f"class_{cls}"
                        cls_dir.mkdir(parents=True, exist_ok=True)
                        for i, img in enumerate(imgs):
                            sample_idx = int(batch_ids[i])
                            fname = f"idx_{sample_idx:06d}_seed_{seed}.png"
                            PIL.Image.fromarray(img, "RGB").save(cls_dir / fname)

                    with pbar_lock:
                        pbar.update(bs)
                else:
                    (seed,) = job
                    gen = torch.Generator(device=dev).manual_seed(seed)

                    if class_idx is not None:
                        class_labels = torch.full((batch_sz,), class_idx, device=dev, dtype=torch.long)
                    else:
                        class_labels = torch.randint(0, num_classes, (batch_sz,),
                                                     device=dev, generator=gen, dtype=torch.long)

                    imgs = _run_sampling(
                        model, vae, diffusion, dev, latent_size, num_classes,
                        batch_sz, class_labels, gen,
                        cfg_scale=cfg_scale, scale_pow=scale_pow,
                        diffusion_steps=diffusion_steps, sampler=sampler,
                        num_sampling_steps=num_sampling_steps,
                    )

                    for i, img in enumerate(imgs):
                        fname = f"seed{seed:04d}"
                        if batch_sz > 1:
                            fname += f"_b{i:02d}"
                        PIL.Image.fromarray(img, "RGB").save(outdir_p / f"{fname}.png")

                    with pbar_lock:
                        pbar.update(1)

    try:
        with ThreadPoolExecutor(max_workers=world_size) as pool:
            futs = [pool.submit(gpu_worker, i) for i in range(world_size)]
            for f in futs:
                f.result()  # surface exceptions
    finally:
        pbar.close()
        for sw in shard_writers:
            if sw is not None:
                sw.close()

    # ------------------------------------------------------------------
    # Optional merge (single-threaded, runs after all workers finish)
    # ------------------------------------------------------------------
    if save_mode == "hdf5" and merge:
        merged_path = Path(output_hdf5) if output_hdf5 else outdir_p / "generated.h5"
        print(f'Merging {world_size} shard(s) → "{merged_path}" ...')
        _merge_shards_to_one_h5(
            merged_path=merged_path,
            shards_dir=shards_dir,
            classes=cls_list,
            samples_per_class=samples_per_class,
            compression=compression_norm,
            chunk_images=hdf5_chunk_images,
            world_size=world_size,
            extra_attrs={
                "checkpoint": os.path.basename(model_path),
                "num_classes": int(num_classes),
                "cfg_scale": float(cfg_scale),
                "num_sampling_steps": int(num_sampling_steps),
                "sampler": sampler,
            },
        )
        size_mb = merged_path.stat().st_size / (1024 * 1024)
        print(f"Merged HDF5: {merged_path} ({size_mb:.1f} MB)")

    print("Done.")


if __name__ == "__main__":
    generate_images()
