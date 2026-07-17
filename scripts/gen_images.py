"""
Generate images using a pretrained DiffiT model (v2 generation contract, §4).

Two run modes:
  per-class mode — pass --samples-per-class, generate N images for every class
                   (or the --classes subset, given as indices or class names)
  seed mode      — pass --seeds, one image per seed (random or fixed class)

Two save modes (per-class mode):
  hdf5  (default) — one per-rank shard ``shards/rank_NNN.h5`` in the RankH5Writer
                    layout (``class_<c>/images|seeds``, images uint8 NHWC),
                    merged by rank 0 into ``<desc>.h5``. Every shard and the
                    merged file carry ``format="generated_images_shard"`` and
                    ``schema_version=1`` so downstream code sniffs any model's
                    output identically. The merge hard-fails on incomplete
                    shards.
  dir             — per-image PNGs under ``class_<c>/`` + a ``classes.json``.

Multi-GPU is self-spawning: ``--gpus N`` launches one worker per GPU via
``torch.multiprocessing`` (the same launch model as training — no torchrun, no
thread pool). Sample indices are block-split per class across ranks.

Determinism: ``seed = base-seed + class·samples_per_class + idx`` — every image
has its own seed, so any subset of the output is reproducible in isolation.

Usage:
    diffit-gen-images --network ckpt.pt \\
        --samples-per-class 1000 --batch-gpu 32 \\
        --image-size 256 --outdir ./out --gpus 2 --desc co11

    diffit-gen-images --network ckpt.pt --save-mode dir \\
        --samples-per-class 100 --classes Ultra_Co11,Ultra_Co25 \\
        --batch-gpu 32 --outdir ./out
"""

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import click
import h5py
import numpy as np
import PIL.Image
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL

import diffit.diffit as diffit_module
from diffit import create_diffusion, diffusion_defaults
from diffit.constants import PIXEL_NORM_HALF, UINT8_MAX, VAE_SCALE_FACTOR
from diffit.dist_util import extract_inference_state_dict, load_state_dict
from diffit.metrics import sample_latents

# Unified h5 signature shared by all four v2 repos (§4).
H5_FORMAT = "generated_images_shard"
H5_SCHEMA_VERSION = 1


def parse_range(s: Union[str, List, None]) -> List[int]:
    """Parse '1,2,5-10' into [1, 2, 5, 6, 7, 8, 9, 10]."""
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


def resolve_classes(spec: Optional[str], num_classes: int, class_names: Optional[List[str]]) -> List[int]:
    """Resolve a --classes spec to validated integer indices.

    Accepts indices / ranges (``0,1,4-6``) or class names (``Ultra_Co11,...``),
    validated against the checkpoint's ``n_classes`` / ``class_names`` metadata.
    ``None``/empty selects every class.
    """
    if spec is None or spec == "":
        return list(range(num_classes))
    name_to_idx = {n: i for i, n in enumerate(class_names)} if class_names else {}
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        m = re.match(r"^(\d+)-(\d+)$", tok)
        if m:
            out.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        elif tok.isdigit():
            out.append(int(tok))
        elif tok in name_to_idx:
            out.append(name_to_idx[tok])
        else:
            raise click.UsageError(
                f"--classes: {tok!r} is neither an index nor a known class name "
                f"({class_names})"
            )
    for c in out:
        if not (0 <= c < num_classes):
            raise click.UsageError(f"--classes: index {c} out of range [0, {num_classes})")
    return out


def _split_indices_block(n: int, rank: int, world_size: int) -> np.ndarray:
    """Contiguous block split — every rank gets ~equal workload."""
    base = n // world_size
    rem = n % world_size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    if start >= end:
        return np.empty((0,), dtype=np.int64)
    return np.arange(start, end, dtype=np.int64)


# ---------------------------------------------------------------------------
# Per-rank HDF5 shard writer (no shared handle, no write lock)
# ---------------------------------------------------------------------------

class RankH5Writer:
    def __init__(self, shard_path, classes, samples_per_class, compression, chunk_images,
                 class_names=None):
        self.shard_path = Path(shard_path)
        self.classes = [int(c) for c in classes]
        self.samples_per_class = int(samples_per_class)
        self.compression = compression
        self.chunk_images = int(chunk_images)
        self.class_names = class_names
        self.f: Optional[h5py.File] = None
        self.initialized = False
        self.img_shape: Optional[Tuple[int, int, int]] = None
        self.d_images: Dict[int, h5py.Dataset] = {}
        self.d_seeds: Dict[int, h5py.Dataset] = {}
        self.d_written: Dict[int, h5py.Dataset] = {}

    def open(self):
        self.shard_path.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(str(self.shard_path), "w")

    def _init(self, img_shape):
        assert self.f is not None
        H, W, C = (int(x) for x in img_shape)
        self.img_shape = (H, W, C)

        self.f.attrs["format"] = H5_FORMAT
        self.f.attrs["schema_version"] = H5_SCHEMA_VERSION
        self.f.attrs["image_shape_hwc"] = self.img_shape
        self.f.attrs["samples_per_class"] = int(self.samples_per_class)
        self.f.attrs["classes"] = np.array(self.classes, dtype=np.int32)
        if self.class_names is not None:
            self.f.attrs["class_names"] = np.array(self.class_names, dtype=h5py.string_dtype())

        chunks0 = max(1, min(self.chunk_images, self.samples_per_class))
        chunks_meta = max(1, min(chunks0 * 4, self.samples_per_class))

        for c in self.classes:
            g = self.f.create_group(f"class_{c}")
            g.attrs["class_idx"] = int(c)
            g.attrs["samples_per_class"] = int(self.samples_per_class)
            g.attrs["image_shape_hwc"] = self.img_shape
            if self.class_names is not None and c < len(self.class_names):
                g.attrs["class_name"] = self.class_names[c]

            self.d_images[c] = g.create_dataset(
                "images", shape=(self.samples_per_class, H, W, C), dtype=np.uint8,
                chunks=(chunks0, H, W, C),
                compression=self.compression if self.compression else None,
                shuffle=bool(self.compression),
            )
            self.d_seeds[c] = g.create_dataset(
                "seeds", shape=(self.samples_per_class,), dtype=np.int64, chunks=(chunks_meta,),
            )
            self.d_written[c] = g.create_dataset(
                "written", shape=(self.samples_per_class,), dtype=np.bool_, chunks=(chunks_meta,),
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
            raise RuntimeError(f"Shape mismatch: got {images.shape[1:]}, expected {self.img_shape}")
        if sample_idxs.size != images.shape[0] or seeds.size != images.shape[0]:
            raise RuntimeError("idxs/seeds/images batch size mismatch")

        order = np.argsort(sample_idxs)
        sample_idxs, seeds, images = sample_idxs[order], seeds[order], images[order]
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


def _merge_shards_to_one_h5(merged_path, shards_dir, classes, samples_per_class,
                            compression, chunk_images, world_size, class_names, extra_attrs):
    """Merge shards into one HDF5 (first shard with written=True wins).

    Hard-fails if any (class, idx) slot is unwritten across all shards — a
    crashed generation run must not silently feed zero-filled black images into
    the downstream angle pipeline (§4).
    """
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    shard_files = [h5py.File(str(shards_dir / f"rank_{r:03d}.h5"), "r") for r in range(world_size)]
    try:
        img_shape = None
        for sf in shard_files:
            if "image_shape_hwc" in sf.attrs:
                img_shape = tuple(sf.attrs["image_shape_hwc"])
                break
        if img_shape is None:
            raise RuntimeError("No shard contains image shape metadata.")
        H, W, C = (int(x) for x in img_shape)

        total_missing = 0
        with h5py.File(str(merged_path), "w") as out:
            out.attrs["format"] = H5_FORMAT
            out.attrs["schema_version"] = H5_SCHEMA_VERSION
            out.attrs["image_shape_hwc"] = (H, W, C)
            out.attrs["samples_per_class"] = int(samples_per_class)
            out.attrs["classes"] = np.array([int(c) for c in classes], dtype=np.int32)
            out.attrs["world_size"] = int(world_size)
            if class_names is not None:
                out.attrs["class_names"] = np.array(class_names, dtype=h5py.string_dtype())
            for k, v in (extra_attrs or {}).items():
                out.attrs[k] = v

            chunks0 = max(1, min(int(chunk_images), int(samples_per_class)))
            chunks_meta = max(1, min(chunks0 * 4, int(samples_per_class)))

            for c in classes:
                c = int(c)
                g = out.create_group(f"class_{c}")
                g.attrs["class_idx"] = c
                g.attrs["samples_per_class"] = int(samples_per_class)
                g.attrs["image_shape_hwc"] = (H, W, C)
                if class_names is not None and c < len(class_names):
                    g.attrs["class_name"] = class_names[c]

                dimg = g.create_dataset(
                    "images", shape=(samples_per_class, H, W, C), dtype=np.uint8,
                    chunks=(chunks0, H, W, C),
                    compression=compression if compression else None,
                    shuffle=bool(compression),
                )
                dseed = g.create_dataset("seeds", shape=(samples_per_class,), dtype=np.int64, chunks=(chunks_meta,))
                dw = g.create_dataset("written", shape=(samples_per_class,), dtype=np.bool_, chunks=(chunks_meta,))
                dw[:] = False

                for sf in shard_files:
                    grp = sf.get(f"class_{c}", None)
                    if grp is None:
                        continue
                    wmask = np.asarray(grp["written"][:], dtype=bool)
                    need = wmask & (~dw[:])
                    if not need.any():
                        continue
                    idxs = np.nonzero(need)[0]
                    dimg[idxs] = grp["images"][idxs]
                    dseed[idxs] = grp["seeds"][idxs]
                    dw[idxs] = True
                    out.flush()

                written = int(np.count_nonzero(dw[:]))
                missing = int(samples_per_class - written)
                g.attrs["written_count"] = written
                g.attrs["missing_count"] = missing
                total_missing += missing
                out.flush()

        if total_missing:
            os.remove(merged_path)
            raise RuntimeError(
                f"Refusing to write {merged_path.name}: {total_missing} image slot(s) "
                "were never generated (incomplete shards) — the merge hard-fails on "
                "missing samples (§4)."
            )
    finally:
        for sf in shard_files:
            sf.close()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _run_sampling(model, vae, diffusion, dev, num_classes, z, class_labels,
                  *, cfg_scale, scale_pow, diffusion_steps, sampler, num_sampling_steps):
    """Sample a batch from precomputed per-image noise ``z`` → uint8 NHWC images."""
    bs = z.shape[0]
    classes_null = torch.full((bs,), num_classes, device=dev, dtype=torch.long)
    z_cfg = torch.cat([z, z], 0)
    model_kwargs = {
        "y": torch.cat([class_labels, classes_null], 0),
        "cfg_scale": cfg_scale,
        "diffusion_steps": diffusion_steps,
        "scale_pow": scale_pow,
    }
    sample = sample_latents(
        model.forward_with_cfg, diffusion, z_cfg.shape, dev,
        sampler=sampler, num_steps=num_sampling_steps, model_kwargs=model_kwargs, noise=z_cfg,
    )
    sample, _ = sample.chunk(2, dim=0)
    sample = vae.decode(sample / VAE_SCALE_FACTOR).sample
    sample = ((sample + 1) * PIXEL_NORM_HALF).clamp(0, UINT8_MAX).to(torch.uint8)
    return sample.permute(0, 2, 3, 1).cpu().numpy()


def _make_noise(dev, latent_size, per_image_seeds):
    """Stack per-image latents, each drawn from its own seeded generator so any
    single image is reproducible independently of batch size / GPU count (§4)."""
    zs = []
    for s in per_image_seeds:
        g = torch.Generator(device=dev).manual_seed(int(s))
        zs.append(torch.randn(1, 4, latent_size, latent_size, device=dev, generator=g))
    return torch.cat(zs, 0)


# ---------------------------------------------------------------------------
# Worker (one per GPU, self-spawned)
# ---------------------------------------------------------------------------

def worker_fn(rank, c, temp_dir):
    world_size = c["gpus"]
    dev = torch.device(f"cuda:{rank}") if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    if world_size > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, ".torch_distributed_init"))
        dist.init_process_group(backend="nccl", init_method=f"file://{init_file}",
                                rank=rank, world_size=world_size)

    latent_size = c["image_size"] // 8
    num_classes = c["num_classes"]

    model = diffit_module.__dict__[c["model_name"]](
        input_size=latent_size, decode_layer=c["decode_layer"], num_classes=num_classes,
    )
    model.load_state_dict(extract_inference_state_dict(load_state_dict(c["network"], map_location="cpu")))
    model.to(dev).eval()

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{c['vae_decoder']}").to(dev).eval()
    for prm in vae.parameters():
        prm.requires_grad_(False)

    diff_config = diffusion_defaults()
    diff_config["timestep_respacing"] = "" if c["sampler"] == "dpm++" else str(c["num_sampling_steps"])
    diffusion = create_diffusion(**diff_config)
    diffusion_steps = diff_config["diffusion_steps"]

    outdir_p = Path(c["outdir"])
    cls_list = c["classes"]
    spc = c["samples_per_class"]
    base_seed = c["base_seed"]
    batch_gpu = c["batch_gpu"]

    sw = None
    if c["save_mode"] == "hdf5":
        sw = RankH5Writer(
            shard_path=outdir_p / "shards" / f"rank_{rank:03d}.h5",
            classes=cls_list, samples_per_class=spc,
            compression=c["compression"], chunk_images=c["chunk_images"],
            class_names=c["class_names"],
        )
        sw.open()

    with torch.inference_mode():
        for cls in cls_list:
            my_ids = _split_indices_block(spc, rank, world_size)
            for start in range(0, my_ids.size, batch_gpu):
                batch_ids = my_ids[start:start + batch_gpu]
                # Per-image seed: base + class·samples_per_class + idx (§4).
                seeds = [base_seed + cls * spc + int(i) for i in batch_ids]
                z = _make_noise(dev, latent_size, seeds)
                class_labels = torch.full((len(batch_ids),), cls, device=dev, dtype=torch.long)
                imgs = _run_sampling(
                    model, vae, diffusion, dev, num_classes, z, class_labels,
                    cfg_scale=c["cfg_scale"], scale_pow=c["scale_pow"],
                    diffusion_steps=diffusion_steps, sampler=c["sampler"],
                    num_sampling_steps=c["num_sampling_steps"],
                )
                if c["save_mode"] == "hdf5":
                    sw.write_batch(cls, batch_ids, np.asarray(seeds, dtype=np.int64), imgs)
                else:
                    cls_dir = outdir_p / f"class_{cls}"
                    cls_dir.mkdir(parents=True, exist_ok=True)
                    for i, img in enumerate(imgs):
                        sidx = int(batch_ids[i])
                        PIL.Image.fromarray(img, "RGB").save(cls_dir / f"idx_{sidx:06d}_seed_{seeds[i]}.png")

    if sw is not None:
        sw.close()

    if world_size > 1:
        dist.barrier()
        if rank == 0:
            _rank0_merge(c)
        if dist.is_initialized():
            dist.destroy_process_group()
    else:
        _rank0_merge(c)


def _rank0_merge(c):
    outdir_p = Path(c["outdir"])
    if c["save_mode"] == "dir":
        # dir mode: write a classes.json manifest next to the class_<c>/ folders.
        manifest = {"classes": [int(x) for x in c["classes"]]}
        if c["class_names"] is not None:
            manifest["class_names"] = list(c["class_names"])
        with open(outdir_p / "classes.json", "wt") as f:
            json.dump(manifest, f, indent=2)
        return
    if not c["merge"]:
        return
    merged_path = outdir_p / f"{c['desc']}.h5"
    print(f'Merging {c["gpus"]} shard(s) → "{merged_path}" ...')
    _merge_shards_to_one_h5(
        merged_path=merged_path, shards_dir=outdir_p / "shards",
        classes=c["classes"], samples_per_class=c["samples_per_class"],
        compression=c["compression"], chunk_images=c["chunk_images"],
        world_size=c["gpus"], class_names=c["class_names"],
        extra_attrs={
            "checkpoint": os.path.basename(c["network"]),
            "num_classes": int(c["num_classes"]),
            "cfg_scale": float(c["cfg_scale"]),
            "num_sampling_steps": int(c["num_sampling_steps"]),
            "sampler": c["sampler"],
        },
    )
    print(f"Merged HDF5: {merged_path}")


# ---------------------------------------------------------------------------
# Seed mode (single process; always PNGs)
# ---------------------------------------------------------------------------

def _run_seed_mode(c, seeds, class_idx):
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    latent_size = c["image_size"] // 8
    num_classes = c["num_classes"]
    model = diffit_module.__dict__[c["model_name"]](
        input_size=latent_size, decode_layer=c["decode_layer"], num_classes=num_classes,
    )
    model.load_state_dict(extract_inference_state_dict(load_state_dict(c["network"], map_location="cpu")))
    model.to(dev).eval()
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{c['vae_decoder']}").to(dev).eval()
    for prm in vae.parameters():
        prm.requires_grad_(False)
    diff_config = diffusion_defaults()
    diff_config["timestep_respacing"] = "" if c["sampler"] == "dpm++" else str(c["num_sampling_steps"])
    diffusion = create_diffusion(**diff_config)
    diffusion_steps = diff_config["diffusion_steps"]
    outdir_p = Path(c["outdir"])
    outdir_p.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for s in seeds:
            z = _make_noise(dev, latent_size, [s])
            if class_idx is not None:
                cls = class_idx
            else:
                g = torch.Generator(device=dev).manual_seed(int(s))
                cls = int(torch.randint(0, num_classes, (1,), device=dev, generator=g).item())
            class_labels = torch.full((1,), cls, device=dev, dtype=torch.long)
            imgs = _run_sampling(
                model, vae, diffusion, dev, num_classes, z, class_labels,
                cfg_scale=c["cfg_scale"], scale_pow=c["scale_pow"],
                diffusion_steps=diffusion_steps, sampler=c["sampler"],
                num_sampling_steps=c["num_sampling_steps"],
            )
            PIL.Image.fromarray(imgs[0], "RGB").save(outdir_p / f"seed{int(s):04d}.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--network", "--model-path", "network", required=True, type=str, help="Path to model checkpoint (--model-path is an alias)")
@click.option("--outdir", required=True, type=str, metavar="DIR", help="Output directory")
@click.option("--samples-per-class", type=int, default=None, help="Per-class mode: N images per class")
@click.option("--classes", type=str, default=None, help="Per-class mode: subset by index or class name (default: all)")
@click.option("--seeds", type=parse_range, default=None, help="Seed mode: list of seeds (e.g. '0,1,4-6')")
@click.option("--class-idx", type=int, default=None, help="Seed mode: fixed class label (random if unset)")
@click.option("--batch-gpu", type=int, default=32, show_default=True, help="Forward-pass batch size per GPU")
@click.option("--gpus", type=int, default=1, show_default=True, help="Number of GPUs (self-spawning workers)")
@click.option("--base-seed", "--seed", "base_seed", type=int, default=0, show_default=True, help="Determinism base: seed = base + class·spc + idx")
@click.option("--save-mode", type=click.Choice(["hdf5", "dir"], case_sensitive=False), default="hdf5", show_default=True)
@click.option("--desc", type=str, default="generated", show_default=True, help="Merged HDF5 filename stem (<desc>.h5)")
@click.option("--merge/--no-merge", default=True, show_default=True, help="hdf5 mode: merge per-rank shards")
@click.option("--hdf5-compression", type=click.Choice(["lzf", "gzip", "none"], case_sensitive=False), default="lzf", show_default=True)
@click.option("--hdf5-chunk-images", type=int, default=256, show_default=True)
@click.option("--image-size", type=int, default=256, show_default=True, help="Image resolution")
@click.option("--model", "model_name", type=str, default="Diffit", show_default=True, help="Model constructor name")
@click.option("--cfg-scale", type=float, default=4.4, show_default=True, help="Classifier-free guidance scale")
@click.option("--steps", "--num-sampling-steps", "num_sampling_steps", type=int, default=250, show_default=True, help="Diffusion sampling steps (--num-sampling-steps is an alias)")
@click.option("--sampler", type=click.Choice(["dpm++", "unipc", "ddim", "ddpm"]), default="ddim", show_default=True)
@click.option("--scale-pow", type=float, default=4.0, show_default=True, help="Power for cosine CFG schedule")
@click.option("--vae-decoder", type=click.Choice(["ema", "mse"]), default="ema", show_default=True)
@click.option("--decode-layer", type=int, default=None, help="Decode layer override")
@click.option("--num-classes", type=int, default=None, help="Override num_classes (default: from checkpoint)")
def generate_images(**kw):
    """Generate images using a pretrained DiffiT model."""
    save_mode = kw["save_mode"].lower()
    compression = None if kw["hdf5_compression"].lower() == "none" else kw["hdf5_compression"].lower()

    seeds = kw["seeds"]
    spc = kw["samples_per_class"]
    if (seeds and spc is not None) or (not seeds and spc is None):
        raise click.UsageError("Provide exactly one of --seeds (seed mode) or --samples-per-class (per-class mode).")
    per_class_mode = spc is not None
    if save_mode == "hdf5" and not per_class_mode:
        raise click.UsageError("--save-mode=hdf5 requires per-class mode (--samples-per-class).")

    # Load checkpoint once (metadata + weights) on CPU.
    raw = load_state_dict(kw["network"], map_location="cpu")
    state = extract_inference_state_dict(raw)
    ckpt_num_classes = int(state["y_embedder.embedding_table.weight"].shape[0]) - 1
    class_names = raw.get("class_names") if isinstance(raw, dict) else None
    num_classes = kw["num_classes"]
    if num_classes is None:
        num_classes = ckpt_num_classes
    elif num_classes != ckpt_num_classes:
        raise click.UsageError(
            f"--num-classes={num_classes} conflicts with checkpoint (implies {ckpt_num_classes})"
        )

    c = dict(
        network=kw["network"], outdir=kw["outdir"],
        model_name=kw["model_name"], decode_layer=kw["decode_layer"],
        num_classes=num_classes, class_names=class_names, image_size=kw["image_size"],
        cfg_scale=kw["cfg_scale"], scale_pow=kw["scale_pow"], sampler=kw["sampler"],
        num_sampling_steps=kw["num_sampling_steps"], vae_decoder=kw["vae_decoder"],
        batch_gpu=kw["batch_gpu"], gpus=max(1, kw["gpus"]), base_seed=kw["base_seed"],
        save_mode=save_mode, desc=kw["desc"], merge=kw["merge"],
        compression=compression, chunk_images=kw["hdf5_chunk_images"],
    )

    Path(kw["outdir"]).mkdir(parents=True, exist_ok=True)

    if not per_class_mode:
        print(f"Seed mode: {len(seeds)} seeds")
        _run_seed_mode(c, seeds, kw["class_idx"])
        print("Done.")
        return

    c["classes"] = resolve_classes(kw["classes"], num_classes, class_names)
    c["samples_per_class"] = spc
    print(f"Per-class mode: {len(c['classes'])} classes × {spc} samples "
          f"(block-split across {c['gpus']} GPU worker(s))")

    torch.multiprocessing.set_start_method("spawn", force=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        if c["gpus"] == 1:
            worker_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=worker_fn, args=(c, temp_dir), nprocs=c["gpus"])
    print("Done.")


if __name__ == "__main__":
    generate_images()
