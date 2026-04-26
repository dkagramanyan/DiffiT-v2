"""
Per-class sample generation across DiffiT 256² training checkpoints.

Script form of `experiments/notebooks/generate_class_samples.ipynb`.

For each requested checkpoint kimg, generates `--samples-per-class` images
for every class and writes them to a single HDF5 file. The on-disk layout
matches `san-v2/gen_images.py::RankH5Writer` (per-class groups, with
`images` / `seeds` / `written` datasets, lzf-compressed and chunked) and
the loop is resumable: existing `written=True` indices are skipped.

Multi-GPU: one model + VAE copy is loaded per device, batches are
pre-sharded round-robin across GPUs, and each GPU is driven by a single
worker thread (so two threads never touch the same model).

Usage:
    python experiments/generate_class_samples.py \\
        --run-dir=/.../training-runs/00017-diffit-256-gpus2-batch192 \\
        --kimgs 4435 6451 8064 12096 16128 \\
        --samples-per-class=1000 \\
        --batch-size=32 \\
        --gpus 0 1
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import h5py
import numpy as np
import torch
from diffusers.models import AutoencoderKL
from tqdm.auto import tqdm

# Walk up from this file until we find the directory containing the
# `diffit` package — works whether the script lives in experiments/ or
# experiments/notebooks/.
REPO_ROOT = next(
    p for p in Path(__file__).resolve().parents
    if (p / 'diffit' / '__init__.py').exists()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import diffit.diffit as diffit_module
from diffit import create_diffusion, diffusion_defaults


# ---------------------------------------------------------------------------
# Model + sampling helpers
# ---------------------------------------------------------------------------

def load_diffit_model(ckpt_path, device, model_name='Diffit'):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'ema' in ckpt:
        state, image_size = ckpt['ema'], ckpt.get('image_size', 256)
    elif isinstance(ckpt, dict) and 'model' in ckpt:
        state, image_size = ckpt['model'], ckpt.get('image_size', 256)
    else:
        state, image_size = ckpt, 256

    # +1 slot is the CFG null token
    num_classes = state['y_embedder.embedding_table.weight'].shape[0] - 1
    latent_size = image_size // 8

    model = diffit_module.__dict__[model_name](
        input_size=latent_size, num_classes=num_classes,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f'  load_state_dict: missing={len(missing)} unexpected={len(unexpected)}')
    model.to(device).eval()
    return model, image_size, num_classes


@torch.inference_mode()
def generate_class_batch(model, vae, diffusion, device, class_idx, batch_size,
                         latent_size, num_classes, seed, *,
                         cfg_scale, scale_pow, diffusion_steps, use_ddim):
    """Generate a batch of `batch_size` images for a single class on `device`.
    Returns uint8 NHWC numpy."""
    gen = torch.Generator(device=device).manual_seed(int(seed))

    z = torch.randn(batch_size, 4, latent_size, latent_size,
                    device=device, generator=gen)
    classes      = torch.full((batch_size,), class_idx,    device=device, dtype=torch.long)
    classes_null = torch.full((batch_size,), num_classes,  device=device, dtype=torch.long)

    z_cfg = torch.cat([z, z], 0)
    model_kwargs = {
        'y':               torch.cat([classes, classes_null], 0),
        'cfg_scale':       cfg_scale,
        'diffusion_steps': diffusion_steps,
        'scale_pow':       scale_pow,
    }

    sample_fn = diffusion.ddim_sample_loop if use_ddim else diffusion.p_sample_loop
    sample = sample_fn(
        model.forward_with_cfg,
        z_cfg.shape, z_cfg,
        clip_denoised=False, progress=False,
        model_kwargs=model_kwargs, device=device,
    )
    sample, _ = sample.chunk(2, dim=0)

    sample = vae.decode(sample / 0.18215).sample
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    return sample.permute(0, 2, 3, 1).contiguous().cpu().numpy()


# ---------------------------------------------------------------------------
# HDF5 writer (matches san-v2/gen_images.py::RankH5Writer layout, no sharding)
# ---------------------------------------------------------------------------

class H5Writer:
    def __init__(self, h5_path, classes, samples_per_class, image_shape_hwc,
                 compression='lzf', chunk_images=256, kimg=None, checkpoint=None):
        self.h5_path           = Path(h5_path)
        self.classes           = [int(c) for c in classes]
        self.samples_per_class = int(samples_per_class)
        self.image_shape_hwc   = tuple(int(x) for x in image_shape_hwc)
        self.compression       = None if (compression is None or str(compression).lower() == 'none') else str(compression).lower()
        self.chunk_images      = int(chunk_images)
        self.kimg              = kimg
        self.checkpoint        = checkpoint

        self.f = None
        self.d_images  = {}
        self.d_seeds   = {}
        self.d_written = {}

    def __enter__(self):
        self.h5_path.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(str(self.h5_path), 'a')

        H, W, C = self.image_shape_hwc

        self.f.attrs['format']            = 'diffit_generated_images'
        self.f.attrs['image_shape_hwc']   = (H, W, C)
        self.f.attrs['samples_per_class'] = self.samples_per_class
        self.f.attrs['classes']           = np.array(self.classes, dtype=np.int32)
        if self.kimg       is not None: self.f.attrs['kimg']       = int(self.kimg)
        if self.checkpoint is not None: self.f.attrs['checkpoint'] = str(self.checkpoint)

        chunks0     = max(1, min(self.chunk_images,    self.samples_per_class))
        chunks_meta = max(1, min(self.chunk_images * 4, self.samples_per_class))

        for c in self.classes:
            gname = f'class_{c}'
            if gname in self.f:
                g = self.f[gname]
            else:
                g = self.f.create_group(gname)
                g.attrs['class_idx']         = int(c)
                g.attrs['samples_per_class'] = self.samples_per_class
                g.attrs['image_shape_hwc']   = (H, W, C)
                g.create_dataset(
                    'images',
                    shape=(self.samples_per_class, H, W, C),
                    dtype=np.uint8,
                    chunks=(chunks0, H, W, C),
                    compression=self.compression,
                    shuffle=bool(self.compression),
                )
                g.create_dataset('seeds',   shape=(self.samples_per_class,),
                                 dtype=np.int64, chunks=(chunks_meta,))
                g.create_dataset('written', shape=(self.samples_per_class,),
                                 dtype=np.bool_, chunks=(chunks_meta,))
                g['written'][:] = False

            self.d_images[c]  = g['images']
            self.d_seeds[c]   = g['seeds']
            self.d_written[c] = g['written']

        self.f.flush()
        return self

    def written_mask(self, class_idx):
        return np.asarray(self.d_written[int(class_idx)][:], dtype=bool)

    def write_batch(self, class_idx, sample_idxs, seeds, images):
        sample_idxs = np.asarray(sample_idxs, dtype=np.int64)
        seeds       = np.asarray(seeds,       dtype=np.int64)
        images      = np.asarray(images,      dtype=np.uint8)

        if tuple(images.shape[1:]) != tuple(self.image_shape_hwc):
            raise RuntimeError(f'Shape mismatch: got {images.shape[1:]}, expected {self.image_shape_hwc}')
        if sample_idxs.size != images.shape[0] or seeds.size != images.shape[0]:
            raise RuntimeError('idxs/seeds/images batch size mismatch')

        # h5py requires monotonic indices for fancy assignment.
        order = np.argsort(sample_idxs)
        sample_idxs = sample_idxs[order]
        seeds       = seeds[order]
        images      = images[order]

        c = int(class_idx)
        self.d_images[c][sample_idxs, :, :, :] = images
        self.d_seeds[c][sample_idxs]           = seeds
        self.d_written[c][sample_idxs]         = True
        self.f.flush()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.f is None:
            return
        for c in self.classes:
            written = int(np.count_nonzero(self.d_written[c][:]))
            g = self.f[f'class_{c}']
            g.attrs['written_count'] = written
            g.attrs['missing_count'] = self.samples_per_class - written
        self.f.flush()
        self.f.close()
        self.f = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--run-dir', type=Path, required=True,
                   help='Training run directory containing network-snapshot-*.pt files')
    p.add_argument('--out-dir', type=Path, required=True,
                   help='Output directory for kimg_*.h5 files')
    p.add_argument('--kimgs', type=int, nargs='+', required=True,
                   help='Snapshot kimg ids to generate from (e.g. 4435 6451 8064)')

    p.add_argument('--samples-per-class', type=int, default=1000)
    p.add_argument('--batch-size',        type=int, default=32,
                   help='Samples per forward pass per GPU')
    p.add_argument('--num-sampling-steps', type=int, default=250)
    p.add_argument('--cfg-scale',  type=float, default=4.4)
    p.add_argument('--scale-pow',  type=float, default=4.0)
    p.add_argument('--use-ddim',   action='store_true')
    p.add_argument('--vae-decoder', choices=['ema', 'mse'], default='ema')
    p.add_argument('--base-seed',  type=int, default=0,
                   help='seed = base_seed + kimg*1e6 + cls*1e3 + first_sample_idx')

    p.add_argument('--classes', type=int, nargs='+', default=None,
                   help='Subset of classes to generate (default: all classes in checkpoint)')
    p.add_argument('--gpus', type=int, nargs='+', default=None,
                   help='GPU ids to use (default: cuda:0..cuda:1, capped at 2)')

    p.add_argument('--hdf5-compression', choices=['lzf', 'gzip', 'none'], default='lzf')
    p.add_argument('--hdf5-chunk-images', type=int, default=256)

    return p.parse_args()


def main():
    args = parse_args()

    if args.gpus is not None:
        gpu_ids = args.gpus
    elif torch.cuda.is_available():
        gpu_ids = list(range(min(2, torch.cuda.device_count())))
    else:
        gpu_ids = []
    devices = [torch.device(f'cuda:{i}') for i in gpu_ids] if gpu_ids else [torch.device('cpu')]
    torch.backends.cuda.matmul.allow_tf32 = True

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Verify all checkpoints exist before starting a long run.
    ckpt_paths = {}
    for k in args.kimgs:
        p = args.run_dir / f'network-snapshot-{k:06d}.pt'
        if not p.exists():
            raise FileNotFoundError(f'Missing checkpoint: {p}')
        ckpt_paths[k] = p
        print(f'  ok  {p.name}')
    print(f'\nOutput dir: {args.out_dir}')
    print(f'Devices:    {devices}')

    # One VAE per device.
    vaes = {}
    for dev in devices:
        v = AutoencoderKL.from_pretrained(f'stabilityai/sd-vae-ft-{args.vae_decoder}').to(dev).eval()
        for prm in v.parameters():
            prm.requires_grad_(False)
        vaes[dev] = v

    diff_config = diffusion_defaults()
    diff_config['timestep_respacing'] = str(args.num_sampling_steps)
    diffusion = create_diffusion(**diff_config)
    diffusion_steps = diff_config['diffusion_steps']
    print(f'VAE loaded on {len(vaes)} device(s); diffusion ready '
          f'(steps={args.num_sampling_steps}, base_steps={diffusion_steps})')

    compression = None if args.hdf5_compression == 'none' else args.hdf5_compression

    for kimg in args.kimgs:
        ckpt_path = ckpt_paths[kimg]
        h5_path   = args.out_dir / f'kimg_{kimg:06d}.h5'
        print(f'\n=== kimg={kimg}  ({ckpt_path.name}) -> {h5_path.name} ===')

        # Load one model copy per device.
        t_load = time.time()
        models = []
        image_size = num_classes = None
        for dev in devices:
            m, image_size, num_classes = load_diffit_model(str(ckpt_path), dev)
            models.append(m)
        latent_size = image_size // 8
        print(f'  loaded on {len(devices)} device(s) in {time.time() - t_load:.1f}s  '
              f'image_size={image_size}  num_classes={num_classes}  latent_size={latent_size}')

        classes = list(range(num_classes)) if args.classes is None else list(args.classes)

        with H5Writer(
            h5_path,
            classes=classes,
            samples_per_class=args.samples_per_class,
            image_shape_hwc=(image_size, image_size, 3),
            compression=compression,
            chunk_images=args.hdf5_chunk_images,
            kimg=kimg,
            checkpoint=ckpt_path.name,
        ) as writer:
            pending_per_class = {c: np.where(~writer.written_mask(c))[0] for c in classes}
            total_pending = sum(int(p.size) for p in pending_per_class.values())
            already_done  = len(classes) * args.samples_per_class - total_pending
            if already_done:
                print(f'  resuming: {already_done} samples already on disk')

            jobs = []  # (cls, batch_ids, seed)
            for cls in classes:
                pending = pending_per_class[cls]
                for start in range(0, pending.size, args.batch_size):
                    batch_ids = pending[start:start + args.batch_size]
                    seed = args.base_seed + kimg * 10**6 + cls * 10**3 + int(batch_ids[0])
                    jobs.append((cls, batch_ids, seed))

            # Pre-shard jobs across GPUs so each worker thread is pinned to one
            # device — prevents concurrent access to the same model.
            jobs_per_gpu = [[] for _ in devices]
            for i, job in enumerate(jobs):
                jobs_per_gpu[i % len(devices)].append(job)

            write_lock = Lock()
            pbar = tqdm(total=total_pending, desc=f'kimg={kimg}', unit='img', smoothing=0.05)

            def gpu_worker(gpu_idx):
                dev   = devices[gpu_idx]
                mdl   = models[gpu_idx]
                vae_d = vaes[dev]
                for cls, batch_ids, seed in jobs_per_gpu[gpu_idx]:
                    bs = int(batch_ids.size)
                    imgs = generate_class_batch(
                        mdl, vae_d, diffusion, dev, cls, bs,
                        latent_size, num_classes, seed,
                        cfg_scale=args.cfg_scale,
                        scale_pow=args.scale_pow,
                        diffusion_steps=diffusion_steps,
                        use_ddim=args.use_ddim,
                    )
                    seeds_arr = np.full(bs, seed, dtype=np.int64)
                    with write_lock:
                        writer.write_batch(cls, batch_ids, seeds_arr, imgs)
                        pbar.update(bs)

            try:
                with ThreadPoolExecutor(max_workers=len(devices)) as pool:
                    futs = [pool.submit(gpu_worker, i) for i in range(len(devices))]
                    for f in futs:
                        f.result()  # surface exceptions
            finally:
                pbar.close()

        for m in models:
            del m
        torch.cuda.empty_cache()
        print(f'  done kimg={kimg}  ->  {h5_path}')

    print('\nAll checkpoints done.')


if __name__ == '__main__':
    main()
