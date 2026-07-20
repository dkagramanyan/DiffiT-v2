#!/usr/bin/env python
"""Driver harness for DiffiT-v2 — build, drive, and eyeball the model.

DiffiT-v2 has no GUI: it's a latent-diffusion transformer driven by a `click`
CLI. This driver gives a future agent two programmatic handles on the running
model, matching the two layers PRs here actually touch:

  smoke     Direct-invocation. Builds a REAL DiffiT on the GPU at a tiny config
            (no checkpoint / dataset / VAE needed), runs a forward pass for both
            learn_sigma modes, a training-loss step, and every reverse-diffusion
            sampler (ddim / ddpm / dpm++ / unipc). Prints shapes + finite checks.
            This is the fast inner-loop check for changes to the attention /
            TMSA / RoPE / diffusion / sampler code. ~15 s on an RTX 3090.

  generate  End-to-end. Runs the real `scripts.gen_images` seed-mode pipeline
            from a checkpoint (model load -> sampling -> SD-VAE decode -> PNG),
            then verifies the PNG is a genuine image (non-constant pixels) rather
            than a blank/NaN frame. This is the "look at the output" path.

Run from the repo root inside the `diffit` conda env. Exits non-zero on any
failure so it works as a CI gate.

    python .claude/skills/run-diffit-v2/driver.py smoke
    python .claude/skills/run-diffit-v2/driver.py generate \
        --network ./training-runs/00018-diffit-512-gpus4-batch256/network-snapshot-019251.pt \
        --image-size 512
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Repo root is three levels up (.claude/skills/run-diffit-v2/driver.py). Put it
# first on sys.path so `import diffit` / `-m scripts...` work regardless of cwd
# (the package is used editable-from-source; it is not pip-installed).
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def _p(msg):
    print(f"[driver] {msg}", flush=True)


def cmd_smoke(args):
    import torch

    from diffit import create_diffusion, diffusion_defaults
    from diffit.diffit import DiffiT
    from diffit.metrics import sample_latents

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _p(f"device: {dev}"
       + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else ""))

    # Tiny but architecturally real config: head_dim = 64/4 = 16 (RoPE needs %4).
    cfg = dict(input_size=8, patch_size=2, in_channels=4,
               hidden_size=64, depth=2, num_heads=4, num_classes=10)
    torch.manual_seed(0)

    # --- forward contract, both sigma modes -------------------------------
    for learn_sigma, exp_c in ((True, 8), (False, 4)):
        model = DiffiT(learn_sigma=learn_sigma, **cfg).to(dev).eval()
        x = torch.randn(2, 4, 8, 8, device=dev)
        t = torch.randint(0, 1000, (2,), device=dev)
        y = torch.randint(0, 10, (2,), device=dev)
        with torch.no_grad():
            out = model(x, t, y)
        assert out.shape == (2, exp_c, 8, 8), out.shape
        assert torch.isfinite(out).all()
        _p(f"forward learn_sigma={learn_sigma}: out {tuple(out.shape)} finite OK")

    # --- training loss step ----------------------------------------------
    model = DiffiT(learn_sigma=True, **cfg).to(dev).eval()
    diff = create_diffusion(**diffusion_defaults())
    x = torch.randn(2, 4, 8, 8, device=dev)
    t = torch.randint(0, 1000, (2,), device=dev)
    y = torch.randint(0, 10, (2,), device=dev)
    losses = diff.training_losses(model, x, t, {"y": y})
    assert torch.isfinite(losses["loss"]).all()
    _p(f"training_losses: loss={losses['loss'].mean().item():.4f} finite OK")

    # --- every sampler through the real generation dispatcher -------------
    # sample_latents is the exact inner loop scripts/gen_images.py uses.
    shape = (2, 4, 8, 8)
    noise = torch.randn(*shape, device=dev)

    def model_fn(xt, ts):
        return model(xt, ts, torch.zeros(xt.shape[0], dtype=torch.long, device=dev))

    for sampler in ("dpm++", "unipc", "ddim", "ddpm"):
        dc = diffusion_defaults()
        # dpm++/unipc need the full 1000-step schedule; ddim/ddpm want the
        # step count baked into the SpacedDiffusion via respacing.
        dc["timestep_respacing"] = "" if sampler in ("dpm++", "unipc") else "8"
        d = create_diffusion(**dc)
        out = sample_latents(model_fn, d, shape, dev, sampler=sampler,
                             num_steps=8, model_kwargs={}, noise=noise)
        assert out.shape == shape, (sampler, out.shape)
        assert torch.isfinite(out).all(), sampler
        _p(f"sampler {sampler:>5}: latent {tuple(out.shape)} finite OK")

    _p("SMOKE PASSED")


def cmd_generate(args):
    import numpy as np
    import PIL.Image

    out = Path(args.outdir)
    n_seeds = args.seeds
    _p(f"generating {n_seeds} image(s) at {args.image_size}px from {args.network}")
    cmd = [
        sys.executable, "-m", "scripts.gen_images",
        "--network", args.network,
        "--seeds", f"0-{n_seeds - 1}",
        "--class-idx", str(args.class_idx),
        "--save-mode", "dir",
        "--outdir", str(out),
        "--image-size", str(args.image_size),
        "--sampler", args.sampler,
        "--steps", str(args.steps),
        "--cfg-scale", str(args.cfg_scale),
    ]
    # HF_HUB_OFFLINE keeps the cached SD-VAE from hitting the network.
    env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    import os
    r = subprocess.run(cmd, env={**os.environ, **env}, cwd=str(REPO_ROOT))
    if r.returncode != 0:
        _p("gen_images FAILED")
        sys.exit(1)

    pngs = sorted(out.glob("seed*.png"))
    assert pngs, f"no PNGs written to {out}"
    for png in pngs:
        arr = np.asarray(PIL.Image.open(png))
        std = float(arr.std())
        assert arr.shape == (args.image_size, args.image_size, 3), arr.shape
        assert std > 1.0, f"{png.name} looks blank/constant (std={std:.3f})"
        _p(f"{png.name}: {arr.shape} pixel-std={std:.1f} -> real image OK")
    _p(f"GENERATE PASSED — {len(pngs)} image(s) in {out}")


def main():
    ap = argparse.ArgumentParser(description="DiffiT-v2 driver harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("smoke", help="direct-invocation model/sampler smoke (no ckpt)")

    g = sub.add_parser("generate", help="end-to-end checkpoint -> PNG + verify")
    g.add_argument("--network", required=True, help="path to a DiffiT snapshot .pt")
    g.add_argument("--outdir", default="/tmp/diffit_driver_gen")
    g.add_argument("--seeds", type=int, default=2, help="number of images (seeds 0..N-1)")
    g.add_argument("--class-idx", type=int, default=0)
    g.add_argument("--image-size", type=int, default=512)
    g.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm", "dpm++", "unipc"])
    g.add_argument("--steps", type=int, default=30)
    g.add_argument("--cfg-scale", type=float, default=1.49)

    args = ap.parse_args()
    {"smoke": cmd_smoke, "generate": cmd_generate}[args.cmd](args)


if __name__ == "__main__":
    main()
