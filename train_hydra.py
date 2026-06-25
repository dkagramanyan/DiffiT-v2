"""Hydra entry point for DiffiT training.

This is a thin wrapper around ``scripts/train.py``: the click CLI there is the
single source of truth for every option and its default. We introspect those
click options to seed defaults, overlay whatever ``configs/config.yaml`` (and
command-line overrides) provide, then hand the merged dict to the same
``train.launch_from_opts`` the click entry point uses -- so the Hydra and CLI
paths produce identical runs.

Usage:
    python train_hydra.py outdir=./runs cfg=diffit-256 \\
        data=./datasets/imagenet_256x256.zip gpus=2 batch_gpu=42

    # override any train.py option by its Python name (dashes become underscores):
    python train_hydra.py outdir=./runs cfg=diffit-256 data=... gpus=2 batch_gpu=42 \\
        combra_metrics=false save_inference_only=true snap=100
"""

import click
import hydra
from omegaconf import DictConfig, OmegaConf

from scripts import train


def _cli_defaults():
    """Default value of every ``train.py`` click option, keyed by its Python name."""
    return {p.name: p.default for p in train.main.params if isinstance(p, click.Option)}


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Start from the CLI defaults, then overlay the resolved Hydra config so any
    # option the user did not set keeps its train.py default and new flags
    # propagate automatically.
    opts = _cli_defaults()
    opts.update(OmegaConf.to_container(cfg, resolve=True))
    train.launch_from_opts(opts)


if __name__ == "__main__":
    main()
