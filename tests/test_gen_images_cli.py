"""CLI contract for `diffit-gen-images` mode selection (no model, no GPU).

Seed mode can only write a directory, but `--save-mode` defaults to `hdf5`, so
every first `--seeds` attempt used to fail with
"--save-mode=hdf5 requires per-class mode" -- naming a flag the caller never set.
The default now falls back to `dir`; an *explicit* `--save-mode hdf5` is still a
contradiction and still errors.
"""

import click
from click.testing import CliRunner

from scripts.gen_images import generate_images


def _run(args):
    # --network is loaded after the mode checks, so a nonexistent path is enough
    # to prove which check fired first.
    return CliRunner().invoke(
        generate_images, ["--network", "/nonexistent.pt", "--outdir", "out", *args]
    )


def test_seed_mode_falls_back_to_dir():
    res = _run(["--seeds", "0-1"])
    assert not isinstance(res.exception, click.UsageError), res.output
    assert isinstance(res.exception, FileNotFoundError)  # got past mode selection


def test_explicit_hdf5_in_seed_mode_still_errors():
    res = _run(["--seeds", "0-1", "--save-mode", "hdf5"])
    assert isinstance(res.exception, SystemExit)
    assert "requires per-class mode" in res.output


def test_exactly_one_of_seeds_or_samples_per_class():
    assert "exactly one of" in _run([]).output
    assert "exactly one of" in _run(["--seeds", "0-1", "--samples-per-class", "2"]).output


# --- checkpoint metadata recovery -----------------------------------------------
#
# Old `00018-*` snapshots are bare EMA state_dicts: no n_classes / resolution /
# class_names wrapper. n_classes is recoverable from y_embedder, but the resolution
# is not -- DiffiT's axial RoPE loads at ANY size, so passing the default
# --image-size 256 to a 512 checkpoint produced noise with no error. train.py
# already writes training_options.json next to every snapshot, so the answer was
# on disk all along.

import json

import torch

from scripts.gen_images import _sidecar_meta


def _bare_checkpoint(tmp_path, n_classes=3, image_size=None):
    run = tmp_path / "00018-diffit-512-gpus4-batch256"
    run.mkdir()
    # Only the tensor gen_images reads for the class count; the model is never built
    # in these tests because the resolution check runs first.
    torch.save({"y_embedder.embedding_table.weight": torch.zeros(n_classes + 1, 8)},
               run / "network-snapshot-019251.pt")
    if image_size is not None:
        (run / "training_options.json").write_text(json.dumps({"image_size": image_size}))
    return run / "network-snapshot-019251.pt"


def test_sidecar_meta_reads_the_recorded_resolution(tmp_path):
    ckpt = _bare_checkpoint(tmp_path, image_size=512)
    assert _sidecar_meta(str(ckpt)) == {"resolution": 512}


def test_sidecar_meta_tolerates_a_run_dir_without_options(tmp_path):
    ckpt = _bare_checkpoint(tmp_path, image_size=None)
    assert _sidecar_meta(str(ckpt)) == {}


def test_conflicting_image_size_is_an_error(tmp_path):
    ckpt = _bare_checkpoint(tmp_path, image_size=512)
    res = CliRunner().invoke(generate_images, [
        "--network", str(ckpt), "--outdir", str(tmp_path / "out"),
        "--seeds", "0-1", "--image-size", "256",
    ])
    assert isinstance(res.exception, SystemExit)
    assert "conflicts with the resolution recorded" in res.output
