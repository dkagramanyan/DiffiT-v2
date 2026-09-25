"""--augment: random dihedral augmentation of training items (CPU only)."""

import ast
import inspect
import io
import json
import zipfile
from collections import Counter

import numpy as np
import pytest
import torch
from PIL import Image

from diffit.image_datasets import ZipImageDataset, dihedral_transform, load_data, random_dihedral

GROUP = [(k, f) for f in (False, True) for k in range(4)]


def _asym(res=8, seed=0):
    """A CHW uint8 image with no symmetry, so all 8 transforms differ."""
    return np.random.default_rng(seed).integers(0, 256, (3, res, res), dtype=np.uint8)


def _which(img, out):
    """Index in GROUP of the transform that maps img to out (exactly one)."""
    hits = [i for i, (k, f) in enumerate(GROUP) if np.array_equal(dihedral_transform(img, k, f), out)]
    assert len(hits) == 1, hits
    return hits[0]


def _tiny_zip(path, n=6, res=8, classes=("a", "b", "c")):
    labels = []
    with zipfile.ZipFile(path, "w") as zf:
        for i in range(n):
            buf = io.BytesIO()
            Image.fromarray(_asym(res, seed=i).transpose(1, 2, 0)).save(buf, format="PNG")
            name = f"{i:05d}.png"
            zf.writestr(name, buf.getvalue())
            labels.append([name, i % len(classes)])
        zf.writestr("dataset.json", json.dumps({"labels": labels, "class_names": list(classes)}))
    return str(path)


def test_group_has_eight_distinct_elements_matching_numpy():
    img = _asym()
    outs = [dihedral_transform(img, k, f) for k, f in GROUP]
    assert len({o.tobytes() for o in outs}) == 8
    assert np.array_equal(dihedral_transform(img, 1, False), np.rot90(img, 1, axes=(1, 2)))
    assert np.array_equal(dihedral_transform(img, 0, True), img[:, :, ::-1])
    for o in outs:
        assert o.dtype == np.uint8 and o.shape == img.shape and o.flags.c_contiguous


def test_non_square_rejected():
    with pytest.raises(AssertionError):
        dihedral_transform(np.zeros((3, 4, 6), np.uint8), 1, False)


def test_random_dihedral_uniform_and_seeded():
    img = _asym()
    torch.manual_seed(123)
    seq = [_which(img, random_dihedral(img)) for _ in range(8000)]
    counts = Counter(seq)
    assert set(counts) == set(range(8))
    # 1000 expected per element; binomial sd ~30, so +-150 is a > 4 sigma band.
    assert all(850 <= c <= 1150 for c in counts.values()), counts
    torch.manual_seed(123)
    assert [_which(img, random_dihedral(img)) for _ in range(200)] == seq[:200]


def test_zip_dataset_augment_preserves_dtype_shape_labels(tmp_path):
    zp = _tiny_zip(tmp_path / "d.zip")
    plain = ZipImageDataset(zp, 8, num_classes=3, class_cond=True)
    aug = ZipImageDataset(zp, 8, num_classes=3, class_cond=True, augment=True, cache_in_ram=True)
    torch.manual_seed(0)
    seen = set()
    for _ in range(20):
        for i in range(len(plain)):
            x0, y0 = plain[i]
            x1, y1 = aug[i]
            assert x1.dtype == np.uint8 and x1.shape == x0.shape == (3, 8, 8)
            np.testing.assert_array_equal(y1["y"], y0["y"])
            seen.add(_which(x0, x1))
    assert seen == set(range(8))  # cached bytes, fresh draw every access


def test_augment_off_is_identity(tmp_path):
    zp = _tiny_zip(tmp_path / "d.zip")
    ds = ZipImageDataset(zp, 8, num_classes=3, class_cond=True, augment=False)
    for i in range(len(ds)):
        np.testing.assert_array_equal(ds[i][0], _asym(8, seed=i))


def test_loader_with_workers_is_deterministic_per_seed(tmp_path):
    zp = _tiny_zip(tmp_path / "d.zip")

    def first_batches(seed):
        torch.manual_seed(seed)
        it = load_data(data_dir=zp, batch_size=3, image_size=8, num_classes=3, class_cond=True,
                       num_workers=2, augment=True)
        return [next(it)[0].numpy() for _ in range(4)]

    a, b, c = first_batches(7), first_batches(7), first_batches(8)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, y) for x, y in zip(a, c))


def _training_loop_calls():
    import scripts.train as train

    tree = ast.parse(inspect.getsource(train.training_loop))
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


def _name(call):
    f = call.func
    return f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)


def _kw(call, key):
    return next((k.value for k in call.keywords if k.arg == key), None)


def test_reference_and_only_training_loader_see_augment():
    calls = _training_loop_calls()
    ref = [c for c in calls if _name(c) == "precompute_combra_reference"]
    assert len(ref) == 1
    assert isinstance(_kw(ref[0], "dihedral"), ast.Name) and _kw(ref[0], "dihedral").id == "augment"
    loaders = [c for c in calls if _name(c) == "load_data"]
    augmented = [c for c in loaders if _kw(c, "augment") is not None]
    assert len(augmented) == 1 and _kw(augmented[0], "augment").id == "augment"
    # The augmented one is the distributed training loader; reals grid / reference are not.
    assert _kw(augmented[0], "cache_in_ram") is not None


def test_cli_augment_flag_reaches_config(tmp_path):
    from click.testing import CliRunner

    import scripts.train as train

    zp = _tiny_zip(tmp_path / "d.zip")
    base = ["--outdir", str(tmp_path), "--cfg", "diffit-256", "--data", zp,
            "--gpus", "1", "--batch-gpu", "2", "--dry-run"]
    for extra, want in (([], True), (["--augment", "False"], False)):
        res = CliRunner().invoke(train.main, base + extra)
        assert res.exit_code == 0, res.output
        assert f'"augment": {str(want).lower()}' in res.output
