"""h5 writer contract for `diffit-gen-images` (no model, no GPU).

Two guarantees, both cheap to lose silently:
- `class_names` is mandatory — an h5 with no class attribution is useless to the
  downstream angle pipeline, so the writer and the merge refuse to construct one.
- `--no-merge` skips the merge, not the hard-fail on incomplete shards — a
  crashed run must not leave shards on disk that look finished (§4).
"""

import h5py
import numpy as np
import pytest

from scripts.gen_images import RankH5Writer, _merge_shards_to_one_h5, _validate_shards

NAMES = ["Ultra_Co11", "Ultra_Co25"]


def _write_shard(tmp_path, spc=4, upto=None):
    """One rank-0 shard with slots [0, upto) written per class (all if None)."""
    sw = RankH5Writer(
        shard_path=tmp_path / "shards" / "rank_000.h5",
        classes=[0, 1], samples_per_class=spc,
        compression=None, chunk_images=2, class_names=NAMES,
    )
    sw.open()
    n = spc if upto is None else upto
    idxs = np.arange(n)
    imgs = np.zeros((n, 2, 2, 3), dtype=np.uint8)
    for c in (0, 1):
        sw.write_batch(c, idxs, idxs, imgs)
    sw.close()
    return tmp_path / "shards"


def test_writer_refuses_missing_class_names(tmp_path):
    with pytest.raises(ValueError, match="class_names"):
        RankH5Writer(
            shard_path=tmp_path / "rank_000.h5",
            classes=[0], samples_per_class=1,
            compression=None, chunk_images=1, class_names=None,
        )


def test_merge_refuses_missing_class_names(tmp_path):
    shards_dir = _write_shard(tmp_path)
    with pytest.raises(ValueError, match="class_names"):
        _merge_shards_to_one_h5(
            merged_path=tmp_path / "merged.h5", shards_dir=shards_dir,
            classes=[0, 1], samples_per_class=4, compression=None,
            chunk_images=2, world_size=1, class_names=None, extra_attrs=None,
        )


def test_validate_shards_passes_a_complete_shard(tmp_path):
    shards_dir = _write_shard(tmp_path)
    _validate_shards(shards_dir, classes=[0, 1], samples_per_class=4, world_size=1)


def test_validate_shards_raises_on_missing_slots(tmp_path):
    shards_dir = _write_shard(tmp_path, upto=3)  # slot 3 of each class unwritten
    with pytest.raises(RuntimeError, match=r"rank_000\.h5: 2 slot"):
        _validate_shards(shards_dir, classes=[0, 1], samples_per_class=4, world_size=1)


def test_validate_shards_does_not_trust_missing_count_attrs(tmp_path):
    # A crashed run may die after close() stamped missing_count=0; the masks are
    # the ground truth.
    shards_dir = _write_shard(tmp_path)
    with h5py.File(shards_dir / "rank_000.h5", "r+") as f:
        f["class_0/written"][1] = False
    with pytest.raises(RuntimeError, match=r"rank_000\.h5: 1 slot"):
        _validate_shards(shards_dir, classes=[0, 1], samples_per_class=4, world_size=1)


def test_writer_refuses_short_class_names(tmp_path):
    # A name list shorter than the class set silently dropped class_name attrs.
    with pytest.raises(ValueError, match="missing a name"):
        RankH5Writer(
            shard_path=tmp_path / "rank_000.h5",
            classes=[0, 1, 2], samples_per_class=1,
            compression=None, chunk_images=1, class_names=NAMES,
        )
    with pytest.raises(ValueError, match="missing a name"):
        _merge_shards_to_one_h5(
            merged_path=tmp_path / "merged.h5", shards_dir=tmp_path / "shards",
            classes=[0, 1, 2], samples_per_class=1, compression=None,
            chunk_images=1, world_size=1, class_names=NAMES, extra_attrs=None,
        )


def test_zero_sample_rank_deletes_empty_shard_and_merge_succeeds(tmp_path):
    # samples_per_class < world_size leaves rank 1 with an empty index block: its
    # close() must delete the group-less shard (not KeyError), and the merge must
    # tolerate the absent file while still merging every sample from rank 0.
    shards_dir = _write_shard(tmp_path, spc=1)
    empty = RankH5Writer(
        shard_path=shards_dir / "rank_001.h5",
        classes=[0, 1], samples_per_class=1,
        compression=None, chunk_images=1, class_names=NAMES,
    )
    empty.open()
    empty.close()
    assert not (shards_dir / "rank_001.h5").exists()

    merged = tmp_path / "merged.h5"
    _merge_shards_to_one_h5(
        merged_path=merged, shards_dir=shards_dir,
        classes=[0, 1], samples_per_class=1, compression=None,
        chunk_images=1, world_size=2, class_names=NAMES, extra_attrs=None,
    )
    with h5py.File(merged, "r") as f:
        assert bool(f["class_0/written"][:].all()) and bool(f["class_1/written"][:].all())

    # --no-merge's validator must likewise accept the deleted empty shard.
    _validate_shards(shards_dir, classes=[0, 1], samples_per_class=1, world_size=2)


def test_validate_shards_raises_on_missing_shard_that_owned_samples(tmp_path):
    # An absent shard is only legitimate for a zero-sample rank; a rank that
    # owned samples and left no file is a crashed worker.
    shards_dir = _write_shard(tmp_path, spc=4)  # rank 0 of 2 writes only its block
    with pytest.raises(RuntimeError, match=r"rank_001\.h5"):
        _validate_shards(shards_dir, classes=[0, 1], samples_per_class=4, world_size=2)
