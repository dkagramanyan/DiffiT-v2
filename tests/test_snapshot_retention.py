"""Snapshot retention: the --snapshot-keep-last N newest snapshots plus the best by
each of combra_fid / combra_fd_dinov2 / combra_cmmd (lower is better, nan ignored);
N = 1 keeps at most 4 files, N = 0 keeps everything."""

import math
import os

import pytest

pytest.importorskip("torch")  # the training-loop module imports torch at module level


def _run(tmp_path, evals, keep_last=1):
    """Replay the training loop's save -> update -> prune sequence over ``evals``."""
    from scripts.train import prune_inference_snapshots, update_best_snapshots

    best = {}
    for kimg, metrics in evals:
        path = os.path.join(tmp_path, f"diffit-snapshot-{kimg:06d}-inference.pt")
        open(path, "wb").close()
        update_best_snapshots(best, metrics, path)
        prune_inference_snapshots(tmp_path, keep_last, best)
        if keep_last:
            n_snaps = sum(f.startswith("diffit-snapshot-") for f in os.listdir(tmp_path))
            assert n_snaps <= keep_last + 3
    return best, sorted(os.listdir(tmp_path))


def _m(fid, fd, cmmd):
    return {"combra_fid": fid, "combra_fd_dinov2": fd, "combra_cmmd": cmmd, "combra_w1": 0.0}


def test_keeps_last_and_best_per_metric(tmp_path):
    best, kept = _run(tmp_path, [
        (200, _m(30.0, 500.0, 0.9)),
        (400, _m(10.0, 600.0, 0.8)),   # best fid
        (600, _m(20.0, 300.0, 0.7)),   # best fd_dinov2
        (800, _m(25.0, 400.0, 0.5)),   # best cmmd
        (1000, _m(40.0, 700.0, 1.0)),  # last, best at nothing
    ])
    assert kept == [f"diffit-snapshot-{k:06d}-inference.pt" for k in (400, 600, 800, 1000)]
    assert {k: os.path.basename(p)[16:22] for k, (_, p) in best.items()} == {
        "combra_fid": "000400", "combra_fd_dinov2": "000600", "combra_cmmd": "000800",
    }


def test_one_snapshot_serves_several_roles(tmp_path):
    _, kept = _run(tmp_path, [
        (200, _m(30.0, 500.0, 0.9)),
        (400, _m(10.0, 300.0, 0.5)),   # best at all three
        (600, _m(20.0, 400.0, 0.7)),
    ])
    assert kept == ["diffit-snapshot-000400-inference.pt", "diffit-snapshot-000600-inference.pt"]


def test_nan_and_missing_metrics_are_ignored(tmp_path):
    best, kept = _run(tmp_path, [
        (200, _m(30.0, math.nan, 0.9)),
        (400, _m(math.nan, 300.0, math.nan)),
        (600, None),                    # failed / disabled eval: last only
        (800, {"combra_fid": 50.0}),
    ])
    assert best["combra_fid"][0] == 30.0
    assert best["combra_fd_dinov2"][0] == 300.0
    assert best["combra_cmmd"][0] == 0.9
    assert kept == [f"diffit-snapshot-{k:06d}-inference.pt" for k in (200, 400, 800)]


def test_no_eval_keeps_only_last(tmp_path):
    best, kept = _run(tmp_path, [(k, None) for k in (200, 400, 600)])
    assert best == {}
    assert kept == ["diffit-snapshot-000600-inference.pt"]


def test_tie_keeps_earlier_and_other_files_untouched(tmp_path):
    (tmp_path / "stats.jsonl").write_text("")
    best, kept = _run(tmp_path, [
        (200, _m(10.0, 300.0, 0.5)),
        (400, _m(10.0, 300.0, 0.5)),
        (600, _m(20.0, 400.0, 0.7)),
    ])
    assert all(os.path.basename(p) == "diffit-snapshot-000200-inference.pt" for _, p in best.values())
    assert kept == ["diffit-snapshot-000200-inference.pt", "diffit-snapshot-000600-inference.pt",
                    "stats.jsonl"]


def test_keep_last_n_newest_plus_best(tmp_path):
    _, kept = _run(tmp_path, [
        (200, _m(10.0, 300.0, 0.5)),   # best at all three
        (400, _m(20.0, 400.0, 0.7)),
        (600, _m(30.0, 500.0, 0.9)),
        (800, _m(40.0, 600.0, 1.0)),
    ], keep_last=2)
    assert kept == [f"diffit-snapshot-{k:06d}-inference.pt" for k in (200, 600, 800)]


def test_keep_last_zero_keeps_everything(tmp_path):
    _, kept = _run(tmp_path, [(k, _m(float(k), float(k), float(k))) for k in (200, 400, 600)],
                   keep_last=0)
    assert len(kept) == 3
