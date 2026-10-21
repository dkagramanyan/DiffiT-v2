"""The row this repo writes to ``stats.jsonl`` must be readable by combra.

``combra.metrics.load_fid_by_kimg`` reads ``Metrics/combra_fid`` and
``Progress/kimg`` from the same JSON line, and shape-filters away any record whose
values are not plain scalars -- silently, returning ``{}``. That is exactly how
san-v2 and StyleSwin runs produced an unreadable metric history while every
combra-side test passed: the reader was tested against a synthetic *flat* row, and
nothing tested the producer.

This is the producer half. It builds the real row through the training loop's own
function, so a change to the row shape fails here instead of silently emptying the
analysis layer.
"""

import importlib.util
import json
import math

import pytest

pytest.importorskip("torch")  # the training-loop module imports torch at module level

requires_combra = pytest.mark.skipif(
    importlib.util.find_spec("combra") is None, reason="combra is not installed"
)


def _row():
    from scripts.train import build_stats_row

    row = build_stats_row(
        {"Loss/train": 1.25}, kimg=403.2, tick=7,
        sec_per_tick=1.0, sec_per_kimg=2.0, total_sec=10.0, maintenance_sec=0.5,
        cpu_mem_gb=1.0, gpu_mem_gb=2.0, gpu_reserved_gb=3.0, lr=1e-4,
    )
    # The training loop merges the combra metrics into this same row.
    row["Metrics/combra_fid"] = 12.5
    return row



def test_row_contains_only_json_scalars():
    for key, value in _row().items():
        assert isinstance(value, (int, float, str)), (
            f"{key} is {type(value).__name__}, not a JSON scalar -- "
            "load_fid_by_kimg will shape-filter this record away"
        )


@requires_combra
def test_row_round_trips_through_load_fid_by_kimg(tmp_path):
    from combra.metrics import load_fid_by_kimg

    path = tmp_path / "stats.jsonl"
    path.write_text(json.dumps(_row()) + "\n")
    assert load_fid_by_kimg(str(path)) == {"000403": 12.5}


def test_jsonl_line_writes_non_finite_as_null():
    # A bare NaN token is not JSON; the row is sanitized and dumped with allow_nan=False.
    from scripts.train import stats_jsonl_line

    row = dict(_row(), **{"Loss/train": math.nan, "Metrics/combra_fid_best": math.inf})
    fields = json.loads(stats_jsonl_line(row, now=1000.5, start_time=1000.0))
    assert fields["Loss/train"] is None and fields["Metrics/combra_fid_best"] is None
    assert fields["Metrics/combra_fid"] == 12.5  # metrics live in the tick row
    assert fields["timestamp"] == 1000.5 and fields["wall_time"] == 0.5
    assert isinstance(fields["datetime"], str)


def test_metric_sections_reach_jsonl_and_log():
    # combra >= 0.25: two angle sections (v1 = P0, v2 = P6) next to the image metrics.
    # The same tag strings are the stats.jsonl keys, the TensorBoard tags and (minus
    # the section) the .log keys; one .log line per section, Metrics: first.
    from diffit.metrics import metric_tag
    from scripts.train import metrics_log_lines, stats_jsonl_line

    evaluated = {"combra_v1_w1": 1.0, "combra_v1_amp2": math.nan, "combra_v2_w1": 2.0,
                 "combra_v2_pi": 0.25, "combra_fid": 30.0, "combra_cmmd": 0.5,
                 "combra_fd_dinov2": 400.0, "combra_num_fid_samples": 48.0}
    row = {"Loss/train": 1.25, "Progress/kimg": 1.0}
    for name, value in evaluated.items():
        row[metric_tag(name)] = value
    row["Metrics/combra_fid_best"] = math.nan  # no finite FID yet: nan, not inf

    fields = json.loads(stats_jsonl_line(row, now=1.0, start_time=0.0))
    assert fields["Metrics_v1/combra_w1"] == 1.0 and fields["Metrics_v1/combra_amp2"] is None
    assert fields["Metrics_v2/combra_pi"] == 0.25 and fields["Metrics/combra_fid"] == 30.0
    assert not any(k.startswith(("Metrics/combra_v", "Metrics_v1/combra_fid")) for k in fields)

    assert metrics_log_lines(row) == [
        "Metrics: combra_fid 30.0000  combra_cmmd 0.5000  combra_fd_dinov2 400.0000  "
        "combra_num_fid_samples 48.0000  combra_fid_best nan",
        "Metrics v1: combra_w1 1.0000  combra_amp2 nan",
        "Metrics v2: combra_w1 2.0000  combra_pi 0.2500",
    ]
    # The Inception suite (combra off) stays one Metrics: line.
    assert metric_tag("FID") == "Metrics/FID"
    assert metrics_log_lines({metric_tag("IS"): 2.0, metric_tag("FID"): 9.0}) == ["Metrics: IS 2.0000  FID 9.0000"]


def test_tick_status_line_format():
    from scripts.train import format_time, tick_status_line

    assert [format_time(s) for s in (12, 184, 3723, 183840)] == [
        "12s", "3m 04s", "1h 02m 03s", "2d 03h 04m"]
    line = tick_status_line(
        tick=7, kimg=403.2, total_sec=184, sec_per_tick=1.0, sec_per_kimg=2.0,
        maintenance_sec=0.5, cpu_mem_gb=1.0, gpu_mem_gb=2.0, gpu_reserved_gb=3.0,
    )
    assert line == ("tick 7     kimg 403.2     time 3m 04s       sec/tick 1.0      "
                    "sec/kimg 2.00     maintenance 0.5    cpumem 1.00   "
                    "gpumem 2.00   reserved 3.00  ")
