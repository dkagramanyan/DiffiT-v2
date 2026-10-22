"""The launcher checks the combra backbone weights first, before anything is spawned."""

import pytest
from click.testing import CliRunner

import scripts.train as train
from diffit.metrics import HAS_COMBRA

pytestmark = pytest.mark.skipif(not HAS_COMBRA, reason="combra is not installed")

ARGS = ["--outdir", "out", "--cfg", "diffit-256", "--data", "data.zip", "--gpus", "1", "--batch-gpu", "1"]


@pytest.fixture
def launch(monkeypatch):
    """Run the CLI with the dataset check stubbed; record require_backbones / launch_training calls."""
    import combra.metrics

    calls = {"backbones": 0, "launch": 0}
    error = {}

    def fake_require_backbones():
        calls["backbones"] += 1
        if "missing" in error:
            raise RuntimeError(error["missing"])
        return {"fid": "a", "cmmd": "b", "fd_dinov2": "c"}

    def fake_launch_training(**kwargs):
        calls["launch"] += 1

    monkeypatch.setattr(combra.metrics, "require_backbones", fake_require_backbones)
    monkeypatch.setattr(train, "read_class_meta", lambda data: (["a", "b"], None))
    monkeypatch.setattr(train, "launch_training", fake_launch_training)

    def run(*extra, missing=None):
        if missing is not None:
            error["missing"] = missing
        return CliRunner().invoke(train.main, [*ARGS, *extra]), calls

    return run


def test_missing_backbones_abort_before_launch(launch):
    result, calls = launch(missing="run download_models.sh first. Missing:\n  fid: x")
    assert result.exit_code != 0
    assert "download_models.sh" in result.output
    assert calls == {"backbones": 1, "launch": 0}


def test_found_backbones_logged_then_launch(launch):
    result, calls = launch()
    assert result.exit_code == 0, result.output
    assert "Backbone weights found (download_models.sh): fid, cmmd, fd_dinov2" in result.output
    assert calls == {"backbones": 1, "launch": 1}


@pytest.mark.parametrize("extra", [("--combra-metrics", "False"), ("--num-fid-samples", "0")])
def test_not_checked_without_combra_metrics(launch, extra):
    result, calls = launch(*extra, missing="should not be called")
    assert result.exit_code == 0, result.output
    assert calls == {"backbones": 0, "launch": 1}
