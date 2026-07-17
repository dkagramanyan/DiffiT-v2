"""Minimal training logger for DiffiT (v2 convention, §7).

Replaces the vendored OpenAI-baselines logger. There is exactly one console
transcript per run — a rank-0-only ``.log`` file named after the run directory,
every line prefixed ``[YYYY-MM-DD HH:MM:SS]`` — plus stdout. No ``progress.csv``
/ ``progress.json`` sidecars (``stats.jsonl`` is the machine-readable source of
truth) and no interleaving of text records into ``stats.jsonl``.

The scalar-accumulation helpers (:func:`logkv_mean` / :func:`dumpkvs`) average
per-tick loss values; they no longer fan out to any file — the training loop
owns writing scalars to ``stats.jsonl`` and TensorBoard.
"""

import datetime
import os
from collections import defaultdict

_STATE = {
    "file": None,      # rank-0 .log file handle (None on other ranks)
    "name2val": defaultdict(float),
    "name2cnt": defaultdict(int),
}


def configure(log_dir, run_name=None, is_main=True):
    """Open the rank-0 ``<run_name>.log`` transcript in ``log_dir``.

    ``is_main=False`` (non-rank-0 workers) logs to stdout only, so the ``.log``
    file is a clean single-rank transcript.
    """
    _STATE["name2val"] = defaultdict(float)
    _STATE["name2cnt"] = defaultdict(int)
    if _STATE["file"] is not None:
        try:
            _STATE["file"].close()
        except Exception:
            pass
        _STATE["file"] = None
    if is_main and log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        fname = f"{run_name}.log" if run_name else "log.txt"
        _STATE["file"] = open(os.path.join(log_dir, fname), "at")


def log(*args):
    """Print a timestamped line to stdout and (rank 0) the ``.log`` file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] " + " ".join(str(a) for a in args)
    print(line, flush=True)
    f = _STATE["file"]
    if f is not None:
        f.write(line + "\n")
        f.flush()


def logkv_mean(key, val):
    """Accumulate a running mean for ``key`` over the current tick."""
    oldval, cnt = _STATE["name2val"][key], _STATE["name2cnt"][key]
    _STATE["name2val"][key] = oldval * cnt / (cnt + 1) + val / (cnt + 1)
    _STATE["name2cnt"][key] = cnt + 1


def dumpkvs():
    """Return the accumulated means and reset the accumulators."""
    out = dict(_STATE["name2val"])
    _STATE["name2val"] = defaultdict(float)
    _STATE["name2cnt"] = defaultdict(int)
    return out


def close():
    if _STATE["file"] is not None:
        try:
            _STATE["file"].close()
        finally:
            _STATE["file"] = None
