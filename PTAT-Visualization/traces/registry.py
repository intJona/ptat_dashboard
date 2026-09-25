"""Adapter registry + per-run/per-runs-root discovery.

Adding a new trace type = write one more module with sniff(path)/load(path) and add it to
ADAPTERS -- nothing else in this file, or the dashboard, needs to change.
"""

from __future__ import annotations

from pathlib import Path

from . import hwinfo_csv, jsonl_events, ptat, timestamped_log
from .base import Trace

# Order matters: the first adapter whose sniff() returns True for a given file wins.
ADAPTERS = [ptat, hwinfo_csv, timestamped_log, jsonl_events]


def load_run(run_dir: Path) -> list[Trace]:
    """Run every file directly inside run_dir through the adapter registry. Files that no
    adapter recognizes (or that fail to parse) are silently skipped."""
    traces = []
    for path in sorted(run_dir.iterdir()):
        if not path.is_file():
            continue
        for adapter in ADAPTERS:
            try:
                if adapter.sniff(path):
                    trace = adapter.load(path)
                    if trace is not None:
                        traces.append(trace)
                    break
            except Exception:
                continue
    return traces


def discover_runs(runs_root: Path) -> dict:
    """{run_name: run_dir} for every subfolder of runs_root -- one folder per captured run."""
    if not runs_root.exists():
        return {}
    return {p.name: p for p in sorted(runs_root.iterdir()) if p.is_dir()}
