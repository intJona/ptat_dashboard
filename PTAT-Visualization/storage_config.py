"""Resolves where run folders are read from.

Precedence: $PTAT_VIZ_RUNS_DIR env var, then the first non-comment line of .storage-config
(gitignored, machine-local override), then ./runs so a fresh clone stays self-contained.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def resolve_runs_root() -> Path:
    env_override = os.environ.get("PTAT_VIZ_RUNS_DIR")
    if env_override:
        return Path(env_override).expanduser()
    config_file = REPO_ROOT / ".storage-config"
    if config_file.exists():
        for line in config_file.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                return Path(line).expanduser()
    return REPO_ROOT / "runs"
