"""Adapter for a generic jsonl event log (e.g. token_usage_log.jsonl -- one model call per
line). Any jsonl file where each record has a "timestamp" field qualifies; every other field
(prompt_tokens, duration_s, tokens_per_second, ...) rides along as a plain DataFrame column with
no hardcoded field names, so a differently-shaped jsonl log works without code changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from .base import Trace


def sniff(path: Path) -> bool:
    if path.suffix.lower() != ".jsonl":
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline()
        if not first.strip():
            return False
        record = json.loads(first)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(record, dict) and "timestamp" in record


def load(path: Path) -> Optional[Trace]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ts = record.pop("timestamp", None)
            if ts is None:
                continue
            record["_ts"] = pd.Timestamp(ts)
            rows.append(record)
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("_ts").reset_index(drop=True)
    return Trace(name=path.stem, kind="event", df=df, source_path=str(path), static_info=None)
