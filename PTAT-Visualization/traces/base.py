"""Shared data model: every input file becomes a Trace, normalized to a common shape so the
dashboard never hardcodes a source-specific column name.

A Trace is either:
  "sampled" -- continuous instrument data (PTAT telemetry, an HWInfo CSV, ...), rendered as lines.
  "event"   -- sparse timestamped points (a text log, a token-usage jsonl, ...), rendered as
               markers/tick-marks, optionally overlaid on a sampled row.

Every Trace.df carries an absolute "_ts" (pandas Timestamp) column. Nothing computes an elapsed
time axis at load time -- that only happens once for the whole run, in align.align_traces(),
so every trace (whatever file it came from) ends up on the exact same "t" (elapsed seconds) zero
point, chosen once a run's full set of traces is known.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class Trace:
    name: str  # display label, e.g. "PTAT", "HWInfo", or a log/jsonl file's stem
    kind: str  # "sampled" | "event"
    df: pd.DataFrame  # must contain "_ts"; gets "t" (elapsed seconds) added by align_traces()
    source_path: str
    # Near-constant columns (sampled traces) or run-level diagnostics (event traces, e.g.
    # run_status/exception/lnk_error_count) -- metadata, not something you'd plot over time.
    static_info: Optional[dict] = None


def split_static(df: pd.DataFrame, keep: tuple = (), drop: tuple = ()) -> tuple[pd.DataFrame, dict]:
    """Classify every column not in `keep`/`drop` as either "varies over the run" (kept as a
    plot candidate) or "holds one constant value throughout" (pulled into a metadata dict
    instead -- a 796-column PTAT log has ~100 columns like CPUID/TjMax/CPU Name that are never
    worth a time-series row). `keep` columns (e.g. "_ts") pass through untouched, unclassified;
    `drop` columns (e.g. raw Date/Time strings once "_ts" exists) are discarded entirely.
    Returns (df_with_keep_and_varying_columns, {static_col: constant_value})."""
    candidates = [c for c in df.columns if c not in keep and c not in drop]
    static, varying = {}, []
    for col in candidates:
        if df[col].nunique(dropna=True) <= 1:
            non_null = df[col].dropna()
            static[col] = non_null.iloc[0] if not non_null.empty else None
        else:
            varying.append(col)
    return df[list(keep) + varying], static


_UNIT_PATTERN = re.compile(r"[\(\[]([^\)\]]+)[\)\]]\s*$")
# Normalizes PTAT's "(Watts)"/"(Degree C)" and HWInfo's "[W]"/"[C]" spellings onto the same key,
# so same-quantity metrics from different trace sources still land in the same auto-grouped row.
_UNIT_ALIASES = {
    "watts": "W", "w": "W",
    "degree c": "C", "c": "C", "celcius": "C", "celsius": "C",
    "mhz": "MHz",
    "percent": "%", "percentage": "%", "%": "%",
    "joules": "J",
    "volts": "V", "v": "V",
    "seconds": "s", "s": "s",
    "amperes": "A", "ampere": "A",
    "milliseconds": "ms",
}


def extract_unit(metric_name: str) -> Optional[str]:
    """Trailing '(Unit)' or '[Unit]' suffix -> normalized unit key, e.g. 'IA Power(Watts)' -> 'W'
    and 'GPU Power [W]' -> 'W' both resolve to the same key. None if there's no unit suffix."""
    m = _UNIT_PATTERN.search(metric_name)
    if not m:
        return None
    raw = m.group(1).strip().lower()
    return _UNIT_ALIASES.get(raw, m.group(1).strip())


# PTAT's per-core-repeated column naming conventions -- recognized ahead of the generic
# Category-Metric split below so e.g. "CPU3-Frequency(MHz)" groups under category "CPU",
# metric "Frequency(MHz)", core 3 instead of being its own one-off category "CPU3".
_CPU_CSTATE = re.compile(r"^CPU(\d+) CState Residency-(.+)$")
_CPU_WORKLOAD = re.compile(r"^CPU Workload-CPU(\d+)\(Percentage\)$")
_CPU_TOPOLOGY_CORE = re.compile(r"^CPU Topology-(Physical Core|Die Id for Core)-(\d+)$")
_CPU_METRIC = re.compile(r"^CPU(\d+)-(.+)$")


def classify_columns(columns, default_category: str) -> dict:
    """Group value-column names into {category: {metric: {core_or_None: column_name}}}.
    Falls back to a plain Category-Metric split on the first '-', or to `default_category`
    for columns with no dash at all (e.g. every column of a flat-schema trace like HWInfo)."""
    tree: dict = {}

    def add(category, metric, core, column):
        tree.setdefault(category, {}).setdefault(metric, {})[core] = column

    for col in columns:
        if col in ("t", "_ts"):
            continue
        m = _CPU_CSTATE.match(col)
        if m:
            add("CState Residency", m.group(2), int(m.group(1)), col)
            continue
        m = _CPU_WORKLOAD.match(col)
        if m:
            add("CPU Workload", "Workload(Percentage)", int(m.group(1)), col)
            continue
        m = _CPU_TOPOLOGY_CORE.match(col)
        if m:
            add("CPU Topology", m.group(1), int(m.group(2)), col)
            continue
        m = _CPU_METRIC.match(col)
        if m:
            add("CPU", m.group(2), int(m.group(1)), col)
            continue
        if "-" in col:
            category, metric = col.split("-", 1)
            add(category, metric, None, col)
        else:
            add(default_category, col, None, col)
    return tree
