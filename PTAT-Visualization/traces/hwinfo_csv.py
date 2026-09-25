"""Adapter for an HWInfo CSV export (e.g. tracking a discrete GPU PTAT itself can't see).

Recognized by a Date+Time column pair plus at least one "GPU ..." column -- deliberately not
tied to any specific GPU vendor's exact column set, since HWInfo's schema depends on which
sensors the logged hardware exposes. Unlike PTAT, HWInfo has no single elapsed-ms counter, so
every row's Date+Time is parsed individually.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .base import Trace, split_static

_TIMESTAMP_FORMATS = (
    "%d.%m.%Y %H:%M:%S.%f", "%d.%m.%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S.%f", "%d/%m/%Y %H:%M:%S",
)


def _read_header(path: Path) -> list:
    return pd.read_csv(path, nrows=0, encoding="cp1252").columns.tolist()


def sniff(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    try:
        header = _read_header(path)
    except Exception:
        return False
    return "Date" in header and "Time" in header and any(c.startswith("GPU ") for c in header)


def _parse_timestamp(date_value, time_value) -> Optional[datetime]:
    text = f"{str(date_value).strip()} {str(time_value).strip()}"
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def load(path: Path) -> Optional[Trace]:
    df = pd.read_csv(path, low_memory=False, encoding="cp1252")
    if "Date" not in df.columns or "Time" not in df.columns:
        return None

    timestamps = [_parse_timestamp(d, t) for d, t in zip(df["Date"], df["Time"])]
    valid = [ts is not None for ts in timestamps]
    df = df.loc[valid].reset_index(drop=True)
    timestamps = [ts for ts in timestamps if ts is not None]
    if not timestamps:
        return None
    df["_ts"] = pd.to_datetime(pd.Series(timestamps))
    df = df.sort_values("_ts").reset_index(drop=True)

    for col in df.columns:
        if col in ("Date", "Time", "_ts"):
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        # Only replace with the numeric version if every non-null value actually converted --
        # otherwise this is a genuinely textual column (e.g. a status string) and should stay one.
        if converted.notna().sum() >= df[col].notna().sum():
            df[col] = converted

    value_df, static_info = split_static(df, keep=("_ts",), drop=("Date", "Time"))
    return Trace(name="HWInfo", kind="sampled", df=value_df, source_path=str(path), static_info=static_info)
