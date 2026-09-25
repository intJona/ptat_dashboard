"""Run-level time alignment: gives every Trace in a run the same elapsed-seconds zero point.

Each adapter only ever produces absolute wall-clock timestamps ("_ts") -- none of them decide
where t=0 is, since that requires seeing every trace in the run at once (and, optionally, a
user-chosen milestone event). That decision lives here, exactly once per run.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .base import Trace


def align_traces(traces: list[Trace], origin: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """Add/overwrite a "t" (elapsed seconds) column on every trace, relative to `origin`.
    If `origin` is omitted, uses the earliest sample among "sampled" traces (falling back to
    the earliest timestamp among ALL traces if there are no sampled ones) -- i.e. "whole
    capture" behavior. Pass an explicit `origin` (e.g. a milestone event's own timestamp) to
    re-zero the run at that instant instead; safe to call repeatedly as the user changes it.
    """
    if origin is None:
        non_empty = [tr for tr in traces if not tr.df.empty]
        sampled_starts = [tr.df["_ts"].min() for tr in non_empty if tr.kind == "sampled"]
        candidates = sampled_starts or [tr.df["_ts"].min() for tr in non_empty]
        if not candidates:
            origin = pd.Timestamp.now()
        else:
            origin = min(candidates)
    for tr in traces:
        if not tr.df.empty:
            tr.df["t"] = (tr.df["_ts"] - origin).dt.total_seconds()
    return origin
