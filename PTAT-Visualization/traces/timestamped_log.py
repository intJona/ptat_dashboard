"""Adapter for a freeform timestamped text log (e.g. a combined terminal log covering an agent
+ its MCP server) -> a sparse "event" trace.

Any line starting with "YYYY-MM-DD HH:MM:SS" becomes one event row. A small set of known
recognizers gives specific lines a meaningful (kind, label) -- inference calls, tool calls,
verification results -- but an unrecognized timestamped line still becomes a plain "log" event
(label = the rest of that line) rather than being dropped, so a brand-new log format is usable
immediately, just with plainer labels than a purpose-built recognizer would give it.

Also extracts a few run-health diagnostics specific to this kind of coding-agent log (reached a
normal end marker vs. an unhandled Python traceback vs. MSVC linker errors) into static_info,
surfaced by the dashboard as a run-status card rather than a plottable series.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from .base import Trace

_TS_PATTERN = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*(?P<rest>.*)$")

# (kind, compiled pattern, optional named group to use as the label). Checked in order; the
# first match wins. Add more recognizers here for other log formats -- unmatched lines still
# get a generic "log" event via the fallback in _classify_line(), they just won't have a
# specifically-parsed label.
_LABELERS = [
    ("inference", re.compile(r"POST http://localhost:\d+/v\d+/chat/completions"), None),
    ("tool_call", re.compile(r" - (?:\w+) - INFO - CALLING (?P<detail>.*)$"), "detail"),
    ("tool_call", re.compile(r" - agent_framework - INFO - Function name: (?P<tool>[A-Za-z_]\w*)$"), "tool"),
    ("verification", re.compile(r" - CODE_VERIFICATION - INFO - (?P<result>\{.*\})$"), "result"),
]
_END_MARKER = "agentic flow: ----- END ------"
_TRACEBACK_START = "Traceback (most recent call last):"
_LNK_ERROR = re.compile(r"error LNK\d+")


def sniff(path: Path) -> bool:
    if path.suffix.lower() != ".txt":
        return False
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(20):
                line = f.readline()
                if not line:
                    break
                if _TS_PATTERN.match(line):
                    return True
    except OSError:
        return False
    return False


def _classify_line(rest: str) -> tuple:
    for kind, pattern, group in _LABELERS:
        m = pattern.search(rest)
        if m:
            return kind, (m.group(group) if group else kind)
    return "log", rest[:160]


def load(path: Path) -> Optional[Trace]:
    rows = []
    reached_end, exception_line, lnk_error_count, in_traceback = False, None, 0, False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.rstrip("\n")
            m = _TS_PATTERN.match(stripped)
            if m:
                kind, label = _classify_line(m.group("rest"))
                rows.append({"_ts": pd.Timestamp(m.group("ts")), "kind": kind, "label": label})
            if _END_MARKER in stripped:
                reached_end = True
            if stripped.strip() == _TRACEBACK_START:
                in_traceback = True
                continue
            if in_traceback and stripped and not stripped.startswith((" ", "\t", "File ")):
                exception_line = stripped
                in_traceback = False
            lnk_error_count += len(_LNK_ERROR.findall(stripped))
    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("_ts").reset_index(drop=True)
    static_info = {
        "run_status": "COMPLETED" if reached_end else "FAILED",
        "exception": exception_line,
        "lnk_error_count": lnk_error_count,
    }
    return Trace(name=path.stem, kind="event", df=df, source_path=str(path), static_info=static_info)
