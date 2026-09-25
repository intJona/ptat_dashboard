# PTAT-Visualization

Local Streamlit + Plotly dashboard for exploring telemetry/event traces captured around a
benchmark or agent run -- an Intel PTAT xlsx/CSV export, plus (optionally) an HWInfo CSV for a
discrete GPU, a timestamped terminal log, and a `token_usage_log.jsonl`.

Nothing in the dashboard hardcodes a source-specific column name. Every input file is turned
into a normalized `Trace` by a small adapter in [`traces/`](traces/) (see `traces/registry.py`),
and [`traces/base.py`](traces/base.py)'s `classify_columns()` groups whatever columns a trace
happens to have into the same `Category -> Metric [-> core]` picker tree, regardless of which
file they came from. Adding a new trace type is a new adapter module, not a dashboard change.

## Setup

If this repo lives under a long path (e.g. a deeply nested OneDrive folder), create the venv
outside it instead of as a local `.venv` -- streamlit's own installed files can exceed Windows'
260-char path limit otherwise, failing mid-install with an `OSError`/"Long Path" hint:

```powershell
python -m venv C:\pyvenvs\ptat-viz
C:\pyvenvs\ptat-viz\Scripts\python.exe -m pip install -r requirements.txt
```

(A short project path can just use a local `.venv` as usual.)

## Adding a run

Create one folder per captured run under `runs/` (see [Storage location](#storage-location)),
containing whichever of these files you have for that run:

| File | Adapter | Required columns |
|---|---|---|
| `*.xlsx` / `*.csv` | `traces/ptat.py` | `Relative Time(mS)` (+ `Date`, `Time`) |
| `*.csv` | `traces/hwinfo_csv.py` | `Date`, `Time`, at least one `GPU ...` column |
| `*.txt` | `traces/timestamped_log.py` | lines starting with `YYYY-MM-DD HH:MM:SS` |
| `*.jsonl` | `traces/jsonl_events.py` | one JSON object per line, each with a `timestamp` field |

A sample run (`runs/2DeveloperAgentDGPURun1/`) is included with just a PTAT xlsx.

## Running the dashboard

```powershell
C:\pyvenvs\ptat-viz\Scripts\streamlit.exe run dashboard.py
```

Pick one or more runs in the sidebar, then Categories -> Metrics to plot. Metrics sharing a unit
(PTAT's `(Watts)`, HWInfo's `[W]`, ...) are auto-grouped onto the same row so e.g. CPU IA Power
and a discrete GPU's power overlay by default with no extra configuration.

## Notable features

- **Per-core aggregation** -- metrics repeated per CPU core (`CPUn-Frequency(MHz)`, ...) get a
  "which cores" picker plus an "average across cores" toggle.
- **Stress window** -- a start/end substring is matched against any log trace's event labels
  (defaults match this repo's own agent log format, but are plain text boxes -- edit them for a
  different log). Each selected run is independently re-zeroed at its own start marker, so
  multiple runs line up on "time since the workload started" rather than wall-clock time.
- **Reference/limit lines** -- draw a dashed horizontal line at a column's first value (e.g. a
  `Turbo Parameters-MSR Power Limit_1 Power(Watts)` / PL1 limit); it lands automatically on
  whichever row already shares that column's unit.
- **Dual Y-axis** -- combine any two auto-grouped rows (e.g. NPU Power + NPU Temperature) onto
  one panel with a secondary axis.
- **Category breakdown** -- combine several columns (e.g. every `CPUn-Throttling Reason`) and
  bucket rows by a priority-ordered keyword list into a time-in-state pie chart.
- **Event overlays** -- a log/jsonl trace's events can render as their own tick-mark row, or as
  vertical markers drawn directly on top of an existing metric row (e.g. tool-call markers over
  a Power row).

## Storage location

Runs are read from `runs/` inside the repo by default. To point at a shared/network location
instead, copy `.storage-config.example` to `.storage-config` (gitignored) and set the path
there, or set the `PTAT_VIZ_RUNS_DIR` environment variable. Precedence: env var >
`.storage-config` > `runs/`.
