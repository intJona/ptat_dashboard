#!/usr/bin/env python3
"""
dashboard.py - local Streamlit + Plotly viewer for arbitrary telemetry/event traces captured
around an agent/benchmark run (PTAT xlsx/CSV, an HWInfo CSV for a discrete GPU, a timestamped
terminal log, a token_usage_log.jsonl, ...).

Nothing here hardcodes a source-specific column name -- see traces/ for the adapter registry
that turns any recognized file into a normalized Trace, and traces/base.py's classify_columns()
for how ~800 PTAT columns (or a dozen flat HWInfo ones) become the same Category -> Metric
[-> core] picker tree regardless of which file they came from. Adding a new trace type is a new
module in traces/, not a dashboard change.

Run layout: one folder per captured run under RUNS_ROOT (see storage_config.py), holding
whichever of the recognized files happen to exist for that run -- e.g.:
    runs/my_run/2Dev...DGPURun1....xlsx   (PTAT)
    runs/my_run/hwinfo.CSV                (optional, discrete GPU)
    runs/my_run/TerminalsLog.txt          (optional, agent+tool-call log)
    runs/my_run/token_usage_log.jsonl     (optional, per-call token usage)

Usage:
  streamlit run dashboard.py
"""

import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from storage_config import resolve_runs_root
from traces.align import align_traces
from traces.base import classify_columns, extract_unit
from traces.registry import discover_runs, load_run

RUNS_ROOT = resolve_runs_root()

# Same recipe as PCIe-Visualization's dashboard.py: one vivid hex per family (matplotlib
# "tab10" order), shaded per-series by shade_color()/shade_factor() rather than sampled from a
# sequential colorscale (whose usable range compresses well before either end).
COLOR_FAMILIES = ["Blues", "Oranges", "Greens", "Reds", "Purples", "Greys", "Cyans", "Pinks"]
BASE_HEX = {
    "Blues": "#1f77b4", "Oranges": "#ff7f0e", "Greens": "#2ca02c", "Reds": "#d62728",
    "Purples": "#9467bd", "Greys": "#7f7f7f", "Cyans": "#17becf", "Pinks": "#e377c2",
}
EVENT_KIND_COLORS = {
    "inference": "#1f77b4", "tool_call": "#d62728", "verification": "#2ca02c", "log": "#7f7f7f",
}


def hue_for(rank):
    return COLOR_FAMILIES[rank % len(COLOR_FAMILIES)]


def shade_color(base_hex, factor):
    r, g, b = (int(base_hex[i:i + 2], 16) / 255 for i in (1, 3, 5))
    if factor <= 1.0:
        r, g, b = r * factor, g * factor, b * factor
    else:
        extra = factor - 1.0
        r, g, b = r + (1 - r) * extra, g + (1 - g) * extra, b + (1 - b) * extra
    r, g, b = (min(max(c, 0.0), 1.0) for c in (r, g, b))
    return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"


def shade_factor(i, n):
    return 1.0 if n <= 1 else 0.55 + (1.15 - 0.55) * (i / (n - 1))


@st.cache_data(show_spinner="Loading run...")
def get_run_traces(run_dir_str: str):
    """Load + default-align every recognized file in a run folder. Cached on the folder path --
    re-alignment to a milestone (cheap: just recomputes 't' from each trace's own '_ts') happens
    OUTSIDE this function on every rerun instead, so changing the milestone patterns in the
    sidebar never re-reads any file from disk."""
    traces = load_run(Path(run_dir_str))
    align_traces(traces)
    return traces


def find_milestone_ts(traces, pattern: str):
    """First event (across every trace with a 'label' column) whose label contains `pattern`
    (case-insensitive), by timestamp order. None if `pattern` is blank or nothing matches."""
    if not pattern:
        return None
    best = None
    for tr in traces:
        if tr.kind != "event" or "label" not in tr.df.columns:
            continue
        hits = tr.df[tr.df["label"].astype(str).str.contains(pattern, case=False, regex=False)]
        if not hits.empty:
            ts = hits["_ts"].min()
            best = ts if best is None else min(best, ts)
    return best


def numeric_metric_tree(run_traces: dict, default_category_for=lambda tr: tr.name):
    """Union, across every selected run's every trace, of NUMERIC columns only -- text/event
    columns like a log's 'label'/'kind' are deliberately excluded here (they're handled by the
    separate event-overlay controls, not plotted as a metric line). Returns
    {(trace_name, category): {metric: set_of_available_cores_or_{None}}}, merged across runs so
    e.g. 'PTAT: CPU: Frequency(MHz)' shows up once even if 3 runs all have it."""
    tree = {}
    for run_name, traces in run_traces.items():
        for tr in traces:
            numeric_cols = [c for c in tr.df.columns if c not in ("t", "_ts")
                             and pd.api.types.is_numeric_dtype(tr.df[c])]
            classified = classify_columns(numeric_cols, default_category_for(tr))
            for category, metrics in classified.items():
                key = (tr.name, category)
                for metric, core_map in metrics.items():
                    tree.setdefault(key, {}).setdefault(metric, set()).update(core_map.keys())
    return tree


def resolve_series(run_traces: dict, trace_name, category, metric, core_selection, average_cores):
    """For every selected run, resolve one picked (trace, category, metric, core-choice) into
    concrete (run_name, column_label, t, y) series ready to plot."""
    out = []
    for run_name, traces in run_traces.items():
        for tr in traces:
            if tr.name != trace_name:
                continue
            numeric_cols = [c for c in tr.df.columns if c not in ("t", "_ts")
                             and pd.api.types.is_numeric_dtype(tr.df[c])]
            classified = classify_columns(numeric_cols, tr.name)
            core_map = classified.get(category, {}).get(metric)
            if not core_map:
                continue
            if list(core_map.keys()) == [None]:
                out.append((run_name, metric, tr.df["t"], tr.df[core_map[None]]))
                continue
            cores = sorted(c for c in core_map if c is not None and c in core_selection)
            if not cores:
                continue
            if average_cores:
                cols = [core_map[c] for c in cores]
                out.append((run_name, f"{metric} (avg of {len(cols)} cores)", tr.df["t"], tr.df[cols].mean(axis=1)))
            else:
                for c in cores:
                    out.append((run_name, f"{metric} [core {c}]", tr.df["t"], tr.df[core_map[c]]))
    return out


def group_by_unit(picked_series):
    """{group_key: [(run_name, label, t, y), ...]} preserving first-appearance order. Metrics
    sharing a unit (PTAT's '(Watts)', HWInfo's '[W]', ...) land in the same group automatically
    -- this is what lets e.g. CPU IA Power and an HWInfo discrete-GPU Power overlay by default
    without the user manually wiring them together."""
    groups = {}
    for (trace_name, category, metric), series in picked_series:
        unit = extract_unit(metric)
        key = unit or f"metric:{category}:{metric}"
        groups.setdefault(key, []).extend(series)
    return groups


def main():
    st.set_page_config(layout="wide", page_title="Agent Run Telemetry Dashboard")
    st.title("Agent Run Telemetry Dashboard")

    run_options = discover_runs(RUNS_ROOT)
    if not run_options:
        st.warning(f"No run folders found under {RUNS_ROOT}/. Drop a run folder (PTAT xlsx, "
                   f"optional HWInfo CSV / log / jsonl) in there.")
        return

    st.sidebar.header("Runs")
    selected_run_names = st.sidebar.multiselect(
        "Run folders", sorted(run_options), default=sorted(run_options)[:1],
    )
    if not selected_run_names:
        st.info("Select at least one run.")
        return
    run_rank = {name: i for i, name in enumerate(selected_run_names)}

    run_traces = {name: get_run_traces(str(run_options[name])) for name in selected_run_names}

    st.sidebar.header("Stress window (optional)")
    st.sidebar.caption("Substring match against any log trace's event labels. Leave blank to "
                        "use the whole capture with no shading. Applied independently per run "
                        "(each run is re-zeroed at its own start marker).")
    start_pattern = st.sidebar.text_input("Start marker contains", value="starting agentic flow")
    end_pattern = st.sidebar.text_input("End marker contains", value="agentic flow: ----- END")
    zoom_to_window = st.sidebar.checkbox("Zoom to stress window", value=bool(start_pattern or end_pattern))
    pre_buffer = st.sidebar.number_input("Pre-buffer (s)", value=25.0, step=5.0)
    post_buffer = st.sidebar.number_input("Post-buffer (s)", value=25.0, step=5.0)

    run_windows = {}  # run_name -> (start_s, end_s) or None
    for run_name, traces in run_traces.items():
        start_ts = find_milestone_ts(traces, start_pattern)
        end_ts = find_milestone_ts(traces, end_pattern)
        if start_ts is not None:
            align_traces(traces, origin=start_ts)
            run_windows[run_name] = (0.0, (end_ts - start_ts).total_seconds() if end_ts is not None else None)
        else:
            run_windows[run_name] = None

    # Run-status card (from timestamped_log's diagnostics, if a log trace is present).
    for run_name, traces in run_traces.items():
        for tr in traces:
            if tr.static_info and "run_status" in tr.static_info:
                info = tr.static_info
                if info["run_status"] == "FAILED":
                    msg = f"**{run_name}** ({tr.name}): FAILED"
                    if info.get("exception"):
                        msg += f" -- `{info['exception']}`"
                    if info.get("lnk_error_count"):
                        msg += f" -- {info['lnk_error_count']} MSVC linker error(s)"
                    st.error(msg)
                else:
                    st.success(f"**{run_name}** ({tr.name}): COMPLETED")

    tree = numeric_metric_tree(run_traces)
    if not tree:
        st.info("No plottable numeric columns found in the selected run(s).")
        return

    st.sidebar.header("Metrics")
    category_labels = {f"{trace}: {category}": (trace, category) for trace, category in sorted(tree)}
    selected_category_labels = st.sidebar.multiselect("Categories", sorted(category_labels))

    picked_series = []  # [((trace_name, category, metric), [(run_name, label, t, y), ...])]
    for cat_label in selected_category_labels:
        trace_name, category = category_labels[cat_label]
        metrics = tree[(trace_name, category)]
        chosen_metrics = st.sidebar.multiselect(
            f"{cat_label} -- metrics", sorted(metrics), key=f"metrics_{cat_label}",
        )
        for metric in chosen_metrics:
            cores = sorted(c for c in metrics[metric] if c is not None)
            core_selection, average_cores = cores, False
            if cores:
                with st.sidebar.expander(f"{metric} -- cores ({len(cores)} available)"):
                    average_cores = st.checkbox("Average across cores", key=f"avg_{cat_label}_{metric}")
                    if not average_cores:
                        core_selection = st.multiselect(
                            "Which cores", cores, default=cores, key=f"cores_{cat_label}_{metric}",
                        )
            series = resolve_series(run_traces, trace_name, category, metric, set(core_selection), average_cores)
            if series:
                picked_series.append(((trace_name, category, metric), series))

    if not picked_series:
        st.info("Select at least one metric in the sidebar.")
        return

    groups = group_by_unit(picked_series)
    panel_keys = list(groups)

    st.sidebar.header("Reference / limit lines")
    st.sidebar.caption("Draws a dashed horizontal line at a column's first value (e.g. a PL1 "
                        "power limit) -- lands automatically on whichever panel shares its unit.")
    ref_line_options = {}
    for run_name, traces in run_traces.items():
        for tr in traces:
            for col in tr.df.columns:
                if col in ("t", "_ts") or not pd.api.types.is_numeric_dtype(tr.df[col]):
                    continue
                ref_line_options[f"{run_name} | {tr.name} | {col}"] = (run_name, tr.name, col)
    chosen_ref_lines = st.sidebar.multiselect("Add reference line(s)", sorted(ref_line_options))

    st.sidebar.header("Combine two rows onto dual Y-axis")
    dual_primary = st.sidebar.selectbox("Primary row", ["(none)"] + panel_keys)
    dual_secondary = st.sidebar.selectbox(
        "Secondary row", ["(none)"] + [k for k in panel_keys if k != dual_primary],
    )

    st.sidebar.header("Event overlays")
    event_trace_names = sorted({tr.name for traces in run_traces.values() for tr in traces
                                 if tr.kind == "event" and "label" in tr.df.columns})
    event_overlay_choice = {}
    for name in event_trace_names:
        event_overlay_choice[name] = st.sidebar.selectbox(
            f"'{name}' events", ["Off", "Own row"] + panel_keys, key=f"event_{name}",
        )

    # --- Build the figure -------------------------------------------------------------------
    row_defs = []  # list of dicts describing each subplot row
    merged = {dual_primary, dual_secondary} if dual_primary != "(none)" and dual_secondary != "(none)" else set()
    for key in panel_keys:
        if key in merged and key != dual_primary:
            continue  # folded into the primary row below
        if key == dual_primary and dual_secondary != "(none)":
            row_defs.append({"kind": "dual", "primary": key, "secondary": dual_secondary})
        else:
            row_defs.append({"kind": "single", "key": key})
    own_row_events = [name for name, choice in event_overlay_choice.items() if choice == "Own row"]
    for name in own_row_events:
        row_defs.append({"kind": "event", "name": name})

    n_rows = len(row_defs)
    specs = [[{"secondary_y": r["kind"] == "dual"}] for r in row_defs]
    row_titles = []
    for r in row_defs:
        if r["kind"] == "single":
            row_titles.append(r["key"])
        elif r["kind"] == "dual":
            row_titles.append(f"{r['primary']} / {r['secondary']}")
        else:
            row_titles.append(f"events: {r['name']}")

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.4 / max(n_rows, 1),
        specs=specs, subplot_titles=row_titles,
    )

    def add_series_to_row(series, row, secondary_y=False):
        for run_name, label, t, y in series:
            base_hex = BASE_HEX[hue_for(run_rank[run_name])]
            same_run_count = sum(1 for s in series if s[0] == run_name)
            idx_in_run = [s[0] for s in series if s[0] == run_name].index(run_name) if same_run_count == 1 else None
            # Shade by this series' rank among same-run lines on this row, for contrast when a
            # run contributes more than one line to the same panel (e.g. several cores).
            same_run_series = [s for s in series if s[0] == run_name]
            shade = shade_factor(same_run_series.index((run_name, label, t, y)), len(same_run_series))
            color = shade_color(base_hex, shade)
            fig.add_trace(
                go.Scatter(x=t, y=y, mode="lines", name=f"{run_name}: {label}",
                           line=dict(color=color, width=2)),
                row=row, col=1, secondary_y=secondary_y,
            )

    for i, r in enumerate(row_defs, start=1):
        if r["kind"] == "single":
            add_series_to_row(groups[r["key"]], i)
        elif r["kind"] == "dual":
            add_series_to_row(groups[r["primary"]], i, secondary_y=False)
            add_series_to_row(groups[r["secondary"]], i, secondary_y=True)
        else:
            name = r["name"]
            for run_name, traces in run_traces.items():
                for tr in traces:
                    if tr.name != name or tr.kind != "event":
                        continue
                    kinds = sorted(tr.df["kind"].unique()) if "kind" in tr.df.columns else ["event"]
                    y_for_kind = {k: j for j, k in enumerate(kinds)}
                    kind_col = tr.df["kind"] if "kind" in tr.df.columns else pd.Series(["event"] * len(tr.df))
                    fig.add_trace(
                        go.Scatter(
                            x=tr.df["t"], y=kind_col.map(y_for_kind), mode="markers",
                            marker=dict(
                                color=[EVENT_KIND_COLORS.get(k, "#888888") for k in kind_col],
                                symbol="line-ns-open", size=10,
                            ),
                            text=tr.df["label"] if "label" in tr.df.columns else None,
                            name=f"{run_name}: {name}",
                        ),
                        row=i, col=1,
                    )
            fig.update_yaxes(row=i, col=1, showticklabels=False)

    # Overlay events onto an existing metric row (vertical dashed lines), e.g. tool-call markers
    # drawn over a Power row -- same idea as PCIe-Visualization's TTFT-on-token-curve markers.
    row_index_by_key = {}
    for i, r in enumerate(row_defs, start=1):
        if r["kind"] == "single":
            row_index_by_key[r["key"]] = i
        elif r["kind"] == "dual":
            row_index_by_key[r["primary"]] = i
            row_index_by_key[r["secondary"]] = i
    for name, choice in event_overlay_choice.items():
        if choice in ("Off", "Own row"):
            continue
        target_row = row_index_by_key.get(choice)
        if target_row is None:
            continue
        for run_name, traces in run_traces.items():
            for tr in traces:
                if tr.name != name or tr.kind != "event":
                    continue
                for _, ev in tr.df.iterrows():
                    fig.add_vline(
                        x=ev["t"], row=target_row, col=1, line_dash="dash", line_width=1,
                        line_color=EVENT_KIND_COLORS.get(ev.get("kind", "log"), "#888888"),
                        opacity=0.6,
                    )

    # Reference/limit lines -- first value of the chosen column, drawn on whichever row already
    # shares that column's unit (falls back to being skipped if no such row is currently shown).
    for opt in chosen_ref_lines:
        run_name, trace_name, col = ref_line_options[opt]
        for tr in run_traces[run_name]:
            if tr.name != trace_name or col not in tr.df.columns:
                continue
            non_null = tr.df[col].dropna()
            if non_null.empty:
                continue
            unit = extract_unit(col)
            target_row = row_index_by_key.get(unit)
            if target_row is None:
                continue
            fig.add_hline(
                y=float(non_null.iloc[0]), row=target_row, col=1, line_dash="dash",
                line_color="black", annotation_text=f"{run_name}: {col}",
            )

    # Stress-window shading, per run (each run zeroed at its own start marker above).
    for run_name, window in run_windows.items():
        if window is None or window[1] is None:
            continue
        for i in range(1, n_rows + 1):
            fig.add_vrect(x0=window[0], x1=window[1], row=i, col=1,
                          fillcolor=BASE_HEX[hue_for(run_rank[run_name])], opacity=0.08, line_width=0)

    if zoom_to_window:
        starts = [w[0] - pre_buffer for w in run_windows.values() if w is not None]
        ends = [w[1] + post_buffer for w in run_windows.values() if w is not None and w[1] is not None]
        if starts and ends:
            fig.update_xaxes(range=[min(starts), max(ends)])

    fig.update_layout(height=280 * n_rows, showlegend=True, xaxis_title=None)
    fig.update_xaxes(title_text="Elapsed time (s)", row=n_rows, col=1)
    st.plotly_chart(fig, use_container_width=True)

    render_categorical_breakdown(run_traces, run_windows, run_rank)
    render_summary(picked_series, run_windows)


def render_categorical_breakdown(run_traces, run_windows, run_rank):
    st.sidebar.header("Category breakdown (time-in-state)")
    st.sidebar.caption("E.g. combine every 'CPUn-Throttling Reason' column and bucket by "
                        "keyword to reproduce a throttling-reason pie.")
    run_name = st.sidebar.selectbox("Run", list(run_traces), key="breakdown_run")
    trace_name = st.sidebar.selectbox(
        "Trace", sorted({tr.name for tr in run_traces[run_name]}), key="breakdown_trace",
    )
    col_pattern = st.sidebar.text_input("Column name contains/regex", key="breakdown_pattern")
    keyword_rules = st.sidebar.text_area(
        "Priority-ordered keyword -> category (one per line: 'Category: kw1,kw2')",
        key="breakdown_rules",
        help="Checked top to bottom; the first line whose keyword appears anywhere in a row's "
             "combined column values wins. Rows matching nothing are 'No match'.",
    )
    if not col_pattern:
        return

    tr = next((t for t in run_traces[run_name] if t.name == trace_name), None)
    if tr is None:
        return
    matched_cols = [c for c in tr.df.columns if c not in ("t", "_ts") and re.search(col_pattern, c)]
    if not matched_cols:
        st.sidebar.caption("No columns match that pattern.")
        return

    df = tr.df
    window = run_windows.get(run_name)
    if window is not None and window[1] is not None:
        df = df[(df["t"] >= window[0]) & (df["t"] <= window[1])]
    combined = df[matched_cols].astype(str).agg(";".join, axis=1)

    rules = []
    for line in keyword_rules.splitlines():
        if ":" not in line:
            continue
        name, kws = line.split(":", 1)
        rules.append((name.strip(), [k.strip() for k in kws.split(",") if k.strip()]))

    if rules:
        category = pd.Series("No match", index=combined.index)
        for name, keywords in reversed(rules):
            hits = combined.apply(lambda text: any(k in text for k in keywords))
            category = category.mask(hits, name)
    else:
        category = combined

    counts = category.value_counts(normalize=True) * 100
    fig = go.Figure(go.Pie(labels=counts.index, values=counts.values))
    fig.update_layout(title=f"{run_name}: {trace_name} -- {col_pattern}", height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_summary(picked_series, run_windows):
    st.header("Summary")
    rows = []
    for (trace_name, category, metric), series in picked_series:
        for run_name, label, t, y in series:
            window = run_windows.get(run_name)
            if window is not None and window[1] is not None:
                mask = (t >= window[0]) & (t <= window[1])
                y = y[mask]
            if y.empty:
                continue
            rows.append({
                "Run": run_name, "Trace": trace_name, "Metric": label,
                "Min": y.min(), "Mean": y.mean(), "Max": y.max(),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
