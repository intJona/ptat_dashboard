"""Standalone smoke test for the trace adapters -- no Streamlit involved. Run with:

    .venv\\Scripts\\python.exe smoke_test.py

Loads every run under runs/ and prints each Trace's shape/columns/static_info so adapter bugs
surface before booting the full dashboard on top of them.
"""

from storage_config import resolve_runs_root
from traces.align import align_traces
from traces.registry import discover_runs, load_run


def main():
    runs_root = resolve_runs_root()
    print(f"runs root: {runs_root}")
    runs = discover_runs(runs_root)
    if not runs:
        print("No runs found.")
        return

    for name, run_dir in runs.items():
        print(f"\n=== run: {name} ({run_dir}) ===")
        traces = load_run(run_dir)
        if not traces:
            print("  (no recognized files)")
            continue
        origin = align_traces(traces)
        print(f"  origin: {origin}")
        for tr in traces:
            print(f"\n  --- trace: {tr.name} (kind={tr.kind}, source={tr.source_path}) ---")
            print(f"      shape: {tr.df.shape}")
            print(f"      columns[:15]: {list(tr.df.columns)[:15]}")
            if "t" in tr.df.columns:
                print(f"      t range: {tr.df['t'].min()} .. {tr.df['t'].max()}")
            if tr.static_info:
                keys = list(tr.static_info)[:10]
                print(f"      static_info keys[:10]: {keys}")


if __name__ == "__main__":
    main()
