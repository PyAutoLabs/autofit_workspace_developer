"""
Aggregate per-N profile summaries into a single baseline.json per package,
and distill cProfile dumps into N10_hotspots.txt.

Run from the autofit_workspace_developer worktree root after the sweep:

    python3 aggregate_profiles.py
"""

import json
import pstats
import sys
from io import StringIO
from pathlib import Path


def aggregate_package(pkg_root):
    profiles_dir = pkg_root / "profiles"
    if not profiles_dir.exists():
        print(f"  (no profiles dir at {profiles_dir})")
        return

    rows = []
    for summary_path in sorted(profiles_dir.glob("N*_summary.json")):
        # Skip the cProfile sweep's summary so the table stays clean — the
        # cProfile run is duplicate work used only for attribution.
        name = summary_path.stem.replace("_summary", "")
        if "_cprof" in name:
            continue
        with open(summary_path) as f:
            rows.append(json.load(f))

    rows.sort(key=lambda r: r["n_datasets"])
    baseline_path = profiles_dir / "baseline.json"
    with open(baseline_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  wrote {baseline_path} ({len(rows)} rows)")

    pstats_path = profiles_dir / "N10.pstats"
    if pstats_path.exists():
        buf = StringIO()
        stats = pstats.Stats(str(pstats_path), stream=buf)
        stats.strip_dirs().sort_stats("cumulative").print_stats(30)
        hotspots_path = profiles_dir / "N10_hotspots.txt"
        with open(hotspots_path, "w") as f:
            f.write(buf.getvalue())
        print(f"  wrote {hotspots_path}")


def main():
    root = Path(__file__).parent
    for pkg in ["graphical", "ep"]:
        print(f"== {pkg} ==")
        aggregate_package(root / pkg)


if __name__ == "__main__":
    main()
