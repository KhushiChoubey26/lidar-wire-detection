"""Convenience script — run the full pipeline on every .parquet file found.

Usage
-----
    python scripts/run_case.py                          # all datasets in cwd
    python scripts/run_case.py --data-dir /path/to/    # specific directory
    python scripts/run_case.py --plot                   # show plots

This script is intentionally simple: it delegates all logic to the
wire_catenary package and just handles argument parsing and output formatting.
"""

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-process all .parquet wire datasets.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Directory containing lidar_cable_points_*.parquet files (default: current dir).",
    )
    parser.add_argument("--plot", action="store_true", help="Show plots for each dataset.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write per-dataset CSV result files (optional).",
    )
    args = parser.parse_args()

    datasets = sorted(args.data_dir.glob("lidar_cable_points_*.parquet"))
    if not datasets:
        print(f"No datasets found in {args.data_dir}", file=sys.stderr)
        return 1

    from wire_catenary.io import load_points
    from wire_catenary.pipeline import run_dataset
    from wire_catenary.visualize import plot_results
    import pandas as pd

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    overall_summary = []

    for path in datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset : {path.name}")

        try:
            results = run_dataset(path)
        except Exception as exc:
            print(f"Error processing {path.name}: {exc}", file=sys.stderr)
            continue

        print(f"Wires   : {len(results)}")

        if results:
            print(f"\n{'ID':>4}  {'N pts':>7}  {'c':>8}  {'u0':>8}  {'v0':>8}  {'RMSE':>8}")
            print("-" * 52)
            for r in results:
                p = r.params
                print(
                    f"{r.wire_id:>4}  {r.n_points:>7}  {p.c:>8.3f}  "
                    f"{p.u0:>8.3f}  {p.v0:>8.3f}  {p.rmse:>8.4f}"
                )
                overall_summary.append(
                    {
                        "dataset": path.stem,
                        "wire_id": r.wire_id,
                        "n_points": r.n_points,
                        "c": p.c,
                        "u0": p.u0,
                        "v0": p.v0,
                        "rmse": p.rmse,
                    }
                )
        else:
            print("No valid wire clusters were detected.")

        if args.out_dir:
            rows = [
                {
                    "wire_id": r.wire_id,
                    "n_points": r.n_points,
                    "c": r.params.c,
                    "u0": r.params.u0,
                    "v0": r.params.v0,
                    "rmse": r.params.rmse,
                }
                for r in results
            ]
            out_path = args.out_dir / f"{path.stem}_results.csv"
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"Saved  : {out_path}")

        if args.plot and results:
            points = load_points(path)
            plot_results(points, results, title=path.name)

    print(f"\n{'=' * 60}")
    print("Summary across all datasets:")
    print(f"  Total wires detected: {len(overall_summary)}")

    if args.out_dir and overall_summary:
        summary_path = args.out_dir / "all_results_summary.csv"
        pd.DataFrame(overall_summary).to_csv(summary_path, index=False)
        print(f"  Summary CSV written: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())