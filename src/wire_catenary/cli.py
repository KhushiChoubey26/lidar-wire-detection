"""CLI entry point: python -m wire_catenary <parquet> [--plot] [--out results.csv]"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wire_catenary",
        description="Detect wires in a LiDAR .parquet point cloud and fit 3D catenaries.",
    )
    parser.add_argument("parquet", type=Path, help="Path to a .parquet point cloud file.")
    parser.add_argument("--plot", action="store_true", help="Show interactive 3D + 2D plots.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional CSV path to write per-wire results.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=50_000,
        help="Downsample to at most this many points (default: 50000).",
    )
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.parquet.exists():
        print(f"Error: file not found — {args.parquet}", file=sys.stderr)
        return 1

    try:
        from .io import load_points
        from .pipeline import run_dataset

        print(f"Loading {args.parquet.name}...")
        results = run_dataset(args.parquet, max_points=args.max_points)

        print(f"\nDetected {len(results)} wire(s)\n")

        if results:
            print(f"{'ID':>4}  {'N pts':>7}  {'c':>8}  {'u0':>8}  {'v0':>8}  {'RMSE (m)':>10}")
            print("-" * 56)
            for r in results:
                p = r.params
                print(
                    f"{r.wire_id:>4}  {r.n_points:>7}  {p.c:>8.3f}  "
                    f"{p.u0:>8.3f}  {p.v0:>8.3f}  {p.rmse:>10.4f}"
                )
        else:
            print("No valid wire clusters were detected.")

        if args.out:
            rows = [
                {
                    "wire_id": r.wire_id,
                    "n_points": r.n_points,
                    "c": r.params.c,
                    "u0": r.params.u0,
                    "v0": r.params.v0,
                    "rmse": r.params.rmse,
                    "u_min": r.u_min,
                    "u_max": r.u_max,
                }
                for r in results
            ]
            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(f"\nResults written to {args.out}")

        if args.plot and results:
            from .visualize import plot_results
            points = load_points(args.parquet)
            plot_results(points, results, title=args.parquet.name)

        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())