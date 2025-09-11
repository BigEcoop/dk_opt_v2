#!/usr/bin/env python3
# src/scripts/salary_analysis.py

import argparse
import pandas as pd
from pathlib import Path

from optimizer.data_fetch import fetch_dk_salaries, fetch_fd_salaries
from optimizer.analysis import compare_fd_dk_salaries


def cmd_compare_salaries(args):
    # 1) Fetch DK + FD salaries
    dk = fetch_dk_salaries(
        draft_group_id=args.draft_group_id,
        week=args.week,
        contest_type_id=args.contest_type_id,
    )
    fd = fetch_fd_salaries(week=args.week)

    # 2) Ensure 'week' column exists for merging
    if "week" not in dk.columns:
        dk["week"] = args.week
    if "week" not in fd.columns:
        fd["week"] = args.week

    # 3) Compare and flag outliers
    df = compare_fd_dk_salaries(dk, fd, threshold=args.threshold)

    # 4) Compute percent difference relative to mean
    avg_diff = df["salary_diff"].mean()
    df["percent_diff"] = df["salary_diff"] / avg_diff * 100

    # 5) Assign business‐named tiers based on quartiles
    q1, q2, q3 = df["percent_diff"].quantile([0.25, 0.50, 0.75])
    df["DK_EXTREME_OVERPRICED"] = df["percent_diff"] <= q1
    df["DK_OVERPRICED"]         = (df["percent_diff"] > q1) & (df["percent_diff"] <= q2)
    df["DK_VALUE"]              = (df["percent_diff"] > q2) & (df["percent_diff"] <= q3)
    df["DK_EXTREME_VALUE"]      = df["percent_diff"] > q3

    print(f"\n-- FD vs DK Comparison (Week {args.week}, Δ>{args.threshold}) --")
    print(df.head())

    # 6) Write CSV to src/output/fd_vs_dk_w{week}.csv
    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "src" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.output) if args.output else out_dir / f"fd_vs_dk_w{args.week}.csv"
    df.to_csv(out_path, index=False)
    print(f"▶ Saved comparison to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="🔍 DFS Salary Comparison Utility"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "compare-salaries",
        help="Fetch DK+FD salaries, compare, flag outliers & tier by percent_diff"
    )
    p.add_argument("--week",            type=int, required=True)
    p.add_argument("--draft-group-id",  type=int, required=True)
    p.add_argument("--contest-type-id", type=int, default=21)
    p.add_argument(
        "--threshold",
        type=float,
        default=300,
        help="Flag any |salary_diff| > this value"
    )
    p.add_argument(
        "--output",
        help="Custom path for comparison CSV (defaults to src/output)"
    )
    p.set_defaults(func=cmd_compare_salaries)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()