#!/usr/bin/env python3

import argparse
import pandas as pd
from pathlib import Path

from optimizer.data_fetch import fetch_dk_salaries, fetch_fd_salaries
from optimizer.analysis.salary_analysis import (
    compare_fd_dk_salaries,
    track_weekly_salary_changes,
    compare_salary_position_rankings,
    compute_value_score,
)


def cmd_compare_salaries(args):
    # 1) Fetch live DK/FD salaries
    dk = fetch_dk_salaries(
        draft_group_id=args.draft_group_id,
        week=args.week,
        contest_type_id=args.contest_type_id,
    )
    fd = fetch_fd_salaries(week=args.week)

    # 2) Compare and flag
    df = compare_fd_dk_salaries(dk, fd, threshold=args.threshold)

    print(f"\n-- FD vs DK Comparison (Week {args.week}, Δ>{args.threshold}) --")
    print(df.head())

    # 3) Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        # project root = two levels up from this script
        proj_root = Path(__file__).resolve().parent.parent
        out_dir = (
            proj_root
            / "src"
            / "optimizer"
            / "data"
            / "output"
            / "analysis"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"fd_vs_dk_w{args.week}.csv"

    # 4) Write CSV
    df.to_csv(out_path, index=False)
    print(f"▶ Saved comparison to {out_path}")


def cmd_track_changes(args):
    hist = pd.read_csv(args.hist_file)
    df = track_weekly_salary_changes(hist)

    print("\n-- Weekly Salary Changes --")
    print(df.tail())

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"▶ Saved weekly changes to {out}")


def cmd_compare_positions(args):
    dk = fetch_dk_salaries(
        draft_group_id=args.draft_group_id,
        week=args.week,
        contest_type_id=args.contest_type_id,
    )
    pos = pd.read_csv(args.position_file)

    df = compare_salary_position_rankings(dk, pos)

    print(f"\n-- Salary vs Position Rank (Week {args.week}) --")
    print(df.head())

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"▶ Saved position comparison to {out}")


def cmd_compute_value(args):
    dk = fetch_dk_salaries(
        draft_group_id=args.draft_group_id,
        week=args.week,
        contest_type_id=args.contest_type_id,
    )
    proj = pd.read_csv(args.projections_file)

    df = compute_value_score(dk, proj)

    print(f"\n-- Value Scores (Week {args.week}) --")
    print(df.head())

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"▶ Saved value scores to {out}")


def main():
    parser = argparse.ArgumentParser(
        description="🔍 DFS Salary Analysis Utilities"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # compare-salaries
    p1 = sub.add_parser(
        "compare-salaries",
        help="Fetch live DK/FD salaries, compare & flag outliers"
    )
    p1.add_argument("--week",            type=int, required=True)
    p1.add_argument("--draft-group-id",  type=int, required=True)
    p1.add_argument("--contest-type-id", type=int, default=21)
    p1.add_argument(
        "--threshold",
        type=float,
        default=300,
        help="Flag any |salary_diff| > this value"
    )
    p1.add_argument(
        "--output",
        help="Path to write comparison CSV (defaults to data/output/analysis/...)"
    )
    p1.set_defaults(func=cmd_compare_salaries)

    # track-changes
    p2 = sub.add_parser(
        "track-changes",
        help="Load historical salaries CSV, compute weekly Δs"
    )
    p2.add_argument(
        "--hist-file",
        required=True,
        help="Path to processed/all_salaries.csv"
    )
    p2.add_argument(
        "--output",
        help="Write weekly changes to this CSV path"
    )
    p2.set_defaults(func=cmd_track_changes)

    # compare-positions
    p3 = sub.add_parser(
        "compare-positions",
        help="Compare DK salaries vs. position rankings"
    )
    p3.add_argument("--week",            type=int, required=True)
    p3.add_argument("--draft-group-id",  type=int, required=True)
    p3.add_argument("--contest-type-id", type=int, default=21)
    p3.add_argument(
        "--position-file",
        required=True,
        help="Path to position_rankings.csv"
    )
    p3.add_argument(
        "--output",
        help="Write position comparison to this CSV path"
    )
    p3.set_defaults(func=cmd_compare_positions)

    # compute-value
    p4 = sub.add_parser(
        "compute-value",
        help="Compute DK value scores from projections"
    )
    p4.add_argument("--week",            type=int, required=True)
    p4.add_argument("--draft-group-id",  type=int, required=True)
    p4.add_argument("--contest-type-id", type=int, default=21)
    p4.add_argument(
        "--projections-file",
        required=True,
        help="Path to projections.csv"
    )
    p4.add_argument(
        "--output",
        help="Write value scores to this CSV path"
    )
    p4.set_defaults(func=cmd_compute_value)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()