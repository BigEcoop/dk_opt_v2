#!/usr/bin/env python3
# src/scripts/salary_analysis.py

import argparse
import pandas as pd
from pathlib import Path

from optimizer.data_fetch            import fetch_dk_salaries, fetch_fd_salaries
from optimizer.analysis              import compare_fd_dk_salaries
from optimizer.salary_ingestor_csv   import (
    get_draftkings_salaries_csv,
    get_fanduel_salaries_csv,
)

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

    # 5) Assign business-named tiers based on quartiles
    q1, q2, q3 = df["percent_diff"].quantile([0.25, 0.50, 0.75])
    df["DK_EXTREME_OVERPRICED"] = df["percent_diff"] <= q1
    df["DK_OVERPRICED"]         = (df["percent_diff"] > q1) & (df["percent_diff"] <= q2)
    df["DK_VALUE"]              = (df["percent_diff"] > q2) & (df["percent_diff"] <= q3)
    df["DK_EXTREME_VALUE"]      = df["percent_diff"] > q3

    # 5.1) Merge in DK IDs from raw DK fallback file
    project_root = Path(__file__).resolve().parents[2]
    id_file = project_root / "src" / "data" / "salary_input" / f"dk_{args.year}_w{args.week}.csv"

    if id_file.exists():
        id_df = pd.read_csv(id_file)
        cols = id_df.columns.tolist()

        # Detect player-name column (contains "name" but not "team")
        name_candidates = [c for c in cols if "name" in c.lower() and "team" not in c.lower()]
        # Detect team column (contains "team")
        team_candidates = [c for c in cols if "team" in c.lower()]
        # Detect ID column (exactly "id", case-insensitive)
        id_candidates   = [c for c in cols if c.strip().lower() == "id"]

        if name_candidates and team_candidates and id_candidates:
            exact_name = [c for c in name_candidates if c.lower() == "name"]
            name_col = exact_name[0] if exact_name else name_candidates[0]
            team_col = team_candidates[0]
            id_col   = id_candidates[0]

            id_df = (
                id_df[[name_col, team_col, id_col]]
                .rename(columns={
                    name_col: "player_name",
                    team_col: "team",
                    id_col:   "ID"
                })
            )
            df = df.merge(id_df, on=["player_name", "team"], how="left")
        else:
            print(f"⚠️ Could not merge ID from {id_file}. Found columns: {cols}")

        # 5.1.1) Append DST entries from salary_input
        sal_in = pd.read_csv(id_file)
        dst_in = sal_in[sal_in["Position"] == "DST"]
        if not dst_in.empty:
            dst_rows = []
            for _, r in dst_in.iterrows():
                dst_rows.append({
                    "player_name":         r["Name"],
                    "team":                r["TeamAbbrev"],
                    "dk_salary":           r["Salary"],
                    "fd_salary":           r["Salary"],
                    "salary_diff":         0.0,
                    "percent_diff":        0.0,
                    "DK_EXTREME_OVERPRICED": False,
                    "DK_OVERPRICED":         False,
                    "DK_VALUE":              False,
                    "DK_EXTREME_VALUE":      False,
                    "ID":                   str(r["ID"]),
                })
            df = pd.concat([df, pd.DataFrame(dst_rows)], ignore_index=True)
    else:
        print(f"⚠️ Could not merge ID from {id_file}, file not found")

    # 5.2) Round all numeric columns to two decimals
    df = df.round(2)

    # 6) Write comparison CSV
    out_dir = project_root / "src" / "output" / "salary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / f"fd_vs_dk_w{args.week}.csv"
    df.to_csv(out_path, index=False)
    print(f"▶ Saved comparison to {out_path}")

def cmd_track_changes(args):
    """
    1) Load raw DK CSVs for week-1 & week, compute dk_salary_change & dk_salary_percent.
    2) Load raw FD  CSVs for week-1 & week, compute fd_salary_change & fd_salary_percent.
    3) Left-join these four new columns onto fd_vs_dk_w{week}.csv via player_name+team.
    """
    week = args.week
    year = args.year

    project_root = Path(__file__).resolve().parents[2]
    out_dir      = project_root / "src" / "output" / "salary"
    comp_file    = out_dir / f"fd_vs_dk_w{week}.csv"

    if not comp_file.exists():
        raise FileNotFoundError(f"Comparison CSV not found: {comp_file}")

    # -- DraftKings change --
    prev_dk = get_draftkings_salaries_csv(week - 1, year)[["player_name", "team", "salary"]]
    curr_dk = get_draftkings_salaries_csv(week,     year)[["player_name", "team", "salary"]]
    prev_dk = prev_dk.rename(columns={"salary": "dk_prev_salary"})
    curr_dk = curr_dk.rename(columns={"salary": "dk_salary"})

    dk_delta = pd.merge(curr_dk, prev_dk, on=["player_name", "team"], how="left")
    dk_delta["dk_salary_change"]  = dk_delta["dk_salary"] - dk_delta["dk_prev_salary"]
    dk_delta["dk_salary_percent"] = (dk_delta["dk_salary_change"] / dk_delta["dk_prev_salary"] * 100).round(2)
    dk_delta = dk_delta[["player_name", "team", "dk_salary_change", "dk_salary_percent"]]

    # -- FanDuel change --
    prev_fd = get_fanduel_salaries_csv(week - 1, year)[["player_name", "team", "salary"]]
    curr_fd = get_fanduel_salaries_csv(week,     year)[["player_name", "team", "salary"]]
    prev_fd = prev_fd.rename(columns={"salary": "fd_prev_salary"})
    curr_fd = curr_fd.rename(columns={"salary": "fd_salary"})

    fd_delta = pd.merge(curr_fd, prev_fd, on=["player_name", "team"], how="left")
    fd_delta["fd_salary_change"]  = fd_delta["fd_salary"] - fd_delta["fd_prev_salary"]
    fd_delta["fd_salary_percent"] = (fd_delta["fd_salary_change"] / fd_delta["fd_prev_salary"] * 100).round(2)
    fd_delta = fd_delta[["player_name", "team", "fd_salary_change", "fd_salary_percent"]]

    # -- Merge deltas onto comparison CSV --
    comp_df = pd.read_csv(comp_file)
    merged  = comp_df.merge(dk_delta, on=["player_name", "team"], how="left") \
                     .merge(fd_delta, on=["player_name", "team"], how="left")

    merged.to_csv(comp_file, index=False)
    print(f"▶ Appended salary-change columns to {comp_file}")

def main():
    parser = argparse.ArgumentParser(
        description="🔍 DFS Salary Analysis Utility"
    )
    subs = parser.add_subparsers(dest="cmd", required=True)

    # compare-salaries
    p1 = subs.add_parser(
        "compare-salaries",
        help="Fetch DK+FD salaries, compare, flag & tier by percent_diff"
    )
    p1.add_argument("--week",           type=int,   required=True)
    p1.add_argument("--year",           type=int,   default=2025)
    p1.add_argument("--draft-group-id", type=int,   required=True)
    p1.add_argument("--contest-type-id",type=int,   default=21)
    p1.add_argument(
        "--threshold",
        type=float,
        default=300,
        help="Flag any |salary_diff| > this value"
    )
    p1.add_argument(
        "--output",
        help="Custom path for comparison CSV (defaults to src/output/salary)"
    )
    p1.set_defaults(func=cmd_compare_salaries)

    # track-changes
    p2 = subs.add_parser(
        "track-changes",
        help="Append DK/FD raw salary changes to comparison CSV"
    )
    p2.add_argument("--week", type=int, required=True)
    p2.add_argument("--year", type=int, default=2025)
    p2.set_defaults(func=cmd_track_changes)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()