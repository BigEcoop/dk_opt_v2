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

FD_TO_DK_TEAM = {
    "JAC": "JAX",
}

def cmd_compare_salaries(args):
    project_root = Path(__file__).resolve().parents[2]
    local_path   = project_root / "src" / "data" / "salary_input" / f"dk_{args.year}_w{args.week}.csv"
    if local_path.exists():
        dk = pd.read_csv(local_path)
        dk = dk.rename(columns={
            "Name":       "player_name",
            "TeamAbbrev": "team",
            "Position":   "position",
            "Salary":     "dk_salary",
        })
    else:
        dk = fetch_dk_salaries(
            draft_group_id=args.draft_group_id,
            week=args.week,
            contest_type_id=args.contest_type_id,
        )

    print("⚡ teams in DK feed:", sorted(dk["team"].astype(str).unique()))

    fd = fetch_fd_salaries(week=args.week)
    fd["team"] = (
        fd["team"]
          .astype(str)
          .str.upper()
          .str.strip()
          .replace(FD_TO_DK_TEAM)
    )
    if "position" not in fd.columns and "roster_position" in fd.columns:
        fd = fd.rename(columns={"roster_position": "position"})
    fd["position"] = fd["position"].fillna("DST").astype(str).str.upper().str.strip()

    if "week" not in dk.columns:
        dk["week"] = args.week
    if "week" not in fd.columns:
        fd["week"] = args.week

    # Rename FD salary to fd_salary for consistency
    if "salary" in fd.columns:
        fd = fd.rename(columns={"salary": "fd_salary"})

    df = compare_fd_dk_salaries(dk, fd, threshold=args.threshold)

    avg_diff = df["salary_diff"].mean()
    df["percent_diff"] = df["salary_diff"] / avg_diff * 100
    q1, q2, q3 = df["percent_diff"].quantile([0.25, 0.50, 0.75])
    df["DK_EXTREME_OVERPRICED"] = df["percent_diff"] <= q1
    df["DK_OVERPRICED"]         = (df["percent_diff"] > q1) & (df["percent_diff"] <= q2)
    df["DK_VALUE"]              = (df["percent_diff"] > q2) & (df["percent_diff"] <= q3)
    df["DK_EXTREME_VALUE"]      = df["percent_diff"] > q3

    id_file = project_root / "src" / "data" / "salary_input" / f"dk_{args.year}_w{args.week}.csv"
    if id_file.exists():
        id_df = pd.read_csv(id_file)
        cols = id_df.columns.tolist()
        name_cols = [c for c in cols if "name" in c.lower() and "team" not in c.lower()]
        team_cols = [c for c in cols if "team" in c.lower()]
        id_cols   = [c for c in cols if c.strip().lower() == "id"]
        if name_cols and team_cols and id_cols:
            nc = name_cols[0] if len(name_cols)==1 else "Name"
            tc = team_cols[0]
            ic = id_cols[0]
            id_df = (
                id_df[[nc, tc, ic]]
                 .rename(columns={nc:"player_name", tc:"team", ic:"ID"})
            )
            df = df.merge(id_df, on=["player_name","team"], how="left")
        sal_in = pd.read_csv(id_file)
        dst_in = sal_in[sal_in.get("Position","")=="DST"]
        if not dst_in.empty:
            dst_rows = []
            for _, r in dst_in.iterrows():
                dst_rows.append({
                    "player_name":           r["Name"],
                    "team":                  r["TeamAbbrev"],
                    "dk_salary":             r["Salary"],
                    "fd_salary":             r["Salary"],
                    "salary_diff":           0.0,
                    "percent_diff":          0.0,
                    "DK_EXTREME_OVERPRICED": False,
                    "DK_OVERPRICED":         False,
                    "DK_VALUE":              False,
                    "DK_EXTREME_VALUE":      False,
                    "ID":                    str(r["ID"]),
                })
            df = pd.concat([df, pd.DataFrame(dst_rows)], ignore_index=True)
    else:
        print(f"⚠️ Could not merge ID from {id_file}")

    df = df.round(2)
    out_dir = project_root / "src" / "output" / "salary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else out_dir / f"fd_vs_dk_w{args.week}.csv"
    df.to_csv(out_path, index=False)
    print(f"▶ Saved comparison to {out_path}")

def cmd_track_changes(args):
    week = args.week
    year = args.year
    project_root = Path(__file__).resolve().parents[2]
    comp_file    = project_root / "src" / "output" / "salary" / f"fd_vs_dk_w{week}.csv"
    if not comp_file.exists():
        raise FileNotFoundError(f"Comparison CSV not found: {comp_file}")

    prev_dk = get_draftkings_salaries_csv(week-1, year)[["player_name","team","salary"]].rename(columns={"salary":"dk_prev_salary"})
    curr_dk = get_draftkings_salaries_csv(week,   year)[["player_name","team","salary"]].rename(columns={"salary":"dk_salary"})
    dk_delta = pd.merge(curr_dk, prev_dk, on=["player_name","team"], how="left")
    dk_delta["dk_salary_change"]  = dk_delta["dk_salary"] - dk_delta["dk_prev_salary"]
    dk_delta["dk_salary_percent"] = (dk_delta["dk_salary_change"]/dk_delta["dk_prev_salary"]*100).round(2)

    prev_fd = get_fanduel_salaries_csv(week-1, year)[["player_name","team","salary"]].rename(columns={"salary":"fd_prev_salary"})
    curr_fd = get_fanduel_salaries_csv(week,   year)[["player_name","team","salary"]].rename(columns={"salary":"fd_salary"})
    fd_delta = pd.merge(curr_fd, prev_fd, on=["player_name","team"], how="left")
    fd_delta["fd_salary_change"]  = fd_delta["fd_salary"] - fd_delta["fd_prev_salary"]
    fd_delta["fd_salary_percent"] = (fd_delta["fd_salary_change"]/fd_delta["fd_prev_salary"]*100).round(2)

    # load existing comparison and drop old salary columns to avoid _x/_y conflicts
    comp_df = pd.read_csv(comp_file)
    comp_df = comp_df.drop(columns=["dk_salary","fd_salary"], errors="ignore")

    merged = (
        comp_df
          .merge(dk_delta, on=["player_name","team"], how="left")
          .merge(fd_delta, on=["player_name","team"], how="left")
    )

    merged.to_csv(comp_file, index=False)
    print(f"▶ Appended salary-change columns to {comp_file}")

def main():
    parser = argparse.ArgumentParser(description="🔍 DFS Salary Analysis Utility")
    subs   = parser.add_subparsers(dest="cmd", required=True)

    p1 = subs.add_parser("compare-salaries", help="Fetch DK+FD, compare & tier")
    p1.add_argument("--week",           type=int, required=True)
    p1.add_argument("--year",           type=int, default=2025)
    p1.add_argument("--draft-group-id", type=int, required=True)
    p1.add_argument("--contest-type-id",type=int, default=21)
    p1.add_argument("--threshold",      type=float, default=300)
    p1.add_argument("--output",         help="Custom path for comparison CSV")
    p1.set_defaults(func=cmd_compare_salaries)

    p2 = subs.add_parser("track-changes", help="Append DK/FD salary changes")
    p2.add_argument("--week", type=int, required=True)
    p2.add_argument("--year", type=int, default=2025)
    p2.set_defaults(func=cmd_track_changes)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()