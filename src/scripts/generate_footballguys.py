#!/usr/bin/env python3
# src/scripts/generate_footballguys.py

import json
import pandas as pd
import argparse
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR    = PROJECT_ROOT / "src" / "data" / "projection_data" / "footballguys"
OUTPUT_ROOT  = PROJECT_ROOT / "src" / "output" / "projections"

def main():
    parser = argparse.ArgumentParser(
        description="Generate Footballguys projections for a given week"
    )
    parser.add_argument(
        "--week",
        type=int,
        required=True,
        help="Week number to process (e.g. 2 for w2)"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Season year (defaults to 2025)"
    )
    args = parser.parse_args()

    YEAR = args.year
    WEEK = args.week

    # build paths
    infile = INPUT_DIR / f"footballguys_{YEAR}_w{WEEK}.csv"
    if not infile.exists():
        raise FileNotFoundError(f"Footballguys input not found: {infile}")

    out_dir  = OUTPUT_ROOT / f"{YEAR}_w{WEEK}"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_out = out_dir / f"footballguys_base_{YEAR}_w{WEEK}.json"

    # load
    print(f"→ Loading {infile}")
    df = pd.read_csv(infile)

    # keep only QB, RB, WR, TE
    df = df[df["pos"].str.lower().isin(["qb", "rb", "wr", "te"])]

    # select only the columns monte_carlo_sims.py uses
    cols_to_keep = [
        "id", "name", "pos", "team",
        "pass-att", "pass-cmp", "pass-int", "pass-td", "pass-yds",
        "rush-car", "rush-td", "rush-yds",
        "rec-tgt", "rec-rec", "rec-td", "rec-yds",
        "fum-lost",
        "pr-td", "pr-yds",
        "kr-td", "kr-yds"
    ]
    df = df[cols_to_keep]

    # write JSON
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    with open(json_out, "w") as fp:
        json.dump({"players": records}, fp, indent=2)
    print(f"✔ Wrote JSON → {json_out}")

if __name__ == "__main__":
    main()