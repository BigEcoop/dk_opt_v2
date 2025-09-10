#!/usr/bin/env python3
"""
generate_diehard_baselines.py

Loads Diehard week-specific projections for all players (Week 1)
and writes them to dh_base.json under …/data/.
"""

import json
import pandas as pd
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("/home/bigec/projects/dk_opt/src/optimizer/data")
CSV_FILE     = DATA_DIR / "diehard_projections.csv"
XLSX_FILE    = DATA_DIR / "diehard_projections.xlsx"
OUTPUT_JSON  = DATA_DIR / "dh_base.json"
TARGET_WEEK  = 1

def load_diehard() -> pd.DataFrame:
    """
    Read the Diehard projections file (CSV or XLSX).
    Filter to TARGET_WEEK if a 'Week' column exists.
    """
    if CSV_FILE.exists():
        df = pd.read_csv(CSV_FILE)
    elif XLSX_FILE.exists():
        df = pd.read_excel(XLSX_FILE, sheet_name=0, engine="openpyxl")
    else:
        raise FileNotFoundError(
            f"Diehard file not found. Place diehard_projections.csv or .xlsx in {DATA_DIR}"
        )

    # If there's a 'Week' column, filter to that week
    if "Week" in df.columns:
        df = df[df["Week"] == TARGET_WEEK].copy()

    return df

def main():
    print(f"→ Loading Diehard projections for Week {TARGET_WEEK}…")
    df = load_diehard()

    # Replace NaN with None so JSON has true nulls
    records = df.where(pd.notnull(df), None).to_dict(orient="records")

    # Write out the JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump({"players": records}, f, indent=2)

    print(f"✔ Wrote {len(records)} players to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()