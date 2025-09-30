#!/usr/bin/env python3
# src/scripts/gen_ownership_json.py

import argparse
import json
import re
import pandas as pd
from pathlib import Path

def normalize(name: str) -> str:
    """
    Lowercase, strip leading/trailing whitespace,
    and drop any character that is not a–z or space.
    This must match the normalize() used in optimize.py.
    """
    return re.sub(r'[^a-z ]', '', name.strip().lower())

def main(year: int, week: int):
    # locate project directories
    project_root = Path(__file__).resolve().parents[2]
    ownership_dir = (
        project_root
        / "src"
        / "data"
        / "projection_data"
        / "ownership"
    )

    # 1) Load the Excel for this week
    xlsx_path = ownership_dir / f"{year}_w{week}.xlsx"
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Ownership Excel not found: {xlsx_path}")
    df = pd.read_excel(xlsx_path, engine="openpyxl")

    # 2) Keep only the columns we care about
    df = df[["PLAYER", "POS", "TEAM", "RST%"]].copy()

    # 3) Convert RST% to a decimal ownership fraction
    df["ownership"] = df["RST%"] / 100.0

    # 4) Normalize PLAYER names exactly as optimize.py does
    df["name_norm"] = df["PLAYER"].apply(normalize)

    # 5) Build a normalized-name → ownership dict
    out_dict = dict(zip(df["name_norm"], df["ownership"]))

    # 6) Write JSON keyed by the normalized name
    out_path = ownership_dir / f"{year}_w{week}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_dict, indent=2))

    print(f"✅ Wrote normalized ownership JSON → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ownership JSON keyed by normalized PLAYER name"
    )
    parser.add_argument("--year", type=int,  required=True, help="Season year (e.g. 2025)")
    parser.add_argument("--week", type=int,  required=True, help="NFL week number (e.g. 3)")
    args = parser.parse_args()
    main(args.year, args.week)