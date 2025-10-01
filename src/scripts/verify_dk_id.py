#!/usr/bin/env python3
# verify_dk_id.py

import argparse
import pandas as pd
from pathlib import Path

def _find_pool_file(base_dir: Path, week: int, year: int, ext: str) -> Path:
    """
    Look for optimizer_pool files named either
      optimizer_pool_<year>_w<week>.<ext>
    or
      optimizer_pool_w<week>.<ext>
    """
    fn_with_year = f"optimizer_pool_{year}_w{week}.{ext}"
    p_with_year  = base_dir / fn_with_year
    if p_with_year.exists():
        return p_with_year

    fn           = f"optimizer_pool_w{week}.{ext}"
    p            = base_dir / fn
    if p.exists():
        return p

    raise FileNotFoundError(
        f"No pool file found. Tried:\n  {p_with_year}\n  {p}"
    )

def main():
    parser = argparse.ArgumentParser(
        description="Verify that every row in your optimizer pool has a non-blank dk_id"
    )
    parser.add_argument(
        "--week", type=int, required=True,
        help="Week number to verify"
    )
    parser.add_argument(
        "--year", type=int, required=True,
        help="Season year (used if filenames include year)"
    )
    args = parser.parse_args()

    base = Path("src/output/optimizer_pool")
    csv_path  = _find_pool_file(base, args.week, args.year, "csv")
    json_path = _find_pool_file(base, args.week, args.year, "json")

    # Load both files with dk_id as string to preserve blanks
    df_csv  = pd.read_csv(csv_path,  dtype={"dk_id": str})
    df_json = pd.read_json(json_path, dtype={"dk_id": str})

    # Count blank or missing dk_id entries
    miss_csv  = df_csv["dk_id"].fillna("").eq("").sum()
    miss_json = df_json["dk_id"].fillna("").eq("").sum()

    print(f"\nCSV file:  {csv_path}")
    print(f"JSON file: {json_path}\n")
    print(f"→ CSV missing dk_id:  {miss_csv}")
    print(f"→ JSON missing dk_id: {miss_json}")

    if (miss_csv + miss_json) == 0:
        print("\n✅ All rows have a valid dk_id!")
    else:
        print("\n❌ Some rows are missing a dk_id – please investigate above counts.")

if __name__ == "__main__":
    main()