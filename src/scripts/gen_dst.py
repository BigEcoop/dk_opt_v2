#!/usr/bin/env python3
# src/scripts/gen_dst.py

import argparse
import json
import pandas as pd
from pathlib import Path

def generate_dst_json(year: int, week: int):
    # 1) Locate your excel file
    excel_path = (
        Path(__file__).resolve().parents[2]
        / "src" / "data" / "projection_data" / "dst"
        / f"dst_{year}_w{week}.xlsx"
    )
    if not excel_path.exists():
        raise FileNotFoundError(f"No DST Excel at {excel_path}")

    # 2) Read and clean
    df = pd.read_excel(excel_path, engine="openpyxl")
    df = df[["PLAYER", "POS", "VAL", "RST%"]]
    
    # 3) Compute decimal ownership and sort by VAL desc
    df["ownership"] = df["RST%"] / 100.0
    df = df.sort_values("VAL", ascending=False).reset_index(drop=True)
    
    # 4) Assign rank (1 = highest VAL)
    df["rank"] = df.index + 1
    
    # 5) Build output dict
    out = {}
    for _, row in df.iterrows():
        out[row["PLAYER"]] = {
            "pos":       row["POS"],
            "val":       float(row["VAL"]),
            "ownership": float(row["ownership"]),
            "rank":      int(row["rank"])
        }
    
    # 6) Write JSON next to your Excel
    out_path = excel_path.with_suffix(".json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote DST JSON → {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=2025, help="Season year")
    p.add_argument("--week", type=int, required=True, help="NFL week")
    args = p.parse_args()
    generate_dst_json(args.year, args.week)