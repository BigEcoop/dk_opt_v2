#!/usr/bin/env python3
# src/scripts/assign_ids.py

import re
import argparse
from pathlib import Path

import pandas as pd
from rapidfuzz import process, fuzz

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FG_CSV_DIR   = PROJECT_ROOT / "src" / "data" / "projection_data" / "footballguys"
FP_ROOT      = PROJECT_ROOT / "src" / "data" / "projection_data" / "fantasypros"

def clean_name(raw: str) -> str:
    """Uppercase, strip Jr/Sr/II/III/IV, remove punctuation, collapse spaces."""
    if pd.isna(raw):
        return ""
    s = raw.upper()
    s = re.sub(r"\b(JR|SR|II|III|IV)\b", "", s)
    s = re.sub(r"[^A-Z ]+", "", s)
    return re.sub(r"\s+", " ", s).strip()

def fuzzy_find_id(name_c: str, lookup: dict, cutoff: int = 85) -> str | None:
    """Return the closest matching FG ID for a clean name, or None if below cutoff."""
    match = process.extractOne(name_c, lookup.keys(), scorer=fuzz.WRatio, score_cutoff=cutoff)
    return lookup[match[0]] if match else None

def generate_new_id(full_name: str, used_ids: set) -> str:
    """
    Build a new ID: first 4 of last name + first 2 of first name + two-digit suffix.
    Increments suffix until an unused ID is found.
    """
    parts = full_name.strip().split()
    if len(parts) < 2:
        base_last = parts[0][:4].upper().ljust(4, "X")
        base_first = parts[0][:2].upper().ljust(2, "X")
    else:
        first, last = parts[0], parts[-1]
        base_last  = last[:4].upper().ljust(4, "X")
        base_first = first[:2].upper().ljust(2, "X")

    base = base_last + base_first
    for i in range(100):
        suffix = f"{i:02d}"
        candidate = base + suffix
        if candidate not in used_ids:
            used_ids.add(candidate)
            return candidate
    raise RuntimeError(f"Exhausted ID space for {full_name}")

def main():
    p = argparse.ArgumentParser(
        description="Stamp FantasyPros CSVs with Footballguys IDs (exact, fuzzy, or generated)"
    )
    p.add_argument("--year", type=int, required=True, help="Season year")
    p.add_argument("--week", type=int, required=True, help="Week number")
    args = p.parse_args()
    year, week = args.year, args.week

    # 1) Load FG lookup table
    fg_csv = FG_CSV_DIR / f"footballguys_{year}_w{week}.csv"
    if not fg_csv.exists():
        raise FileNotFoundError(f"Footballguys CSV not found: {fg_csv}")
    df_fg = pd.read_csv(fg_csv, usecols=["id", "name", "team"])
    df_fg["name_clean"]  = df_fg["name"].apply(clean_name)
    df_fg["team_clean"]  = df_fg["team"].astype(str).str.strip().str.upper()

    used_ids = set(df_fg["id"].tolist())
    fg_lookup = dict(zip(df_fg["name_clean"], df_fg["id"]))

    # 2) Prepare FP CSV directory and output folder
    fp_csv_dir = FP_ROOT / f"{year}_w{week}" / "csv"
    if not fp_csv_dir.exists():
        raise FileNotFoundError(f"FantasyPros CSV folder not found: {fp_csv_dir}")
    out_dir = FP_ROOT / f"{year}_w{week}" / "w_id"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3) Process each position CSV
    for csv_path in sorted(fp_csv_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)

        # normalize header names
        df.columns = df.columns.str.strip().str.lower()
        if "player" in df.columns:
            df = df.rename(columns={"player": "name"})
        if "name" not in df.columns or "team" not in df.columns:
            raise KeyError(f"Missing name/team in {csv_path.name}")

        # clean up fields
        df["name"]       = df["name"].astype(str).str.strip()
        df["team"]       = df["team"].astype(str).str.strip().str.upper()
        df["name_clean"] = df["name"].apply(clean_name)
        df["team_clean"] = df["team"]

        # exact merge
        df = df.merge(
            df_fg[["id", "name_clean", "team_clean"]],
            on=["name_clean", "team_clean"],
            how="left"
        )
        missing_exact = int(df["id"].isna().sum())

        # fuzzy fill
        mask = df["id"].isna()
        df.loc[mask, "id"] = df.loc[mask, "name_clean"].apply(
            lambda nc: fuzzy_find_id(nc, fg_lookup)
        )
        missing_fuzzy = int(df["id"].isna().sum())

        # generate IDs for any still missing
        mask2 = df["id"].isna()
        for idx in df[mask2].index:
            new_id = generate_new_id(df.at[idx, "name"], used_ids)
            df.at[idx, "id"] = new_id

        missing_final = int(df["id"].isna().sum())

        print(
            f"{csv_path.name}: {len(df)} rows, "
            f"{missing_exact} missing after exact, "
            f"{missing_fuzzy} after fuzzy, "
            f"{missing_final} after generate"
        )

        # write stamped CSV
        out_csv = out_dir / f"{csv_path.stem}_with_id.csv"
        df.to_csv(out_csv, index=False)
        print(f"✔ Wrote → {out_csv.name}")

if __name__ == "__main__":
    main()