#!/usr/bin/env python3
# src/scripts/add_dk_id.py

import argparse
import json
import html
import re
import pandas as pd
from pathlib import Path
from rapidfuzz import process, fuzz

# one‐off name overrides: former name → current name
NAME_OVERRIDES = {
    "Marquise Brown": "Hollywood Brown",
}

# ── FULL TEAM NAME → DRAFTKINGS TEAM CODE ────────────────────────────
TEAM_CODE = {
    "Arizona Cardinals":     "ARI","Atlanta Falcons":      "ATL",
    "Baltimore Ravens":      "BAL","Buffalo Bills":         "BUF",
    "Carolina Panthers":     "CAR","Chicago Bears":         "CHI",
    "Cincinnati Bengals":    "CIN","Cleveland Browns":      "CLE",
    "Dallas Cowboys":        "DAL","Denver Broncos":        "DEN",
    "Detroit Lions":         "DET","Green Bay Packers":     "GB",
    "Houston Texans":        "HOU","Indianapolis Colts":     "IND",
    "Jacksonville Jaguars":  "JAX","Kansas City Chiefs":     "KC",
    "Las Vegas Raiders":     "LV","Los Angeles Chargers":   "LAC",
    "Los Angeles Rams":      "LAR","Miami Dolphins":        "MIA",
    "Minnesota Vikings":     "MIN","New England Patriots":  "NE",
    "New Orleans Saints":    "NO","New York Giants":        "NYG",
    "New York Jets":         "NYJ","Philadelphia Eagles":    "PHI",
    "Pittsburgh Steelers":   "PIT","San Francisco 49ers":   "SF",
    "Seattle Seahawks":      "SEA","Tampa Bay Buccaneers":  "TB",
    "Tennessee Titans":      "TEN","Washington Commanders":  "WAS"
}

def normalize_name(s: str) -> str:
    s = html.unescape(str(s))
    s = re.sub(r"\b(II|III|IV|Jr\.?|Sr\.?)\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^\w\s']+", "", s)
    return s.upper().strip()

ROOT       = Path(__file__).resolve().parents[2]
SIMS_DIR   = ROOT / "src/output/sims"
SALARY_DIR = ROOT / "src/output/salary"

def load_salary(week: int) -> pd.DataFrame:
    sal = pd.read_csv(SALARY_DIR / f"fd_vs_dk_w{week}.csv")
    sal = sal.rename(columns={"player_name":"name","position":"pos","ID":"dk_id"})

    sal["team_raw"]   = sal["team"].astype(str).str.strip()
    sal["team"]       = sal["team_raw"].map(TEAM_CODE).fillna(sal["team_raw"].str.upper())
    sal["pos"]        = sal["pos"].fillna("DST").astype(str).str.upper().str.strip()
    sal["name_norm"]  = sal["name"].apply(normalize_name)
    sal = build_salary_ids(sal)

    return sal[["name","name_norm","team","pos","id","dk_id"]]

def build_salary_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    prefixes = []
    for nm in df["name"]:
        parts = nm.replace(" DST","").split()
        first, last = parts[0], parts[-1]
        prefixes.append((last[:4] + first[:2]).upper())
    df["prefix"] = prefixes
    df["suffix"] = df.groupby("prefix").cumcount().apply(lambda i: f"{i:02d}")
    df["id"]     = df["prefix"] + df["suffix"]
    df.drop(columns=["prefix","suffix","team_raw"], inplace=True)
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--week", type=int, required=True)
    p.add_argument("--year", type=int, required=True)
    args = p.parse_args()

    sal_df      = load_salary(args.week)
    sims_folder = SIMS_DIR / f"{args.year}_w{args.week}_all"
    excel_pth   = sims_folder / "player_summary.xlsx"
    json_pth    = sims_folder / "player_summary.json"

    # 1) Load & normalize Excel
    xlsx = pd.read_excel(excel_pth)
    xlsx["player"]    = xlsx["player"].replace(NAME_OVERRIDES)
    xlsx["team"]      = xlsx["team"].str.upper().str.strip()
    xlsx["pos"]       = xlsx["pos"].fillna("DST").str.upper().str.strip()
    xlsx["name_norm"] = xlsx["player"].apply(normalize_name)

    # 2) Load JSON
    sims_json = json.loads(json_pth.read_text())

    # 3) Exact merge on normalized name, team, pos
    merged = xlsx.merge(
        sal_df,
        left_on=["name_norm","team","pos"],
        right_on=["name_norm","team","pos"],
        how="left",
        suffixes=("","_sal")
    )

    # 4) Fill any existing JSON dk_id if exact merge failed
    merged["dk_id"] = merged["dk_id"].fillna(
        merged["player"].map(lambda p: sims_json.get(p, {}).get("dk_id",""))
    )

    # 5) Fuzzy fallback
    missing = merged["dk_id"].isna()
    fuzzy = 0
    for i, row in merged[missing].iterrows():
        cands = sal_df[(sal_df.team == row.team) & (sal_df.pos == row.pos)]
        if cands.empty:
            continue
        match = process.extractOne(row.name_norm, cands["name_norm"], scorer=fuzz.WRatio)
        if match and match[1] >= 85:
            dk = cands.loc[cands["name_norm"] == match[0], "dk_id"].iloc[0]
            merged.at[i, "dk_id"] = dk
            fuzzy += 1

    # 6) One‐off override for Hollywood Brown
    try:
        hb_id = sal_df.loc[sal_df["name"] == "Hollywood Brown", "dk_id"].iat[0]
        merged.loc[merged["player"] == "Hollywood Brown", "dk_id"] = hb_id
        print(f"💥 Forced dk_id {hb_id} for Hollywood Brown")
    except IndexError:
        print("⚠️ Could not find Hollywood Brown in salary table")

    exact = merged["dk_id"].notna().sum() - fuzzy
    total = merged["dk_id"].notna().sum()
    print(f"🔍 Exact mapped {exact}; fuzzy mapped {fuzzy}; total {total}")

    # 7) Write back Excel
    xlsx["dk_id"] = merged["dk_id"]
    xlsx.to_excel(excel_pth, index=False)
    print(f"✅ Updated Excel: {excel_pth}")

    # 8) Write back JSON with Python-native ints
    for player, rec in list(sims_json.items()):
        lookup = NAME_OVERRIDES.get(player, player)
        mask   = merged["player"] == lookup
        if mask.any():
            raw_val = merged.loc[mask, "dk_id"].iat[0]
            # cast any numpy scalar to Python int
            try:
                rec["dk_id"] = int(raw_val)
            except Exception:
                rec["dk_id"] = raw_val
        else:
            print(f"⚠️ No dk_id found for '{player}' (tried '{lookup}')")

    json_pth.write_text(json.dumps(sims_json, indent=2))
    print(f"✅ Updated JSON:  {json_pth}")

if __name__ == "__main__":
    main()