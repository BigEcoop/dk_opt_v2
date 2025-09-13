#!/usr/bin/env python3
# src/scripts/merge_projections.py

import json
import pandas as pd
import argparse
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parents[2]
FG_JSON_DIR    = PROJECT_ROOT / "src" / "output" / "projections"
FP_CSV_ROOT    = PROJECT_ROOT / "src" / "data" / "projection_data" / "fantasypros"
OUTPUT_ROOT    = FG_JSON_DIR

# map hyphenated column names → internal snake_case stat keys
COL_MAP = {
  "pass-att":      "pass_attempts",
  "pass-cmp":      "pass_completions",
  "pass-yds":      "passing_yds",
  "pass-td":       "passing_tds",
  "pass-int":      "interceptions",
  "rush-car":      "rushing_attempts",
  "rush-yds":      "rushing_yds",
  "rush-td":       "rushing_tds",
  "rec-tgt":       "receiving_targets",
  "rec-rec":       "receptions",
  "rec-yds":       "receiving_yds",
  "rec-td":        "receiving_tds",
  "fum-lost":      "fumbles",
  "pr-yds":        "punt_return_yards",
  "pr-td":         "punt_return_tds",
  "kr-yds":        "kick_return_yds",
  "kr-td":         "kick_return_tds",
  "fpts":          "fantasy_points"
}

# stats your sims consume
ALL_STATS = [
  "pass_attempts","pass_completions","passing_yds","passing_tds","interceptions",
  "rushing_attempts","rushing_yds","rushing_tds",
  "receiving_targets","receptions","receiving_yds","receiving_tds",
  "fumbles","punt_return_yards","punt_return_tds",
  "kick_return_yds","kick_return_tds"
]

def main():
    p = argparse.ArgumentParser(description="Merge Footballguys + FantasyPros")
    p.add_argument("--year",  type=int, default=2025, help="Season year")
    p.add_argument("--week",  type=int, required=True, help="Week number")
    args = p.parse_args()
    YEAR, WEEK = args.year, args.week

    # ── Paths ───────────────────────────────────────────────────────────────────
    fg_json = FG_JSON_DIR / f"{YEAR}_w{WEEK}" / f"footballguys_base_{YEAR}_w{WEEK}.json"
    fp_dir  = FP_CSV_ROOT / f"{YEAR}_w{WEEK}" / "w_id"
    out_dir = OUTPUT_ROOT   / f"{YEAR}_w{WEEK}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"baseline_{YEAR}_w{WEEK}.json"
    out_csv  = out_dir / f"baseline_{YEAR}_w{WEEK}.csv"

    # ── 1. Load & normalize Footballguys JSON ─────────────────────────────────
    fg_records = json.loads(fg_json.read_text())["players"]
    df_fg = pd.json_normalize(fg_records)

    # flatten stats.* → stat columns, rename via COL_MAP, then suffix _fg
    df_fg.columns = [c.replace("stats.", "") for c in df_fg.columns]
    df_fg = df_fg.rename(columns={k: COL_MAP[k] for k in df_fg.columns if k in COL_MAP})
    df_fg = df_fg.add_suffix("_fg")

    # restore merge-keys
    df_fg = df_fg.rename(columns={
      "id_fg":   "id",
      "name_fg": "name",
      "team_fg": "team",
      "pos_fg":  "pos"
    })

    # ── 2. Load stamped FantasyPros CSVs ───────────────────────────────────────
    if not fp_dir.exists():
        raise FileNotFoundError(f"FantasyPros w_id folder missing: {fp_dir}")

    frames = []
    for csv_path in sorted(fp_dir.glob("*.csv")):
        df = pd.read_csv(csv_path)

        # normalize headers to hyphens
        df.columns = df.columns.str.strip().str.lower().str.replace("_","-")

        # rename 'player' → 'name' if present
        if "player" in df.columns:
            df = df.rename(columns={"player":"name"})

        # require id,name,team
        if not {"id","name","team"}.issubset(df.columns):
            raise KeyError(f"Missing id/name/team in {csv_path.name}")

        # tag position from filename: fantasypros_pos_year_wweek_with_id
        pos = csv_path.stem.split("_")[1]
        df["pos"] = pos

        # map hyphen stats → internal keys, suffix _fp
        rename_map = {k:v for k,v in COL_MAP.items() if k in df.columns}
        df = df.rename(columns=rename_map)
        fp_stats = list(rename_map.values()) + ["fantasy_points"]
        df = df.rename(columns={c: f"{c}_fp" for c in fp_stats})

        frames.append(df)

    df_fp = pd.concat(frames, ignore_index=True)

    # ── 3. Merge on id (outer to include extra FP players) ────────────────────
    df_merge = df_fg.merge(
        df_fp,
        on="id",
        how="outer",
        suffixes=("_fg","_fp"),
        indicator=True
    )

    # ── 4. Build merged player list ───────────────────────────────────────────
    out_players = []
    for _, row in df_merge.iterrows():
        stats = {}
        # average or pick single for each stat
        for st in ALL_STATS:
            fg_val = row.get(f"{st}_fg")
            fp_val = row.get(f"{st}_fp")
            if pd.notna(fg_val) or pd.notna(fp_val):
                stats[st] = float(pd.Series([fg_val, fp_val]).dropna().mean())
        # ensure every stat is present
        for st in ALL_STATS:
            stats.setdefault(st, 0.0)

        # carry through fantasy_points if present
        fp_pts = row.get("fantasy_points_fp")
        if pd.notna(fp_pts):
            stats["fantasy_points"] = float(fp_pts)

        # choose name/team/pos from FG if available, else FP
        name = row["name_fg"] if pd.notna(row.get("name_fg")) else row.get("name")
        team = row["team_fg"] if pd.notna(row.get("team_fg")) else row.get("team")
        pos  = row["pos_fg"]  if pd.notna(row.get("pos_fg"))  else row.get("pos")

        out_players.append({
            "name":  name,
            "id":    row["id"],
            "team":  team,
            "pos":   pos,
            "stats": stats
        })

    # ── 5. Write outputs ───────────────────────────────────────────────────────
    out_json.write_text(json.dumps({"players": out_players}, indent=2))
    print(f"Wrote JSON → {out_json}")

    df_out = pd.json_normalize(out_players)
    df_out.to_csv(out_csv, index=False)
    print(f"Wrote CSV  → {out_csv}")

if __name__ == "__main__":
    main()