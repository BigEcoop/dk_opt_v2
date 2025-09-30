#!/usr/bin/env python3
# src/scripts/optimize.py

import argparse
import json
import pandas as pd
from pathlib import Path

# ── FULL TEAM NAME → DRAFTKINGS TEAM CODE ────────────────────────────
TEAM_CODE = {
  "Arizona Cardinals":"ARI","Atlanta Falcons":"ATL","Baltimore Ravens":"BAL",
  "Buffalo Bills":"BUF","Carolina Panthers":"CAR","Chicago Bears":"CHI",
  "Cincinnati Bengals":"CIN","Cleveland Browns":"CLE","Dallas Cowboys":"DAL",
  "Denver Broncos":"DEN","Detroit Lions":"DET","Green Bay Packers":"GB",
  "Houston Texans":"HOU","Indianapolis Colts":"IND","Jacksonville Jaguars":"JAX",
  "Kansas City Chiefs":"KC","Las Vegas Raiders":"LV",
  "Los Angeles Chargers":"LAC","Los Angeles Rams":"LAR","Miami Dolphins":"MIA",
  "Minnesota Vikings":"MIN","New England Patriots":"NE","New Orleans Saints":"NO",
  "New York Giants":"NYG","New York Jets":"NYJ","Philadelphia Eagles":"PHI",
  "Pittsburgh Steelers":"PIT","San Francisco 49ers":"SF","Seattle Seahawks":"SEA",
  "Tampa Bay Buccaneers":"TB","Tennessee Titans":"TEN","Washington Commanders":"WAS"
}

# reverse map: code → full team name
CODE_TO_TEAM = {v: k for k, v in TEAM_CODE.items()}

# ── OUTPUT SCHEMA & DEFAULTS ─────────────────────────────────────────
UNIFIED_SCHEMA = [
  "dk_id","name","team","pos",
  "dk_salary","fd_salary","salary_diff","percent_diff",
  "proj_mean","proj_ceiling","proj_floor",
  "implied_team_total","game_spread",
  "home_win_prob","away_win_prob",
  "def_vs_pos","proj_own","exposure_limit"
]

DEFAULTS = {
  "dk_salary": 0, "fd_salary": 0, "salary_diff": 0.0, "percent_diff": 0.0,
  "proj_mean": 0.0, "proj_ceiling": 0.0, "proj_floor": 0.0,
  "implied_team_total": 0.0, "game_spread": 0.0,
  "home_win_prob": 0.0, "away_win_prob": 0.0,
  "def_vs_pos": 16, "proj_own": 0.0, "exposure_limit": 7
}

# ── PATHS ─────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
SIMS_DIR    = ROOT / "src/output/sims"
SALARY_DIR  = ROOT / "src/output/salary"
DEF_RANKS   = ROOT / "src/data/projection_data/defensive_ranks"
OWNERSHIP   = ROOT / "src/data/projection_data/ownership"
DST_DIR     = ROOT / "src/data/projection_data/dst"


def load_salary(week: int) -> pd.DataFrame:
    """
    Load FD vs DK salary CSV, rename columns, treat blank 'pos' as DST,
    and rename DST rows to '<Full Team Name> DST'.
    """
    sal = pd.read_csv(SALARY_DIR / f"fd_vs_dk_w{week}.csv")
    sal = sal.rename(columns={
      "player_name": "name",
      "position":    "pos",
      "ID":          "dk_id"
    })
    sal["dk_id"] = sal["dk_id"].astype(str)

    # blank pos → DST
    sal["pos"] = sal["pos"].fillna("").astype(str)
    dst_mask = sal["pos"] == ""
    sal.loc[dst_mask, "pos"] = "DST"

    # map code back to full name for DST entries
    sal.loc[dst_mask, "name"] = (
      sal.loc[dst_mask, "team"].map(CODE_TO_TEAM) + " DST"
    )
    return sal


def load_player_summary(path: Path) -> pd.DataFrame:
    """
    Load player_summary.json keyed by player name; extract dk_points.
    """
    raw = json.loads(path.read_text())
    rows = []
    for player_name, rec in raw.items():
        pts = rec.get("dk_points", {})
        rows.append({
          "name":          player_name,
          "proj_mean":     pts.get("mean", 0.0),
          "proj_ceiling":  pts.get("ceiling", 0.0),
          "proj_floor":    pts.get("floor", 0.0),
        })
    return pd.DataFrame(rows)


def merge_betting(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    bet = pd.read_csv(path)
    home = bet[["home","home_win_prob","home_avg_score","mean_spread"]].rename(
      columns={
        "home":           "team",
        "home_avg_score": "implied_team_total",
        "mean_spread":    "game_spread"
      }
    )
    away = bet[["away","away_win_prob"]].rename(columns={"away": "team"})
    df = df.merge(home, on="team", how="left")
    df = df.merge(away, on="team", how="left")
    return df


def merge_def_ranks(df: pd.DataFrame, week: int, year: int) -> pd.DataFrame:
    p = DEF_RANKS / f"def_rank_{year}_w{week}.json"
    if not p.exists():
        return df
    raw = json.loads(p.read_text())
    dr  = pd.DataFrame(raw).T.reset_index().rename(columns={"index": "team"})
    dr  = dr.rename(columns={
      "vs_qb": "def_vs_qb", "vs_rb": "def_vs_rb",
      "vs_wr": "def_vs_wr", "vs_te": "def_vs_te"
    })
    df = df.merge(dr, on="team", how="left")
    df["pos"] = df["pos"].fillna("").astype(str)
    df["def_vs_pos"] = df.apply(
      lambda r: r.get(f"def_vs_{r['pos'].lower()}", DEFAULTS["def_vs_pos"]),
      axis=1
    )
    return df


def merge_ownership(df: pd.DataFrame, week: int, year: int) -> pd.DataFrame:
    p = OWNERSHIP / f"{year}_w{week}.json"
    if not p.exists():
        return df
    raw = json.loads(p.read_text())
    om  = {k.lower(): v for k, v in raw.items()}
    df["proj_own"] = df["name"].str.lower().map(om).fillna(DEFAULTS["proj_own"])
    return df


def inject_dst(df: pd.DataFrame, week: int, year: int) -> pd.DataFrame:
    """
    Use 'val' to rank DST in solver but display manual 'proj_mean' if provided.
    """
    p = DST_DIR / f"dst_{year}_w{week}.json"
    if not p.exists():
        return df
    raw = json.loads(p.read_text())

    for full, rec in raw.items():
        code = TEAM_CODE.get(full)
        if not code:
            continue
        mask = (df["pos"] == "DST") & (df["team"] == code)

        # solver uses rec['val'], display uses rec['proj_mean'] if set
        solver_val = rec.get("val", DEFAULTS["proj_mean"])
        display_mean = rec.get("proj_mean", solver_val)

        df.loc[mask, ["proj_mean","proj_ceiling","proj_floor"]] = display_mean
        df.loc[mask, "def_vs_pos"] = rec.get("rank", DEFAULTS["def_vs_pos"])
        df.loc[mask, "proj_own"]   = rec.get("ownership", DEFAULTS["proj_own"])
        df.loc[mask, "name"]       = f"{full} DST"
        df.loc[mask, "team"]       = full

    return df


def build_master_pool(week: int, year: int) -> pd.DataFrame:
    # 1) Load salary (with DST rows labeled)
    sal = load_salary(week)

    # 2) Load player projections keyed by name
    sims = SIMS_DIR / f"{year}_w{week}_all"
    ps   = load_player_summary(sims / "player_summary.json")

    # 3) Merge salary + projections on 'name'
    df = sal.merge(ps, on="name", how="left")

    # 4) Fill missing proj_* with defaults
    for col in ("proj_mean","proj_ceiling","proj_floor"):
        df[col] = df[col].fillna(DEFAULTS[col])

    # 5) Merge betting odds, defensive ranks, ownership
    df = merge_betting(df, sims / "betting.csv")
    df = merge_def_ranks(df, week, year)
    df = merge_ownership(df, week, year)

    # 6) Inject DST projection values
    df = inject_dst(df, week, year)

    # 7) Fill any other missing defaults & exposure_limit
    for col, default in DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)
    df["exposure_limit"] = DEFAULTS["exposure_limit"]

    # 8) Reorder & drop zero-salary rows
    out = df[UNIFIED_SCHEMA].copy()
    out = out[out["dk_salary"] > 0].reset_index(drop=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--year", type=int, required=True)
    args = parser.parse_args()

    pool = build_master_pool(args.week, args.year)
    od   = ROOT / "src/output/optimizer_pool"
    od.mkdir(parents=True, exist_ok=True)

    pool.to_csv(od / f"optimizer_pool_w{args.week}.csv", index=False)
    pool.to_json(
      od / f"optimizer_pool_w{args.week}.json",
      orient="records", indent=2
    )
    print(f"\n✅ Wrote {len(pool)} rows for week {args.week}")


if __name__ == "__main__":
    main()