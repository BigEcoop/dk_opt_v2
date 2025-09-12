#!/usr/bin/env python3
# src/scripts/monte_carlo_sims.py

import json
import random
import statistics
import argparse
import pandas as pd
import nfl_data_py as nfl
import urllib.error
from pathlib import Path
from typing import Dict, List, Tuple

# ── PROJECT ROOT & PATHS ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# raw baselines under src/output/projections/{year}_w{week}/footballguys_base_{year}_w{week}.json
PROJS_ROOT   = PROJECT_ROOT / "src" / "output" / "projections"

# scoring JSON remains constant at src/data/scoring.json
SCORING_JSON = PROJECT_ROOT / "src" / "data" / "scoring.json"

# root for simulation outputs
SIMS_ROOT    = PROJECT_ROOT / "src" / "output" / "sims"

# ── SIM CONSTANTS ───────────────────────────────────────────────────────────────
DEF_POSITIONS  = {"CB", "DE", "DT", "LB", "PK", "TD", "S"}
TEAM_SCALE     = 5.87
ODDS_URL       = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
DIFF_THRESHOLD = 0.05

team_map = {"LA": "LAR", "LAR": "LAR", "FA": None}

key_map = {
    "pass_att":  "pass_attempts",    "pass_cmp": "pass_completions",
    "pass_td":   "passing_tds",      "pass_yds": "passing_yds",
    "pass_int":  "interceptions",    "rush_car": "rushing_attempts",
    "rush_td":   "rushing_tds",      "rush_yds": "rushing_yds",
    "rec_tgt":   "receiving_targets","rec_rec": "receptions",
    "rec_td":    "receiving_tds",    "rec_yds": "receiving_yds",
    "fum_lost":  "fumbles",          "pr_yds": "punt_return_yards",
    "pr_tds":    "punt_return_tds",  "kr_yds": "kick_return_yds",
    "kr_tds":    "kick_return_tds"
}

OFFENSE_POSITIONS = {"qb", "rb", "wr", "te"}
OUTPUT_CATEGORIES  = list(key_map.values())

# ── HELPERS ────────────────────────────────────────────────────────────────────
def load_json(path: Path) -> Dict:
    return json.loads(path.read_text())

def write_json(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))

def write_csv(path: Path, header: List[str], rows: List[List]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")

# ── PREPARE PLAYERS ────────────────────────────────────────────────────────────
def prepare_players(baseline_path: Path, week: int) -> List[Dict]:
    raw = load_json(baseline_path)["players"]
    try:
        year = int(baseline_path.stem.split("_")[2])
        inj_df = nfl.import_injuries([year])
    except (urllib.error.HTTPError, ValueError):
        inj_df = pd.DataFrame(columns=["player_name", "injury_status"])
    injured = set(
        inj_df[inj_df["injury_status"].isin({"questionable", "out", "doubtful"})]
             ["player_name"]
    )

    players: List[Dict] = []
    for rec in raw:
        name   = rec.get("name") or rec.get("Player") or rec.get("player")
        pid    = rec.get("id") or rec.get("ID")
        raw_tm = rec.get("team") or rec.get("Team")
        team   = team_map.get(raw_tm, raw_tm)
        pos    = (rec.get("pos") or rec.get("POS") or rec.get("position") or "").lower()

        if not (name and pid and team) or team is None:
            continue
        if name in injured or pos not in OFFENSE_POSITIONS:
            continue

        stats: Dict[str, float] = {}
        for k, v in rec.items():
            norm = k.strip().lower().replace(" ", "_").replace("-", "_")
            if norm in key_map:
                try:
                    stats[key_map[norm]] = float(v)
                except (ValueError, TypeError):
                    stats[key_map[norm]] = 0.0
        if not stats:
            continue

        players.append({
            "player":   name,
            "id":       pid,
            "team":     team,
            "position": pos,
            "stats":    stats
        })
    return players

# ── SCORING & SIMULATION ───────────────────────────────────────────────────────
def score_statline(statline: Dict[str, float], scoring: Dict) -> float:
    o, y, b = scoring["offense"], scoring["yardage"], scoring["bonuses"]
    pts = 0.0
    pts += statline.get("passing_tds", 0)         * o["passing_td"]
    pts += statline.get("interceptions", 0)       * o["interception"]
    pts += statline.get("passing_yds", 0)         * y["passing"]
    if statline.get("passing_yds", 0) >= 300:
        pts += b["passing_300_yds"]
    pts += statline.get("rushing_tds", 0)         * o["rushing_td"]
    pts += statline.get("rushing_yds", 0)         * y["rushing"]
    if statline.get("rushing_yds", 0) >= 100:
        pts += b["rushing_100_yds"]
    pts += statline.get("receiving_tds", 0)       * o["receiving_td"]
    pts += statline.get("receptions", 0)          * o["reception"]
    pts += statline.get("receiving_yds", 0)       * y["receiving"]
    if statline.get("receiving_yds", 0) >= 100:
        pts += b["receiving_100_yds"]
    pts += statline.get("fumbles", 0)             * o["fumble_lost"]
    pts += statline.get("two_point_conversion", 0)* o.get("two_point_conversion", 0)
    return pts

def simulate_team_score(team_stats: Dict[str, float]) -> int:
    td_count = (
        team_stats.get("passing_tds", 0) +
        team_stats.get("rushing_tds", 0) +
        team_stats.get("receiving_tds", 0)
    )
    pts = td_count * 6
    pat_successes = sum(1 for _ in range(int(td_count)) if random.random() < 0.95)
    pts += pat_successes
    fg_attempts  = int(random.uniform(0, 4.75))
    fg_successes = sum(1 for _ in range(fg_attempts) if random.random() < 0.85)
    pts += fg_successes * 3
    if random.random() < 0.03:
        pts += 2  # safety
    return pts

def simulate_for(
    matchups: List[Tuple[str, str]],
    teams:    Dict[str, List[Dict]],
    scoring:  Dict,
    sims:     int
) -> Tuple[
    Dict[str, Dict[str, List[float]]],
    Dict[str, Dict[str, List[float]]]
]:
    player_agg = {
        p["player"]: {c: [] for c in OUTPUT_CATEGORIES + ["dk_points"]}
        for roster in teams.values() for p in roster
    }
    team_agg = {t: {"wins": 0, "scores": [], "spreads": []} for t in teams}

    for away, home in matchups:
        home_roster = teams[home]
        away_roster = teams[away]

        for _ in range(sims):
            def run_lineup(roster):
                out = {}
                for p in roster:
                    jitter = {
                        st: max(0.0, random.gauss(mu, max(mu * 0.2, 1)))
                        for st, mu in p["stats"].items()
                    }
                    dk_pts = score_statline(jitter, scoring)
                    out[p["player"]] = {"stats": jitter, "dk": dk_pts}
                return out

            h_out = run_lineup(home_roster)
            a_out = run_lineup(away_roster)

            h_stats = {
                c: sum(info["stats"].get(c, 0) for info in h_out.values())
                for c in OUTPUT_CATEGORIES
            }
            a_stats = {
                c: sum(info["stats"].get(c, 0) for info in a_out.values())
                for c in OUTPUT_CATEGORIES
            }

            h_score = simulate_team_score(h_stats)
            a_score = simulate_team_score(a_stats)

            winner = home if h_score >= a_score else away
            spread = h_score - a_score

            for pl, info in {**h_out, **a_out}.items():
                for c, val in info["stats"].items():
                    player_agg[pl][c].append(val)
                player_agg[pl]["dk_points"].append(info["dk"])

            team_agg[winner]["wins"]      += 1
            team_agg[home]["scores"].append(h_score)
            team_agg[away]["scores"].append(a_score)
            team_agg[home]["spreads"].append(spread)
            team_agg[away]["spreads"].append(-spread)

    return player_agg, team_agg

# ── SUMMARIES & ENTRYPOINT ─────────────────────────────────────────────────────
def summarize_players(
    player_agg: Dict[str, Dict[str, List[float]]],
    meta:       Dict[str, Dict[str, str]]
) -> Dict[str, Dict]:
    summary: Dict[str, Dict] = {}
    for pl, stats in player_agg.items():
        m = meta.get(pl, {})
        rec = {"id": m.get("id", ""), "team": m.get("team", ""), "pos": m.get("pos", "")}
        for c, vals in stats.items():
            if not vals:
                continue
            rec[c] = {
                "floor":   min(vals),
                "mean":    statistics.mean(vals),
                "ceiling": max(vals)
            }
        rec["total_tds"] = (
            rec.get("rushing_tds", {}).get("mean", 0) +
            rec.get("receiving_tds", {}).get("mean", 0)
        )
        summary[pl] = rec
    return summary

def summarize_teams(
    team_agg: Dict[str, Dict[str, List[float]]],
    sims:     int
) -> Dict[str, Dict]:
    summary: Dict[str, Dict] = {}
    for t, agg in team_agg.items():
        if not agg["scores"]:
            summary[t] = {"win_prob": 0.0, "mean_score": 0.0, "mean_spread": 0.0}
        else:
            total_scores  = sum(agg["scores"])
            total_spreads = sum(agg["spreads"])
            summary[t] = {
                "win_prob":    agg["wins"] / sims,
                "mean_score":  total_scores  / (sims * TEAM_SCALE),
                "mean_spread": total_spreads / (sims * TEAM_SCALE)
            }
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--baseline",
        type=Path,
        help=(
            "Baseline JSON; defaults to "
            "src/output/projections/{year}_w{week}/footballguys_base_{year}_w{week}.json"
        )
    )
    parser.add_argument(
        "-s", "--scoring",
        type=Path,
        help="Scoring JSON; defaults to src/data/scoring.json"
    )
    parser.add_argument(
        "-w", "--week",
        type=int,
        default=1,
        help="Week number"
    )
    parser.add_argument(
        "-n", "--sims",
        type=int,
        default=1000,
        help="Number of simulations"
    )
    parser.add_argument(
        "-k", "--odds-key",
        type=str,
        default="1e8f9c8a0b084887612c523ebc7b9c6f",
        help="Your TheOddsAPI key"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Season year"
    )
    parser.add_argument(
        "--days",
        nargs="+",
        choices=[
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday",
            "Sunday", "All"
        ],
        default=["Sunday", "Monday"],
        help=(
            "Which weekdays to simulate "
            "(e.g. Thursday or Sunday Tuesday). 'All' = every game."
        )
    )
    args = parser.parse_args()

    # resolve baseline & scoring JSON paths
    baseline = args.baseline or (
        PROJS_ROOT / f"{args.year}_w{args.week}" /
        f"footballguys_base_{args.year}_w{args.week}.json"
    )
    scoring  = args.scoring or SCORING_JSON

    if not baseline.exists():
        raise FileNotFoundError(f"Baseline JSON not found: {baseline}")
    if not scoring.exists():
        raise FileNotFoundError(f"Scoring JSON not found: {scoring}")

    # label folder by weekday (or "all")
    label = args.days[0] if len(args.days) == 1 else "all"
    sims_dir = SIMS_ROOT / f"{args.year}_w{args.week}_{label}"
    sims_dir.mkdir(parents=True, exist_ok=True)

    # define output paths under sims_dir
    PLAYER_SUMMARY = sims_dir / "player_summary.json"
    TEAM_SUMMARY   = sims_dir / "team_summary.json"
    BETTING_CSV    = sims_dir / "betting.csv"
    BETTING_XLSX   = sims_dir / "betting.xlsx"

    # load & prepare players
    players     = prepare_players(baseline, args.week)
    player_meta = {
        p["player"]: {"id": p["id"], "team": p["team"], "pos": p["position"]}
        for p in players
    }

    scoring_rules = load_json(scoring)["scoring"]

    # load schedule and filter by week
    sched_full = nfl.import_schedules([args.year])
    sched_full["home_team"] = sched_full["home_team"].map(lambda t: team_map.get(t, t))
    sched_full["away_team"] = sched_full["away_team"].map(lambda t: team_map.get(t, t))
    week_sched = sched_full[sched_full["week"] == args.week]

    # filter by days argument
    if "All" in args.days:
        main_games = week_sched
    else:
        main_games = week_sched[week_sched["weekday"].isin(args.days)]
    main_matchups = [(r.away_team, r.home_team) for r in main_games.itertuples()]

    # assemble teams dict
    teams: Dict[str, List[Dict]] = {}
    for p in players:
        teams.setdefault(p["team"], []).append(p)
    for away, home in main_matchups:
        teams.setdefault(away, [])
        teams.setdefault(home, [])

    # run monte carlo sims
    pa_main, ta_main = simulate_for(main_matchups, teams, scoring_rules, args.sims)

    # write player_summary.json
    ps_main = summarize_players(pa_main, player_meta)
    write_json(PLAYER_SUMMARY, ps_main)
    print(f"Wrote {PLAYER_SUMMARY}")

    # export player_summary CSV & XLSX
    df_players = pd.json_normalize(
        [{"player": name, **stats} for name, stats in ps_main.items()],
        sep="_"
    )
    csv_path  = sims_dir / "player_summary.csv"
    xlsx_path = sims_dir / "player_summary.xlsx"
    df_players.to_csv(csv_path, index=False)
    df_players.to_excel(xlsx_path, index=False)
    print(f"Wrote {csv_path} & {xlsx_path}")

    # write team_summary.json
    ts_main = summarize_teams(ta_main, args.sims)
    write_json(TEAM_SUMMARY, ts_main)
    print(f"Wrote {TEAM_SUMMARY}")

    # write betting.csv & betting.xlsx
    header = [
        "type", "home", "away",
        "home_win_prob", "away_win_prob",
        "home_avg_score", "away_avg_score",
        "mean_total", "mean_spread"
    ]
    rows = []
    for r in main_games.itertuples():
        h, a = r.home_team, r.away_team
        hh, aa = ts_main[h], ts_main[a]
        rows.append([
            "GAME",
            h, a,
            round(hh["win_prob"], 3),
            round(aa["win_prob"], 3),
            round(hh["mean_score"], 1),
            round(aa["mean_score"], 1),
            round(hh["mean_score"] + aa["mean_score"], 1),
            round(hh["mean_score"] - aa["mean_score"], 1)
        ])
    write_csv(BETTING_CSV, header, rows)
    pd.read_csv(BETTING_CSV).to_excel(BETTING_XLSX, index=False)
    print(f"Wrote {BETTING_CSV} & {BETTING_XLSX}")