#!/usr/bin/env python3
# src/scripts/monte_carlo_sims.py

import json
import random
import statistics
import argparse
import pandas as pd
import nfl_data_py as nfl
import urllib.error
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# ── PROJECT ROOT & PATHS ───────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[2]
PROJS_ROOT    = PROJECT_ROOT / "src" / "output" / "projections"
SCORING_JSON  = PROJECT_ROOT / "src" / "data" / "scoring.json"
SIMS_ROOT     = PROJECT_ROOT / "src" / "output" / "sims"
ESPN_DIR      = PROJECT_ROOT / "src" / "data" / "projection_data" / "espn"
TEAM_PROFILE  = PROJECT_ROOT / "src" / "data" / "projection_data" / "2025_team.json"

# ── TUNING CONSTANTS ────────────────────────────────────────────────────────────
DISPERSION_K   = 13      # overdispersion param for turnovers
ESPN_WEIGHT    = 0.04    # weight for ESPN total‐score calibration
FG_LAMBDA      = 1.2     # Poisson mean FG attempts
SCORE_DAMPEN   = 0.75    # global dampening of raw scores
CLAMP_MULT     = 2.0     # clamp raw yardage at [0, μ*CLAMP_MULT]

team_map = {"LA": "LAR", "LAR": "LAR", "FA": None}

OUTPUT_CATEGORIES = [
    "pass_attempts","pass_completions","passing_yds","passing_tds","interceptions",
    "rushing_attempts","rushing_yds","rushing_tds",
    "receiving_targets","receptions","receiving_yds","receiving_tds",
    "fumbles","punt_return_yards","punt_return_tds","kick_return_yards","kick_return_tds"
]
OFFENSE_POSITIONS = {"qb", "rb", "wr", "te"}

# ── OVERDISPERSION SAMPLER ─────────────────────────────────────────────────────
def sample_nb(mu: float, k: int = DISPERSION_K) -> int:
    """Negative‐Binomial(mean=mu, var=mu + μ²/k) sampler for low‐count stats."""
    if mu <= 0:
        return 0
    p = k / (k + mu)
    return np.random.negative_binomial(k, p)

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
    j   = load_json(baseline_path)
    raw = j.get("players", [])

    try:
        year   = int(baseline_path.stem.split("_")[2])
        inj_df = nfl.import_injuries([year])
    except (urllib.error.HTTPError, ValueError):
        inj_df = pd.DataFrame(columns=["player_name","injury_status"])
    injured = set(
        inj_df[inj_df["injury_status"].isin({"questionable","out","doubtful"})]
             ["player_name"]
    )

    players: List[Dict] = []
    for rec in raw:
        name   = rec.get("name") or rec.get("player") or rec.get("Player")
        pid    = rec.get("id")   or rec.get("ID")
        raw_tm = rec.get("team") or rec.get("Team")
        team   = team_map.get(raw_tm, raw_tm)
        pos    = (rec.get("pos") or rec.get("POS") or rec.get("position") or "").lower()

        if not (name and pid and team) or team is None:
            continue
        if name in injured or pos not in OFFENSE_POSITIONS:
            continue

        stats_raw = rec.get("stats", {})
        stats: Dict[str, float] = {}
        for st, mu in stats_raw.items():
            if st in OUTPUT_CATEGORIES:
                try:
                    stats[st] = float(mu)
                except (ValueError, TypeError):
                    stats[st] = 0.0

        if stats:
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
    pts += statline.get("passing_tds", 0)   * o["passing_td"]
    pts += statline.get("interceptions", 0) * o["interception"]
    pts += statline.get("passing_yds", 0)   * y["passing"]
    if statline.get("passing_yds", 0) >= 300:
        pts += b["passing_300_yds"]
    pts += statline.get("rushing_tds", 0)   * o["rushing_td"]
    pts += statline.get("rushing_yds", 0)   * y["rushing"]
    if statline.get("rushing_yds", 0) >= 100:
        pts += b["rushing_100_yds"]
    pts += statline.get("receiving_tds", 0) * o["receiving_td"]
    pts += statline.get("receptions", 0)    * o["reception"]
    pts += statline.get("receiving_yds", 0) * y["receiving"]
    if statline.get("receiving_yds", 0) >= 100:
        pts += b["receiving_100_yds"]
    pts += statline.get("fumbles", 0)       * o["fumble_lost"]
    return pts

def simulate_team_score(team_stats: Dict[str, float]) -> int:
    td_count = (
        team_stats.get("passing_tds", 0) +
        team_stats.get("rushing_tds", 0) +
        team_stats.get("receiving_tds", 0)
    )
    # Each TD = 6 points + PAT (98%)
    pts = td_count * 6
    pts += sum(1 for _ in range(int(td_count)) if random.random() < 0.98)

    # Field goals
    fg_attempts  = np.random.poisson(lam=FG_LAMBDA)
    fg_successes = sum(1 for _ in range(fg_attempts) if random.random() < 0.85)
    pts += fg_successes * 3

    # Two-point conversions
    two_pt_tries = np.random.binomial(int(td_count), 0.10)
    two_pt_made  = sum(1 for _ in range(two_pt_tries) if random.random() < 0.50)
    pts += two_pt_made * 2

    return pts

def simulate_for(
    matchups: List[Tuple[str, str]],
    teams:    Dict[str, List[Dict]],
    scoring:  Dict,
    sims:     int,
    espn_map: Dict[str, Dict]
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
        key = f"{away}-{home}"
        for _ in range(sims):
            def run_lineup(roster):
                out = {}
                for p in roster:
                    jitter = {}
                    for st, mu in p["stats"].items():
                        if st in ["interceptions", "fumbles"]:
                            jitter[st] = sample_nb(mu)
                        elif st in ["passing_tds", "rushing_tds", "receiving_tds"]:
                            jitter[st] = np.random.poisson(lam=mu)
                        else:
                            raw = random.gauss(mu, max(mu * 0.12, 1))
                            jitter[st] = max(0.0, min(raw, mu * CLAMP_MULT))
                    dk_pts = score_statline(jitter, scoring)
                    out[p["player"]] = {"stats": jitter, "dk": dk_pts}
                return out

            h_out = run_lineup(teams[home])
            a_out = run_lineup(teams[away])

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

            # ESPN-based rescaling
            raw_total = h_score + a_score
            espn      = espn_map.get(key, {})
            if espn.get("home_proj") is not None and espn.get("away_proj") is not None and raw_total > 0:
                espn_total = espn["home_proj"] + espn["away_proj"]
                scale      = (espn_total / raw_total) ** ESPN_WEIGHT
                h_score    = round(h_score * scale)
                a_score    = round(a_score * scale)

            # global dampening
            h_score = round(h_score * SCORE_DAMPEN)
            a_score = round(a_score * SCORE_DAMPEN)

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
        rec = {"id": m.get("id",""), "team": m.get("team",""), "pos": m.get("pos","")}
        for c, vals in stats.items():
            if not vals:
                continue
            summary_vals = {
                "floor":   min(vals),
                "mean":    statistics.mean(vals),
                "ceiling": max(vals)
            }
            rec[c] = summary_vals
        rec["total_tds"] = (
            rec.get("rushing_tds",{}).get("mean",0) +
            rec.get("receiving_tds",{}).get("mean",0)
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
            summary[t] = {"win_prob":0.0, "mean_score":0.0, "mean_spread":0.0}
        else:
            tot_s  = sum(agg["scores"])
            tot_sp = sum(agg["spreads"])
            summary[t] = {
                "win_prob":    agg["wins"] / sims,
                "mean_score":  tot_s       / sims,
                "mean_spread": tot_sp      / sims
            }
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b","--baseline", type=Path, help="Baseline JSON")
    parser.add_argument("-s","--scoring",  type=Path, help="Scoring JSON")
    parser.add_argument("-w","--week",     type=int, default=1, help="Week number")
    parser.add_argument("-n","--sims",     type=int, default=1000, help="Number of sims")
    parser.add_argument("-k","--odds-key", type=str, default=None, help="TheOddsAPI key")
    parser.add_argument("--year",          type=int, default=2025, help="Season year")
    parser.add_argument(
        "--days", nargs="+",
        choices=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday","All"],
        default=["Sunday","Monday"],
        help="Weekdays to simulate; 'All'=every game"
    )
    parser.add_argument(
        "--matchup", action="append", nargs=2, metavar=("AWAY","HOME"),
        help="Filter specific games; e.g. --matchup KC PHI"
    )
    args = parser.parse_args()

    dvoa_map = load_json(
        PROJECT_ROOT / "src" / "data" / "projection_data" / f"dvoa_{args.year}.json"
    )
    espn_map = {}
    ep = ESPN_DIR / f"espn_{args.year}_w{args.week}.json"
    if ep.exists():
        espn_map = load_json(ep)
    team_profile = {}
    if TEAM_PROFILE.exists():
        team_profile = load_json(TEAM_PROFILE)

    baseline = args.baseline or (
        PROJS_ROOT / f"{args.year}_w{args.week}" /
        f"baseline_{args.year}_w{args.week}.json"
    )
    scoring  = args.scoring or SCORING_JSON

    if not baseline.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline}")
    if not scoring.exists():
        raise FileNotFoundError(f"Scoring not found: {scoring}")

    players     = prepare_players(baseline, args.week)
    player_meta = {p["player"]: {"id": p["id"], "team": p["team"], "pos": p["position"]} 
                   for p in players}

    sched_full = nfl.import_schedules([args.year])
    sched_full["home_team"] = sched_full["home_team"].map(lambda t: team_map.get(t,t))
    sched_full["away_team"] = sched_full["away_team"].map(lambda t: team_map.get(t,t))
    week_sched = sched_full[sched_full["week"] == args.week]

    if args.matchup:
        mask = False
        for aw, hm in args.matchup:
            mask |= ((week_sched.away_team == aw) & (week_sched.home_team == hm))
        main_games = week_sched[mask]
    elif "All" in args.days:
        main_games = week_sched
    else:
        main_games = week_sched[week_sched["weekday"].isin(args.days)]

    main_matchups = [(r.away_team, r.home_team) for r in main_games.itertuples()]

    teams: Dict[str, List[Dict]] = {}
    for p in players:
        teams.setdefault(p["team"], []).append(p)
    for aw, hm in main_matchups:
        teams.setdefault(aw, []); teams.setdefault(hm, [])

    scoring_rules  = load_json(scoring)["scoring"]
    pa_main, ta_main = simulate_for(main_matchups, teams, scoring_rules, args.sims, espn_map)

    label    = args.days[0] if len(args.days) == 1 else "all"
    sims_dir = SIMS_ROOT / f"{args.year}_w{args.week}_{label}"
    sims_dir.mkdir(parents=True, exist_ok=True)

    # player summary
    PLAYER_SUMMARY = sims_dir / "player_summary.json"
    ps_main        = summarize_players(pa_main, player_meta)
    write_json(PLAYER_SUMMARY, ps_main)
    pd.json_normalize(
        [{"player": n, **s} for n, s in ps_main.items()], sep="_"
    ).to_excel(sims_dir / "player_summary.xlsx", index=False)

    # betting lines
    header = [
        "type","home","away",
        "home_win_prob","away_win_prob",
        "home_avg_score","away_avg_score",
        "mean_total","mean_spread"
    ]
    rows = []
    ts   = summarize_teams(ta_main, args.sims)
    for r in main_games.itertuples():
        h, a = r.home_team, r.away_team
        hh, aa = ts[h], ts[a]
        rows.append([
            "GAME", h, a,
            round(hh["win_prob"],3), round(aa["win_prob"],3),
            round(hh["mean_score"],1), round(aa["mean_score"],1),
            round(hh["mean_score"]+aa["mean_score"],1),
            round(hh["mean_score"]-aa["mean_score"],1)
        ])
    write_csv(sims_dir/"betting.csv", header, rows)
    pd.read_csv(sims_dir/"betting.csv").to_excel(sims_dir/"betting.xlsx", index=False)
    print(f"Wrote {PLAYER_SUMMARY} & betting.csv to {sims_dir}")