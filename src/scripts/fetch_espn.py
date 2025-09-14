#!/usr/bin/env python3
# src/scripts/fetch_espn.py

import argparse
import json
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from typing import Dict, List

# ── CONFIG ───────────────────────────────────────────────────────────────────────
ESPN_URL = (
    "https://www.espn.com/fantasy/football/story/"
    "_/id/46205300/nfl-fantasy-football-2025-ultimate-playbook-"
    "projections-win-probabilities-week-2-shadow-reports-mike-clay"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/117.0.0.0 Safari/537.36"
    )
}

FULL_NAME_TO_ABBR = {
    "Cardinals": "ARI", "Rams": "LAR", "49ers": "SF", "Seahawks": "SEA",
    "Raiders": "LV", "Broncos": "DEN", "Chiefs": "KC", "Chargers": "LAC",
    "Cowboys": "DAL", "Giants": "NYG", "Commanders": "WAS", "Eagles": "PHI",
    "Panthers": "CAR", "Saints": "NO", "Buccaneers": "TB", "Falcons": "ATL",
    "Packers": "GB", "Bears": "CHI", "Vikings": "MIN", "Lions": "DET",
    "Bills": "BUF", "Dolphins": "MIA", "Patriots": "NE", "Jets": "NYJ",
    "Steelers": "PIT", "Browns": "CLE", "Ravens": "BAL", "Bengals": "CIN",
    "Titans": "TEN", "Colts": "IND", "Jaguars": "JAX", "Texans": "HOU"
}

def fetch_espn_matchups(url: str) -> Dict[str, Dict]:
    """
    Scrape ESPN’s Mike Clay Playbook for projected score, OU, win%
    Returns mapping "AWAY-HOME" -> {away_proj, home_proj, over_under, away_win_prob, home_win_prob}
    """
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # gather all paragraph tags for sequential parsing
    paras: List[str] = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    games: Dict[str, Dict] = {}

    for idx, text in enumerate(paras):
        # look for projected score line
        if not re.search(r"Projected\s+score:", text, flags=re.IGNORECASE):
            continue

        # parse "Projected score: Team1 28, Team2 22"
        m = re.search(
            r"Projected\s+score:\s*([A-Za-z ]+?)\s*(\d+),\s*([A-Za-z ]+?)\s*(\d+)",
            text,
            flags=re.IGNORECASE
        )
        if not m:
            continue
        full1, s1, full2, s2 = m.groups()
        abbr1 = FULL_NAME_TO_ABBR.get(full1.strip())
        abbr2 = FULL_NAME_TO_ABBR.get(full2.strip())
        if not (abbr1 and abbr2):
            continue

        # ESPN lists home team first in this line
        home_abbr, away_abbr = abbr1, abbr2
        home_proj = float(s1)
        away_proj = float(s2)

        # initialize optional fields
        over_under = None
        away_wp = None
        home_wp = None

        # search next few paragraphs for OU and win%
        for follow in paras[idx+1 : idx+6]:
            # Over/Under
            m_ou = re.search(r"Over/?Under:\s*([\d.]+)", follow, flags=re.IGNORECASE)
            if m_ou and over_under is None:
                over_under = float(m_ou.group(1))

            # Win probability
            m_wp = re.search(
                r"Win\s+probability:\s*([A-Za-z ]+?)\s*(\d+)%", follow, flags=re.IGNORECASE
            )
            if m_wp:
                team_full, pct = m_wp.groups()
                pct_f = float(pct) / 100.0
                wp_abbr = FULL_NAME_TO_ABBR.get(team_full.strip())
                if wp_abbr == away_abbr:
                    away_wp, home_wp = pct_f, 1 - pct_f
                else:
                    home_wp, away_wp = pct_f, 1 - pct_f

            # break if all found
            if over_under is not None and away_wp is not None:
                break

        key = f"{away_abbr}-{home_abbr}"
        games[key] = {
            "away_proj":     away_proj,
            "home_proj":     home_proj,
            "over_under":    over_under,
            "away_win_prob": away_wp,
            "home_win_prob": home_wp
        }

    return games

def main():
    parser = argparse.ArgumentParser(description="Fetch ESPN weekly projections")
    parser.add_argument("--year", type=int, required=True, help="Season year")
    parser.add_argument("--week", type=int, required=True, help="NFL week number")
    parser.add_argument("--url",  type=str, default=ESPN_URL, help="Playbook URL")
    args = parser.parse_args()

    data = fetch_espn_matchups(args.url)

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "src" / "data" / "projection_data" / "espn"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"espn_{args.year}_w{args.week}.json"
    out_path.write_text(json.dumps(data, indent=2))

    print(f"Wrote ESPN projections to {out_path}")

if __name__ == "__main__":
    main()