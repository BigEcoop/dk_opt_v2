#!/usr/bin/env python3
# src/optimizer/engine.py

import json
from pathlib import Path
import pandas as pd


def load_scoring_config() -> dict:
    """
    Load DraftKings scoring JSON from src/data/scoring.json
    and return the parsed dict.
    """
    # __file__ is src/optimizer/engine.py
    # Move up two levels to src/, then into data/scoring.json
    scoring_path = Path(__file__).resolve().parent.parent / "data" / "scoring.json"
    with scoring_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Pull out only the "scoring" section once at import time
SCORING_JSON = load_scoring_config()
SCORING_RULES = SCORING_JSON["scoring"]


class ScoreEngine:
    """
    A fast, data‐driven fantasy‐points engine.

    - On init, flatten the nested SCORING_RULES into a simple mapping
      of column names → weights.
    - score_players() then dot‐products the DataFrame columns
      against those weights in one vectorized pass.
    """

    def __init__(self):
        s = SCORING_RULES
        self.rules: dict[str, float] = {
            "passing_yards":   s["yardage"]["passing"],
            "rushing_yards":   s["yardage"]["rushing"],
            "receiving_yards": s["yardage"]["receiving"],
            "passing_tds":     s["offense"]["passing_td"],
            "rushing_tds":     s["offense"]["rushing_td"],
            "receiving_tds":   s["offense"]["receiving_td"],
            "interception":    s["offense"]["interception"],
            "fumble_lost":     s["offense"]["fumble_lost"],
            "receptions":      s["offense"]["reception"],
        }

    @classmethod
    def load_config(cls) -> "ScoreEngine":
        """
        Instantiate the engine with rules pulled from scoring.json.
        """
        return cls()

    def score_players(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute a 'fantasy_points' Series by dotting df[rules.keys()]
        against the corresponding weights in self.rules.
        """
        metrics = list(self.rules.keys())
        missing = set(metrics) - set(df.columns)
        if missing:
            raise KeyError(f"Missing metric columns in DataFrame: {missing}")

        weights = pd.Series(self.rules)
        fantasy_points = df[metrics].dot(weights)
        fantasy_points.name = "fantasy_points"
        return fantasy_points