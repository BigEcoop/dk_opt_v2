#!/usr/bin/env python3
# src/optimizer/salary_ingestor_csv.py

import pandas as pd
from pathlib import Path

# assume you always run from project root C:/Projects/dk_opt
PROJECT_ROOT = Path.cwd()

# directory where salary CSVs live for this version
SALARY_INPUT_DIR = PROJECT_ROOT / "src" / "data" / "salary_input"


def _to_int(s: pd.Series) -> pd.Series:
    """Convert to integer, raising on error."""
    return pd.to_numeric(s, errors="raise", downcast="integer")


def _to_float(s: pd.Series) -> pd.Series:
    """Convert to float, raising on error."""
    return pd.to_numeric(s, errors="raise", downcast="float")


def get_draftkings_salaries_csv(week: int, year: int) -> pd.DataFrame:
    """
    Load DraftKings salaries for week {week}, {year} from
    src/data/salary_input/dk_{year}_w{week}.csv,
    normalize columns, split 'game' into away/home/kickoff.
    """
    path = SALARY_INPUT_DIR / f"dk_{year}_w{week}.csv"
    if not path.exists():
        raise FileNotFoundError(f"DraftKings CSV not found at: {path}")

    df = pd.read_csv(path)

    # normalize & rename to our schema
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    df = df.rename(columns={
        "Name":             "player_name",
        "ID":               "player_id",
        "Position":         "position",
        "Roster_Position":  "roster_position",
        "Salary":           "salary",
        "Game_Info":        "game",
        "TeamAbbrev":       "team",
        "AvgPointsPerGame": "avg_points_per_game",
    })

    # convert types
    df["player_id"] = _to_int(df["player_id"])
    df["salary"] = _to_int(df["salary"])
    df["avg_points_per_game"] = _to_float(df["avg_points_per_game"])

    # split 'game' into away/home/kickoff
    parts = df["game"].str.extract(
        r"(?P<away>\w{3})@(?P<home>\w{3}) (?P<kickoff>.+)"
    )
    df["away"] = parts["away"]
    df["home"] = parts["home"]
    df["kickoff"] = pd.to_datetime(
        parts["kickoff"],
        format="%m/%d/%Y %I:%M%p ET",
        errors="coerce"
    )

    return df


def get_fanduel_salaries_csv(week: int, year: int) -> pd.DataFrame:
    """
    Load FanDuel salaries for week {week}, {year} from
    src/data/salary_input/fd_{year}_w{week}.csv,
    normalize columns to our schema.
    """
    path = SALARY_INPUT_DIR / f"fd_{year}_w{week}.csv"
    if not path.exists():
        raise FileNotFoundError(f"FanDuel CSV not found at: {path}")

    df = pd.read_csv(path)

    # normalize headers
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    # build player_name from Nickname if available, else First_Name + Last_Name
    df["player_name"] = df.apply(
        lambda r: r["Nickname"]
                  if pd.notna(r.get("Nickname")) and r["Nickname"]
                  else f"{r.get('First_Name','')} {r.get('Last_Name','')}".strip(),
        axis=1
    )

    # extract numeric ID (split on dash if present)
    df["player_id"] = (
        df["Id"].astype(str)
          .str.split("-", 1)
          .str[-1]
          .astype(int)
    )

    # rename to our schema
    df = df.rename(columns={
        "Position":             "position",
        "Salary":               "salary",
        "FPPG":                 "avg_points_per_game",
        "Team":                 "team",
        "Opponent":             "opponent",
    })

    return df