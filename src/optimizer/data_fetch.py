#!/usr/bin/env python3
# src/optimizer/data_fetch.py

import pandas as pd
import nfl_data_py as nfl
import requests
from typing import Sequence, List
from io import StringIO
from pathlib import Path

from optimizer.salary_ingestor_csv1 import (
    get_draftkings_salaries_csv,
    get_fanduel_salaries_csv,
)

# project root (…/dk_opt)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def fetch_weekly_data(
    years: Sequence[int],
    week: int,
    columns: List[str],
) -> pd.DataFrame:
    """
    Fetch weekly NFL data for one or more season years and a specific week,
    then select only the requested columns.
    """
    # 1) Pull in raw data
    df = nfl.import_weekly_data(list(years))

    # 2) Normalize synonyms
    rename_map: dict[str, str] = {}
    if "display_name" in df.columns:
        rename_map["display_name"] = "player_name"
    if "player" in df.columns and "player_name" not in df.columns:
        rename_map["player"] = "player_name"
    if "weekNum" in df.columns:
        rename_map["weekNum"] = "week"
    if "week_number" in df.columns:
        rename_map["week_number"] = "week"
    if "pos" in df.columns and "position" not in df.columns:
        rename_map["pos"] = "position"
    if "position_code" in df.columns:
        rename_map["position_code"] = "position"

    if rename_map:
        df = df.rename(columns=rename_map)

    # 3) Ensure 'week' exists
    if "week" not in df.columns:
        raise ValueError(f"No 'week' column after renaming. Columns: {df.columns.tolist()}")

    # 4) Filter by week
    df_filtered = df[df["week"] == week]
    if df_filtered.empty:
        raise ValueError(f"No data for years={years}, week={week}.")

    # 5) Verify requested columns
    missing = [c for c in columns if c not in df_filtered.columns]
    if missing:
        raise ValueError(
            f"Missing requested columns: {missing}. "
            f"Available: {df_filtered.columns.tolist()}"
        )

    # 6) Return only what was asked for
    return df_filtered[columns].reset_index(drop=True)


def fetch_dk_salaries(
    draft_group_id: int,
    week: int,
    contest_type_id: int = 21
) -> pd.DataFrame:
    """
    Pulls the available-players CSV from DraftKings and returns a
    DataFrame with columns: player_name, position, team, salary, week.

    If local CSV exists at src/data/salary_input/dk_2025_w{week}.csv, use that
    via get_draftkings_salaries_csv.
    """
    # CSV fallback to local file
    csv_path = PROJECT_ROOT / "src" / "data" / "salary_input" / f"dk_2025_w{week}.csv"
    if csv_path.exists():
        return get_draftkings_salaries_csv(week, 2025)

    # Otherwise fetch live from DraftKings
    sess = requests.Session()
    url = (
        "https://www.draftkings.com/lineup/getavailableplayerscsv"
        f"?contestTypeId={contest_type_id}"
        f"&draftGroupId={draft_group_id}"
    )
    resp = sess.get(url)
    resp.raise_for_status()

    # parse CSV text into DataFrame
    df = pd.read_csv(StringIO(resp.text))

    # normalize columns
    df = df.rename(columns={
        "Name":       "player_name",
        "Position":   "position",
        "TeamAbbrev": "team",
        "Salary":     "salary",
    })

    # cast salary to int and tag week
    df["salary"] = df["salary"].astype(int)
    df["week"] = week

    return df[["player_name", "position", "team", "salary", "week"]]


def fetch_fd_salaries(week: int) -> pd.DataFrame:
    """
    Downloads FanDuel’s current NFL salary slate,
    and returns player_name, position, team, salary, week.

    If local CSV exists at src/data/salary_input/fd_2025_w{week}.csv, use that
    via get_fanduel_salaries_csv.
    """
    # CSV fallback to local file
    csv_path = PROJECT_ROOT / "src" / "data" / "salary_input" / f"fd_2025_w{week}.csv"
    if csv_path.exists():
        return get_fanduel_salaries_csv(week, 2025)

    FD_SALARY_URL = "https://api.fanduel.com/contests/v1/current/sport/nfl/salaries"

    # create session and prime with homepage (sets cookies)
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.fanduel.com/",
        "Origin": "https://www.fanduel.com",
        "Connection": "keep-alive",
        "X-Requested-With": "XMLHttpRequest"
    })
    sess.get("https://www.fanduel.com/")

    # now fetch the salaries JSON
    resp = sess.get(FD_SALARY_URL)
    resp.raise_for_status()

    data = resp.json().get("data", [])
    df = pd.DataFrame(data).rename(columns={
        "firstName":  "first_name",
        "lastName":   "last_name",
        "position":   "position",
        "teamAbbrev": "team",
        "salary":     "salary",
    })

    # combine first + last name, tag week
    df["player_name"] = (
        df["first_name"].fillna("") + " " + df["last_name"].fillna("")
    ).str.strip()
    df["week"] = week

    return df[["player_name", "position", "team", "salary", "week"]]