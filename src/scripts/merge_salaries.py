#!/usr/bin/env python3
# src/scripts/merge_salaries.py

from optimizer.salary_ingestor_csv1 import (
    get_draftkings_salaries_csv,
    get_fanduel_salaries_csv,
)
import pandas as pd


def main(week: int, year: int):
    # Load DraftKings and FanDuel salary tables
    dk = get_draftkings_salaries_csv(week, year)
    fd = get_fanduel_salaries_csv(week, year)

    # Rename overlapping columns
    dk = dk.rename(
        columns={
            "player_id": "player_id_dk",
            "salary": "salary_dk",
            "avg_points_per_game": "avg_points_per_game_dk",
        }
    )
    fd = fd.rename(
        columns={
            "player_id": "player_id_fd",
            "salary": "salary_fd",
            "avg_points_per_game": "avg_points_per_game_fd",
        }
    )

    # Select FanDuel columns you need (excluding their game fields)
    fd_sel = fd[
        [
            "player_name",
            "team",
            "player_id_fd",
            "salary_fd",
            "avg_points_per_game_fd",
            "opponent",
        ]
    ]

    # Merge on player_name + team
    merged = pd.merge(
        dk,
        fd_sel,
        on=["player_name", "team"],
        how="inner",
    )

    # Reorder to include game date/time and original DK columns
    cols = [
        "position",
        "player_name",
        "player_id_dk",
        "roster_position",
        "salary_dk",
        "game",
        "team",
        "away",
        "home",
        "kickoff",
        "avg_points_per_game_dk",
        "player_id_fd",
        "salary_fd",
        "avg_points_per_game_fd",
        "opponent",
    ]
    merged = merged[cols]

    # Output preview and row count
    print(merged.head())
    print(f"Rows merged: {len(merged)}")


if __name__ == "__main__":
    import sys

    week, year = 1, 2025
    if len(sys.argv) == 3:
        week, year = map(int, sys.argv[1:])
    main(week, year)