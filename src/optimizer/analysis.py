# src/optimizer/analysis.py

import pandas as pd

def compare_fd_dk_salaries(
    dk_df: pd.DataFrame,
    fd_df: pd.DataFrame,
    threshold: int
) -> pd.DataFrame:
    """
    Compare DraftKings vs FanDuel salaries.

    Parameters
    ----------
    dk_df : pd.DataFrame
        DK salaries with columns [player_name, position, team, salary, week]
    fd_df : pd.DataFrame
        FD salaries with columns [player_name, position, team, salary, week]
    threshold : int
        Absolute dollar amount above/below the mean difference at which to flag.

    Returns
    -------
    pd.DataFrame
        Columns:
          - player_name, position, team, week
          - dk_salary, fd_salary
          - salary_diff            (fd_salary - dk_salary)
          - avg_diff               (scalar repeated; mean(salary_diff) )
          - diff_from_avg          (salary_diff - avg_diff)
          - flagged (bool)         True if |diff_from_avg| >= threshold
    """
    # 1) Rename and merge on player/week keys
    left = dk_df.rename(columns={"salary": "dk_salary"})
    right = fd_df.rename(columns={"salary": "fd_salary"})
    merged = pd.merge(
        left, right,
        on=["player_name", "position", "team", "week"],
        how="inner"
    )

    # 2) Compute raw difference and average
    merged["salary_diff"] = merged["fd_salary"] - merged["dk_salary"]
    avg_diff = merged["salary_diff"].mean()

    # 3) Compute deviation and flag outliers
    merged["avg_diff"] = avg_diff
    merged["diff_from_avg"] = merged["salary_diff"] - avg_diff
    merged["flagged"] = merged["diff_from_avg"].abs() >= threshold

    # 4) Reorder columns for clarity
    cols = [
        "player_name", "position", "team", "week",
        "dk_salary", "fd_salary",
        "salary_diff", "avg_diff", "diff_from_avg", "flagged"
    ]
    return merged[cols]