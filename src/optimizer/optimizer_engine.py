#!/usr/bin/env python3
# src/optimizer/optimizer_engine.py

import pulp
from pulp import PULP_CBC_CMD, LpStatus

def optimize_lineup(
    df,
    score_col="proj_mean",
    no_good_cuts=None
):
    """
    Build and solve a single DraftKings lineup:
      - 1 QB, 2 RB, 3 WR, ≥1 TE, 1 DST, 1 FLEX (RB/WR/TE)
      - Salary ≤ 50,000
      - Maximize the given score_col
    no_good_cuts: list of iterables of ids to exclude exact prior lineups
    """
    model = pulp.LpProblem("DK_Lineup", pulp.LpMaximize)

    # Decision variables
    vars = {
        row.id: pulp.LpVariable(f"x_{row.id}", cat="Binary")
        for _, row in df.iterrows()
    }

    # Objective
    model += pulp.lpSum(
        vars[row.id] * row[score_col] for _, row in df.iterrows()
    ), "TotalScore"

    # Salary cap
    model += pulp.lpSum(
        vars[row.id] * row.dk_salary for _, row in df.iterrows()
    ) <= 50000, "SalaryCap"

    # Positional constraints
    model += pulp.lpSum(
        vars[row.id] for _, row in df[df.pos == "QB"].iterrows()
    ) == 1, "QB"
    model += pulp.lpSum(
        vars[row.id] for _, row in df[df.pos == "RB"].iterrows()
    ) == 2, "RBs"
    model += pulp.lpSum(
        vars[row.id] for _, row in df[df.pos == "WR"].iterrows()
    ) == 3, "WRs"
    model += pulp.lpSum(
        vars[row.id] for _, row in df[df.pos == "TE"].iterrows()
    ) >= 1, "TE_Min"
    model += pulp.lpSum(
        vars[row.id] for _, row in df[df.pos == "DST"].iterrows()
    ) == 1, "DST"

    skill_pool = df[df.pos.isin(["RB", "WR", "TE"])]
    model += pulp.lpSum(
        vars[row.id] for _, row in skill_pool.iterrows()
    ) == 7, "SkillTotal"

    model += pulp.lpSum(
        vars[row.id] for _, row in df.iterrows()
    ) == 9, "TotalRoster"

    # No‐good cuts: exclude exact previous lineups
    if no_good_cuts:
        for idx, cut in enumerate(no_good_cuts):
            model += pulp.lpSum(vars[i] for i in cut) <= len(cut) - 1, f"NoGood_{idx}"

    status = model.solve(PULP_CBC_CMD(msg=False))
    if LpStatus[model.status] != "Optimal":
        raise RuntimeError(f"No optimal solution: status={LpStatus[model.status]}")

    chosen = [pid for pid, var in vars.items() if var.value() == 1]
    return df[df.id.isin(chosen)]