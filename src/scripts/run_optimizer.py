#!/usr/bin/env python3
# src/scripts/run_optimizer.py

import sys
import argparse
import json
import random
import csv
import re
import html
from pathlib import Path

import pandas as pd
from pulp import (
    LpProblem,
    LpMaximize,
    LpVariable,
    lpSum,
    PULP_CBC_CMD,
    LpStatus
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SALARY_MIN = 49400
SALARY_MAX = 50000

TEAM_CODE = {
    "Arizona Cardinals":     "ARI",
    "Atlanta Falcons":       "ATL",
    "Baltimore Ravens":      "BAL",
    "Buffalo Bills":         "BUF",
    "Carolina Panthers":     "CAR",
    "Chicago Bears":         "CHI",
    "Cincinnati Bengals":    "CIN",
    "Cleveland Browns":      "CLE",
    "Dallas Cowboys":        "DAL",
    "Denver Broncos":        "DEN",
    "Detroit Lions":         "DET",
    "Green Bay Packers":     "GB",
    "Houston Texans":        "HOU",
    "Indianapolis Colts":    "IND",
    "Jacksonville Jaguars":  "JAX",
    "Kansas City Chiefs":    "KC",
    "Las Vegas Raiders":     "LV",
    "Los Angeles Chargers":  "LAC",
    "Los Angeles Rams":      "LAR",
    "Miami Dolphins":        "MIA",
    "Minnesota Vikings":     "MIN",
    "New England Patriots":  "NE",
    "New Orleans Saints":    "NO",
    "New York Giants":       "NYG",
    "New York Jets":         "NYJ",
    "Philadelphia Eagles":   "PHI",
    "Pittsburgh Steelers":   "PIT",
    "San Francisco 49ers":   "SF",
    "Seattle Seahawks":      "SEA",
    "Tampa Bay Buccaneers":  "TB",
    "Tennessee Titans":      "TEN",
    "Washington Commanders": "WAS"
}

def own_multiplier(own):
    if own <= 0.09:
        return 1.05
    elif own <= 0.10:
        return 1.00
    elif own <= 0.18:
        return 0.97
    elif own <= 0.21:
        return 0.94
    else:
        return 0.90

def def_multiplier(rank):
    if rank >= 26:
        return 1.10
    elif rank >= 20:
        return 1.05
    elif rank >= 9:
        return 1.00
    elif rank >= 5:
        return 0.97
    else:
        return 0.94

def normalize_name(s):
    """Strip suffixes, HTML entities, punctuation; uppercase."""
    s = html.unescape(str(s))
    s = re.sub(r"\b(II|III|IV|JR\.?|SR\.?)\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"[^\w\s']+", "", s)
    return s.upper().strip()

def main():
    parser = argparse.ArgumentParser(
        description="Build bulk DraftKings lineups with extended salary & uniqueness cycling"
    )
    parser.add_argument("--count", type=int, default=100,
                        help="Number of lineups to generate")
    parser.add_argument("--week", type=int, required=True, help="NFL Week")
    parser.add_argument("--year", type=int, required=True, help="Season Year")
    parser.add_argument("--boosts", type=Path,
                        help="JSON mapping dk_id to multiplier (optional)")
    args = parser.parse_args()

    # 1) Load pool & drop zero-salary rows
    pool_csv = PROJECT_ROOT / "src/output/optimizer_pool" / f"optimizer_pool_w{args.week}.csv"
    df = pd.read_csv(pool_csv)
    df = df[df["dk_salary"] > 0].reset_index(drop=True)
    print(f"🔍 Pool size after salary>0 filter: {len(df)}")

    # 2) POST-HOC SIMS PATCH (only apply when sim_pts>0)
    sims_xlsx = PROJECT_ROOT / "src/output/sims" / f"{args.year}_w{args.week}_all" / "player_summary.xlsx"
    if sims_xlsx.exists():
        sims = pd.read_excel(sims_xlsx)

        # Normalize pool and sims names, teams, positions
        df["name_norm"]        = df["name"].apply(normalize_name)
        df["team_code_norm"]   = df["team"].astype(str).str.upper().str.strip()
        df["pos_norm"]         = df["pos"].astype(str).str.upper().str.strip()

        sims["name_norm"]      = sims["player"].apply(normalize_name)
        sims["team_merge_norm"]= sims["team"].astype(str).str.upper().str.strip()
        sims["pos_norm"]       = sims["pos"].astype(str).str.upper().str.strip()

        # Merge on normalized keys
        df = df.merge(
            sims[["name_norm", "team_merge_norm", "pos_norm", "dk_points_mean"]],
            left_on=["name_norm", "team_code_norm", "pos_norm"],
            right_on=["name_norm", "team_merge_norm", "pos_norm"],
            how="left"
        )

        # Patch projections where sim exists
        patched_mask = df["dk_points_mean"].fillna(0) > 0
        print(f"🔍 Rows with dk_points_mean>0 (to patch): {patched_mask.sum()}")
        df.loc[patched_mask, "proj_mean"] = df.loc[patched_mask, "dk_points_mean"]

    # 3) Identify & export true ghosts before dropping
    ghosts = df[(df["proj_mean"] == 0) & df["dk_points_mean"].isnull()]
    ghosts_csv = PROJECT_ROOT / "src/output" / f"ghosts_removed_w{args.week}_y{args.year}.csv"
    ghosts_csv.parent.mkdir(parents=True, exist_ok=True)
    ghosts.to_csv(ghosts_csv, index=False)
    print(f"🔍 Exported {len(ghosts)} ghost rows to {ghosts_csv}")

    # Drop ghosts and helper cols
    mask_keep = ~((df["proj_mean"] == 0) & df["dk_points_mean"].isnull())
    df = df[mask_keep].reset_index(drop=True)
    print(f"🔍 Pool size after removing ghosts: {len(df)}")
    df.drop(columns=[
        "team_merge", "sim_pts",
        "name_norm", "team_code_norm", "pos_norm",
        "team_merge_norm", "dk_points_mean"
    ], inplace=True, errors="ignore")

    # 4) Positions & teams
    df["pos"] = df["pos"].fillna("").astype(str)
    df.loc[df["pos"] == "", "pos"] = "DST"
    if df["pos"].value_counts().get("DST", 0) < 1:
        print("❌ No DST found in pool")
        sys.exit(1)
    df["team_code"] = df["team"].map(TEAM_CODE).fillna(df["team"]).str.upper()

    # 5) Boosts/fades
    boosts = {}
    if args.boosts and args.boosts.exists():
        boosts = {str(k): v for k, v in json.loads(args.boosts.read_text()).items()}
    df["score_adj"] = df.apply(
        lambda r: (
            r["proj_mean"]
            * boosts.get(str(r["dk_id"]), 1.0)
            * own_multiplier(r["proj_own"])
            * def_multiplier(r["def_vs_pos"])
        ), axis=1
    )

    # 6) Metrics
    df["fantasy_value"] = df["score_adj"] / (df["dk_salary"] / 1000)
    df["h_value"]       = (df["score_adj"] ** (1/3)) / df["dk_salary"] * 2000
    df["ceiling_score"] = df["proj_ceiling"] / df["proj_ceiling"].max()

    # 7) Betting sims
    sims_file = PROJECT_ROOT / "src/output/sims" / f"{args.year}_w{args.week}_all" / "betting.csv"
    if sims_file.exists():
        sims2 = pd.read_csv(sims_file)
        sims2 = sims2[sims2["type"] == "GAME"]
        implied_map, total_map = {}, {}
        for _, row in sims2.iterrows():
            implied_map[row["home"]] = row["home_avg_score"]
            implied_map[row["away"]] = row["away_avg_score"]
            total_map[row["home"]]   = row["mean_total"]
            total_map[row["away"]]   = row["mean_total"]
        df["implied_team_total"] = df["team_code"].map(implied_map).fillna(0.0)
        df["game_total"]         = df["team_code"].map(total_map).fillna(0.0)
    else:
        df["implied_team_total"] = 0.0
        df["game_total"]         = 0.0

    df["implied_norm"]    = df["implied_team_total"] / df["implied_team_total"].max()
    df["game_total_norm"] = df["game_total"]         / df["game_total"].max()

    # 8) Price flags
    pricing_file = PROJECT_ROOT / "src/output/salary" / f"fd_vs_dk_w{args.week}.csv"
    price_map = {}
    if pricing_file.exists():
        pricing = pd.read_csv(pricing_file)
        for _, row in pricing.iterrows():
            dkid = str(row["ID"])
            w = 0.0
            if row.get("DK_EXTREME_VALUE", False):      w += 0.75
            if row.get("DK_VALUE", False):              w += 0.40
            if row.get("DK_OVERPRICED", False):         w -= 0.40
            if row.get("DK_EXTREME_OVERPRICED", False): w -= 0.75
            price_map[dkid] = w
    df["price_weight"] = df["dk_id"].map(lambda i: price_map.get(str(i), 0.0))

    # 9) Structure & exposures
    struct         = json.loads((PROJECT_ROOT / "src/data/lineup_structure.json").read_text())
    stack_usage    = struct["stack_usage"]
    stack_usage["naked_qb_rate"] = 0.07
    bringback_use  = struct["bringback_usage"]
    flex_dist      = struct["flex_position_distribution"]

    exposures   = {str(int(dkid)): 0 for dkid in df["dk_id"]}
    all_lineups = []

    # 10) Generate lineups
    solver = PULP_CBC_CMD(msg=False, timeLimit=30)
    prior_lineups = []
    trial = 0
    attempts = 0
    max_attempts = args.count * 10

    schedule = [
        (4, SALARY_MIN),    (4, SALARY_MIN - 200),
        (5, SALARY_MIN),    (5, SALARY_MIN - 200),
        (6, SALARY_MIN),    (6, SALARY_MIN - 200),
        (4, SALARY_MIN - 400),(5, SALARY_MIN - 400),(6, SALARY_MIN - 400),
        (7, SALARY_MIN),    (7, SALARY_MIN - 200),(7, SALARY_MIN - 400),
        (8, SALARY_MIN),    (8, SALARY_MIN - 200),(8, SALARY_MIN - 400),
        (9, SALARY_MIN),    (9, SALARY_MIN - 200),(9, SALARY_MIN - 400),
    ]
    schedule_index = 0
    current_cap, salary_floor_current = schedule[schedule_index]
    failures_since_toggle = 0
    cap_counts = {}
    salary_floor_counts = {}

    while trial < args.count and attempts < max_attempts:
        attempts += 1
        if attempts % 10 == 0:
            print(f"🔄 Attempt {attempts}  |  Valid lineups so far: {trial}")

        model = LpProblem("DKLineup", LpMaximize)
        vars_map = {
            int(r.dk_id): LpVariable(f"x_{r.dk_id}", cat="Binary")
            for _, r in df.iterrows()
        }

        # a) exposure caps
        for dkid_str, count in exposures.items():
            dkid = int(dkid_str)
            own  = df.loc[df["dk_id"] == dkid, "proj_own"].iloc[0]
            max_allowed = 5 if own > 0.20 else 7
            if count >= max_allowed:
                model += vars_map[dkid] == 0, f"exp_cap_{dkid}_{attempts}"

        # b) stack vs naked
        if random.random() >= stack_usage["naked_qb_rate"]:
            choice = random.choices(
                list(stack_usage["stack_depths"].keys()),
                list(stack_usage["stack_depths"].values()), k=1
            )[0]
            depth_map = {"single_stack":2, "double_stack":3, "triple+_stack":4}
            n_stack = depth_map[choice]
            team_to_stack = random.choice(df["team_code"].unique())

            model += lpSum(
                vars_map[i] for i in vars_map
                if df.loc[df["dk_id"] == i, "team_code"].iloc[0] == team_to_stack
            ) >= n_stack, f"stack_depth_{attempts}"

            pu = stack_usage["positions_used"]
            if random.random() < pu["wr_in_stack"]:
                model += lpSum(
                    vars_map[i] for i in vars_map
                    if (
                        df.loc[df["dk_id"] == i, "team_code"].iloc[0] == team_to_stack and
                        df.loc[df["dk_id"] == i, "pos"].iloc[0]         == "WR"
                    )
                ) >= 1, f"stack_WR_{attempts}"
            if random.random() < pu["te_in_stack"]:
                model += lpSum(
                    vars_map[i] for i in vars_map
                    if (
                        df.loc[df["dk_id"] == i, "team_code"].iloc[0] == team_to_stack and
                        df.loc[df["dk_id"] == i, "pos"].iloc[0]         == "TE"
                    )
                ) >= 1, f"stack_TE_{attempts}"
            if random.random() < pu["rb_in_stack"]:
                model += lpSum(
                    vars_map[i] for i in vars_map
                    if (
                        df.loc[df["dk_id"] == i, "team_code"].iloc[0] == team_to_stack and
                        df.loc[df["dk_id"] == i, "pos"].iloc[0]         == "RB"
                    )
                ) >= 1, f"stack_RB_{attempts}"

            if random.random() < bringback_use["rate"]:
                pb = random.choices(
                    list(bringback_use["positions"].keys()),
                    list(bringback_use["positions"].values()), k=1
                )[0].upper()
                model += lpSum(
                    vars_map[i] for i in vars_map
                    if (
                        df.loc[df["dk_id"] == i, "team_code"].iloc[0] == team_to_stack and
                        df.loc[df["dk_id"] == i, "pos"].iloc[0]         == pb
                    )
                ) >= 1, f"bringback_{pb}_{attempts}"

        # c) flex distribution
        flex_pick = random.choices(
            ["WR","RB","TE"],
            [flex_dist["wr"], flex_dist["rb"], flex_dist["te"]], k=1
        )[0]
        if flex_pick == "WR":
            model += lpSum(
                vars_map[i] for i in vars_map
                if df.loc[df["dk_id"] == i, "pos"].iloc[0] == "WR"
            ) >= 4, f"flex_WR_{attempts}"
        elif flex_pick == "RB":
            model += lpSum(
                vars_map[i] for i in vars_map
                if df.loc[df["dk_id"] == i, "pos"].iloc[0] == "RB"
            ) >= 3, f"flex_RB_{attempts}"
        else:
            model += lpSum(
                vars_map[i] for i in vars_map
                if df.loc[df["dk_id"] == i, "pos"].iloc[0] == "TE"
            ) >= 2, f"flex_TE_{attempts}"

        # UNIQUE: dynamic cap
        for idx, prev_ids in enumerate(prior_lineups):
            model += lpSum(vars_map[i] for i in prev_ids) <= current_cap, f"unique_{idx}_{attempts}"

        # d) objective & core constraints
        model += lpSum(
            vars_map[i] * (
                  df.loc[df["dk_id"] == i, "score_adj"].iloc[0]
                + df.loc[df["dk_id"] == i, "fantasy_value"].iloc[0]
                + df.loc[df["dk_id"] == i, "h_value"].iloc[0]
                + 0.02 * df.loc[df["dk_id"] == i, "ceiling_score"].iloc[0]
                + 1.00 * df.loc[df["dk_id"] == i, "implied_norm"].iloc[0]
                + 0.75 * df.loc[df["dk_id"] == i, "game_total_norm"].iloc[0]
                + df.loc[df["dk_id"] == i, "price_weight"].iloc[0]
            )
            for i in vars_map
        ), "TotalCombinedScore"
        model += lpSum(
            vars_map[i] * df.loc[df["dk_id"] == i, "dk_salary"].iloc[0]
            for i in vars_map
        ) <= SALARY_MAX, "SalaryCap"
        model += lpSum(
            vars_map[i] * df.loc[df["dk_id"] == i, "dk_salary"].iloc[0]
            for i in vars_map
        ) >= salary_floor_current, "SalaryFloor"
        model += lpSum(
            vars_map[i] for i in vars_map
            if df.loc[df["dk_id"] == i, "pos"].iloc[0] == "QB"
        ) == 1, "QB"
        model += lpSum(
            vars_map[i] for i in vars_map
            if df.loc[df["dk_id"] == i, "pos"].iloc[0] == "DST"
        ) == 1, "DST"
        model += lpSum(
            vars_map[i] for i in vars_map
            if df.loc[df["dk_id"] == i, "pos"].iloc[0] == "RB"
        ) >= 2, "MinRB"
        model += lpSum(
            vars_map[i] for i in vars_map
            if df.loc[df["dk_id"] == i, "pos"].iloc[0] == "WR"
        ) >= 3, "MinWR"
        model += lpSum(
            vars_map[i] for i in vars_map
            if df.loc[df["dk_id"] == i, "pos"].iloc[0] == "TE"
        ) >= 1, "MinTE"
        model += lpSum(vars_map.values()) == 9, "TotalPlayers"

        status = model.solve(solver)
        if status != 1:
            # only count *failed* solves toward cycling
            failures_since_toggle += 1
            if failures_since_toggle >= 70:
                schedule_index = min(schedule_index + 1, len(schedule) - 1)
                current_cap, salary_floor_current = schedule[schedule_index]
                print(
                    f"⚠️ Switching to cap={current_cap} and "
                    f"salary floor={salary_floor_current} after 70 consecutive fails"
                )
                failures_since_toggle = 0
            continue

        # reset on success
        failures_since_toggle = 0

        cap_counts[current_cap] = cap_counts.get(current_cap, 0) + 1
        salary_floor_counts[salary_floor_current] = (
            salary_floor_counts.get(salary_floor_current, 0) + 1
        )

        selected = [i for i, v in vars_map.items() if v.value() == 1]
        trial += 1
        prior_lineups.append(set(selected))
        for dkid in selected:
            exposures[str(dkid)] += 1

        # ── BUILD ORDERED LINEUP ───────────────────────────────────────────────
        lineup   = df[df["dk_id"].isin(selected)].copy()
        qb_row   = lineup[lineup.pos == "QB"]
        rb_rows  = lineup[lineup.pos == "RB"].head(2)
        wr_rows  = lineup[lineup.pos == "WR"].head(3)
        te_row   = lineup[lineup.pos == "TE"].head(1)
        dst_row  = lineup[lineup.pos == "DST"]
        used     = set(qb_row.index) | set(rb_rows.index) \
                 | set(wr_rows.index) | set(te_row.index) \
                 | set(dst_row.index)
        flex_idx = [i for i in lineup.index if i not in used]
        flex_rows= lineup.loc[flex_idx]

        ordered = pd.concat([
            qb_row, rb_rows, wr_rows,
            te_row, flex_rows, dst_row
        ]).reset_index(drop=True)

        ordered["FPTS"]         = ordered["score_adj"].round(2)
        ordered["FantasyValue"] = ordered["fantasy_value"].round(2)
        ordered["HValue"]       = ordered["h_value"].round(2)

        total_sal  = ordered["dk_salary"].sum()
        total_fpts = ordered["FPTS"].sum()

        ordered["lineup_salary"] = total_sal
        ordered["lineup_fpts"]   = total_fpts
        ordered["iteration"]     = trial

        all_lineups.append(ordered)

    # end while

    if trial < args.count:
        print(f"⚠️ Could only generate {trial} unique lineups in {attempts} attempts")
    else:
        print(f"✅ Generated {trial} unique lineups in {attempts} attempts")

    # ── FULL OUTPUT ──────────────────────────────────────────────────────────────
    full = pd.concat(all_lineups, ignore_index=True)
    summary = (
        full
        .groupby("iteration")[["lineup_fpts", "lineup_salary"]]
        .first()
        .reset_index()
        .sort_values("lineup_fpts", ascending=False)
    )
    summary["rank"] = range(1, len(summary) + 1)
    rank_map       = dict(zip(summary["iteration"], summary["rank"]))
    full["rank"]   = full["iteration"].map(rank_map)

    def reorder(df_lu):
        idxs = []
        idxs += df_lu[df_lu.pos == "QB"].index.tolist()
        idxs += df_lu[df_lu.pos == "RB"].index.tolist()[:2]
        idxs += df_lu[df_lu.pos == "WR"].index.tolist()[:3]
        rest = [i for i in df_lu.index if i not in idxs]
        idxs += rest
        return df_lu.loc[idxs]

    full = (
        full
        .groupby("iteration", sort=False, group_keys=False)
        .apply(reorder)
        .reset_index(drop=True)
    )

    # insert TOTAL rows
    total_rows = []
    for _, row in summary.iterrows():
        total_rows.append({
            **{c: "" for c in full.columns},
            "iteration": row["iteration"],
            "rank":      row["rank"],
            "name":      "TOTAL",
            "dk_salary": row["lineup_salary"],
            "FPTS":      row["lineup_fpts"]
        })
    totals_df            = pd.DataFrame(total_rows)
    totals_df["is_total"] = True
    full["is_total"]      = False
    full_out = (
        pd.concat([full, totals_df], ignore_index=True)
        .sort_values(["rank", "iteration", "is_total"], ascending=[True, True, True])
        .drop(columns=["is_total"])
        .reset_index(drop=True)
    )

    # insert blank line after each TOTAL
    rows = []
    for _, r in full_out.iterrows():
        rows.append(r)
        if r["name"] == "TOTAL":
            rows.append(pd.Series({c: "" for c in full_out.columns}))
    full_df = pd.DataFrame(rows, columns=full_out.columns)

    # write bulk lineups CSV
    out_dir = PROJECT_ROOT / "src" / "output" / "lineups" / f"{args.year}_w_{args.week}"
    out_dir.mkdir(parents=True, exist_ok=True)
    full_csv = out_dir / f"bulk_lineups_{args.year}_w{args.week}.csv"
    full_df.to_csv(full_csv, index=False)

    # ── QUICK OUTPUT ─────────────────────────────────────────────────────────────
    quick    = full_out[["dk_id", "name", "team", "pos", "dk_salary", "FPTS"]].copy()
    qrows    = []
    for _, r in quick.iterrows():
        qrows.append(r)
        if r["name"] == "TOTAL":
            qrows.append(pd.Series({c: "" for c in quick.columns}))
    quick_df  = pd.DataFrame(qrows, columns=quick.columns)
    quick_csv = out_dir / f"quick_bulk_lineups_{args.year}_week_{args.week}.csv"
    quick_df.to_csv(quick_csv, index=False)

    # ── DK ENTRY FORM CSV ───────────────────────────────────────────────────────
    template_csv = (
        PROJECT_ROOT
        / "src" / "data" / "dk_entry_form"
        / f"dk_bulk_entry_{args.year}_w{args.week}.csv"
    )
    entry_csv = out_dir / f"dk_entry_{args.year}_w{args.week}.csv"

    with open(template_csv, newline="") as fin:
        reader = csv.reader(fin)
        rows   = list(reader)

    for lineup_idx, (_, grp) in enumerate(full_out.groupby("iteration", sort=False), start=1):
        ids = grp.loc[grp.name != "TOTAL", "dk_id"].astype(int).astype(str).tolist()
        if lineup_idx < len(rows):
            rows[lineup_idx][:9] = ids
        else:
            rows.append(ids + [""] * (len(rows[0]) - 9))

    with open(entry_csv, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerows(rows)

    # ── FINAL STATUS MESSAGE ───────────────────────────────────────────────────
    print(
        f"\n✅ Generated {trial} lineups and wrote:\n"
        f"   • {full_csv}\n"
        f"   • {quick_csv}\n"
        f"   • {entry_csv}"
    )

    # ── QUICK PROJECTION SUMMARY ────────────────────────────────────────────────
    scores = full.groupby("iteration")["lineup_fpts"].first()
    print("\n🏈  Projections distribution:")
    print(scores.describe().round(1))
    print("\nLowest 5 lineups:")
    print(scores.nsmallest(5))

    # ── UNIQUE CAP & SALARY DEBUG ───────────────────────────────────────────────
    print("\n⚙️ Lineups by unique-cap level:")
    for cap in sorted(cap_counts):
        print(f"  cap={cap}: {cap_counts[cap]}")
    print("\n💰 Lineups by salary floor:")
    for floor in sorted(salary_floor_counts):
        print(f"  floor={floor}: {salary_floor_counts[floor]}")

if __name__ == "__main__":
    main()