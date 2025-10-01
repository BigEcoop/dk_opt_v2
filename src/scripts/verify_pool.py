#!/usr/bin/env python3
# verify_pool.py

import argparse
import pandas as pd
import json
from pathlib import Path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--week",  type=int, default=4)
    p.add_argument("--year",  type=int, default=2025)
    args = p.parse_args()

    # build your file-paths from args.week & args.year
    sal_path    = Path(f"src/output/salary/fd_vs_dk_w{args.week}.csv")
    csv_path    = Path(f"src/output/optimizer_pool/optimizer_pool_w{args.week}.csv")
    json_path   = Path(f"src/output/optimizer_pool/optimizer_pool_w{args.week}.json")

    # 1) Salary
    sal = pd.read_csv(sal_path, dtype={"ID":str})
    sal = sal.rename(columns={"ID":"dk_id","player_name":"name"})
    sal = sal[sal.dk_salary>0][["dk_id","name"]]

    # 2) Pools
    pool_csv  = pd.read_csv(csv_path, dtype={"dk_id":str})
    pool_json = pd.read_json(json_path, dtype={"dk_id":str})

    # 3) Diff
    sal_ids   = set(sal["dk_id"])
    csv_ids   = set(pool_csv["dk_id"])
    json_ids  = set(pool_json["dk_id"])
    miss_csv  = sal_ids - csv_ids
    miss_json = sal_ids - json_ids

    print(f"Paid players: {len(sal_ids)}")
    print(f"CSV rows:     {len(csv_ids)}")
    print(f"JSON rows:    {len(json_ids)}\n")

    if miss_csv:
        print("❌ Missing in CSV:", miss_csv)
    else:
        print("✅ All in CSV")
    if miss_json:
        print("❌ Missing in JSON:", miss_json)
    else:
        print("✅ All in JSON")