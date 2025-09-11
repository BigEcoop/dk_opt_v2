#!/usr/bin/env python3
# src/optimizer/__main__.py

import argparse
import logging
import sys

from optimizer.data_loader import load_scoring
from optimizer.ingestion.salary_ingestor_csv import (
    get_draftkings_salaries_csv,
    get_fanduel_salaries_csv,
)
from optimizer.analysis.salary_analysis import compare_fd_dk_salaries


def main():
    parser = argparse.ArgumentParser(
        description="DFS salary merge & compare"
    )
    parser.add_argument(
        "--week",
        type=int,
        required=True,
        help="Week number (e.g. 2)",
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Season year (e.g. 2025)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=300,
        help="Salary diff threshold for flagging",
    )
    parser.add_argument(
        "--output",
        help="Where to write fd_vs_dk CSV (defaults to src/output)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Merge
    dk = get_draftkings_salaries_csv(args.week, args.year)
    fd = get_fanduel_salaries_csv(args.week, args.year)
    merged = compare_fd_dk_salaries(dk, fd, threshold=args.threshold)

    # Output
    if args.output:
        out_path = args.output
    else:
        proj_root = Path(__file__).resolve().parent.parent
        out_dir = proj_root / "output"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"fd_vs_dk_w{args.week}.csv"

    merged.to_csv(out_path, index=False)
    print(f"Saved merged salaries to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Error in optimizer CLI")
        sys.exit(1)