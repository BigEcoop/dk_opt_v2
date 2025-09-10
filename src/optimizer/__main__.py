# src/dk_opt/__main__.py

import argparse
import logging
import sys

from dk_opt.optimizer import optimize
from dk_opt.projections import load_projections
from dk_opt.salary import load_salaries
from dk_opt.utils import save_lineups


def main():
    parser = argparse.ArgumentParser(
        description="DFS optimizer for DraftKings NFL contests"
    )
    parser.add_argument(
        "--salary-file",
        required=True,
        help="Path to the CSV file with player salaries",
    )
    parser.add_argument(
        "--projection-file",
        required=True,
        help="Path to the CSV file with player projections",
    )
    parser.add_argument(
        "--output-file",
        default="lineups.csv",
        help="Where to save the generated lineups (CSV)",
    )
    parser.add_argument(
        "--num-lineups",
        type=int,
        default=150,
        help="Number of lineups to generate",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    logging.info("Loading salary data from %s", args.salary_file)
    salaries = load_salaries(args.salary_file)

    logging.info("Loading projection data from %s", args.projection_file)
    projections = load_projections(args.projection_file)

    logging.info("Optimizing %d lineups", args.num_lineups)
    lineups = optimize(
        salaries,
        projections,
        num_lineups=args.num_lineups,
    )

    logging.info("Saving %d lineups to %s", len(lineups), args.output_file)
    save_lineups(lineups, args.output_file)

    print(f"Generated {len(lineups)} lineups. Saved to {args.output_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Unhandled exception in CLI:")
        sys.exit(1)