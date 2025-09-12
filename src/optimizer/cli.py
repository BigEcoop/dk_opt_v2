# src/optimizer/cli.py

import click
import pandas as pd
import nfl_data_py as nfl

from optimizer.data_fetch import fetch_weekly_data
from optimizer.engine1 import score_df


@click.group()
def main():
    """dk_opt command-line interface."""
    pass


@main.command("fetch-data")
@click.option(
    "-y", "--years",
    type=int,
    multiple=True,
    required=True,
    help="One or more season years to fetch (e.g. -y 2023 -y 2024)."
)
@click.option(
    "-w", "--week",
    type=int,
    required=True,
    help="The week number to fetch (1–18)."
)
@click.option(
    "-c", "--columns",
    type=str,
    multiple=True,
    required=True,
    help="Columns to include in the output (e.g. -c player_name -c position)."
)
@click.option(
    "-o", "--output", "output_path",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, resolve_path=True),
    required=False,
    help="Path to write the fetched CSV (instead of stdout)."
)
def fetch_data(years, week, columns, output_path):
    """
    Fetch weekly NFL data and either print or save the selected columns.
    """
    df = fetch_weekly_data(list(years), week, list(columns))
    if output_path:
        df.to_csv(output_path, index=False)
        click.echo(f"✅ Fetched data saved to {output_path}")
    else:
        click.echo(df.to_csv(index=False))


@main.command("list-columns")
@click.option(
    "-y", "--years",
    type=int,
    multiple=True,
    required=True,
    help="One or more season years to inspect (e.g. -y 2023 -y 2024)."
)
@click.option(
    "-w", "--week",
    type=int,
    required=True,
    help="The week number to inspect (1–18)."
)
def list_columns(years, week):
    """
    List all available column names in nfl_data_py.import_weekly_data for debugging.
    """
    df = nfl.import_weekly_data(list(years))
    click.echo("Available columns:")
    for col in df.columns:
        click.echo(f"  • {col}")


@main.command("score-data")
@click.option(
    "-i", "--input", "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    required=True,
    help="Path to raw stats CSV."
)
@click.option(
    "-o", "--output", "output_path",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, resolve_path=True),
    required=True,
    help="Where to save the scored CSV."
)
def score_data(input_path, output_path):
    """
    Load raw stats CSV, compute fantasy points, and save with a new column 'fantasy_points'.
    """
    df = pd.read_csv(input_path)
    scored = score_df(df)
    scored.to_csv(output_path, index=False)
    click.echo(f"✅ Scored data saved to {output_path}")


if __name__ == "__main__":
    main()