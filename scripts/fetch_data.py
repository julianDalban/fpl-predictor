"""
Fetch FPL historical data from vaastav/Fantasy-Premier-League.
"""

import sys
from pathlib import Path
import requests
import pandas as pd

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "vaastav/Fantasy-Premier-League/master/data"
)

SEASONS = ["2021-22", "2022-23", "2023-24"]

# Files to fetch per season and their subpaths within the repo
FILES = {
    "merged_gw.csv": "gws/merged_gw.csv",       # gameweek-level rows
    "cleaned_players.csv": "cleaned_players.csv", # season-level summaries
    "players_raw.csv": "players_raw.csv",         # positional encoding
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch_file(season: str, filename: str, subpath: str) -> Path:
    """Download a single CSV and return the local path."""
    url = f"{BASE_URL}/{season}/{subpath}"
    dest = DATA_DIR / season / filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Fetching {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    dest.write_bytes(resp.content)
    return dest


def validate(path: Path, min_rows: int = 10) -> pd.DataFrame:
    """Read the CSV and do a quick sanity check."""
    df = pd.read_csv(path, low_memory=False)
    assert len(df) >= min_rows, f"{path.name}: only {len(df)} rows"
    return df


def main() -> None:
    print(f"Output directory: {DATA_DIR}\n")
    summary = []

    for season in SEASONS:
        print(f"--- {season} ---")
        for filename, subpath in FILES.items():
            path = fetch_file(season, filename, subpath)
            df = validate(path)
            summary.append(
                {"season": season, "file": filename,
                 "rows": len(df), "cols": len(df.columns)}
            )
            print(f"    {filename}: {len(df):,} rows x {len(df.columns)} cols")
        print()

    # Quick cross-season consistency check on merged_gw columns
    gw_cols = {}
    for season in SEASONS:
        df = pd.read_csv(DATA_DIR / season / "merged_gw.csv", nrows=0)
        gw_cols[season] = set(df.columns)

    shared = set.intersection(*gw_cols.values())
    for season in SEASONS:
        diff = gw_cols[season] - shared
        if diff:
            print(f"WARNING: {season} has extra columns not in all seasons: {diff}")

    print("Column intersection across all seasons:")
    print(f"  {len(shared)} shared columns")
    print(f"  {sorted(shared)}\n")

    print("Done. You can now delete this script.")


if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        print(f"\nHTTP error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        sys.exit(1)