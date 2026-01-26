"""
Data ingestion module.

Functions for loading raw UFC data from CSV files into pandas DataFrames.
No transformations are applied at this stage beyond basic CSV parsing.
"""

import shutil
from pathlib import Path

import pandas as pd

from ufc_predictor.config import UFC_DATA_SOURCE_DIR, RAW_DATA_DIR

# CSV file mappings based on ufc-data/ contents
FIGHTS_CSV = "ufc_fight_results.csv"  # Main fight outcomes
FIGHTERS_CSV = "ufc_fighter_tott.csv"  # Fighter physical attributes
EVENTS_CSV = "ufc_event_details.csv"  # Event metadata
FIGHT_STATS_CSV = "ufc_fight_stats.csv"  # Per-round statistics
FIGHTER_DETAILS_CSV = "ufc_fighter_details.csv"  # Basic fighter info
FIGHT_DETAILS_CSV = "ufc_fight_details.csv"  # Basic fight listings
ODDS_CSV = "ufc_betting_odds.csv"  # Betting odds (Kaggle dataset)


def load_raw_fights(path: Path | None = None) -> pd.DataFrame:
    """
    Load the primary fight-level CSV into a DataFrame.

    If `path` is None, use the default fights CSV from UFC_DATA_SOURCE_DIR.

    Args:
        path: Path to the raw fights CSV file (optional)

    Returns:
        DataFrame containing raw fight records

    Raises:
        FileNotFoundError: If the specified CSV file does not exist
    """
    if path is None:
        path = UFC_DATA_SOURCE_DIR / FIGHTS_CSV
    if not path.exists():
        raise FileNotFoundError(f"Fights CSV not found at: {path}")
    return pd.read_csv(path)


def load_raw_fighters(path: Path | None = None) -> pd.DataFrame:
    """
    Load the fighter-level metadata CSV into a DataFrame.

    If `path` is None, use the default fighters CSV from UFC_DATA_SOURCE_DIR.

    Args:
        path: Path to the raw fighters CSV file (optional)

    Returns:
        DataFrame containing raw fighter metadata (height, reach, stance, etc.)

    Raises:
        FileNotFoundError: If the specified CSV file does not exist
    """
    if path is None:
        path = UFC_DATA_SOURCE_DIR / FIGHTERS_CSV
    if not path.exists():
        raise FileNotFoundError(f"Fighters CSV not found at: {path}")
    return pd.read_csv(path)


def load_raw_events(path: Path | None = None) -> pd.DataFrame:
    """
    Load the event-level CSV into a DataFrame.

    If `path` is None, use the default events CSV from UFC_DATA_SOURCE_DIR.

    Args:
        path: Path to the raw events CSV file (optional)

    Returns:
        DataFrame containing UFC event information

    Raises:
        FileNotFoundError: If the specified CSV file does not exist
    """
    if path is None:
        path = UFC_DATA_SOURCE_DIR / EVENTS_CSV
    if not path.exists():
        raise FileNotFoundError(f"Events CSV not found at: {path}")
    return pd.read_csv(path)


def load_raw_odds(path: Path | None = None) -> pd.DataFrame:
    """
    Load betting odds CSV from Kaggle dataset.

    If `path` is None, use the default odds CSV from UFC_DATA_SOURCE_DIR.
    Returns empty DataFrame if file is not found (odds are optional).

    Args:
        path: Path to the raw odds CSV file (optional)

    Returns:
        DataFrame containing betting odds data, or empty DataFrame if not found
    """
    if path is None:
        path = UFC_DATA_SOURCE_DIR / ODDS_CSV
    if not path.exists():
        print(f"Warning: Odds file not found at {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def load_raw_fight_stats(path: str) -> pd.DataFrame:
    """
    Load raw per-fight statistics from CSV file.

    Args:
        path: Path to the raw fight stats CSV file

    Returns:
        DataFrame containing detailed fight statistics

    TODO: Implement CSV loading
    """
    pass


def sync_raw_to_data_dir(overwrite: bool = False) -> list[Path]:
    """
    Ensure that copies of the raw UFC CSVs exist under data/raw/.

    - If overwrite is False, do not overwrite existing files under data/raw/.
    - If overwrite is True, overwrite the copies.
    - This does not modify the original files under `ufc-data/`.

    Args:
        overwrite: Whether to overwrite existing files in data/raw/

    Returns:
        List of file paths that were synced (copied)
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    synced = []
    for csv_file in UFC_DATA_SOURCE_DIR.glob("*.csv"):
        dest = RAW_DATA_DIR / csv_file.name
        if not dest.exists() or overwrite:
            shutil.copy2(csv_file, dest)
            synced.append(dest)
    return synced


if __name__ == "__main__":
    synced = sync_raw_to_data_dir()
    if synced:
        print(f"Synced {len(synced)} files to {RAW_DATA_DIR}:")
        for f in synced:
            print(f"  - {f.name}")
    else:
        print("No files synced (all files already exist).")
