"""
Data cleaning module.

Functions for cleaning and standardizing raw UFC data.
Operations include:
- Removing duplicates
- Handling missing values (drop or impute based on column importance)
- Standardizing data types (dates, numerics, categoricals)
- Normalizing categorical labels (weight classes, stances, etc.)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_predictor.config import INTERMEDIATE_DATA_DIR
from ufc_predictor.data.ingest import load_raw_fights, load_raw_fighters, load_raw_events, load_raw_odds


def clean_fights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize raw fight data.

    Operations:
    - Merge with events data to get dates
    - Parse BOUT column to extract fighter names
    - Create binary target variable (1=fighter1 wins, 0=fighter2 wins)
    - Drop draws and no contests
    - Sort chronologically (oldest first)
    - Remove duplicate fight records

    Args:
        df: Raw fights DataFrame

    Returns:
        Cleaned fights DataFrame with columns:
        event, date, fighter1, fighter2, target, weight_class, method, round, time, url
    """
    # Load events data to get dates
    events_df = load_raw_events()

    # Strip whitespace from EVENT column in both dataframes for proper merging
    df["EVENT"] = df["EVENT"].str.strip()
    events_df["EVENT"] = events_df["EVENT"].str.strip()

    # Merge fights with events to get dates
    df = df.merge(events_df[["EVENT", "DATE"]], on="EVENT", how="left")

    # Parse DATE column to datetime
    df["date"] = pd.to_datetime(df["DATE"], format="%B %d, %Y", errors="coerce")

    # Parse BOUT column to extract fighter1 and fighter2
    bout_split = df["BOUT"].str.split(" vs. ", n=1, expand=True)
    df["fighter1"] = bout_split[0]
    df["fighter2"] = bout_split[1]

    # Create binary target column
    # "W/L" = fighter1 wins (target=1), "L/W" = fighter2 wins (target=0)
    # Drop draws ("D/D") and no contests ("NC/NC")
    df = df[df["OUTCOME"].isin(["W/L", "L/W"])].copy()
    df["target"] = (df["OUTCOME"] == "W/L").astype(int)

    # Sort chronologically (oldest first) for ELO processing
    df = df.sort_values("date").reset_index(drop=True)

    # Drop duplicates
    df = df.drop_duplicates(subset=["EVENT", "BOUT"], keep="first")

    # Select and rename columns for downstream use
    df = df[[
        "EVENT", "date", "fighter1", "fighter2", "target",
        "WEIGHTCLASS", "METHOD", "ROUND", "TIME", "URL"
    ]].rename(columns={
        "EVENT": "event",
        "WEIGHTCLASS": "weight_class",
        "METHOD": "method",
        "ROUND": "round",
        "TIME": "time",
        "URL": "url"
    })

    return df


def clean_fighters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize raw fighter metadata.

    Operations:
    - Replace "--" with NaN for missing values
    - Parse HEIGHT to numeric inches
    - Parse REACH to numeric inches
    - Parse DOB to datetime
    - Parse WEIGHT to numeric pounds
    - Standardize STANCE (empty -> NaN)
    - Remove duplicate fighter records

    Args:
        df: Raw fighters DataFrame

    Returns:
        Cleaned fighters DataFrame with columns:
        fighter, height_inches, reach_inches, weight_lbs, stance, dob, url
    """
    df = df.copy()

    # Replace "--" with NaN for missing values
    df.replace("--", np.nan, inplace=True)

    # Parse HEIGHT (e.g., "5' 11\"") to inches
    def parse_height(h):
        if pd.isna(h):
            return np.nan
        match = re.match(r"(\d+)'\s*(\d+)\"", str(h))
        if match:
            feet = int(match.group(1))
            inches = int(match.group(2))
            return feet * 12 + inches
        return np.nan

    df["height_inches"] = df["HEIGHT"].apply(parse_height)

    # Parse REACH (e.g., "72\"") to numeric inches
    def parse_reach(r):
        if pd.isna(r):
            return np.nan
        match = re.match(r"(\d+)\"", str(r))
        if match:
            return int(match.group(1))
        return np.nan

    df["reach_inches"] = df["REACH"].apply(parse_reach)

    # Parse DOB (e.g., "Jul 22, 1989") to datetime
    df["dob"] = pd.to_datetime(df["DOB"], format="%b %d, %Y", errors="coerce")

    # Parse WEIGHT (e.g., "185 lbs.") to numeric pounds
    def parse_weight(w):
        if pd.isna(w):
            return np.nan
        match = re.match(r"(\d+)\s*lbs\.?", str(w))
        if match:
            return int(match.group(1))
        return np.nan

    df["weight_lbs"] = df["WEIGHT"].apply(parse_weight)

    # Standardize STANCE (empty string -> NaN)
    df["stance"] = df["STANCE"].replace("", np.nan)

    # Drop duplicates
    df = df.drop_duplicates(subset=["FIGHTER"], keep="first")

    # Select and rename columns
    df = df[[
        "FIGHTER", "height_inches", "reach_inches", "weight_lbs",
        "stance", "dob", "URL"
    ]].rename(columns={
        "FIGHTER": "fighter",
        "URL": "url"
    })

    return df


def clean_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize raw event data.

    Operations:
    - Remove duplicate events
    - Standardize event date formats
    - Normalize location/venue information

    Args:
        df: Raw events DataFrame

    Returns:
        Cleaned events DataFrame

    TODO: Implement cleaning logic
    """
    pass


def clean_odds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize betting odds data.

    Operations:
    1. Rename columns for consistency (fight_url -> url, odds_1 -> odds_f1, odds_2 -> odds_f2)
    2. Group by fight URL and take mean odds (handles multiple bookies)
    3. Return minimal columns needed for merge

    Args:
        df: Raw odds DataFrame from Kaggle

    Returns:
        Cleaned odds DataFrame with columns: url, odds_f1, odds_f2
    """
    if df.empty:
        return df

    # Rename for clarity
    df = df.rename(columns={
        'fight_url': 'url',
        'odds_1': 'odds_f1',
        'odds_2': 'odds_f2'
    })

    # Group by fight URL and average the odds (handles multiple bookies)
    odds_agg = df.groupby('url').agg({
        'odds_f1': 'mean',
        'odds_f2': 'mean'
    }).reset_index()

    return odds_agg


def merge_odds_with_fights(fights_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge betting odds with fights using URL as join key.

    Uses UFCStats fight URLs for reliable matching across datasets.

    Args:
        fights_df: Cleaned fights DataFrame (must have 'url' column)
        odds_df: Raw odds DataFrame from Kaggle

    Returns:
        Fights DataFrame with odds_f1 and odds_f2 columns added
    """
    if odds_df.empty:
        fights_df['odds_f1'] = np.nan
        fights_df['odds_f2'] = np.nan
        return fights_df

    # Clean odds data
    odds_clean = clean_odds(odds_df)

    # Simple URL-based merge
    merged = fights_df.merge(
        odds_clean[['url', 'odds_f1', 'odds_f2']],
        on='url',
        how='left'
    )

    # Report match rate
    matched = merged['odds_f1'].notna().sum()
    total = len(merged)
    print(f"  Odds merge: {matched}/{total} fights matched ({100*matched/total:.1f}%)")

    return merged


def run_cleaning_pipeline() -> None:
    """
    Run the complete data cleaning pipeline.

    Loads raw data, cleans fights and fighters, and saves to parquet files
    in the intermediate data directory.
    """
    print("Loading raw data...")
    raw_fights = load_raw_fights()
    raw_fighters = load_raw_fighters()

    print(f"Raw fights: {len(raw_fights)} rows")
    print(f"Raw fighters: {len(raw_fighters)} rows")

    print("\nCleaning fights...")
    cleaned_fights = clean_fights(raw_fights)

    print("Cleaning fighters...")
    cleaned_fighters = clean_fighters(raw_fighters)

    print(f"\nCleaned fights: {len(cleaned_fights)} rows")
    print(f"Cleaned fighters: {len(cleaned_fighters)} rows")

    # Referential integrity filter: keep only fights where both fighters exist
    print("\nApplying referential integrity filter...")
    valid_fighters = set(cleaned_fighters["fighter"])
    before_filter = len(cleaned_fights)
    cleaned_fights = cleaned_fights[
        cleaned_fights["fighter1"].isin(valid_fighters)
        & cleaned_fights["fighter2"].isin(valid_fighters)
    ].reset_index(drop=True)
    after_filter = len(cleaned_fights)
    filtered_count = before_filter - after_filter
    print(f"Filtered {filtered_count} fights with missing fighters")
    print(f"Final cleaned fights: {after_filter} rows")

    # Merge betting odds
    print("\nMerging betting odds...")
    raw_odds = load_raw_odds()
    if not raw_odds.empty:
        print(f"Loaded {len(raw_odds)} odds records")
        cleaned_fights = merge_odds_with_fights(cleaned_fights, raw_odds)
    else:
        print("No odds data found - skipping odds merge")
        cleaned_fights['odds_f1'] = np.nan
        cleaned_fights['odds_f2'] = np.nan

    # Create intermediate data directory if needed
    INTERMEDIATE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save to parquet
    fights_path = INTERMEDIATE_DATA_DIR / "cleaned_fights.parquet"
    fighters_path = INTERMEDIATE_DATA_DIR / "cleaned_fighters.parquet"

    print(f"\nSaving cleaned data...")
    cleaned_fights.to_parquet(fights_path, index=False)
    cleaned_fighters.to_parquet(fighters_path, index=False)

    print(f"  Fights saved to: {fights_path}")
    print(f"  Fighters saved to: {fighters_path}")


if __name__ == "__main__":
    run_cleaning_pipeline()
    print("\nCleaning complete!")
