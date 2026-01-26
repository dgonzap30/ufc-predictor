"""
ELO rating system implementation for UFC fighters.

This module implements a standard ELO rating system adapted for UFC fights.
Ratings are updated chronologically after each fight to maintain temporal consistency.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from ufc_predictor.config import ELO_START, K_FACTOR, INTERMEDIATE_DATA_DIR


def initialize_ratings(start_rating: float = 1500) -> Dict[str, float]:
    """
    Initialize an empty ELO ratings dictionary.

    Args:
        start_rating: Initial rating for new fighters (default: 1500)

    Returns:
        Empty dictionary to be populated with fighter ratings

    TODO: Implement initialization logic
    """
    pass


def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calculate expected score (win probability) for fighter A against fighter B.

    Uses standard ELO formula: E_A = 1 / (1 + 10^((R_B - R_A) / 400))

    Args:
        rating_a: ELO rating of fighter A
        rating_b: ELO rating of fighter B

    Returns:
        Expected score for fighter A (probability between 0 and 1)
    """
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_ratings(
    rating_a: float,
    rating_b: float,
    outcome_a: float,
    k_factor: float = K_FACTOR
) -> Tuple[float, float]:
    """
    Update ELO ratings for both fighters after a fight.

    Args:
        rating_a: Current ELO rating of fighter A
        rating_b: Current ELO rating of fighter B
        outcome_a: Actual outcome for fighter A (1.0 = win, 0.0 = loss, 0.5 = draw)
        k_factor: ELO K-factor controlling update magnitude (default: from config)

    Returns:
        Tuple of (new_rating_a, new_rating_b)
    """
    expected_a = expected_score(rating_a, rating_b)
    expected_b = 1 - expected_a

    new_rating_a = rating_a + k_factor * (outcome_a - expected_a)
    new_rating_b = rating_b + k_factor * ((1 - outcome_a) - expected_b)

    return new_rating_a, new_rating_b


def compute_elo_history(
    fights: pd.DataFrame,
    start_rating: float = ELO_START,
    k_factor: float = K_FACTOR
) -> pd.DataFrame:
    """
    Compute historical ELO ratings for all fighters across all fights.

    This function:
    1. Sorts fights chronologically
    2. Initializes all fighters at start_rating
    3. Iterates through fights in order, recording pre-fight ratings
    4. Updates ratings after each fight
    5. Returns augmented fight data with ELO information

    Args:
        fights: DataFrame with fight records, must include:
            - date: Fight date
            - fighter1, fighter2: Fighter names
            - target: Outcome (1 = fighter1 wins, 0 = fighter2 wins)
        start_rating: Initial rating for new fighters
        k_factor: ELO K-factor for updates

    Returns:
        DataFrame with original fight data plus:
            - elo_f1_pre: Pre-fight rating for fighter 1
            - elo_f2_pre: Pre-fight rating for fighter 2
            - elo_diff_pre: elo_f1_pre - elo_f2_pre

    IMPORTANT: Must process fights in chronological order to maintain causality.
    Pre-fight ratings must be recorded BEFORE updating ratings to avoid label leakage.
    """
    # Initialize ratings tracker
    ratings: Dict[str, float] = {}

    # Ensure chronological order
    fights = fights.sort_values("date").reset_index(drop=True)

    # Storage for pre-fight ratings
    elo_f1_pre = []
    elo_f2_pre = []
    elo_diff_pre = []

    for _, fight in fights.iterrows():
        f1, f2 = fight["fighter1"], fight["fighter2"]
        outcome = fight["target"]  # 1 = f1 wins, 0 = f2 wins

        # Get current ratings (or initialize)
        r1 = ratings.get(f1, start_rating)
        r2 = ratings.get(f2, start_rating)

        # SAVE PRE-FIGHT RATINGS FIRST (before updating!)
        elo_f1_pre.append(r1)
        elo_f2_pre.append(r2)
        elo_diff_pre.append(r1 - r2)

        # NOW update ratings based on outcome
        new_r1, new_r2 = update_ratings(r1, r2, outcome, k_factor)
        ratings[f1] = new_r1
        ratings[f2] = new_r2

    # Add columns to dataframe
    fights = fights.copy()
    fights["elo_f1_pre"] = elo_f1_pre
    fights["elo_f2_pre"] = elo_f2_pre
    fights["elo_diff_pre"] = elo_diff_pre

    return fights


def run_elo_pipeline() -> pd.DataFrame:
    """
    Run the complete ELO rating pipeline.

    Loads cleaned fights, computes ELO ratings, and saves to parquet.

    Returns:
        DataFrame with fights and ELO ratings
    """
    print("Loading cleaned fights...")
    fights = pd.read_parquet(INTERMEDIATE_DATA_DIR / "cleaned_fights.parquet")
    print(f"Loaded {len(fights)} fights")

    print("\nComputing ELO ratings...")
    fights_with_elo = compute_elo_history(fights)

    # Save output
    output_path = INTERMEDIATE_DATA_DIR / "fights_with_elo.parquet"
    fights_with_elo.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Print summary stats
    print(f"\nELO Summary:")
    print(f"  Fights processed: {len(fights_with_elo)}")
    unique_fighters = len(set(fights_with_elo["fighter1"]) | set(fights_with_elo["fighter2"]))
    print(f"  Unique fighters rated: {unique_fighters}")
    print(
        f"  ELO diff range: {fights_with_elo['elo_diff_pre'].min():.0f} to "
        f"{fights_with_elo['elo_diff_pre'].max():.0f}"
    )

    return fights_with_elo


if __name__ == "__main__":
    run_elo_pipeline()
