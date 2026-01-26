"""
Feature engineering for UFC fight prediction.

This module creates features from:
- Fighter historical performance (win/loss records, streaks, experience)
- Physical attributes (height, reach, age)
- Matchup dynamics (differences in attributes and experience)
- ELO ratings (from the rating module)

CRITICAL: All features must use only information available BEFORE each fight.
No feature should depend on the outcome of the current or future fights.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from ufc_predictor.config import (
    INTERMEDIATE_DATA_DIR,
    FEATURES_DATA_DIR,
    RECENT_FIGHTS_WINDOW,
)


def _compute_fighter_history_stats(fights: pd.DataFrame) -> pd.DataFrame:
    """
    Compute fighter history stats WITHOUT label leakage.

    Returns DataFrame with columns:
        - fighter, date, fight_idx (unique identifier)
        - position ('f1' or 'f2')
        - total_fights_pre: fights before this one
        - wins_pre: wins before this one
        - losses_pre: losses before this one
        - win_rate_pre: win_rate using only prior fights
        - last_5_wins_pre: wins in last 5 fights (before this one)
        - days_since_last_pre: days since previous fight
        - current_streak_pre: current win/loss streak (positive=wins)
    """
    # Step 1: Melt fights to fighter-centric view
    # Create two rows per fight: one for fighter1, one for fighter2

    f1_view = fights[['date', 'fighter1', 'target', 'method', 'round', 'time']].copy()
    f1_view.columns = ['date', 'fighter', 'won', 'method', 'round', 'time']  # target=1 means f1 won
    f1_view['fight_idx'] = fights.index
    f1_view['position'] = 'f1'
    f1_view['opponent_elo'] = fights['elo_f2_pre'].values

    f2_view = fights[['date', 'fighter2', 'target', 'method', 'round', 'time']].copy()
    f2_view.columns = ['date', 'fighter', 'won', 'method', 'round', 'time']
    f2_view['won'] = 1 - f2_view['won']  # Invert: target=1 means f2 lost
    f2_view['fight_idx'] = fights.index
    f2_view['position'] = 'f2'
    f2_view['opponent_elo'] = fights['elo_f1_pre'].values

    all_fights = pd.concat([f1_view, f2_view], ignore_index=True)
    all_fights = all_fights.sort_values(['fighter', 'date', 'fight_idx'])

    # Step 2: For each fighter, compute cumulative stats using SHIFT
    # shift(1) ensures we only use data from BEFORE current fight

    grouped = all_fights.groupby('fighter', sort=False)

    # Cumulative counts (shifted)
    all_fights['total_fights_pre'] = grouped.cumcount()  # 0-indexed count before this fight
    all_fights['wins_cumsum'] = grouped['won'].cumsum()
    all_fights['wins_pre'] = grouped['wins_cumsum'].shift(1, fill_value=0)
    all_fights['losses_pre'] = all_fights['total_fights_pre'] - all_fights['wins_pre']

    # NEW: Finish tracking (KO and Submission wins)
    all_fights['is_ko_win'] = (all_fights['won'] == 1) & (
        all_fights['method'].isin(['KO/TKO', "TKO - Doctor's Stoppage"])
    )
    all_fights['is_sub_win'] = (all_fights['won'] == 1) & (all_fights['method'] == 'Submission')

    all_fights['ko_wins_cumsum'] = grouped['is_ko_win'].cumsum()
    all_fights['ko_wins_pre'] = grouped['ko_wins_cumsum'].shift(1, fill_value=0)

    all_fights['sub_wins_cumsum'] = grouped['is_sub_win'].cumsum()
    all_fights['sub_wins_pre'] = grouped['sub_wins_cumsum'].shift(1, fill_value=0)

    # NEW: Finish rate and KO win ratio (avoid div by zero)
    all_fights['finish_rate_pre'] = np.where(
        all_fights['wins_pre'] > 0,
        (all_fights['ko_wins_pre'] + all_fights['sub_wins_pre']) / all_fights['wins_pre'],
        0.0
    )
    all_fights['ko_win_ratio_pre'] = np.where(
        all_fights['wins_pre'] > 0,
        all_fights['ko_wins_pre'] / all_fights['wins_pre'],
        0.0
    )

    # NEW: Average fight time in seconds
    def parse_time_to_seconds(row):
        """Convert round and time to total fight duration in seconds."""
        if pd.isna(row['time']):
            return np.nan
        try:
            parts = row['time'].split(':')
            minutes = int(parts[0])
            seconds = int(parts[1])
            time_in_round = minutes * 60 + seconds
            # Total time = (completed rounds * 300) + time in final round
            return (row['round'] - 1) * 300 + time_in_round
        except:
            return np.nan

    all_fights['fight_duration_sec'] = all_fights.apply(parse_time_to_seconds, axis=1)
    all_fights['duration_cumsum'] = grouped['fight_duration_sec'].cumsum()
    all_fights['duration_count'] = grouped['fight_duration_sec'].count()
    all_fights['duration_cumsum_pre'] = grouped['duration_cumsum'].shift(1, fill_value=0)
    all_fights['duration_count_pre'] = grouped['duration_count'].shift(1, fill_value=0)
    all_fights['avg_fight_time_pre'] = np.where(
        all_fights['duration_count_pre'] > 0,
        all_fights['duration_cumsum_pre'] / all_fights['duration_count_pre'],
        0.0
    )

    # NEW: Average opponent ELO (strength of schedule)
    all_fights['opponent_elo_cumsum'] = grouped['opponent_elo'].cumsum()
    all_fights['opponent_elo_cumsum_pre'] = grouped['opponent_elo_cumsum'].shift(1, fill_value=0)
    all_fights['avg_opponent_elo_pre'] = np.where(
        all_fights['total_fights_pre'] > 0,
        all_fights['opponent_elo_cumsum_pre'] / all_fights['total_fights_pre'],
        1500.0  # ELO_START default for debut fighters
    )

    # Win rate (avoid division by zero)
    all_fights['win_rate_pre'] = np.where(
        all_fights['total_fights_pre'] > 0,
        all_fights['wins_pre'] / all_fights['total_fights_pre'],
        0.5  # Default for debut fighters
    )

    # Last N wins (rolling with shift)
    # Use min_periods=1 so we get values even with < WINDOW fights
    window = RECENT_FIGHTS_WINDOW
    all_fights['last_n_wins_raw'] = grouped['won'].rolling(
        window=window, min_periods=1
    ).sum().reset_index(level=0, drop=True)
    all_fights['last_5_wins_pre'] = grouped['last_n_wins_raw'].shift(1, fill_value=0)

    # Days since last fight
    all_fights['prev_date'] = grouped['date'].shift(1)
    all_fights['days_since_last_pre'] = (all_fights['date'] - all_fights['prev_date']).dt.days
    all_fights['days_since_last_pre'] = all_fights['days_since_last_pre'].fillna(-1)  # -1 for debut

    # Current streak (positive = win streak, negative = loss streak)
    # This is more complex - need to track streak that was current BEFORE this fight
    def compute_streak(group):
        streak = []
        current = 0
        prev_streak = 0
        for won in group['won']:
            streak.append(prev_streak)  # Record streak BEFORE this fight
            if won == 1:
                current = current + 1 if current > 0 else 1
            else:
                current = current - 1 if current < 0 else -1
            prev_streak = current
        return pd.Series(streak, index=group.index)

    all_fights['current_streak_pre'] = grouped.apply(compute_streak).reset_index(level=0, drop=True)

    # Select output columns
    result = all_fights[['fighter', 'fight_idx', 'position', 'total_fights_pre',
                         'wins_pre', 'losses_pre', 'win_rate_pre',
                         'last_5_wins_pre', 'days_since_last_pre', 'current_streak_pre',
                         'finish_rate_pre', 'ko_win_ratio_pre',
                         'avg_fight_time_pre', 'avg_opponent_elo_pre']]

    return result


def _merge_fighter_attributes(
    fights: pd.DataFrame,
    fighters: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge fighter physical attributes to fight records.

    Handles missing values with sensible defaults:
        - height_inches: median imputation
        - reach_inches: median imputation
        - stance: 'Unknown' category
        - dob/age: median imputation
    """
    fights = fights.copy()

    # Prepare fighters table with calculated age at each fight
    # We need age AT TIME OF FIGHT, not current age
    fighters_attrs = fighters[['fighter', 'height_inches', 'reach_inches', 'stance', 'dob']].copy()

    # Impute missing values
    median_height = fighters_attrs['height_inches'].median()
    median_reach = fighters_attrs['reach_inches'].median()

    fighters_attrs['height_inches'] = fighters_attrs['height_inches'].fillna(median_height)
    fighters_attrs['reach_inches'] = fighters_attrs['reach_inches'].fillna(median_reach)
    fighters_attrs['stance'] = fighters_attrs['stance'].fillna('Unknown')

    # Merge for fighter1
    fights = fights.merge(
        fighters_attrs.rename(columns={
            'fighter': 'fighter1',
            'height_inches': 'f1_height',
            'reach_inches': 'f1_reach',
            'stance': 'f1_stance',
            'dob': 'f1_dob'
        }),
        on='fighter1',
        how='left'
    )

    # Merge for fighter2
    fights = fights.merge(
        fighters_attrs.rename(columns={
            'fighter': 'fighter2',
            'height_inches': 'f2_height',
            'reach_inches': 'f2_reach',
            'stance': 'f2_stance',
            'dob': 'f2_dob'
        }),
        on='fighter2',
        how='left'
    )

    # Calculate age at fight time
    fights['f1_age'] = (fights['date'] - fights['f1_dob']).dt.days / 365.25
    fights['f2_age'] = (fights['date'] - fights['f2_dob']).dt.days / 365.25

    # Impute missing ages with median
    median_age = pd.concat([fights['f1_age'], fights['f2_age']]).median()
    fights['f1_age'] = fights['f1_age'].fillna(median_age)
    fights['f2_age'] = fights['f2_age'].fillna(median_age)

    # Drop intermediate dob columns
    fights = fights.drop(columns=['f1_dob', 'f2_dob'])

    return fights


def add_fighter_history_features(fights: pd.DataFrame) -> pd.DataFrame:
    """
    Add fighter historical performance features to fight data.

    For each fighter in each fight, compute (using only prior fights):
    - Total fights (experience)
    - Total wins/losses
    - Win rate
    - Current streak (wins or losses)
    - Recent form (e.g., wins in last N fights)
    - Days since last fight (recency)

    Args:
        fights: DataFrame with fight records sorted chronologically

    Returns:
        DataFrame with added history features for both fighters:
            - f1_total_fights, f2_total_fights
            - f1_wins, f2_wins
            - f1_losses, f2_losses
            - f1_win_rate, f2_win_rate
            - f1_streak, f2_streak
            - f1_last_5_wins, f2_last_5_wins
            - f1_days_since_last, f2_days_since_last

    CRITICAL: Must process fights in chronological order and only use
    information from fights that occurred BEFORE the current fight.
    """
    fights = fights.copy()
    fights = fights.sort_values('date').reset_index(drop=True)

    # Compute fighter-level history
    history = _compute_fighter_history_stats(fights)

    # Split into f1 and f2 views
    f1_history = history[history['position'] == 'f1'].set_index('fight_idx')
    f2_history = history[history['position'] == 'f2'].set_index('fight_idx')

    # Rename columns with prefixes
    f1_cols = {
        'total_fights_pre': 'f1_total_fights',
        'wins_pre': 'f1_wins',
        'losses_pre': 'f1_losses',
        'win_rate_pre': 'f1_win_rate',
        'last_5_wins_pre': 'f1_last_5_wins',
        'days_since_last_pre': 'f1_days_since_last',
        'current_streak_pre': 'f1_streak',
        'finish_rate_pre': 'f1_finish_rate',
        'ko_win_ratio_pre': 'f1_ko_win_ratio',
        'avg_fight_time_pre': 'f1_avg_fight_time',
        'avg_opponent_elo_pre': 'f1_avg_opp_elo',
    }
    f2_cols = {
        'total_fights_pre': 'f2_total_fights',
        'wins_pre': 'f2_wins',
        'losses_pre': 'f2_losses',
        'win_rate_pre': 'f2_win_rate',
        'last_5_wins_pre': 'f2_last_5_wins',
        'days_since_last_pre': 'f2_days_since_last',
        'current_streak_pre': 'f2_streak',
        'finish_rate_pre': 'f2_finish_rate',
        'ko_win_ratio_pre': 'f2_ko_win_ratio',
        'avg_fight_time_pre': 'f2_avg_fight_time',
        'avg_opponent_elo_pre': 'f2_avg_opp_elo',
    }

    # Join to fights (using .loc to preserve index alignment)
    for old_col, new_col in f1_cols.items():
        fights[new_col] = f1_history.loc[fights.index, old_col]

    for old_col, new_col in f2_cols.items():
        fights[new_col] = f2_history.loc[fights.index, old_col]

    return fights


def add_matchup_difference_features(fights: pd.DataFrame) -> pd.DataFrame:
    """
    Add matchup-specific difference features.

    Compute differences between fighters for:
    - Physical attributes (height, reach, age)
    - Experience (total fights, win rate)
    - Recent form

    Args:
        fights: DataFrame with fight records and fighter attributes

    Returns:
        DataFrame with added difference features:
            - height_diff (fighter1 - fighter2)
            - reach_diff
            - age_diff
            - experience_diff
            - win_rate_diff
            - streak_diff
            - last_5_wins_diff
            - days_since_last_diff
            - stance_matchup (categorical)
    """
    fights = fights.copy()

    # Physical differences
    fights['height_diff'] = fights['f1_height'] - fights['f2_height']
    fights['reach_diff'] = fights['f1_reach'] - fights['f2_reach']
    fights['age_diff'] = fights['f1_age'] - fights['f2_age']

    # Experience/performance differences
    fights['experience_diff'] = fights['f1_total_fights'] - fights['f2_total_fights']
    fights['win_rate_diff'] = fights['f1_win_rate'] - fights['f2_win_rate']
    fights['streak_diff'] = fights['f1_streak'] - fights['f2_streak']
    fights['last_5_wins_diff'] = fights['f1_last_5_wins'] - fights['f2_last_5_wins']
    fights['days_since_last_diff'] = fights['f1_days_since_last'] - fights['f2_days_since_last']

    # NEW: Style/quality differences
    fights['finish_rate_diff'] = fights['f1_finish_rate'] - fights['f2_finish_rate']
    fights['ko_win_ratio_diff'] = fights['f1_ko_win_ratio'] - fights['f2_ko_win_ratio']
    fights['avg_fight_time_diff'] = fights['f1_avg_fight_time'] - fights['f2_avg_fight_time']
    fights['avg_opponent_elo_diff'] = fights['f1_avg_opp_elo'] - fights['f2_avg_opp_elo']

    # Stance matchup categorical
    fights['stance_matchup'] = fights['f1_stance'] + '_vs_' + fights['f2_stance']

    return fights


def attach_elo_features(
    fights: pd.DataFrame,
    elo_history: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Attach ELO rating features to fight data.

    Merges pre-fight ELO ratings from the rating module:
    - elo_f1_pre: Pre-fight ELO for fighter 1
    - elo_f2_pre: Pre-fight ELO for fighter 2
    - elo_diff_pre: elo_f1_pre - elo_f2_pre

    Args:
        fights: DataFrame with fight records
        elo_history: DataFrame from rating.elo.compute_elo_history() (optional)

    Returns:
        DataFrame with ELO features attached
    """
    if 'elo_f1_pre' in fights.columns and 'elo_diff_pre' in fights.columns:
        # ELO already attached, just return
        return fights

    # Otherwise merge from elo_history
    if elo_history is not None:
        elo_cols = ['date', 'fighter1', 'fighter2', 'elo_f1_pre', 'elo_f2_pre', 'elo_diff_pre']
        fights = fights.merge(
            elo_history[elo_cols],
            on=['date', 'fighter1', 'fighter2'],
            how='left'
        )

    return fights


def build_feature_matrix(
    fights: pd.DataFrame,
    include_elo: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build complete feature matrix and target labels for ML.

    This is the main feature engineering pipeline that:
    1. Adds fighter history features
    2. Merges physical attributes
    3. Adds matchup difference features
    4. Attaches ELO features (if requested)
    5. Encodes categorical variables (weight class, stance)
    6. Selects final feature columns
    7. Extracts target variable

    Args:
        fights: DataFrame with cleaned fight data
        include_elo: Whether to include ELO features

    Returns:
        Tuple of (X, y) where:
            - X: Feature matrix (DataFrame)
            - y: Target labels (Series, 1 = fighter1 wins, 0 = fighter2 wins)

    IMPORTANT: Ensures no label leakage. All features use only
    pre-fight information.
    """
    fights = fights.copy().sort_values('date').reset_index(drop=True)

    # Load fighters data
    fighters = pd.read_parquet(INTERMEDIATE_DATA_DIR / "cleaned_fighters.parquet")

    # Add history features
    print("Adding fighter history features...")
    fights = add_fighter_history_features(fights)

    # Merge physical attributes
    print("Merging physical attributes...")
    fights = _merge_fighter_attributes(fights, fighters)

    # Add difference features
    print("Adding matchup difference features...")
    fights = add_matchup_difference_features(fights)

    # Filter: remove fights where BOTH fighters have no history
    # (we can predict better if at least one fighter has prior fights)
    both_debut = (fights['f1_total_fights'] == 0) & (fights['f2_total_fights'] == 0)
    print(f"Dropping {both_debut.sum()} fights where both fighters are debuting")
    fights = fights[~both_debut].reset_index(drop=True)

    # Define feature columns
    numerical_features = [
        # Physical differences
        'height_diff', 'reach_diff', 'age_diff',
        # Experience differences
        'experience_diff', 'win_rate_diff', 'streak_diff',
        'last_5_wins_diff', 'days_since_last_diff',
        # Style/quality differences
        'finish_rate_diff', 'ko_win_ratio_diff', 'avg_fight_time_diff', 'avg_opponent_elo_diff',
        # Individual stats (useful for interaction effects)
        'f1_total_fights', 'f2_total_fights',
        'f1_win_rate', 'f2_win_rate',
        'f1_finish_rate', 'f2_finish_rate',
        'f1_ko_win_ratio', 'f2_ko_win_ratio',
        'f1_avg_fight_time', 'f2_avg_fight_time',
        'f1_avg_opp_elo', 'f2_avg_opp_elo',
    ]

    if include_elo:
        numerical_features.extend(['elo_diff_pre', 'elo_f1_pre', 'elo_f2_pre'])

    categorical_features = ['weight_class', 'stance_matchup']

    # One-hot encode categoricals
    X = fights[numerical_features].copy()

    for cat_col in categorical_features:
        dummies = pd.get_dummies(fights[cat_col], prefix=cat_col, drop_first=True)
        X = pd.concat([X, dummies], axis=1)

    # Handle any remaining NaNs in numericals
    X = X.fillna(0)

    # Extract target and metadata
    y = fights['target']
    url = fights['url']
    date = fights['date']

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    return X, y, url, date


def run_feature_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run complete feature engineering pipeline.

    1. Load fights_with_elo.parquet
    2. Build feature matrix
    3. Time-based train/test split (80/20 by date)
    4. Save to features_train.parquet and features_test.parquet

    Returns:
        (train_df, test_df) with features and target
    """
    print("Loading fights with ELO...")
    fights = pd.read_parquet(INTERMEDIATE_DATA_DIR / "fights_with_elo.parquet")
    print(f"Loaded {len(fights)} fights")

    print("\nBuilding feature matrix...")
    X, y, url, date = build_feature_matrix(fights, include_elo=True)

    # Combine X and y for easy split
    feature_df = X.copy()
    feature_df['target'] = y.values

    # CRITICAL FIX: Use URL and date from build_feature_matrix (already filtered/aligned)
    feature_df['url'] = url.values
    feature_df['date'] = date.values

    # Time-based split: sort by date, take last 20% as test
    feature_df = feature_df.sort_values('date').reset_index(drop=True)
    split_idx = int(len(feature_df) * 0.8)
    split_date = feature_df.iloc[split_idx]['date']

    # Keep URL in output for backtest alignment, drop date (can be reconstructed)
    train_df = feature_df.iloc[:split_idx].drop(columns=['date'])
    test_df = feature_df.iloc[split_idx:].drop(columns=['date'])

    print(f"\nTime-based split at {split_date}:")
    print(f"  Train: {len(train_df)} fights")
    print(f"  Test: {len(test_df)} fights")

    # Save outputs
    FEATURES_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = FEATURES_DATA_DIR / "features_train.parquet"
    test_path = FEATURES_DATA_DIR / "features_test.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"\nSaved to:")
    print(f"  {train_path}")
    print(f"  {test_path}")

    return train_df, test_df


if __name__ == "__main__":
    run_feature_pipeline()
