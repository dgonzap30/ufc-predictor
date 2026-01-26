"""
Fight prediction and betting analysis module.

Provides functions to predict hypothetical UFC matchups using trained models,
with optional Expected Value (EV) calculations for betting analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime

from ufc_predictor.config import (
    MODELS_DIR,
    INTERMEDIATE_DATA_DIR,
    ELO_START,
    K_FACTOR,
)
from ufc_predictor.models.train import load_model
from ufc_predictor.rating.elo import update_ratings

# Exact feature column names from training data (98 columns)
FEATURE_COLUMNS = [
    "height_diff",
    "reach_diff",
    "age_diff",
    "experience_diff",
    "win_rate_diff",
    "streak_diff",
    "last_5_wins_diff",
    "days_since_last_diff",
    "finish_rate_diff",
    "ko_win_ratio_diff",
    "avg_fight_time_diff",
    "avg_opponent_elo_diff",
    "f1_total_fights",
    "f2_total_fights",
    "f1_win_rate",
    "f2_win_rate",
    "f1_finish_rate",
    "f2_finish_rate",
    "f1_ko_win_ratio",
    "f2_ko_win_ratio",
    "f1_avg_fight_time",
    "f2_avg_fight_time",
    "f1_avg_opp_elo",
    "f2_avg_opp_elo",
    "elo_diff_pre",
    "elo_f1_pre",
    "elo_f2_pre",
    "weight_class_Catch Weight Bout",
    "weight_class_Featherweight Bout",
    "weight_class_Flyweight Bout",
    "weight_class_Heavyweight Bout",
    "weight_class_Light Heavyweight Bout",
    "weight_class_Lightweight Bout",
    "weight_class_Middleweight Bout",
    "weight_class_Open Weight Bout",
    "weight_class_UFC 10 Tournament Title Bout",
    "weight_class_UFC 13 Heavyweight Tournament Title Bout",
    "weight_class_UFC 13 Lightweight Tournament Title Bout",
    "weight_class_UFC 15 Heavyweight Tournament Title Bout",
    "weight_class_UFC 2 Tournament Title Bout",
    "weight_class_UFC 3 Tournament Title Bout",
    "weight_class_UFC 4 Tournament Title Bout",
    "weight_class_UFC 5 Tournament Title Bout",
    "weight_class_UFC 6 Tournament Title Bout",
    "weight_class_UFC 7 Tournament Title Bout",
    "weight_class_UFC 8 Tournament Title Bout",
    "weight_class_UFC Bantamweight Title Bout",
    "weight_class_UFC Featherweight Title Bout",
    "weight_class_UFC Flyweight Title Bout",
    "weight_class_UFC Heavyweight Title Bout",
    "weight_class_UFC Interim Bantamweight Title Bout",
    "weight_class_UFC Interim Featherweight Title Bout",
    "weight_class_UFC Interim Flyweight Title Bout",
    "weight_class_UFC Interim Heavyweight Title Bout",
    "weight_class_UFC Interim Light Heavyweight Title Bout",
    "weight_class_UFC Interim Lightweight Title Bout",
    "weight_class_UFC Interim Middleweight Title Bout",
    "weight_class_UFC Interim Welterweight Title Bout",
    "weight_class_UFC Light Heavyweight Title Bout",
    "weight_class_UFC Lightweight Title Bout",
    "weight_class_UFC Middleweight Title Bout",
    "weight_class_UFC Superfight Championship Bout",
    "weight_class_UFC Welterweight Title Bout",
    "weight_class_UFC Women's Bantamweight Title Bout",
    "weight_class_UFC Women's Featherweight Title Bout",
    "weight_class_UFC Women's Flyweight Title Bout",
    "weight_class_UFC Women's Strawweight Title Bout",
    "weight_class_Ultimate Fighter 25 Welterweight Tournament Title Bout",
    "weight_class_Ultimate Fighter 33 Flyweight Tournament Title Bout",
    "weight_class_Ultimate Fighter 4 Middleweight Tournament Title Bout",
    "weight_class_Ultimate Fighter 4 Welterweight Tournament Title Bout",
    "weight_class_Ultimate Ultimate '95 Tournament Title Bout",
    "weight_class_Ultimate Ultimate '96 Tournament Title Bout",
    "weight_class_Welterweight Bout",
    "weight_class_Women's Bantamweight Bout",
    "weight_class_Women's Featherweight Bout",
    "weight_class_Women's Flyweight Bout",
    "weight_class_Women's Strawweight Bout",
    "stance_matchup_Open Stance_vs_Southpaw",
    "stance_matchup_Orthodox_vs_Open Stance",
    "stance_matchup_Orthodox_vs_Orthodox",
    "stance_matchup_Orthodox_vs_Sideways",
    "stance_matchup_Orthodox_vs_Southpaw",
    "stance_matchup_Orthodox_vs_Switch",
    "stance_matchup_Orthodox_vs_Unknown",
    "stance_matchup_Sideways_vs_Orthodox",
    "stance_matchup_Southpaw_vs_Open Stance",
    "stance_matchup_Southpaw_vs_Orthodox",
    "stance_matchup_Southpaw_vs_Sideways",
    "stance_matchup_Southpaw_vs_Southpaw",
    "stance_matchup_Southpaw_vs_Switch",
    "stance_matchup_Southpaw_vs_Unknown",
    "stance_matchup_Switch_vs_Orthodox",
    "stance_matchup_Switch_vs_Southpaw",
    "stance_matchup_Switch_vs_Switch",
    "stance_matchup_Switch_vs_Unknown",
    "stance_matchup_Unknown_vs_Orthodox",
    "stance_matchup_Unknown_vs_Unknown",
]

# Imputation defaults (medians from training data)
MEDIAN_HEIGHT = 70.0  # inches
MEDIAN_REACH = 72.0  # inches
MEDIAN_AGE = 30.0  # years

# Sniper Strategy: Excluded weight classes (negative ROI)
EXCLUDED_WEIGHT_CLASSES = {
    'Heavyweight Bout',
    'Flyweight Bout',
    'Women\'s Strawweight Bout'
}


def load_inference_data() -> Tuple[any, pd.DataFrame, pd.DataFrame]:
    """
    Load trained model and reference data for inference.

    Returns:
        Tuple of (model, fighters_df, fights_df)
    """
    model = load_model(MODELS_DIR / "calibrated_xgboost.joblib")
    fighters_df = pd.read_parquet(INTERMEDIATE_DATA_DIR / "cleaned_fighters.parquet")
    fights_df = pd.read_parquet(INTERMEDIATE_DATA_DIR / "fights_with_elo.parquet")

    return model, fighters_df, fights_df


def get_fighter_current_state(
    fighter_name: str,
    fights_df: pd.DataFrame,
    fighters_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> Dict:
    """
    Get fighter's current state for prediction.

    Args:
        fighter_name: Fighter name (case-sensitive)
        fights_df: Historical fights DataFrame with ELO ratings
        fighters_df: Fighter attributes DataFrame
        as_of_date: Date for which to compute state

    Returns:
        Dictionary with fighter state including:
        - latest_elo, total_fights, win_rate, last_5_wins, streak
        - days_since_last, height, reach, age, stance
    """
    # Find fighter in fighters table
    fighter_mask = fighters_df['fighter'] == fighter_name
    if not fighter_mask.any():
        # Unknown fighter - return defaults
        print(f"Warning: Fighter '{fighter_name}' not found. Using debut defaults.")
        return {
            'latest_elo': ELO_START,
            'total_fights': 0,
            'wins': 0,
            'win_rate': 0.5,
            'last_5_wins': 0,
            'streak': 0,
            'days_since_last': -1,
            'height': MEDIAN_HEIGHT,
            'reach': MEDIAN_REACH,
            'age': MEDIAN_AGE,
            'stance': 'Unknown',
        }

    fighter_attrs = fighters_df[fighter_mask].iloc[0]

    # Get physical attributes (with imputation)
    height = fighter_attrs['height_inches'] if pd.notna(fighter_attrs['height_inches']) else MEDIAN_HEIGHT
    reach = fighter_attrs['reach_inches'] if pd.notna(fighter_attrs['reach_inches']) else MEDIAN_REACH
    stance = fighter_attrs['stance'] if pd.notna(fighter_attrs['stance']) else 'Unknown'

    # Calculate age at fight date
    if pd.notna(fighter_attrs['dob']):
        age = (as_of_date - fighter_attrs['dob']).days / 365.25
    else:
        age = MEDIAN_AGE

    # Find all fights for this fighter before as_of_date
    mask = ((fights_df['fighter1'] == fighter_name) | (fights_df['fighter2'] == fighter_name))
    mask &= (fights_df['date'] < as_of_date)
    fighter_fights = fights_df[mask].sort_values('date')

    if len(fighter_fights) == 0:
        # Debut fighter (exists in database but no fights yet)
        return {
            'latest_elo': ELO_START,
            'total_fights': 0,
            'wins': 0,
            'win_rate': 0.5,
            'last_5_wins': 0,
            'streak': 0,
            'days_since_last': -1,
            'height': height,
            'reach': reach,
            'age': age,
            'stance': stance,
        }

    # Calculate fight outcomes
    fighter_fights = fighter_fights.copy()
    fighter_fights['won'] = fighter_fights.apply(
        lambda r: r['target'] if r['fighter1'] == fighter_name else 1 - r['target'],
        axis=1
    )

    # Calculate stats
    total_fights = len(fighter_fights)
    wins = fighter_fights['won'].sum()
    win_rate = wins / total_fights if total_fights > 0 else 0.5
    last_5_wins = fighter_fights.tail(5)['won'].sum()

    # Calculate streak (positive for win streak, negative for loss streak)
    streak = 0
    for won in reversed(fighter_fights['won'].tolist()):
        if streak == 0:
            streak = 1 if won else -1
        elif (streak > 0 and won) or (streak < 0 and not won):
            streak += 1 if streak > 0 else -1
        else:
            break

    # Days since last fight
    days_since_last = (as_of_date - fighter_fights['date'].max()).days

    # Compute latest ELO (post-fight ELO from most recent fight)
    last_fight = fighter_fights.iloc[-1]
    if last_fight['fighter1'] == fighter_name:
        pre_elo = last_fight['elo_f1_pre']
        opp_elo = last_fight['elo_f2_pre']
        won_last = last_fight['target']
    else:
        pre_elo = last_fight['elo_f2_pre']
        opp_elo = last_fight['elo_f1_pre']
        won_last = 1 - last_fight['target']

    latest_elo, _ = update_ratings(pre_elo, opp_elo, won_last, K_FACTOR)

    return {
        'latest_elo': latest_elo,
        'total_fights': total_fights,
        'wins': wins,
        'win_rate': win_rate,
        'last_5_wins': last_5_wins,
        'streak': streak,
        'days_since_last': days_since_last,
        'height': height,
        'reach': reach,
        'age': age,
        'stance': stance,
    }


def build_feature_vector(
    f1_state: Dict,
    f2_state: Dict,
    weight_class: str,
) -> pd.DataFrame:
    """
    Build 86-feature vector matching training format.

    Args:
        f1_state: Fighter 1 state dictionary
        f2_state: Fighter 2 state dictionary
        weight_class: Weight class for the fight

    Returns:
        DataFrame with single row and 86 columns in exact training order
    """
    # Calculate all numerical features
    features = {
        'height_diff': f1_state['height'] - f2_state['height'],
        'reach_diff': f1_state['reach'] - f2_state['reach'],
        'age_diff': f1_state['age'] - f2_state['age'],
        'experience_diff': f1_state['total_fights'] - f2_state['total_fights'],
        'win_rate_diff': f1_state['win_rate'] - f2_state['win_rate'],
        'streak_diff': f1_state['streak'] - f2_state['streak'],
        'last_5_wins_diff': f1_state['last_5_wins'] - f2_state['last_5_wins'],
        'days_since_last_diff': f1_state['days_since_last'] - f2_state['days_since_last'],
        # Additional features (set to 0 for simplified inference)
        'finish_rate_diff': 0.0,
        'ko_win_ratio_diff': 0.0,
        'avg_fight_time_diff': 0.0,
        'avg_opponent_elo_diff': 0.0,
        'f1_total_fights': f1_state['total_fights'],
        'f2_total_fights': f2_state['total_fights'],
        'f1_win_rate': f1_state['win_rate'],
        'f2_win_rate': f2_state['win_rate'],
        # Additional per-fighter features (set to 0 for simplified inference)
        'f1_finish_rate': 0.0,
        'f2_finish_rate': 0.0,
        'f1_ko_win_ratio': 0.0,
        'f2_ko_win_ratio': 0.0,
        'f1_avg_fight_time': 0.0,
        'f2_avg_fight_time': 0.0,
        'f1_avg_opp_elo': ELO_START,
        'f2_avg_opp_elo': ELO_START,
        'elo_f1_pre': f1_state['latest_elo'],
        'elo_f2_pre': f2_state['latest_elo'],
        'elo_diff_pre': f1_state['latest_elo'] - f2_state['latest_elo'],
    }

    # One-hot encode weight_class (initialize all to False)
    for col in FEATURE_COLUMNS:
        if col.startswith('weight_class_') and col not in features:
            features[col] = False

    # Set the correct weight class to True (if it exists in training columns)
    weight_col = f'weight_class_{weight_class}'
    if weight_col in features:
        features[weight_col] = True

    # One-hot encode stance_matchup (initialize all to False)
    for col in FEATURE_COLUMNS:
        if col.startswith('stance_matchup_') and col not in features:
            features[col] = False

    # Set the correct stance matchup to True
    stance_matchup = f"{f1_state['stance']}_vs_{f2_state['stance']}"
    stance_col = f'stance_matchup_{stance_matchup}'
    if stance_col in features:
        features[stance_col] = True

    # Return as DataFrame with exact column order
    return pd.DataFrame([features])[FEATURE_COLUMNS]


def american_odds_to_decimal(odds: float) -> float:
    """
    Convert American odds to decimal odds.

    Args:
        odds: American odds (e.g., -150 for favorite, +200 for underdog)

    Returns:
        Decimal odds
    """
    if odds > 0:
        return (odds / 100) + 1
    else:
        return (100 / abs(odds)) + 1


def calculate_ev(prob: float, decimal_odds: float) -> float:
    """
    Calculate Expected Value (EV) of a bet.

    Args:
        prob: Probability of winning (0-1)
        decimal_odds: Decimal odds for the bet

    Returns:
        Expected value as a decimal (e.g., 0.082 for +8.2% EV)
    """
    return (prob * decimal_odds) - 1


def check_sniper_criteria(
    model_prob: float,
    decimal_odds: Optional[float],
    weight_class: str
) -> Dict:
    """
    Check if a bet meets the Sniper Strategy criteria.

    Sniper Strategy Rules (verified +2.2% ROI on real odds):
    1. Model probability > 65% (high confidence)
    2. Fighter is favorite (decimal odds < 2.0, i.e., implied prob > 50%)
    3. Weight class not excluded (Heavyweight, Flyweight, Women's Strawweight)

    Args:
        model_prob: Model's predicted probability for this fighter (0-1)
        decimal_odds: Decimal odds for this fighter (optional)
        weight_class: Weight class of the fight

    Returns:
        Dictionary with:
        - is_sniper_bet: bool (True if all criteria met)
        - reasons: list of strings explaining why bet passed/failed
    """
    reasons = []

    # Criterion 1: High Confidence (>65%)
    if model_prob <= 0.65:
        reasons.append(f"Confidence too low: {model_prob*100:.1f}% (need >65%)")

    # Criterion 2: Betting on Favorite (decimal odds < 2.0)
    if decimal_odds is not None:
        if decimal_odds >= 2.0:
            implied_prob = (1 / decimal_odds) * 100
            reasons.append(f"Betting on underdog (implied prob {implied_prob:.1f}% < 50%)")
    else:
        # If odds not provided, we can't verify this criterion
        reasons.append("Odds not provided (cannot verify favorite status)")

    # Criterion 3: Weight Class Not Excluded
    if weight_class in EXCLUDED_WEIGHT_CLASSES:
        reasons.append(f"{weight_class} is excluded (negative ROI)")

    return {
        "is_sniper_bet": len(reasons) == 0,
        "reasons": reasons
    }


def predict_matchup(
    fighter1_name: str,
    fighter2_name: str,
    weight_class: str = "Heavyweight Bout",
    fight_date: Optional[datetime] = None,
    odds_f1: Optional[float] = None,
    odds_f2: Optional[float] = None,
) -> Dict:
    """
    Predict fight outcome with optional betting EV analysis.

    Args:
        fighter1_name: Fighter 1 name
        fighter2_name: Fighter 2 name
        weight_class: Weight class for the fight
        fight_date: Date for prediction (default: today)
        odds_f1: American odds for Fighter 1 (optional)
        odds_f2: American odds for Fighter 2 (optional)

    Returns:
        Dictionary with:
        - winner: predicted winner name
        - confidence: probability as percentage (0-100)
        - f1_prob: Fighter 1 win probability (0-1)
        - f2_prob: Fighter 2 win probability (0-1)
        - f1_state: Fighter 1 state dict
        - f2_state: Fighter 2 state dict
        - ev_f1: EV for Fighter 1 (if odds provided)
        - ev_f2: EV for Fighter 2 (if odds provided)
        - recommendation: betting recommendation (if odds provided)
    """
    # Default to today if no date provided
    if fight_date is None:
        fight_date = pd.Timestamp.now()
    elif not isinstance(fight_date, pd.Timestamp):
        fight_date = pd.Timestamp(fight_date)

    # Load inference data
    model, fighters_df, fights_df = load_inference_data()

    # Get current state for both fighters
    f1_state = get_fighter_current_state(fighter1_name, fights_df, fighters_df, fight_date)
    f2_state = get_fighter_current_state(fighter2_name, fights_df, fighters_df, fight_date)

    # Build feature vector
    X = build_feature_vector(f1_state, f2_state, weight_class)

    # Make prediction
    proba = model.predict_proba(X)[0]
    f1_prob = proba[1]  # Probability that fighter1 wins
    f2_prob = proba[0]  # Probability that fighter2 wins

    # Determine winner
    if f1_prob > f2_prob:
        winner = fighter1_name
        confidence = f1_prob * 100
    else:
        winner = fighter2_name
        confidence = f2_prob * 100

    result = {
        'winner': winner,
        'confidence': confidence,
        'f1_prob': f1_prob,
        'f2_prob': f2_prob,
        'f1_state': f1_state,
        'f2_state': f2_state,
    }

    # Calculate EV if odds provided
    if odds_f1 is not None and odds_f2 is not None:
        decimal_f1 = american_odds_to_decimal(odds_f1)
        decimal_f2 = american_odds_to_decimal(odds_f2)

        ev_f1 = calculate_ev(f1_prob, decimal_f1)
        ev_f2 = calculate_ev(f2_prob, decimal_f2)

        result['ev_f1'] = ev_f1
        result['ev_f2'] = ev_f2
        result['odds_f1'] = odds_f1
        result['odds_f2'] = odds_f2

        # Betting recommendation
        if ev_f1 > 0 and ev_f1 > ev_f2:
            result['recommendation'] = f"Bet on {fighter1_name} (EV: +{ev_f1*100:.1f}%)"
        elif ev_f2 > 0 and ev_f2 > ev_f1:
            result['recommendation'] = f"Bet on {fighter2_name} (EV: +{ev_f2*100:.1f}%)"
        else:
            result['recommendation'] = "No positive EV bets"

        # Add Sniper Strategy check if there's a betting opportunity
        if result['recommendation'].startswith('Bet'):
            # Determine which fighter is recommended
            if ev_f1 > ev_f2:
                rec_prob = f1_prob
                rec_odds = decimal_f1
            else:
                rec_prob = f2_prob
                rec_odds = decimal_f2

            # Check Sniper Strategy criteria
            sniper_check = check_sniper_criteria(rec_prob, rec_odds, weight_class)
            result['sniper'] = sniper_check

    return result
