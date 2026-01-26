#!/usr/bin/env python3
"""
Sniper Zone Heatmap: Visualize ROI by Weight Class and Confidence.

Shows exactly where the model wins and loses money, proving why we filter
out Heavyweights and low-confidence bets in the Sniper Strategy.

Usage:
    python scripts/plot_sniper_heatmap.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ufc_predictor.config import (
    FEATURES_DATA_DIR,
    INTERMEDIATE_DATA_DIR,
    MODELS_DIR,
    PLOTS_DIR
)


def load_data():
    """Load test features, fights data, and trained model."""
    print("Loading data...")

    # Load test features
    test_df = pd.read_parquet(FEATURES_DATA_DIR / "features_test.parquet")
    test_urls = test_df['url']
    X_test = test_df.drop(columns=["target", "url"])
    y_test = test_df["target"]

    # Load fights data with odds
    fights_df = pd.read_parquet(INTERMEDIATE_DATA_DIR / "fights_with_elo.parquet")

    # Join on URL to ensure proper alignment
    test_fights = fights_df[fights_df['url'].isin(test_urls)].copy()
    test_fights = test_fights.set_index('url').loc[test_urls].reset_index()

    # Load calibrated model
    model = joblib.load(MODELS_DIR / "calibrated_xgboost.joblib")

    print(f"  Test features: {X_test.shape}")
    print(f"  Test fights: {len(test_fights)}")
    print(f"  Model loaded: calibrated_xgboost.joblib")

    return X_test, y_test, test_fights, model


def get_confidence_bin(prob):
    """Assign confidence bin based on model probability."""
    if prob >= 0.70:
        return '70%+'
    elif prob >= 0.65:
        return '65-70%'
    elif prob >= 0.60:
        return '60-65%'
    elif prob >= 0.55:
        return '55-60%'
    else:
        return '50-55%'


def simulate_bets(X_test, y_test, test_fights, model, bet_size=100, edge_threshold=0.05):
    """Simulate betting with predictions and real odds."""
    print("\nSimulating bets with real odds...")

    results = []

    for idx in range(len(X_test)):
        # Get model prediction
        X = X_test.iloc[idx:idx+1]
        probs = model.predict_proba(X)[0]
        prob_f1, prob_f2 = probs[1], probs[0]

        # Get fight info
        fight = test_fights.iloc[idx]

        # Skip if no real odds
        if pd.isna(fight.get('odds_f1')) or pd.isna(fight.get('odds_f2')):
            continue

        # Calculate EV for both fighters
        odds_f1 = fight['odds_f1']
        odds_f2 = fight['odds_f2']

        ev_f1 = (prob_f1 * odds_f1) - 1
        ev_f2 = (prob_f2 * odds_f2) - 1

        # Determine if we should bet
        if ev_f1 > edge_threshold and ev_f1 >= ev_f2:
            bet_on = 'f1'
            model_prob = prob_f1
            odds = odds_f1
        elif ev_f2 > edge_threshold:
            bet_on = 'f2'
            model_prob = prob_f2
            odds = odds_f2
        else:
            continue

        # Resolve bet
        actual_winner = y_test.iloc[idx]
        won_bet = (bet_on == 'f1' and actual_winner == 1) or (bet_on == 'f2' and actual_winner == 0)

        if won_bet:
            profit = bet_size * (odds - 1)
        else:
            profit = -bet_size

        # Record result
        results.append({
            'weight_class': fight['weight_class'],
            'model_prob': model_prob,
            'won': won_bet,
            'profit': profit,
            'bet_size': bet_size
        })

    print(f"  Total bets with real odds: {len(results)}")

    return pd.DataFrame(results)


def create_heatmap_data(bet_results):
    """Create pivot table of ROI by weight class and confidence."""
    print("\nCalculating ROI by weight class and confidence...")

    # Add confidence bins
    bet_results['confidence_bin'] = bet_results['model_prob'].apply(get_confidence_bin)

    # Calculate ROI per group
    def calculate_roi(group):
        total_profit = group['profit'].sum()
        total_wagered = len(group) * group['bet_size'].iloc[0]
        return (total_profit / total_wagered) * 100 if total_wagered > 0 else 0

    # Group by weight class and confidence bin
    heatmap_data = bet_results.groupby(['weight_class', 'confidence_bin']).apply(
        calculate_roi
    ).unstack(fill_value=np.nan)

    # Reorder columns
    column_order = ['50-55%', '55-60%', '60-65%', '65-70%', '70%+']
    heatmap_data = heatmap_data[[col for col in column_order if col in heatmap_data.columns]]

    # Filter to weight classes with at least some data
    heatmap_data = heatmap_data.dropna(how='all')

    # Sort by average ROI (descending)
    heatmap_data['avg_roi'] = heatmap_data.mean(axis=1, skipna=True)
    heatmap_data = heatmap_data.sort_values('avg_roi', ascending=False).drop('avg_roi', axis=1)

    # Limit to top weight classes for readability
    if len(heatmap_data) > 15:
        heatmap_data = heatmap_data.head(15)

    print(f"  Heatmap dimensions: {heatmap_data.shape}")

    return heatmap_data


def plot_heatmap(heatmap_data):
    """Create and save the Sniper Zone heatmap."""
    print("\nCreating heatmap visualization...")

    # Create figure
    plt.figure(figsize=(12, 10))

    # Create heatmap
    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        center=0,
        vmin=-20,
        vmax=20,
        cbar_kws={'label': 'ROI (%)'},
        linewidths=0.5,
        linecolor='gray'
    )

    # Set title and labels
    plt.title(
        'Sniper Zone Heatmap: ROI by Weight Class and Model Confidence\n' +
        '(Green = Profitable, Red = Losses)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    plt.xlabel('Model Confidence', fontsize=12, fontweight='bold')
    plt.ylabel('Weight Class', fontsize=12, fontweight='bold')

    # Add Sniper Zone annotation
    # Find cells with confidence >= 65%
    sniper_cols = [col for col in heatmap_data.columns if col in ['65-70%', '70%+']]
    excluded_classes = ['Heavyweight Bout', 'Flyweight Bout', 'Women\'s Strawweight Bout']

    # Add subtle border to Sniper Zone
    for i, weight_class in enumerate(heatmap_data.index):
        for j, conf_bin in enumerate(heatmap_data.columns):
            if conf_bin in sniper_cols and weight_class not in excluded_classes:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='blue', lw=3))

    # Add text annotation for Sniper Zone
    plt.text(
        0.98, 0.02,
        'Blue border = Sniper Zone\n(>65% confidence, safe weight classes)',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    )

    plt.tight_layout()

    # Save plot
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOTS_DIR / "sniper_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Heatmap saved to: {output_path}")

    return output_path


def main():
    """Main execution function."""
    print("=" * 80)
    print(" SNIPER ZONE HEATMAP GENERATION")
    print("=" * 80)

    # Load data
    X_test, y_test, test_fights, model = load_data()

    # Simulate bets
    bet_results = simulate_bets(X_test, y_test, test_fights, model)

    if len(bet_results) == 0:
        print("\nError: No bets with real odds found. Cannot create heatmap.")
        return

    # Create heatmap data
    heatmap_data = create_heatmap_data(bet_results)

    # Plot and save
    output_path = plot_heatmap(heatmap_data)

    print("\n" + "=" * 80)
    print(" HEATMAP GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput: {output_path}")
    print("\nThe heatmap shows ROI by weight class and model confidence.")
    print("Green cells = Profitable, Red cells = Losses")
    print("Blue borders = Sniper Zone (>65% confidence, safe classes)")
    print("=" * 80)


if __name__ == "__main__":
    main()
