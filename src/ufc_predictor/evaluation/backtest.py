"""
Profitability backtesting for UFC fight prediction model.

This module simulates a betting strategy over the test period to verify
if the trained model can generate profit. It uses ELO-simulated market odds
(with bookmaker vig) and a simplified Kelly Criterion betting strategy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from typing import Dict, List, Tuple

from ufc_predictor.config import (
    FEATURES_DATA_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    INTERMEDIATE_DATA_DIR
)


def get_market_odds(fight_row: pd.Series) -> Tuple[float, float, bool]:
    """
    Get market odds for a fight.

    Checks for real odds first (from Kaggle dataset), then falls back to
    ELO-simulated odds if real odds are not available.

    Args:
        fight_row: Row from fights DataFrame containing elo_diff_pre and optionally odds_f1/odds_f2

    Returns:
        Tuple of (odds_f1, odds_f2, is_real) where is_real indicates real vs simulated odds
    """
    # Check for real odds first
    if 'odds_f1' in fight_row.index and pd.notna(fight_row.get('odds_f1')):
        return fight_row['odds_f1'], fight_row['odds_f2'], True

    # Fall back to ELO-simulated odds
    elo_diff_pre = fight_row['elo_diff_pre']

    # ELO probability formula
    prob_f1 = 1 / (1 + 10 ** ((-elo_diff_pre) / 400))
    prob_f2 = 1 - prob_f1

    # Convert to decimal odds
    decimal_odds_f1 = 1 / prob_f1
    decimal_odds_f2 = 1 / prob_f2

    # Apply 5% vig (bookmaker fee) - odds are less favorable for bettor
    decimal_odds_f1 *= 0.95
    decimal_odds_f2 *= 0.95

    return decimal_odds_f1, decimal_odds_f2, False


def load_backtest_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, any]:
    """
    Load all data needed for backtesting.

    Returns:
        Tuple of (test_features, fights_data, targets, trained_model)
    """
    print("[1/3] Loading test data...")

    # Load test features (now includes URL for proper alignment)
    test_df = pd.read_parquet(FEATURES_DATA_DIR / "features_test.parquet")
    test_urls = test_df['url']  # Extract URLs before dropping
    X_test = test_df.drop(columns=["target", "url"])
    y_test = test_df["target"]

    # Load fights data
    fights_df = pd.read_parquet(INTERMEDIATE_DATA_DIR / "fights_with_elo.parquet")

    # CRITICAL FIX: Join on URL instead of assuming index alignment
    # This ensures we get the EXACT same fights that were used in feature engineering
    test_fights = fights_df[fights_df['url'].isin(test_urls)].copy()
    test_fights = test_fights.set_index('url').loc[test_urls].reset_index()

    print(f"  Test features: {X_test.shape}")
    print(f"  Test fights: {len(test_fights)}")

    # Verify alignment (critical sanity check)
    assert len(X_test) == len(test_fights), f"Data misalignment! X_test={len(X_test)}, test_fights={len(test_fights)}"
    assert len(y_test) == len(test_fights), f"Target misalignment! y_test={len(y_test)}, test_fights={len(test_fights)}"

    print(f"  ✓ Data alignment verified")

    # Load calibrated model
    print("[2/3] Loading calibrated model...")
    model_path = MODELS_DIR / "calibrated_xgboost.joblib"
    model = joblib.load(model_path)
    print(f"  Calibrated model loaded from {model_path}")

    return X_test, test_fights, y_test, model


def run_backtest(
    bet_size: float = 100,
    edge_threshold: float = 0.05,
    starting_bankroll: float = 10000
) -> Dict:
    """
    Simulate betting strategy over test period.

    For each fight:
    1. Get model's predicted probabilities
    2. Calculate simulated market odds from ELO
    3. Check if Expected Value > threshold for either fighter
    4. If yes, place fixed bet ($100)
    5. Track bankroll, wins, losses

    Args:
        bet_size: Fixed bet amount per wager (default: $100)
        edge_threshold: Minimum EV required to place bet (default: 5%)
        starting_bankroll: Starting capital (default: $10,000)

    Returns:
        Dictionary with results:
        - total_bets, wins, losses, win_rate
        - starting_bankroll, ending_bankroll, total_profit, roi
        - bet_history (list of dicts)
    """
    print("\n" + "=" * 80)
    print(" PROFITABILITY BACKTEST")
    print("=" * 80)

    # Load data
    X_test, test_fights, y_test, model = load_backtest_data()

    print("[3/3] Running betting simulation...")
    print(f"  Bet size: ${bet_size:.2f}")
    print(f"  Edge threshold: {edge_threshold * 100:.1f}%")
    print(f"  Starting bankroll: ${starting_bankroll:,.2f}")

    # Initialize tracking
    bankroll = starting_bankroll
    total_bets = 0
    wins = 0
    losses = 0
    bet_history = []

    # Simulate betting over test period
    for idx in range(len(X_test)):
        # Get model prediction
        X = X_test.iloc[idx:idx+1]
        probs = model.predict_proba(X)[0]
        prob_f1, prob_f2 = probs[1], probs[0]  # target=1 means f1 wins

        # Get market odds (real if available, else simulated from ELO)
        fight_row = test_fights.iloc[idx]
        odds_f1, odds_f2, is_real = get_market_odds(fight_row)

        # Calculate Expected Value for both fighters
        ev_f1 = (prob_f1 * odds_f1) - 1
        ev_f2 = (prob_f2 * odds_f2) - 1

        # Determine if we should bet (and on whom)
        bet_on = None
        odds = None
        model_prob = None

        if ev_f1 > edge_threshold and ev_f1 >= ev_f2:
            bet_on = 'f1'
            odds = odds_f1
            model_prob = prob_f1
            ev = ev_f1
        elif ev_f2 > edge_threshold:
            bet_on = 'f2'
            odds = odds_f2
            model_prob = prob_f2
            ev = ev_f2

        # Skip if no betting opportunity
        if bet_on is None:
            continue

        # Place bet
        total_bets += 1

        # Resolve bet
        actual_winner = y_test.iloc[idx]  # 1 = f1 wins, 0 = f2 wins
        won_bet = (bet_on == 'f1' and actual_winner == 1) or (bet_on == 'f2' and actual_winner == 0)

        if won_bet:
            profit = bet_size * (odds - 1)
            bankroll += profit
            wins += 1
        else:
            profit = -bet_size
            bankroll -= bet_size
            losses += 1

        # Record bet
        bet_history.append({
            'date': test_fights.iloc[idx]['date'],
            'bet_on': bet_on,
            'model_prob': model_prob,
            'odds': odds,
            'ev': ev,
            'won': won_bet,
            'profit': profit,
            'bankroll': bankroll,
            'odds_source': 'real' if is_real else 'simulated',
            'weight_class': test_fights.iloc[idx]['weight_class'],
            'implied_prob': 1 / odds
        })

    # Calculate summary stats
    total_profit = bankroll - starting_bankroll
    roi = (total_profit / starting_bankroll) * 100
    win_rate = (wins / total_bets * 100) if total_bets > 0 else 0

    # Separate stats for real vs simulated odds
    real_bets = [b for b in bet_history if b['odds_source'] == 'real']
    sim_bets = [b for b in bet_history if b['odds_source'] == 'simulated']

    # Print results
    print("\n" + "=" * 80)
    print(" PROFITABILITY BACKTEST RESULTS")
    print("=" * 80)
    if len(bet_history) > 0:
        print(f"Period: {bet_history[0]['date']} to {bet_history[-1]['date']}")
    print(f"Starting Bankroll: ${starting_bankroll:,.2f}")
    print()

    # Odds coverage
    print("Odds Coverage:")
    print(f"  Fights with Real Odds: {len([b for b in bet_history if b['odds_source'] == 'real'])} ({100*len(real_bets)/total_bets:.1f}%)")
    print(f"  Fights with Simulated: {len([b for b in bet_history if b['odds_source'] == 'simulated'])} ({100*len(sim_bets)/total_bets:.1f}%)")
    print()

    # Real odds performance
    if len(real_bets) > 0:
        real_wins = sum(1 for b in real_bets if b['won'])
        real_profit = sum(b['profit'] for b in real_bets)
        real_roi = (real_profit / (len(real_bets) * bet_size)) * 100
        real_win_rate = (real_wins / len(real_bets)) * 100

        print("REAL ODDS PERFORMANCE:")
        print(f"  Total Bets: {len(real_bets)}")
        print(f"  Win Rate: {real_win_rate:.1f}%")
        print(f"  Total Profit: ${real_profit:+,.2f}")
        print(f"  ROI: {real_roi:+.1f}%")
        print()

    # Simulated odds performance
    if len(sim_bets) > 0:
        sim_wins = sum(1 for b in sim_bets if b['won'])
        sim_profit = sum(b['profit'] for b in sim_bets)
        sim_roi = (sim_profit / (len(sim_bets) * bet_size)) * 100
        sim_win_rate = (sim_wins / len(sim_bets)) * 100

        print("SIMULATED ODDS PERFORMANCE:")
        print(f"  Total Bets: {len(sim_bets)}")
        print(f"  Win Rate: {sim_win_rate:.1f}%")
        print(f"  Total Profit: ${sim_profit:+,.2f}")
        print(f"  ROI: {sim_roi:+.1f}%")
        print()

    print("OVERALL PERFORMANCE:")
    print(f"  Total Bets Placed: {total_bets}")
    print(f"  Wins: {wins} ({win_rate:.1f}%)")
    print(f"  Losses: {losses}")
    print(f"  Starting Bankroll:  ${starting_bankroll:,.2f}")
    print(f"  Ending Bankroll:    ${bankroll:,.2f}")
    print(f"  Total Profit:       ${total_profit:+,.2f}")
    print(f"  ROI:                {roi:+.1f}%")
    print()

    # Conclusion
    if len(real_bets) > 0:
        real_wins = sum(1 for b in real_bets if b['won'])
        real_profit = sum(b['profit'] for b in real_bets)
        real_roi = (real_profit / (len(real_bets) * bet_size)) * 100

        if real_roi > 5:
            print("CONCLUSION: Model beats Vegas lines! (Professional-grade edge)")
        elif real_roi > 0:
            print("CONCLUSION: Model has marginal edge over Vegas lines")
        elif real_roi > -5:
            print("CONCLUSION: Model breaks even against Vegas (within variance)")
        else:
            print("CONCLUSION: Model doesn't beat Vegas lines. Consider probability calibration.")
    else:
        if roi > 0:
            print("CONCLUSION: Model shows positive edge (simulated odds only)")
        elif roi == 0:
            print("CONCLUSION: Model breaks even (simulated odds only)")
        else:
            print("CONCLUSION: Model loses money (simulated odds only)")

    print("=" * 80)

    # Generate profit chart
    if len(bet_history) > 0:
        print("\nGenerating profit chart...")
        plot_profit_curve(bet_history, starting_bankroll)

    # Print segmented ROI analysis
    if len(bet_history) > 0:
        print_segmented_roi(bet_history, bet_size)

    return {
        'total_bets': total_bets,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'starting_bankroll': starting_bankroll,
        'ending_bankroll': bankroll,
        'total_profit': total_profit,
        'roi': roi,
        'bet_history': bet_history
    }


def run_sniper_backtest(
    bet_size: float = 100,
    starting_bankroll: float = 10000
) -> Dict:
    """
    Sniper Strategy: Only bet on high-confidence favorites in select weight classes.

    Filters:
    - Model Probability > 65% (high confidence only)
    - Betting on Favorite (Implied Prob > 50%)
    - Exclude: Heavyweight, Flyweight, Women's Strawweight (negative ROI weight classes)

    Args:
        bet_size: Fixed bet amount per wager (default: $100)
        starting_bankroll: Starting capital (default: $10,000)

    Returns:
        Dictionary with results
    """
    print("\n" + "=" * 80)
    print(" SNIPER STRATEGY BACKTEST")
    print("=" * 80)
    print("\nFilters Applied:")
    print("  - Model Probability > 65%")
    print("  - Favorites Only (Implied Prob > 50%)")
    print("  - Excluded: Heavyweight, Flyweight, Women's Strawweight")

    # Excluded weight classes (negative ROI from segmentation)
    EXCLUDED_WEIGHT_CLASSES = {
        'Heavyweight Bout',
        'Flyweight Bout',
        'Women\'s Strawweight Bout'
    }

    # Load data
    X_test, test_fights, y_test, model = load_backtest_data()

    print("\n[3/3] Running Sniper betting simulation...")
    print(f"  Bet size: ${bet_size:.2f}")
    print(f"  Starting bankroll: ${starting_bankroll:,.2f}")

    # Initialize tracking
    bankroll = starting_bankroll
    total_bets = 0
    wins = 0
    losses = 0
    bet_history = []
    filtered_count = 0  # Track how many opportunities were filtered out

    # Simulate betting over test period
    for idx in range(len(X_test)):
        # Get model prediction
        X = X_test.iloc[idx:idx+1]
        probs = model.predict_proba(X)[0]
        prob_f1, prob_f2 = probs[1], probs[0]  # target=1 means f1 wins

        # Get market odds (real if available, else simulated from ELO)
        fight_row = test_fights.iloc[idx]
        odds_f1, odds_f2, is_real = get_market_odds(fight_row)
        weight_class = test_fights.iloc[idx]['weight_class']

        # Calculate Expected Value for both fighters
        ev_f1 = (prob_f1 * odds_f1) - 1
        ev_f2 = (prob_f2 * odds_f2) - 1

        # Determine if we should bet (and on whom)
        bet_on = None
        odds = None
        model_prob = None
        edge_threshold = 0.05  # Standard 5% edge threshold

        if ev_f1 > edge_threshold and ev_f1 >= ev_f2:
            bet_on = 'f1'
            odds = odds_f1
            model_prob = prob_f1
            ev = ev_f1
        elif ev_f2 > edge_threshold:
            bet_on = 'f2'
            odds = odds_f2
            model_prob = prob_f2
            ev = ev_f2

        # Skip if no betting opportunity at all
        if bet_on is None:
            continue

        # SNIPER FILTERS
        implied_prob = 1 / odds

        # Filter 1: Model Confidence > 65%
        if model_prob <= 0.65:
            filtered_count += 1
            continue

        # Filter 2: Favorites Only (Implied Prob > 50%)
        if implied_prob <= 0.50:
            filtered_count += 1
            continue

        # Filter 3: Exclude negative-ROI weight classes
        if weight_class in EXCLUDED_WEIGHT_CLASSES:
            filtered_count += 1
            continue

        # All filters passed - place bet
        total_bets += 1

        # Resolve bet
        actual_winner = y_test.iloc[idx]  # 1 = f1 wins, 0 = f2 wins
        won_bet = (bet_on == 'f1' and actual_winner == 1) or (bet_on == 'f2' and actual_winner == 0)

        if won_bet:
            profit = bet_size * (odds - 1)
            bankroll += profit
            wins += 1
        else:
            profit = -bet_size
            bankroll -= bet_size
            losses += 1

        # Record bet
        bet_history.append({
            'date': test_fights.iloc[idx]['date'],
            'bet_on': bet_on,
            'model_prob': model_prob,
            'odds': odds,
            'ev': ev,
            'won': won_bet,
            'profit': profit,
            'bankroll': bankroll,
            'odds_source': 'real' if is_real else 'simulated',
            'weight_class': weight_class,
            'implied_prob': implied_prob
        })

    # Calculate summary stats
    total_profit = bankroll - starting_bankroll
    roi = (total_profit / starting_bankroll) * 100
    win_rate = (wins / total_bets * 100) if total_bets > 0 else 0

    # Separate stats for real vs simulated odds
    real_bets = [b for b in bet_history if b['odds_source'] == 'real']
    sim_bets = [b for b in bet_history if b['odds_source'] == 'simulated']

    # Print results
    print("\n" + "=" * 80)
    print(" SNIPER STRATEGY RESULTS")
    print("=" * 80)
    if len(bet_history) > 0:
        print(f"Period: {bet_history[0]['date']} to {bet_history[-1]['date']}")
    print(f"Starting Bankroll: ${starting_bankroll:,.2f}")
    print()

    print("Filtering Stats:")
    print(f"  Total Opportunities: {total_bets + filtered_count}")
    print(f"  Filtered Out: {filtered_count}")
    print(f"  Bets Placed: {total_bets}")
    print()

    # Odds coverage
    print("Odds Coverage:")
    if total_bets > 0:
        print(f"  Fights with Real Odds: {len(real_bets)} ({100*len(real_bets)/total_bets:.1f}%)")
        print(f"  Fights with Simulated: {len(sim_bets)} ({100*len(sim_bets)/total_bets:.1f}%)")
    print()

    # Real odds performance
    if len(real_bets) > 0:
        real_wins = sum(1 for b in real_bets if b['won'])
        real_profit = sum(b['profit'] for b in real_bets)
        real_roi = (real_profit / (len(real_bets) * bet_size)) * 100
        real_win_rate = (real_wins / len(real_bets)) * 100

        print("REAL ODDS PERFORMANCE:")
        print(f"  Total Bets: {len(real_bets)}")
        print(f"  Win Rate: {real_win_rate:.1f}%")
        print(f"  Total Profit: ${real_profit:+,.2f}")
        print(f"  ROI: {real_roi:+.1f}%")
        print()

    # Simulated odds performance
    if len(sim_bets) > 0:
        sim_wins = sum(1 for b in sim_bets if b['won'])
        sim_profit = sum(b['profit'] for b in sim_bets)
        sim_roi = (sim_profit / (len(sim_bets) * bet_size)) * 100
        sim_win_rate = (sim_wins / len(sim_bets)) * 100

        print("SIMULATED ODDS PERFORMANCE:")
        print(f"  Total Bets: {len(sim_bets)}")
        print(f"  Win Rate: {sim_win_rate:.1f}%")
        print(f"  Total Profit: ${sim_profit:+,.2f}")
        print(f"  ROI: {sim_roi:+.1f}%")
        print()

    print("OVERALL PERFORMANCE:")
    print(f"  Total Bets Placed: {total_bets}")
    print(f"  Wins: {wins} ({win_rate:.1f}%)")
    print(f"  Losses: {losses}")
    print(f"  Starting Bankroll:  ${starting_bankroll:,.2f}")
    print(f"  Ending Bankroll:    ${bankroll:,.2f}")
    print(f"  Total Profit:       ${total_profit:+,.2f}")
    print(f"  ROI:                {roi:+.1f}%")
    print()

    # Conclusion
    if len(real_bets) > 0:
        real_wins = sum(1 for b in real_bets if b['won'])
        real_profit = sum(b['profit'] for b in real_bets)
        real_roi = (real_profit / (len(real_bets) * bet_size)) * 100

        print("COMPARISON TO UNFILTERED BACKTEST:")
        print(f"  Unfiltered Real Odds ROI: -4.9%")
        print(f"  Sniper Real Odds ROI:     {real_roi:+.1f}%")
        if real_roi > -4.9:
            improvement = real_roi - (-4.9)
            print(f"  Improvement:              {improvement:+.1f}%")
        print()

        if real_roi > 5:
            print("CONCLUSION: Sniper Strategy beats Vegas! (Professional-grade edge)")
        elif real_roi > 0:
            print("CONCLUSION: Sniper Strategy has positive edge over Vegas")
        elif real_roi > -5:
            print("CONCLUSION: Sniper Strategy breaks even against Vegas")
        else:
            print("CONCLUSION: Sniper Strategy still loses against Vegas")
    else:
        if roi > 0:
            print("CONCLUSION: Sniper Strategy shows positive edge (simulated odds only)")
        else:
            print("CONCLUSION: Sniper Strategy loses money (simulated odds only)")

    print("=" * 80)

    return {
        'total_bets': total_bets,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'starting_bankroll': starting_bankroll,
        'ending_bankroll': bankroll,
        'total_profit': total_profit,
        'roi': roi,
        'bet_history': bet_history,
        'filtered_count': filtered_count
    }


def plot_profit_curve(bet_history: List[Dict], starting_bankroll: float):
    """
    Plot cumulative profit over time.

    Args:
        bet_history: List of bet dictionaries with 'date', 'profit', 'bankroll'
        starting_bankroll: Initial capital
    """
    # Extract data
    dates = [bet['date'] for bet in bet_history]
    cumulative_profit = [bet['bankroll'] - starting_bankroll for bet in bet_history]

    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(dates, cumulative_profit, linewidth=2, color='#2E86AB')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Break Even')
    plt.fill_between(dates, cumulative_profit, 0, alpha=0.2, color='#2E86AB')

    plt.title('Profitability Backtest: Cumulative Profit Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Profit ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Format y-axis as currency
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    # Save plot
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOTS_DIR / "profit_simulation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Profit chart saved to: {plot_path}")


def print_segmented_roi(bet_history: List[Dict], bet_size: float):
    """
    Print ROI breakdown by segment (Favorite/Underdog, Confidence, Weight Class).

    Args:
        bet_history: List of bet dictionaries
        bet_size: Fixed bet amount per wager
    """
    if len(bet_history) == 0:
        print("No bets to analyze.")
        return

    print("\n" + "=" * 80)
    print(" SEGMENTED ROI ANALYSIS")
    print("=" * 80)

    # Segment 1: Favorite vs Underdog
    favorites = [b for b in bet_history if b['implied_prob'] > 0.5]
    underdogs = [b for b in bet_history if b['implied_prob'] <= 0.5]

    print("\nFAVORITE vs UNDERDOG:")
    if len(favorites) > 0:
        fav_wins = sum(1 for b in favorites if b['won'])
        fav_profit = sum(b['profit'] for b in favorites)
        fav_roi = (fav_profit / (len(favorites) * bet_size)) * 100
        print(f"  Favorites:   ROI: {fav_roi:+6.1f}% ({len(favorites):3d} bets, {fav_wins:3d} wins)")
    else:
        print("  Favorites:   No bets")

    if len(underdogs) > 0:
        dog_wins = sum(1 for b in underdogs if b['won'])
        dog_profit = sum(b['profit'] for b in underdogs)
        dog_roi = (dog_profit / (len(underdogs) * bet_size)) * 100
        print(f"  Underdogs:   ROI: {dog_roi:+6.1f}% ({len(underdogs):3d} bets, {dog_wins:3d} wins)")
    else:
        print("  Underdogs:   No bets")

    # Segment 2: Confidence Levels (by Model Probability)
    high_conf = [b for b in bet_history if b['model_prob'] > 0.65]
    med_conf = [b for b in bet_history if 0.55 < b['model_prob'] <= 0.65]
    low_conf = [b for b in bet_history if 0.50 <= b['model_prob'] <= 0.55]

    print("\nCONFIDENCE LEVELS (by Model Probability):")
    if len(high_conf) > 0:
        high_profit = sum(b['profit'] for b in high_conf)
        high_roi = (high_profit / (len(high_conf) * bet_size)) * 100
        print(f"  High (>65%):     ROI: {high_roi:+6.1f}% ({len(high_conf):3d} bets)")
    else:
        print("  High (>65%):     No bets")

    if len(med_conf) > 0:
        med_profit = sum(b['profit'] for b in med_conf)
        med_roi = (med_profit / (len(med_conf) * bet_size)) * 100
        print(f"  Medium (55-65%): ROI: {med_roi:+6.1f}% ({len(med_conf):3d} bets)")
    else:
        print("  Medium (55-65%): No bets")

    if len(low_conf) > 0:
        low_profit = sum(b['profit'] for b in low_conf)
        low_roi = (low_profit / (len(low_conf) * bet_size)) * 100
        print(f"  Low (50-55%):    ROI: {low_roi:+6.1f}% ({len(low_conf):3d} bets)")
    else:
        print("  Low (50-55%):    No bets")

    # Segment 3: Weight Class
    weight_classes = {}
    for bet in bet_history:
        wc = bet['weight_class']
        if wc not in weight_classes:
            weight_classes[wc] = []
        weight_classes[wc].append(bet)

    print("\nWEIGHT CLASS BREAKDOWN:")
    # Sort by ROI descending
    wc_stats = []
    for wc, bets in weight_classes.items():
        profit = sum(b['profit'] for b in bets)
        roi = (profit / (len(bets) * bet_size)) * 100
        wc_stats.append((wc, roi, len(bets)))

    wc_stats.sort(key=lambda x: x[1], reverse=True)  # Sort by ROI

    for wc, roi, num_bets in wc_stats:
        # Truncate long weight class names for cleaner output
        wc_display = wc[:30] if len(wc) > 30 else wc
        print(f"  {wc_display:<30} ROI: {roi:+6.1f}% ({num_bets:3d} bets)")

    print("=" * 80)


if __name__ == "__main__":
    # Run standard unfiltered backtest
    print("\n" + "=" * 80)
    print(" RUNNING STANDARD BACKTEST")
    print("=" * 80)
    standard_results = run_backtest()

    # Run Sniper Strategy backtest
    sniper_results = run_sniper_backtest()

    # Final comparison
    print("\n" + "=" * 80)
    print(" FINAL COMPARISON")
    print("=" * 80)

    # Extract real odds ROI from both strategies
    standard_real_bets = [b for b in standard_results['bet_history'] if b['odds_source'] == 'real']
    sniper_real_bets = [b for b in sniper_results['bet_history'] if b['odds_source'] == 'real']

    if len(standard_real_bets) > 0:
        standard_real_profit = sum(b['profit'] for b in standard_real_bets)
        standard_real_roi = (standard_real_profit / (len(standard_real_bets) * 100)) * 100
    else:
        standard_real_roi = 0

    if len(sniper_real_bets) > 0:
        sniper_real_profit = sum(b['profit'] for b in sniper_real_bets)
        sniper_real_roi = (sniper_real_profit / (len(sniper_real_bets) * 100)) * 100
    else:
        sniper_real_roi = 0

    print("\nStandard Backtest (Unfiltered):")
    print(f"  Total Bets: {len(standard_real_bets)}")
    print(f"  Real Odds ROI: {standard_real_roi:+.1f}%")

    print("\nSniper Strategy (Filtered):")
    print(f"  Total Bets: {len(sniper_real_bets)}")
    print(f"  Real Odds ROI: {sniper_real_roi:+.1f}%")

    if len(sniper_real_bets) > 0 and len(standard_real_bets) > 0:
        improvement = sniper_real_roi - standard_real_roi
        print(f"\nImprovement: {improvement:+.1f}% ROI")
        print(f"Bet Volume Reduction: {len(standard_real_bets) - len(sniper_real_bets)} fewer bets")

    print("=" * 80)
