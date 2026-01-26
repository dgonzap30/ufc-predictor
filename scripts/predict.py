#!/usr/bin/env python3
"""
CLI script for UFC fight prediction.

Predicts hypothetical matchups using the trained Random Forest model,
with optional betting Expected Value (EV) calculation.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ufc_predictor.inference.predict import predict_matchup


def main():
    """Main entry point for fight prediction CLI."""
    parser = argparse.ArgumentParser(
        description="Predict UFC fight outcomes with optional betting analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic prediction
  python scripts/predict.py --fighter1 "Jon Jones" --fighter2 "Tom Aspinall"

  # With betting odds for EV calculation
  python scripts/predict.py --fighter1 "Jon Jones" --fighter2 "Tom Aspinall" \\
      --odds1 -150 --odds2 +130

  # Specify weight class and date
  python scripts/predict.py --fighter1 "Jon Jones" --fighter2 "Tom Aspinall" \\
      --weight-class "Heavyweight Bout" --date 2025-12-31
        """
    )

    parser.add_argument(
        "--fighter1",
        required=True,
        help="Fighter 1 name (case-sensitive)"
    )

    parser.add_argument(
        "--fighter2",
        required=True,
        help="Fighter 2 name (case-sensitive)"
    )

    parser.add_argument(
        "--weight-class",
        default="Heavyweight Bout",
        help="Weight class for the fight (default: Heavyweight Bout)"
    )

    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Fight date for prediction (YYYY-MM-DD format, default: today)"
    )

    parser.add_argument(
        "--odds1",
        type=float,
        default=None,
        help="American odds for Fighter 1 (e.g., -150 for favorite, +200 for underdog)"
    )

    parser.add_argument(
        "--odds2",
        type=float,
        default=None,
        help="American odds for Fighter 2 (e.g., -150 for favorite, +200 for underdog)"
    )

    args = parser.parse_args()

    # Parse date if provided
    fight_date = None
    if args.date:
        try:
            fight_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            print(f"Error: Invalid date format '{args.date}'. Use YYYY-MM-DD.")
            sys.exit(1)

    # Make prediction
    try:
        result = predict_matchup(
            fighter1_name=args.fighter1,
            fighter2_name=args.fighter2,
            weight_class=args.weight_class,
            fight_date=fight_date,
            odds_f1=args.odds1,
            odds_f2=args.odds2,
        )

        # Print results
        print("\n" + "=" * 60)
        print(" UFC FIGHT PREDICTION")
        print("=" * 60)
        print(f"{args.fighter1} vs {args.fighter2}")
        print(f"Weight Class: {args.weight_class}")
        if fight_date:
            print(f"Date: {args.date}")
        else:
            print(f"Date: {datetime.now().strftime('%Y-%m-%d')} (today)")

        print(f"\nPREDICTION: {result['winner']} wins")
        print(f"Confidence: {result['confidence']:.1f}%")

        print(f"\nDetailed Probabilities:")
        print(f"  {args.fighter1}: {result['f1_prob']*100:.1f}%")
        print(f"  {args.fighter2}: {result['f2_prob']*100:.1f}%")

        # Fighter stats
        print(f"\n{args.fighter1} Stats:")
        f1 = result['f1_state']
        print(f"  ELO: {f1['latest_elo']:.1f}")
        print(f"  Record: {f1['wins']}-{f1['total_fights']-f1['wins']} ({f1['win_rate']*100:.1f}% win rate)")
        print(f"  Streak: {f1['streak']} {'wins' if f1['streak'] > 0 else 'losses' if f1['streak'] < 0 else 'N/A'}")
        print(f"  Days since last fight: {f1['days_since_last']}")

        print(f"\n{args.fighter2} Stats:")
        f2 = result['f2_state']
        print(f"  ELO: {f2['latest_elo']:.1f}")
        print(f"  Record: {f2['wins']}-{f2['total_fights']-f2['wins']} ({f2['win_rate']*100:.1f}% win rate)")
        print(f"  Streak: {f2['streak']} {'wins' if f2['streak'] > 0 else 'losses' if f2['streak'] < 0 else 'N/A'}")
        print(f"  Days since last fight: {f2['days_since_last']}")

        # Betting analysis
        if 'ev_f1' in result:
            print("\n" + "-" * 60)
            print("BETTING ANALYSIS")
            print("-" * 60)

            ev_f1 = result['ev_f1']
            ev_f2 = result['ev_f2']

            print(f"{args.fighter1}:")
            print(f"  Odds: {result['odds_f1']:+.0f}")
            print(f"  EV: {ev_f1*100:+.1f}% {'✓ POSITIVE EV' if ev_f1 > 0 else ''}")

            print(f"\n{args.fighter2}:")
            print(f"  Odds: {result['odds_f2']:+.0f}")
            print(f"  EV: {ev_f2*100:+.1f}% {'✓ POSITIVE EV' if ev_f2 > 0 else ''}")

            print(f"\nRECOMMENDATION: {result['recommendation']}")

        # Sniper Strategy Check
        if 'sniper' in result:
            print()
            if result['sniper']['is_sniper_bet']:
                print("=" * 60)
                print("  🎯 [SNIPER BET FOUND] 🎯")
                print("=" * 60)
                print("  ✅ High confidence (>65%)")
                print("  ✅ Betting on favorite")
                print("  ✅ Weight class approved")
                print("=" * 60)
            else:
                print("=" * 60)
                print("  ⛔ [PASS - NOT A SNIPER BET] ⛔")
                print("=" * 60)
                for reason in result['sniper']['reasons']:
                    print(f"  ❌ {reason}")
                print("-" * 60)
                print("  Skipping: Risk is too high for Sniper Strategy")
                print("=" * 60)

        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\nError making prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
