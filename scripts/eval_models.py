#!/usr/bin/env python3
"""
CLI script to evaluate and compare trained models.

This script loads trained models and test data, computes metrics,
and generates comparison reports.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ufc_predictor import config


def main():
    """
    Main entry point for model evaluation.

    TODO: Implement evaluation CLI.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate and compare UFC prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all models in models/ directory
  python scripts/eval_models.py

  # Evaluate specific models
  python scripts/eval_models.py --elo-model models/elo_baseline.pkl --ml-model models/logreg_model.pkl

  # Generate detailed report
  python scripts/eval_models.py --detailed-report
        """
    )

    parser.add_argument(
        "--test-features-path",
        type=Path,
        default=config.FEATURES_DATA_DIR / "features_test.parquet",
        help="Path to test features"
    )

    parser.add_argument(
        "--elo-model",
        type=Path,
        default=config.MODELS_DIR / "elo_baseline.pkl",
        help="Path to ELO baseline model"
    )

    parser.add_argument(
        "--ml-model",
        type=Path,
        default=config.MODELS_DIR / "logreg_model.pkl",
        help="Path to ML model"
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="Path to save comparison report (default: output/reports/comparison_{timestamp}.md)"
    )

    parser.add_argument(
        "--detailed-report",
        action="store_true",
        help="Generate detailed report with calibration plots"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("UFC Model Evaluation")
    print("=" * 60)
    print()
    print("TODO: Evaluation implementation in progress")
    print()
    print("Configuration:")
    print(f"  Test features:    {args.test_features_path}")
    print(f"  ELO model:        {args.elo_model}")
    print(f"  ML model:         {args.ml_model}")
    print(f"  Detailed report:  {args.detailed_report}")
    print()

    # TODO: Implement evaluation logic
    # from ufc_predictor.models.train import load_model
    # from ufc_predictor.models.predict import predict_proba
    # from ufc_predictor.evaluation.compare import compare_elo_vs_model, generate_comparison_report
    #
    # # Load test data
    # X_test, y_test = load_features(args.test_features_path)
    #
    # # Load models
    # elo_model = load_model(args.elo_model)
    # ml_model = load_model(args.ml_model)
    #
    # # Generate predictions
    # elo_probs = predict_proba(elo_model, X_test)
    # ml_probs = predict_proba(ml_model, X_test)
    #
    # # Compare models
    # comparison = compare_elo_vs_model(elo_probs, ml_probs, y_test)
    #
    # # Generate report
    # report_path = args.report_path or (config.REPORTS_DIR / f"comparison_{timestamp}.md")
    # generate_comparison_report(comparison, report_path)
    #
    # print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
