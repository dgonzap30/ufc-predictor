#!/usr/bin/env python3
"""
CLI script to train a specific ML model.

This script loads pre-engineered features and trains a specified model type.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ufc_predictor import config


def main():
    """
    Main entry point for model training.

    TODO: Implement model training CLI.
    """
    parser = argparse.ArgumentParser(
        description="Train a UFC fight prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train logistic regression
  python scripts/train_model.py --model-type logreg

  # Train random forest with custom parameters
  python scripts/train_model.py --model-type random_forest --n-estimators 200

  # Load features from custom path
  python scripts/train_model.py --features-path data/features/custom_features.parquet
        """
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="logreg",
        choices=["logreg", "random_forest", "gradient_boosting"],
        help="Type of model to train"
    )

    parser.add_argument(
        "--features-path",
        type=Path,
        default=config.FEATURES_DATA_DIR / "features_train.parquet",
        help="Path to training features"
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path to save trained model (default: models/{model_type}_model.pkl)"
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds"
    )

    # Model-specific hyperparameters
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of estimators for tree-based models"
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth for tree-based models"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("UFC Model Training")
    print("=" * 60)
    print()
    print("TODO: Training implementation in progress")
    print()
    print("Configuration:")
    print(f"  Model type:       {args.model_type}")
    print(f"  Features path:    {args.features_path}")
    print(f"  CV folds:         {args.cv_folds}")
    print()

    # TODO: Implement training logic
    # from ufc_predictor.models.train import train_ml_model, save_model
    #
    # # Load features
    # X_train, y_train = load_features(args.features_path)
    #
    # # Train model
    # model = train_ml_model(X_train, y_train, model_type=args.model_type)
    #
    # # Save model
    # output_path = args.output_path or (config.MODELS_DIR / f"{args.model_type}_model.pkl")
    # save_model(model, output_path)
    #
    # print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    main()
