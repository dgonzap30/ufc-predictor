#!/usr/bin/env python3
"""
CLI script to run the complete UFC prediction pipeline.

This script orchestrates the end-to-end process:
- Data loading and cleaning
- ELO computation
- Feature engineering
- Model training
- Evaluation and reporting
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ufc_predictor.pipeline.run_pipeline import run_full_pipeline
from ufc_predictor import config


def main():
    """
    Main entry point for pipeline execution.

    TODO: Implement CLI argument parsing and pipeline invocation.
    """
    parser = argparse.ArgumentParser(
        description="Run the UFC fight prediction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python scripts/run_pipeline.py

  # Specify test split date
  python scripts/run_pipeline.py --test-start-date 2022-01-01

  # Use random forest model
  python scripts/run_pipeline.py --model-type random_forest

  # Custom K-factor for ELO
  python scripts/run_pipeline.py --k-factor 40
        """
    )

    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=config.RAW_DATA_DIR,
        help="Directory containing raw CSV files"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.OUTPUT_DIR,
        help="Directory for output files"
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="logreg",
        choices=["logreg", "random_forest", "gradient_boosting"],
        help="Type of ML model to train"
    )

    parser.add_argument(
        "--test-start-date",
        type=str,
        default=None,
        help="Date to start test set (YYYY-MM-DD format)"
    )

    parser.add_argument(
        "--k-factor",
        type=float,
        default=config.K_FACTOR,
        help="ELO K-factor for rating updates"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print verbose progress messages"
    )

    args = parser.parse_args()

    # Run the complete pipeline
    success = run_full_pipeline()

    # Exit with appropriate code
    if success:
        sys.exit(0)
    else:
        print("\nPipeline failed. See error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
