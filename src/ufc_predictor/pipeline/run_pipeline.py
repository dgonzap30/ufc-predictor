"""
End-to-end pipeline orchestration.

This module provides the main pipeline function that coordinates all steps:
1. Load raw data
2. Clean and validate
3. Compute ELO ratings
4. Engineer features
5. Train baseline + ML models
6. Evaluate and compare
7. Generate reports
"""

from typing import Optional, Dict, Any
from pathlib import Path

from ufc_predictor.data.ingest import sync_raw_to_data_dir
from ufc_predictor.data.cleaning import run_cleaning_pipeline
from ufc_predictor.data.validation import run_validation
from ufc_predictor.rating.elo import run_elo_pipeline
from ufc_predictor.features.engineering import run_feature_pipeline
from ufc_predictor.models.train import run_experiments
from ufc_predictor.evaluation.evaluate import run_evaluation
from ufc_predictor.evaluation.backtest import run_backtest


def run_full_pipeline() -> bool:
    """
    Execute the complete UFC prediction pipeline end-to-end.

    Runs all 7 stages sequentially:
    1. Data Ingestion - Sync raw CSVs to data/raw/
    2. Data Cleaning - Clean and standardize data
    3. Data Validation - Check integrity and generate report
    4. ELO Ratings - Compute historical ELO ratings
    5. Feature Engineering - Build feature matrices
    6. Model Training - Train baseline and ML models
    7. Evaluation - Generate metrics, plots, and reports

    Returns:
        True if pipeline completed successfully, False if any stage failed
    """
    print("\n" + "=" * 80)
    print(" UFC FIGHT PREDICTION - END-TO-END PIPELINE ")
    print("=" * 80)

    stages = [
        ("Stage 1: Data Ingestion", sync_raw_to_data_dir),
        ("Stage 2: Data Cleaning", run_cleaning_pipeline),
        ("Stage 3: Data Validation", run_validation),
        ("Stage 4: ELO Ratings", run_elo_pipeline),
        ("Stage 5: Feature Engineering", run_feature_pipeline),
        ("Stage 6: Model Training", run_experiments),
        ("Stage 7: Evaluation", run_evaluation),
        ("Stage 8: Profitability Backtest", run_backtest),
    ]

    for i, (stage_name, stage_func) in enumerate(stages, 1):
        print(f"\n{'=' * 80}")
        print(f"  {stage_name}")
        print(f"{'=' * 80}\n")

        try:
            result = stage_func()

            # Log result if applicable (most functions return None)
            if result is not None and stage_name == "Stage 3: Data Validation":
                # Validation returns a dict with results, just informational
                pass

        except Exception as e:
            print(f"\n{'!' * 80}")
            print(f"  ERROR in {stage_name}")
            print(f"{'!' * 80}")
            print(f"  {type(e).__name__}: {e}")
            print(f"{'!' * 80}\n")
            return False

    # Success
    print(f"\n{'=' * 80}")
    print("  PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'=' * 80}")
    print("\nGenerated outputs:")
    print("  - Cleaned data:       data/intermediate/")
    print("  - ELO ratings:        data/intermediate/")
    print("  - Feature matrices:   data/features/")
    print("  - Trained models:     models/")
    print("  - Evaluation plots:   output/plots/")
    print("  - Evaluation report:  output/reports/final_evaluation.md")
    print(f"{'=' * 80}\n")

    return True


def run_end_to_end(
    raw_data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    model_type: str = "logreg",
    test_start_date: Optional[str] = None,
    k_factor: float = 32,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run the complete UFC prediction pipeline.

    Pipeline steps:
    1. **Data Ingestion**
       - Load raw fights, fighters, and events from CSV files

    2. **Data Cleaning**
       - Remove duplicates
       - Handle missing values
       - Standardize types and formats

    3. **Data Validation**
       - Check referential integrity
       - Validate value ranges
       - Generate validation report

    4. **ELO Computation**
       - Compute historical ELO ratings chronologically
       - Record pre-fight ratings for each bout

    5. **Feature Engineering**
       - Add fighter history features
       - Add matchup difference features
       - Attach ELO features
       - Build final feature matrix

    6. **Train/Test Split**
       - Split data chronologically
       - Ensure temporal consistency (train on past, test on future)

    7. **Model Training**
       - Train ELO baseline model
       - Train full ML model
       - Save models to disk

    8. **Evaluation**
       - Compute metrics for both models
       - Compare ELO vs ML performance
       - Generate calibration analysis

    9. **Reporting**
       - Save metrics to JSON
       - Generate markdown comparison report
       - Create visualizations (optional)

    Args:
        raw_data_dir: Directory containing raw CSV files (default: config.RAW_DATA_DIR)
        output_dir: Directory for outputs (default: config.OUTPUT_DIR)
        model_type: Type of ML model to train ("logreg", "random_forest", "gradient_boosting")
        test_start_date: Date to split train/test (default: use config.TEST_START_DATE)
        k_factor: ELO K-factor (default: 32)
        verbose: Print progress messages

    Returns:
        Dictionary containing:
        {
            "validation_report": Dict,
            "elo_metrics": Dict,
            "model_metrics": Dict,
            "comparison": Dict,
            "model_paths": {
                "elo_baseline": str,
                "ml_model": str
            }
        }

    TODO: Implement complete pipeline orchestration
    - Import and call functions from all modules
    - Add progress logging
    - Handle errors gracefully
    - Save intermediate results
    """
    pass


def run_data_pipeline(
    raw_data_dir: Path,
    output_dir: Path,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run only the data processing portion of the pipeline.

    Steps:
    1. Load raw data
    2. Clean data
    3. Validate data
    4. Save cleaned data to intermediate/

    Args:
        raw_data_dir: Directory with raw CSV files
        output_dir: Directory for outputs
        verbose: Print progress

    Returns:
        Dictionary with paths to cleaned data and validation report

    TODO: Implement data-only pipeline
    """
    pass


def run_training_pipeline(
    features_path: Path,
    models_dir: Path,
    model_type: str = "logreg",
    verbose: bool = True
) -> Dict[str, str]:
    """
    Run only the model training portion of the pipeline.

    Assumes features have already been engineered and saved.

    Steps:
    1. Load feature matrices
    2. Train baseline model
    3. Train ML model
    4. Save models

    Args:
        features_path: Path to saved feature matrices
        models_dir: Directory to save models
        model_type: Type of ML model
        verbose: Print progress

    Returns:
        Dictionary with paths to saved models

    TODO: Implement training-only pipeline
    """
    pass


if __name__ == "__main__":
    print("TODO: This module should be called from scripts/run_pipeline.py")
    print()
    print("Intended usage:")
    print("  python scripts/run_pipeline.py [options]")
    print()
    print("The pipeline will:")
    print("  1. Load and clean raw UFC data")
    print("  2. Compute ELO ratings")
    print("  3. Engineer features")
    print("  4. Train baseline + ML models")
    print("  5. Evaluate and compare performance")
    print("  6. Generate reports")
