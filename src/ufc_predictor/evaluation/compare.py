"""
Model comparison utilities.

Functions for comparing ELO baseline performance vs ML models.
"""

import numpy as np
from typing import Dict, Any


def compare_elo_vs_model(
    elo_probs: np.ndarray,
    model_probs: np.ndarray,
    y_true: np.ndarray
) -> Dict[str, Any]:
    """
    Compare ELO baseline performance vs ML model performance.

    Computes metrics for both models and calculates improvements.

    Args:
        elo_probs: Win probabilities from ELO ratings
        model_probs: Win probabilities from ML model
        y_true: True fight outcomes

    Returns:
        Dictionary with comparison results:
        {
            "elo_metrics": {
                "accuracy": float,
                "brier_score": float,
                "log_loss": float,
                ...
            },
            "model_metrics": {
                "accuracy": float,
                "brier_score": float,
                "log_loss": float,
                ...
            },
            "improvements": {
                "accuracy_delta": float,  # model - elo
                "brier_delta": float,     # elo - model (negative = better)
                "log_loss_delta": float,  # elo - model (negative = better)
            },
            "winner": str  # "elo" or "model" based on overall performance
        }

    TODO: Implement comparison logic using metrics module
    """
    pass


def generate_comparison_report(
    comparison: Dict[str, Any],
    output_path: str
) -> None:
    """
    Generate human-readable comparison report.

    Creates a markdown or text report summarizing:
    - Metrics for both models
    - Improvements achieved by ML model
    - Calibration analysis
    - Recommendations

    Args:
        comparison: Output from compare_elo_vs_model()
        output_path: Path to save report file

    TODO: Implement report generation
    - Format metrics in readable tables
    - Add interpretation and insights
    - Save to output/reports/
    """
    pass


def analyze_upset_predictions(
    elo_probs: np.ndarray,
    model_probs: np.ndarray,
    y_true: np.ndarray,
    upset_threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Analyze how well each model predicts upsets (underdog wins).

    An upset is when the fighter with lower pre-fight probability wins.

    Args:
        elo_probs: Win probabilities from ELO
        model_probs: Win probabilities from ML model
        y_true: True outcomes
        upset_threshold: Threshold for defining underdogs (default: <30% prob)

    Returns:
        Dictionary with upset prediction analysis

    TODO: Implement upset analysis
    - Identify upsets in test set
    - Compare model performance on upsets vs favorites
    """
    pass
