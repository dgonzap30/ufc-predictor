"""
Metrics computation for model evaluation.

This module computes standard classification metrics with a focus on
probability quality (calibration) in addition to accuracy.
"""

import numpy as np
from typing import Dict, Any, Optional


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Metrics computed:
    - Accuracy: Proportion of correct predictions
    - Brier Score: Mean squared error of probabilities (lower is better)
    - Log Loss: Cross-entropy loss (lower is better)
    - ROC-AUC: Area under ROC curve (if probabilities provided)
    - Precision, Recall, F1: Standard classification metrics

    Args:
        y_true: True labels (1 = fighter1 wins, 0 = fighter2 wins)
        y_pred: Predicted labels (binary)
        y_proba: Predicted probabilities for fighter1 winning (optional)

    Returns:
        Dictionary with all computed metrics:
        {
            "accuracy": float,
            "brier_score": float,  # Only if y_proba provided
            "log_loss": float,     # Only if y_proba provided
            "roc_auc": float,      # Only if y_proba provided
            "precision": float,
            "recall": float,
            "f1": float
        }

    TODO: Implement metrics computation using sklearn.metrics
    """
    pass


def compute_brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Compute Brier score (mean squared error of probabilities).

    Brier score measures the quality of probabilistic predictions.
    Lower is better (0 = perfect, 1 = worst possible).

    Args:
        y_true: True binary outcomes
        y_proba: Predicted probabilities

    Returns:
        Brier score

    TODO: Implement Brier score: mean((y_true - y_proba)^2)
    """
    pass


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Compute calibration metrics for probability predictions.

    Assesses whether predicted probabilities match observed frequencies.
    E.g., of all predictions with 70% confidence, do 70% actually win?

    Args:
        y_true: True binary outcomes
        y_proba: Predicted probabilities
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with:
        {
            "calibration_error": float,  # Expected Calibration Error (ECE)
            "bin_edges": array,
            "bin_accuracies": array,
            "bin_confidences": array,
            "bin_counts": array
        }

    TODO: Implement calibration analysis
    """
    pass
