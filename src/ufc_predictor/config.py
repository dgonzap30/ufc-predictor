"""
Central configuration for the UFC predictor project.

This module contains paths, constants, and default parameters used throughout the project.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERMEDIATE_DATA_DIR = DATA_DIR / "intermediate"
FEATURES_DATA_DIR = DATA_DIR / "features"

# Model directory
MODELS_DIR = PROJECT_ROOT / "models"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = OUTPUT_DIR / "reports"
METRICS_DIR = OUTPUT_DIR / "metrics"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Logs directory
LOGS_DIR = PROJECT_ROOT / "logs"

# Source data directory (original CSVs from ufc-data/, read-only)
UFC_DATA_SOURCE_DIR = PROJECT_ROOT / "ufc-data"

# ELO rating parameters
ELO_START = 1500  # Initial rating for new fighters
K_FACTOR = 32     # ELO update rate (higher = more volatile)

# Train/test split parameters
# TODO: Define the cutoff strategy (date-based or percentage-based)
TEST_START_YEAR = None  # Placeholder: e.g., 2022 for fights from 2022 onwards as test set
TEST_START_DATE = None  # Placeholder: e.g., "2022-01-01"

# Model training parameters
RANDOM_SEED = 42
DEFAULT_MODEL_TYPE = "logreg"  # Options: "logreg", "random_forest", "gradient_boosting"

# Feature engineering parameters
RECENT_FIGHTS_WINDOW = 5  # Number of recent fights to consider for form features
