# UFC Predictor

A machine learning system for predicting UFC fight outcomes using ELO ratings and rich fighter features.

## Overview

This project implements a two-tier prediction system:

1. **ELO Baseline** - A transparent, skill rating-based prediction model that serves as the minimum performance benchmark
2. **ML Models** - Advanced models trained on engineered features (fighter history, physical attributes, matchup dynamics) to improve upon the ELO baseline

## Project Structure

```
ufc-predictor/
├── data/
│   ├── raw/              # Original CSV files (read-only)
│   ├── intermediate/     # Cleaned and validated data
│   └── features/         # ML-ready feature matrices
├── models/               # Serialized trained models
├── notebooks/            # Exploratory analysis
├── output/
│   ├── reports/          # Human-readable evaluation reports
│   ├── metrics/          # Machine-readable metrics (JSON/CSV)
│   └── plots/            # Visualizations
├── logs/                 # Pipeline execution logs
├── scripts/              # CLI tools for running pipeline steps
└── src/ufc_predictor/    # Main Python package
    ├── data/             # Data ingestion, cleaning, validation
    ├── rating/           # ELO rating system
    ├── features/         # Feature engineering
    ├── models/           # Model training and inference
    ├── evaluation/       # Metrics and comparison tools
    └── pipeline/         # End-to-end orchestration
```

## Installation

```bash
# Install the package in editable mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## Usage

### Run the full pipeline

```bash
python scripts/run_pipeline.py
```

### Train a specific model

```bash
python scripts/train_model.py --model-type logreg
```

### Evaluate models

```bash
python scripts/eval_models.py
```

## Key Features

- **Time-based train/test split** - Models evaluated on chronologically future fights
- **No label leakage** - Features use only pre-fight information
- **Comprehensive evaluation** - Accuracy, Brier score, log loss, ROC-AUC
- **Direct ELO vs ML comparison** - Side-by-side performance metrics

## Documentation

For detailed project context, architecture decisions, and implementation guidelines, see:
- [PROJECT_CONTEXT_UFC.md](docs/PROJECT_CONTEXT_UFC.md)

## Development Status

This project is currently in the scaffolding phase. Module implementations are in progress.
