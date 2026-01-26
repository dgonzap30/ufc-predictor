# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a UFC fight prediction system using a two-tier approach:
1. **ELO baseline** - Transparent skill rating system as minimum performance benchmark
2. **Calibrated XGBoost** - Advanced model trained on 98 engineered features with isotonic calibration

**Sniper Strategy**: Profitable betting strategy (+2.2% ROI on real Vegas odds) that filters bets to:
- High confidence only (model probability > 65%)
- Favorites only (implied probability > 50%)
- Exclude negative-ROI weight classes (Heavyweight, Flyweight, Women's Strawweight)

## Development Commands

### Installation
```bash
# Install package in editable mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Running the System
```bash
# Run full pipeline (ingestion → cleaning → ELO → features → training → evaluation → backtest)
python scripts/run_pipeline.py

# Make predictions on hypothetical matchups
python scripts/predict.py --fighter1 "Islam Makhachev" --fighter2 "Dustin Poirier" \
    --weight-class "Lightweight Bout" --odds1 -300 --odds2 +250

# Run profitability backtest (standard strategy)
python src/ufc_predictor/evaluation/backtest.py

# Generate Sniper Zone heatmap visualization
python scripts/plot_sniper_heatmap.py
```

### Code Quality
```bash
# Format code
black src/ scripts/

# Lint code
ruff src/ scripts/
```

### Configuration
- Line length: 100 characters
- Python versions: >=3.9

## Architecture & Data Flow

### Pipeline Stages
The system follows a strict sequential flow to avoid label leakage:

1. **Ingestion** (`src/ufc_predictor/data/ingest.py`)
   - Load raw CSVs from `data/raw/`
   - No transformations, minimal parsing

2. **Cleaning** (`src/ufc_predictor/data/cleaning.py`)
   - Remove duplicates, handle missing values
   - Standardize data types and categorical labels
   - Output to `data/intermediate/`

3. **Validation** (`src/ufc_predictor/data/validation.py`)
   - Check referential integrity between fights and fighters
   - Flag impossible values (negative stats, future dates)
   - Generate validation reports in `output/reports/`

4. **ELO Rating** (`src/ufc_predictor/rating/elo.py`)
   - Process fights chronologically
   - Track `{fighter_id: rating}` dictionary
   - Initialize new fighters at `ELO_START` (1500)
   - Record pre-fight ELO for both fighters: `elo_1`, `elo_2`, `elo_diff`
   - Update ratings after each fight using `K_FACTOR` (32)

5. **Feature Engineering** (`src/ufc_predictor/features/engineering.py`)
   - Build fighter history features using only pre-fight information:
     - Experience (prior fights count)
     - Win rate, recent form (last N fights)
     - Recency (days since last fight)
   - Create matchup features:
     - Physical differences (reach, height, age)
     - Experience and win rate differences
     - Weight class, stance combinations
   - Attach ELO information
   - Output to `data/features/` as `features_train.parquet` and `features_test.parquet`

6. **Model Training** (`src/ufc_predictor/models/train.py`)
   - Use time-based split (train on older fights, test on recent)
   - Train baseline model (logistic regression on `elo_diff` only)
   - Train XGBoost with tuned hyperparameters on all 98 features
   - Wrap with `CalibratedClassifierCV` (isotonic method, 5-fold CV)
   - Save calibrated model as `calibrated_xgboost.joblib`

7. **Evaluation** (`src/ufc_predictor/evaluation/*.py`)
   - Compute metrics: accuracy, Brier score, log loss, ROC-AUC
   - Compare ELO baseline vs ML models side-by-side
   - Generate reports in `output/reports/` and `output/metrics/`

8. **Profitability Backtest** (`src/ufc_predictor/evaluation/backtest.py`)
   - Simulate betting strategy over test period with real Vegas odds
   - Calculate ROI, win rate, total profit/loss
   - Segment analysis by weight class, confidence, favorite/underdog
   - Validate Sniper Strategy filters

9. **Sniper Strategy Inference** (`src/ufc_predictor/inference/predict.py`)
   - Enforce profitable betting rules in production predictions
   - Filter: Model probability > 65%, Favorites only, Exclude HW/Fly/W-Straw
   - CLI provides clear SNIPER BET / PASS signals

10. **Pipeline Orchestration** (`src/ufc_predictor/pipeline/run_pipeline.py`)
   - Wraps stages 1-9 into single callable pipeline
   - Used by CLI scripts

### Directory Structure

```
data/
├── raw/              # Original CSV files (READ-ONLY - never modify programmatically)
├── intermediate/     # Cleaned data (cleaned_fights.parquet, cleaned_fighters.parquet)
└── features/         # ML-ready matrices (features_train.parquet, features_test.parquet)

models/               # Serialized models (elo_baseline.json, logreg_model.pkl, etc.)

output/
├── reports/          # Human-readable reports (Markdown, text, HTML)
├── metrics/          # Machine-readable metrics (JSON/CSV)
└── plots/            # Visualizations

logs/                 # Pipeline execution logs

src/ufc_predictor/    # Main Python package
├── data/             # Ingestion, cleaning, validation
├── rating/           # ELO rating system
├── features/         # Feature engineering
├── models/           # Training and inference
├── evaluation/       # Metrics and comparison tools
└── pipeline/         # End-to-end orchestration
```

## Critical Constraints

### No Label Leakage
**All features must use only pre-fight information.**
- No feature can depend on outcomes of current or future fights
- All aggregates computed from history strictly before fight date
- When computing fighter stats (win rate, recent form, etc.), use only fights that occurred before the current fight

### Data Integrity
- `data/raw/` is **read-only** - never modify programmatically
- Always process fights in chronological order for ELO updates
- Maintain referential integrity between fights and fighters tables

### Module Boundaries
Keep modules single-responsibility:
- Data logic in `data/` and `features/`
- Rating logic in `rating/`
- Training logic in `models/`
- Evaluation logic in `evaluation/`
- Orchestration logic in `pipeline/`

## Key Configuration

Located in `src/ufc_predictor/config.py`:

```python
# ELO parameters
ELO_START = 1500      # Initial rating for new fighters
K_FACTOR = 32         # ELO update rate (higher = more volatile)

# Feature engineering
RECENT_FIGHTS_WINDOW = 5  # Number of recent fights for form features

# Model training
RANDOM_SEED = 42
DEFAULT_MODEL_TYPE = "xgboost"
```

### Sniper Strategy Configuration

Located in `src/ufc_predictor/inference/predict.py` and `src/ufc_predictor/evaluation/backtest.py`:

```python
# Excluded weight classes (negative ROI)
EXCLUDED_WEIGHT_CLASSES = {
    'Heavyweight Bout',
    'Flyweight Bout',
    'Women\'s Strawweight Bout'
}

# Sniper Strategy thresholds
CONFIDENCE_THRESHOLD = 0.65  # Model probability must be > 65%
FAVORITE_THRESHOLD = 2.0     # Decimal odds must be < 2.0 (implied prob > 50%)
EDGE_THRESHOLD = 0.05        # Minimum EV required to place bet (5%)
```

## Core Entities

**Fight**
- `fight_id`, `date`, `fighter1_id`, `fighter2_id`
- Outcome: `1` = fighter1 wins, `0` = fighter2 wins (draws/NC may be dropped)

**Fighter**
- `fighter_id`, `height`, `reach`, `age`, `stance`, `weight_class`

## Evaluation Protocol

- **Time-based split**: Train on fights up to cutoff date, test on fights after
- **Always compare**: ELO baseline vs ML model on same test set
- **Store all experiments** with metadata: date, data version, model type, hyperparameters, feature set, full metrics

## Models & Performance

### Current Production Model
- **Model**: Calibrated XGBoost (`calibrated_xgboost.joblib`)
- **Features**: 98 engineered features including:
  - Physical attributes (height, reach, age differences)
  - Fighter history (win rate, streak, finish rate, KO ratio)
  - Experience metrics (total fights, recency, avg opponent ELO)
  - ELO ratings (pre-fight, difference)
  - Weight class and stance matchup (one-hot encoded)
- **Calibration**: Isotonic calibration via 5-fold cross-validation
- **Test Set Performance**:
  - Accuracy: 61.3%
  - Log Loss: 0.656
  - Brier Score: 0.216

### Backtest Results (Real Vegas Odds)

**Unfiltered Strategy:**
- Total Bets: 865
- Win Rate: 38.4%
- ROI: -4.9% (Market Neutral - close to break-even)

**Sniper Strategy (Filtered):**
- Total Bets: 157 (82% reduction)
- Win Rate: 63.1%
- **ROI: +2.2%** ✅ **Positive edge against Vegas**

**Sniper Strategy Filters:**
1. Model Probability > 65% (High Confidence: +16.9% ROI)
2. Betting on Favorites (Implied Prob > 50%: +7.7% ROI)
3. Exclude: Heavyweight (-7.2%), Flyweight (-11.9%), Women's Strawweight (-9.0%)

## Future Possibilities

- Predict method of victory (KO/TKO, submission, decision)
- Time-decayed ELO (recent fights weighted more heavily)
- Style embeddings based on per-fight stats
- Simulation of entire cards or hypothetical matchups

## Related Documentation

- [PROJECT_CONTEXT_UFC.md](docs/PROJECT_CONTEXT_UFC.md) - Detailed project context and design decisions
- [README.md](README.md) - Project overview and quick start
