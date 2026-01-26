# UFC Predictor — Project Context

## 1. Project Overview

This project builds a UFC fight prediction system with two main goals:

1. **Baseline rating model**
   - Implement a transparent, ELO-style skill rating system for fighters.
   - Use pre-fight ELO ratings to generate win probabilities.
   - Treat this as the minimum acceptable benchmark for predictive performance.

2. **Machine learning model on top of rich features**
   - Engineer features from fighter history, physical attributes, and matchup context.
   - Train ML models to predict fight outcomes and compare them directly to the ELO baseline.
   - Evaluate both accuracy and probability quality (calibration, Brier score, etc.).

The system should be **reproducible, modular, and extensible**, mirroring the structure and discipline of the existing NFL predictor project.

---

## 2. Data & Directory Layout

All data lives under `ufc_predictor/data` (or `./data` at repo root, depending on environment):

- `data/raw/`
  - Raw CSV files as obtained from external sources (UFC stats scrapers, public datasets, etc.).
  - These files are treated as **read-only**; never overwritten programmatically.

- `data/intermediate/`
  - Cleaned and validated versions of raw data:
    - `cleaned_fights.parquet` or `.csv` — fight-level history.
    - `cleaned_fighters.parquet` or `.csv` — fighter metadata (height, reach, stance, etc.).
  - These are the inputs to feature engineering.

- `data/features/`
  - ML-ready feature matrices:
    - `features_train.parquet`
    - `features_test.parquet`
  - Each row represents a single fight with pre-fight features and a target label.

Other key directories:

- `models/` — serialized models (baseline and ML), e.g. `elo_baseline.json`, `logreg_model.pkl`.
- `output/reports/` — human-readable reports (Markdown, text, HTML) summarizing runs and metrics.
- `output/metrics/` — machine-readable metrics (JSON/CSV) for experiments.
- `logs/` — logs for pipeline runs and experiments.

---

## 3. Core Entities & Definitions

- **Fight**
  A single bout between two fighters, identified by:
  - `fight_id` (if available)
  - `date`
  - `fighter1_id`, `fighter2_id`
  - outcome (e.g. `fighter1_win`, `fighter2_win`, optionally draw/NC)

- **Fighter**
  A person who appears in fights, with attributes such as:
  - height, reach, age at fight time (derived)
  - stance
  - weight class

- **Label / Target**
  For now, the primary target is **binary outcome**:
  - `1` = fighter1 wins
  - `0` = fighter2 wins
  Draws and no contests may be dropped or handled separately.

- **Pre-fight information only**
  All features used for modeling must reflect the state **before the fight happens** to avoid label leakage.

---

## 4. Pipeline Stages & Corresponding Modules

The end-to-end flow is:

1. **Ingestion (`src/ufc_predictor/data/ingest.py`)**
   - Load raw CSVs from `data/raw/` into pandas DataFrames.
   - No transformation beyond the minimum required to read the files.

2. **Cleaning (`src/ufc_predictor/data/cleaning.py`)**
   - Remove duplicates.
   - Handle missing values (drop or impute depending on column criticality).
   - Standardize data types (dates, numeric vs categorical).
   - Normalize categorical labels (e.g. weight class, stance).
   - Output: cleaned fights and fighters tables → `data/intermediate/`.

3. **Validation (`src/ufc_predictor/data/validation.py`)**
   - Check referential integrity between fights and fighters.
   - Flag impossible values (ages, heights, negative stats, future dates).
   - Summarize anomalies in a validation report under `output/reports/`.
   - The goal is to fail fast if data is broken.

4. **ELO Rating (`src/ufc_predictor/rating/elo.py`)**
   - Maintain a rating dictionary `{fighter_id: rating}`.
   - Initialize new fighters at `ELO_START` (e.g. 1500).
   - Iterate fights in chronological order and update ratings after each result.
   - Record pre-fight ratings for both fighters and store them with fight data.
   - Output: fight-level table augmented with `elo_1`, `elo_2`, `elo_diff`.

5. **Feature Engineering (`src/ufc_predictor/features/engineering.py`)**
   - Build fighter-history features (using only history before each fight):
     - prior fights count (experience)
     - prior wins/losses, win rate
     - recent form (e.g. last N fights)
     - recency (days since last fight)
   - Add matchup features:
     - differences in reach, height, age
     - differences in experience and win rate
     - weight class, stance combinations (encoded)
   - Attach ELO information (elo_1, elo_2, elo_diff).
   - Produce:
     - `X` (feature matrix)
     - `y` (fight outcome)
     saved to `data/features/`.

6. **Model Training (`src/ufc_predictor/models/train.py`)**
   - Establish a **time-based split**: train on older fights, test on more recent fights.
   - Train:
     - Baseline classification model (e.g. logistic regression) using only `elo_diff`.
     - Rich-feature models (e.g. logistic regression, random forest, gradient boosting).
   - Save models in `models/`.

7. **Evaluation & Comparison (`src/ufc_predictor/evaluation/*.py`)**
   - Compute metrics for:
     - ELO baseline (as a probability model, if calibrated).
     - ML models.
   - Metrics:
     - Accuracy
     - Brier score (probability quality)
     - Log loss
     - ROC-AUC (optional)
   - Generate comparison reports:
     - ELO vs ML performance
     - Calibration insights
     - Upset prediction behavior

8. **Pipeline Orchestration (`src/ufc_predictor/pipeline/run_pipeline.py`)**
   - Wrap the above into a single callable pipeline:
     - Optionally parametrized by date range, model type, etc.
   - Used by CLI scripts in `scripts/`.

---

## 5. Modeling Goals & Priorities

Short-term priorities:

1. **Get a robust ELO system running**
   - Ratings that evolve sensibly over time.
   - Simple mapping from ELO difference → win probability.
   - Basic evaluation of ELO-only predictions.

2. **Add a simple but strong ML baseline**
   - Logistic regression with:
     - ELO difference
     - A small set of matchup features (reach, age, experience differences).
   - Compare vs ELO-only.

3. **Iterate on features & models**
   - Add more nuanced fighter history features.
   - Try different model classes (random forest, gradient boosting).
   - Focus on improving **probability calibration**, not just accuracy.

Longer-term possibilities:

- Predict method of victory (KO/TKO, submission, decision).
- Time-decayed ELO (recent fights weighted more heavily).
- Style embeddings based on per-fight stats.
- Simulation of entire cards or hypothetical matchups.

---

## 6. Evaluation Protocol

- Use a **time-based split**:
  - Train on fights up to a chosen cutoff date/year.
  - Test on fights strictly after that cutoff.
  - This simulates real forward-looking prediction.

- Always compare:
  - **ELO baseline** vs **ML model** on the same test set.
  - Report metrics side-by-side.

- Store all experiment results under `output/metrics/` and `output/reports/` with:
  - Date of run
  - Data version (e.g. hash or timestamp of raw data)
  - Model type + hyperparameters
  - Feature set description
  - Full metric dictionary

---

## 7. Implementation Guidelines

- Keep modules **single-responsibility**:
  - Data logic in `data/` and `features/`.
  - Rating logic in `rating/`.
  - Training logic in `models/`.
  - Evaluation logic in `evaluation/`.
  - Orchestration logic in `pipeline/`.

- Avoid label leakage:
  - No feature should depend on outcomes of the current or future fights.
  - All aggregates must be computed from history strictly before the fight date.

- Favor readability over cleverness:
  - Clear, explicit transforms.
  - Extensive docstrings and comments, especially around ELO updates and feature construction.

- Maintain parity with the NFL predictor where it makes sense:
  - Similar metric naming/structure.
  - Similar folder naming conventions.
  - This makes cross-sport comparisons and code reuse easier.

---

## 8. How Future Agents Should Use This Context

Any future agent working on this repo should:

1. Read this file first to understand:
   - Where data lives
   - What each module should do
   - How evaluation is structured

2. Respect the boundaries:
   - Don't mix concerns (e.g. don't compute features inside model training).
   - Don't modify `data/raw/` programmatically.

3. When adding or changing behavior:
   - Update this context file if the high-level design changes.
   - Keep directory layout and naming consistent unless there is a strong reason to refactor.

This document is the source of truth for how the UFC predictor is structured and what it's trying to achieve.
