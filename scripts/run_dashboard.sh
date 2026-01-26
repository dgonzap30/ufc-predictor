#!/bin/bash
# Launch the UFC Sniper Dashboard

cd "$(dirname "$0")/.."
source venv/bin/activate

# Install package in editable mode if not already installed
pip install -e . > /dev/null 2>&1

streamlit run src/ufc_predictor/app/Home.py
