#!/usr/bin/env bash
# run_experiment.sh — Full CLR POC pipeline. Run setup_lambda.sh first.
set -euo pipefail

# JN: This is to sidestep some lambda.ai issues
export USE_TF=0

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "=== CLR POC — Full Pipeline ==="

echo "[1/6] Preparing training data..."
python data_prep.py
python format_for_training.py

echo "[2/6] Training: helpful_only..."
python train.py --condition helpful_only

echo "[3/6] Training: mixed_inconsistent..."
python train.py --condition mixed_inconsistent

echo "[4/6] Quick behavioral check..."
python quick_check.py

echo "[5/6] Generating responses for all probes..."
python generate_responses.py

echo "[6/6] Evaluating & visualizing..."
python evaluate.py
python visualize.py

echo ""
echo "=== Pipeline complete ==="
echo "Results in eval/"
