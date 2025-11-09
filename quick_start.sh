#!/usr/bin/env bash
set -e

echo "Card Fraud Detection - Quick Start"
echo "==================================="
echo ""

# Check if data file exists
if [ ! -f "data/creditcard.csv" ]; then
    echo "Error: data/creditcard.csv not found"
    echo "Please place the dataset in the data/ directory"
    exit 1
fi

echo "Step 1: Training baseline model (Logistic Regression)..."
uv run python train.py \
    --data data/creditcard.csv \
    --output-dir models/baseline_logistic \
    --model-type logistic \
    --optimize-threshold

echo ""
echo "Step 2: Training XGBoost model..."
uv run python train.py \
    --data data/creditcard.csv \
    --output-dir models/xgboost_v1 \
    --model-type xgboost \
    --optimize-threshold \
    --skip-splits

echo ""
echo "Step 3: Running batch inference on test set..."
uv run python -m src.inference.batch \
    --model-dir models/xgboost_v1 \
    --preprocessing models/xgboost_v1/preprocessing_pipeline.pkl \
    --input data/splits/test.csv \
    --output predictions/test_predictions.csv

echo ""
echo "Quick start complete!"
echo ""
echo "Results:"
echo "  - Models saved to: models/"
echo "  - Splits saved to: data/splits/"
echo "  - Predictions saved to: predictions/"
echo ""
echo "Next steps:"
echo "  - Review model metrics in models/*/model_info.json"
echo "  - Compare models using MLflow (optional)"
echo "  - Deploy best model using Docker"
