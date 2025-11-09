# Usage Guide

## Training Models

### Basic Training

Train a baseline logistic regression model:

```bash
uv run python train.py \
    --data data/creditcard.csv \
    --output-dir models/baseline_logistic \
    --model-type logistic
```

### Train XGBoost with Threshold Optimization

```bash
uv run python train.py \
    --data data/creditcard.csv \
    --output-dir models/xgboost_v1 \
    --model-type xgboost \
    --optimize-threshold
```

### Train LightGBM

```bash
uv run python train.py \
    --data data/creditcard.csv \
    --output-dir models/lightgbm_v1 \
    --model-type lightgbm \
    --optimize-threshold
```

### Reuse Existing Splits

If you've already created train/val/test splits, skip recreation:

```bash
uv run python train.py \
    --data data/creditcard.csv \
    --output-dir models/xgboost_v2 \
    --model-type xgboost \
    --skip-splits
```

## Batch Inference

Run predictions on a CSV file:

```bash
uv run python -m src.inference.batch \
    --model-dir models/xgboost_v1 \
    --preprocessing models/xgboost_v1/preprocessing_pipeline.pkl \
    --input data/splits/test.csv \
    --output predictions/test_predictions.csv
```

With custom threshold:

```bash
uv run python -m src.inference.batch \
    --model-dir models/xgboost_v1 \
    --preprocessing models/xgboost_v1/preprocessing_pipeline.pkl \
    --input data/new_transactions.csv \
    --output predictions/predictions.csv \
    --threshold 0.3
```

## Running Tests

Run all tests:

```bash
uv run pytest tests/ -v
```

With coverage report:

```bash
uv run pytest tests/ -v --cov=src --cov-report=html
open htmlcov/index.html
```

Run specific test file:

```bash
uv run pytest tests/test_preprocessing.py -v
```

## Docker Usage

### Build Docker Image

```bash
docker build -t card-fraud:latest .
```

### Run Training in Container

```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
    card-fraud:latest \
    uv run python train.py --data /app/data/creditcard.csv --output-dir /app/models/docker_run
```

### Run Inference in Container

```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models -v $(pwd)/predictions:/app/predictions \
    card-fraud:latest \
    uv run python -m src.inference.batch \
    --model-dir /app/models/xgboost_v1 \
    --preprocessing /app/models/xgboost_v1/preprocessing_pipeline.pkl \
    --input /app/data/splits/test.csv \
    --output /app/predictions/docker_predictions.csv
```

## EDA Notebook

### Interactive Mode (Browser)

```bash
marimo edit notebooks/eda_creditcard.py
```

This opens an interactive notebook in your browser where you can:
- Explore data distributions
- Analyze class imbalance
- View correlation heatmaps
- Examine feature importance
- Generate visualizations

### Non-Interactive Mode (Script)

Run the notebook as a Python script:

```bash
marimo run notebooks/eda_creditcard.py
```

## Model Comparison

Compare model performance by checking `model_info.json` files:

```bash
# View baseline model metrics
cat models/baseline_logistic/model_info.json | jq '.metrics'

# View XGBoost model metrics
cat models/xgboost_v1/model_info.json | jq '.metrics'

# Compare F1 scores
echo "Baseline F1: $(cat models/baseline_logistic/model_info.json | jq '.metrics.test.f1_score')"
echo "XGBoost F1:  $(cat models/xgboost_v1/model_info.json | jq '.metrics.test.f1_score')"
```

## Hyperparameter Tuning (Manual)

Create a tuning script or use grid search:

```python
from src.models.train import FraudDetectionModel, evaluate_model
from src.data.preprocessing import preprocess_data, create_preprocessing_pipeline
import pandas as pd

# Load data
train_df = pd.read_csv('data/splits/train.csv')
val_df = pd.read_csv('data/splits/val.csv')

# Preprocess
pipeline = create_preprocessing_pipeline()
X_train, pipeline = preprocess_data(train_df, pipeline, fit=True)
y_train = train_df['Class']
X_val, _ = preprocess_data(val_df, pipeline, fit=False)
y_val = val_df['Class']

# Hyperparameter grid
learning_rates = [0.01, 0.05, 0.1]
max_depths = [3, 5, 7]

best_f1 = 0
best_params = {}

for lr in learning_rates:
    for depth in max_depths:
        model = FraudDetectionModel(
            model_type='xgboost',
            hyperparams={'learning_rate': lr, 'max_depth': depth}
        )
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_val, y_val)
        
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_params = {'learning_rate': lr, 'max_depth': depth}

print(f"Best F1: {best_f1:.4f}")
print(f"Best params: {best_params}")
```

## CI/CD

GitHub Actions will automatically:
- Run tests on every push
- Lint code with ruff
- Type check with mypy

View workflow results at: `.github/workflows/ci.yml`

## Tips

1. **Always use locked dependencies**: `uv sync --locked`
2. **Version control models**: Save git commit hash in `model_info.json`
3. **Monitor data drift**: Check feature distributions in production
4. **Threshold tuning**: Optimize for business metrics (precision vs recall tradeoff)
5. **Stratified splits**: Always use stratified sampling for imbalanced data
