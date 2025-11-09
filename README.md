# Credit Card Fraud Detection - MLOps Pipeline

Production-ready machine learning pipeline for detecting fraudulent credit card transactions using MLOps best practices.

## Project Overview

**Objective**: Binary classification of credit card transactions as legitimate or fraudulent.

**Dataset**: `data/creditcard.csv` (~151 MB, 284,807 transactions)
- **Target variable**: `Class` (0 = legitimate, 1 = fraud)
- **Features**: Time, V1-V28 (PCA-transformed), Amount
- **Class imbalance**: ~0.17% fraud rate (492 fraudulent out of 284,807 total)

**Evaluation Metrics**: F1-score, Precision, Recall, PR-AUC, Confusion Matrix

**Compliance**: PCI DSS and GDPR requirements enforced throughout the pipeline.

## Requirements

- **Local**: macOS (M3 compatible) or Ubuntu ≥21
- **Python**: ≥3.11
- **Package manager**: [uv](https://github.com/astral-sh/uv) (for reproducible environments)
- **Deployment**: Docker, AWS EC2, GitHub Codespaces

## Quick Start

### 1. Setup Environment

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
cd card-fraud

# Create virtual environment and install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### 2. Run Exploratory Data Analysis

```bash
# Launch the marimo EDA notebook
marimo edit notebooks/eda_creditcard.py
```

This will open an interactive notebook in your browser with comprehensive data exploration:
- Dataset overview and quality checks
- Class distribution analysis
- Feature correlation analysis
- Visualization of Amount and V-features by class
- Modeling recommendations

**Alternative**: Run the notebook as a script (non-interactive):

```bash
marimo run notebooks/eda_creditcard.py
```

### 3. Development Workflow

```bash
# Run main application
python main.py

# Run tests (when available)
pytest tests/

# Type checking (when configured)
mypy src/

# Linting (when configured)
ruff check src/
```

## Project Structure

```
card-fraud/
├── data/
│   └── creditcard.csv          # Training dataset (>100 MB, not in git)
├── notebooks/
│   └── eda_creditcard.py       # Marimo EDA notebook (mandatory phase)
├── src/                        # (to be created) Pipeline modules
│   ├── data/                   # Data ingestion and preprocessing
│   ├── features/               # Feature engineering
│   ├── models/                 # Model training and evaluation
│   └── serving/                # Inference and API
├── tests/                      # (to be created) Unit and integration tests
├── .github/
│   └── workflows/              # (to be created) CI/CD pipelines
├── docker/                     # (to be created) Dockerfiles
├── monitoring/                 # (to be created) Prometheus/Grafana configs
├── pyproject.toml              # Project dependencies
├── uv.lock                     # Locked dependency versions
└── README.md                   # This file
```

## MLOps Pipeline Architecture

### Data Strategy
- **Storage**: Local CSV (development), S3 (production)
- **Versioning**: DVC or timestamped snapshots
- **Preprocessing**: Stratified splits, class weighting, feature engineering
- **Compliance**: Encrypted at rest and in transit, access-controlled

### Model Lifecycle
- **Training**: Automated pipeline with experiment tracking (MLflow)
- **Evaluation**: Stratified K-fold CV, holdout test with F1/PR-AUC metrics
- **Registry**: Model versioning with metadata (S3 or MLflow)
- **CI/CD**: Automated testing, building, and deployment

### Deployment
- **Mode**: Batch inference (local or scheduled)
- **Infrastructure**: Docker containers on AWS EC2 (Ubuntu ≥21)
- **Scalability**: Horizontal scaling with load balancing (if real-time needed)

### Monitoring
- **Metrics**: Model performance (F1, drift), data quality, system health
- **Tools**: Prometheus (metrics collection), Grafana (dashboards)
- **Alerts**: Automated notifications for performance degradation

### Retraining
- **Trigger**: Scheduled (monthly) or performance-based
- **Validation**: A/B testing or shadow deployment
- **Rollback**: Version-controlled model artifacts

## Key Findings from EDA

1. **Class Imbalance**: 0.17% fraud rate requires:
   - Stratified sampling
   - Class-weighted models
   - Threshold tuning for precision/recall balance

2. **Feature Engineering Opportunities**:
   - Time-based features (hour, day patterns)
   - Amount transformations (log1p, z-score)
   - Interaction features

3. **Recommended Models**:
   - Baseline: Logistic Regression with class_weight='balanced'
   - Advanced: XGBoost, LightGBM, Random Forest
   - Anomaly detection: Isolation Forest, Autoencoders

4. **Compliance Requirements**:
   - V1-V28 already anonymized (PCA-transformed)
   - No raw cardholder data (PAN, CVV) stored or logged
   - Access controls and audit trails enforced

## Environment Management

This project uses `uv` for reproducible Python environments:

```bash
# Install dependencies (creates .venv and uv.lock)
uv sync

# Add a new dependency
uv add <package-name>

# Update dependencies
uv sync --upgrade

# In CI/CD, use locked dependencies
uv sync --locked
```

**Benefits**:
- Deterministic builds (uv.lock committed to git)
- Fast dependency resolution
- M3 Mac and Linux compatibility
- Easy migration to GitHub Codespaces or AWS EC2

## GitHub Codespaces Setup

To run this project in Codespaces:

1. Open the repository in GitHub
2. Click "Code" → "Create codespace on main"
3. Wait for container to build
4. Run setup commands:

```bash
uv sync
marimo edit notebooks/eda_creditcard.py
```

## Contributing

1. Follow PEP 8 style guidelines
2. Add type hints for all functions
3. Write unit tests for new features
4. Update documentation
5. Use atomic, descriptive git commits

## License

(Add your license here)

## Contact

(Add contact information here)
