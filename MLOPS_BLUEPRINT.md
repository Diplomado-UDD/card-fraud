# Machine Learning Pipeline Blueprint
## Credit Card Fraud Detection

**Date**: November 9, 2025  
**Version**: 1.0  
**Status**: Production-Ready Design

---

## User Objective

Develop a production-ready machine learning system to detect fraudulent credit card transactions with high precision and recall, minimizing financial losses while maintaining customer experience. The system must process transactions in batch mode, comply with PCI DSS and GDPR regulations, and operate reliably across local development (macOS M3), GitHub Codespaces, and AWS EC2 (Ubuntu ≥21) production environments.

**Success Criteria:**
- **Primary Metric**: F1-score ≥ 0.75 on holdout test set (balancing precision and recall)
- **Business Metrics**: 
  - Precision ≥ 0.80 (limit false positives to avoid customer friction)
  - Recall ≥ 0.70 (catch at least 70% of fraudulent transactions)
  - PR-AUC ≥ 0.70 (account for class imbalance)
- **Operational Requirements**:
  - Batch inference processing within 1 hour for daily transaction batches
  - Model retraining pipeline completes within 4 hours
  - System uptime ≥ 99.5%
  - Model performance monitoring with automated alerts for drift detection

**Business Impact:**
- Reduce fraudulent transaction losses by 60-70%
- Decrease false positive rate (improving customer experience)
- Enable proactive fraud prevention through pattern detection
- Maintain regulatory compliance (PCI DSS, GDPR Article 22)

---

## ML Problem Type

**Problem Formulation**: Supervised binary classification

**Target Variable**: `Class`
- 0 = Legitimate transaction
- 1 = Fraudulent transaction

**Input Features**: 31 columns
- `Time`: Seconds elapsed since first transaction (numeric)
- `V1-V28`: PCA-transformed features (anonymized, numeric, pre-scaled)
- `Amount`: Transaction amount in currency units (numeric, right-skewed)

**Dataset Characteristics**:
- **Size**: 284,807 transactions (~151 MB)
- **Class Distribution**: 
  - Legitimate (0): 284,315 (99.83%)
  - Fraud (1): 492 (0.17%)
  - **Imbalance Ratio**: 578:1
- **Data Quality**: No missing values, minimal duplicates
- **Privacy**: Features already anonymized via PCA (PCI/GDPR compliant)

**Evaluation Metrics** (in priority order):
1. **F1-Score**: Primary metric balancing precision and recall
2. **Precision-Recall AUC (PR-AUC)**: Better than ROC-AUC for imbalanced data
3. **Precision & Recall**: Monitor individually for threshold tuning
4. **Confusion Matrix**: Track false positives vs. false negatives
5. **Precision@k**: Operational metric (e.g., top 1000 predicted frauds)

**Rationale**: Given the 578:1 imbalance, accuracy is misleading (predicting all 0's yields 99.83% accuracy but 0% fraud detection). F1-score and PR-AUC are appropriate for this use case. Business context requires balancing precision (avoid flagging legitimate transactions) with recall (catch actual fraud).

**Model Considerations**:
- Class weighting or resampling required
- Stratified train/validation/test splits mandatory
- Threshold optimization for precision/recall tradeoff
- Time-based validation split if temporal ordering matters

---

## Data Strategy

### Data Sources and Storage

**Development Environment:**
- **Location**: Local filesystem (`data/creditcard.csv`)
- **Size**: 151 MB (fits in memory on 16GB RAM)
- **Access**: Direct file I/O with pandas

**Production Environment:**
- **Storage**: AWS S3 bucket with versioning enabled
  - Raw data: `s3://card-fraud-prod/data/raw/creditcard_{timestamp}.csv`
  - Processed data: `s3://card-fraud-prod/data/processed/`
  - Model artifacts: `s3://card-fraud-prod/models/`
- **Security**: 
  - Encryption at rest (S3 SSE-KMS)
  - Encryption in transit (TLS 1.2+)
  - IAM roles with least-privilege access
  - Bucket policies preventing public access
- **Backup**: Cross-region replication for disaster recovery

### Data Pipeline Architecture

**Stage 1: Data Ingestion**
- **Trigger**: Scheduled daily batch (cron or Apache Airflow)
- **Process**:
  1. Download raw transaction CSV from source system or S3
  2. Validate schema (31 columns, expected dtypes)
  3. Compute dataset hash (SHA256) for change detection
  4. Check row count and Class distribution bounds
  5. Store raw data with timestamp in S3 (immutable)
- **Tools**: Python + boto3 (S3), pandas (validation)
- **Failure Handling**: Retry with exponential backoff, alert on repeated failures

**Stage 2: Data Preprocessing**
```python
# Preprocessing steps (deterministic, version-controlled)
1. Validate no missing values (fail if any)
2. Remove exact duplicates (log count)
3. Feature engineering:
   - log1p_Amount = log(1 + Amount)  # Handle skewness
   - Amount_zscore = (Amount - mean) / std
   - Hour_of_day = (Time % 86400) / 3600
   - Hour_sin = sin(2π * Hour_of_day / 24)  # Cyclical encoding
   - Hour_cos = cos(2π * Hour_of_day / 24)
4. Preserve original V1-V28 (already PCA-scaled)
5. Create feature set: V1-V28, Amount, log1p_Amount, Hour_sin, Hour_cos
```

- **Implementation**: Modular pipeline using scikit-learn `Pipeline` and custom transformers
- **Serialization**: Save fitted preprocessing pipeline as `preprocessing_pipeline.pkl` (versioned with model)
- **Validation**: Unit tests for each transformer, integration test on sample data

**Stage 3: Train/Validation/Test Splits**
- **Strategy**: Stratified random split (maintains 0.17% fraud rate in each set)
- **Split Ratios**:
  - Train: 70% (199,365 transactions)
  - Validation: 15% (42,721 transactions)
  - Test: 15% (42,721 transactions)
- **Holdout Test Set**: Locked away until final model evaluation (no peeking)
- **Alternative**: Time-based split if transaction ordering is meaningful (prevent data leakage)
  - Example: Train on first 70% by Time, validate on next 15%, test on final 15%

**Stage 4: Data Versioning**
- **Tool**: DVC (Data Version Control) or timestamped S3 paths
- **Versioning Strategy**:
  - Each training run references specific data version (SHA256 hash)
  - Store metadata: {data_version, row_count, fraud_count, date_created}
  - Track lineage: raw → processed → splits
- **Reproducibility**: Lock data version in experiment tracking (MLflow)

**Compliance (PCI DSS & GDPR)**:
- V1-V28 already anonymized (no cardholder data)
- Do NOT log or store: PAN, CVV, cardholder names, raw transaction details
- Access controls: Role-based access to S3 buckets and model artifacts
- Audit trail: CloudTrail logs all S3 and model access
- Data retention: Define and enforce retention policy (e.g., 7 years for compliance)
- Right to explanation: Implement SHAP or LIME for model interpretability (GDPR Article 22)

---

## Model Lifecycle

### Training Pipeline

**Orchestration**: Apache Airflow or simple Python scripts (start simple, scale as needed)

**Pipeline Stages**:

1. **Environment Setup**
   - Use `uv sync --locked` to recreate exact environment from `uv.lock`
   - Validate Python 3.11+, GPU availability (if applicable)

2. **Data Loading**
   - Load versioned data from S3 or local filesystem
   - Validate schema and apply preprocessing pipeline
   - Generate stratified train/val splits (or load pre-split data)

3. **Model Training**
   - **Baseline Model**: Logistic Regression with `class_weight='balanced'`
     ```python
     from sklearn.linear_model import LogisticRegression
     model = LogisticRegression(
         class_weight='balanced',
         max_iter=1000,
         random_state=42,
         solver='saga',
         n_jobs=-1
     )
     ```
   
   - **Advanced Models**:
     - **XGBoost**: `scale_pos_weight=578` (imbalance ratio)
     - **LightGBM**: `is_unbalance=True`, `metric='auc'`
     - **Random Forest**: `class_weight='balanced_subsample'`
   
   - **Hyperparameter Tuning**:
     - Stratified 5-fold cross-validation on training set
     - Grid search or Bayesian optimization (Optuna)
     - Optimize for F1-score (not accuracy)
     - Search space examples:
       - XGBoost: learning_rate [0.01, 0.1], max_depth [3, 7, 10], n_estimators [100, 500]
       - LightGBM: num_leaves [31, 63, 127], learning_rate [0.01, 0.05, 0.1]

4. **Model Evaluation**
   - **Validation Set Metrics**:
     - F1-score, Precision, Recall, PR-AUC
     - Confusion matrix (TP, FP, TN, FN)
     - Precision-Recall curve
     - ROC curve (for comparison only)
   
   - **Threshold Optimization**:
     - Default threshold: 0.5
     - Tune threshold on validation set to maximize F1 or business metric
     - Example: Find threshold where Precision ≥ 0.80 and Recall is maximized
   
   - **Feature Importance**:
     - SHAP values for model interpretability
     - Log top 10 most important features

5. **Model Acceptance Criteria**
   - F1-score ≥ 0.75 on validation set
   - Precision ≥ 0.80, Recall ≥ 0.70
   - PR-AUC ≥ 0.70
   - No significant performance gap between train and validation (detect overfitting)
   - If criteria not met: iterate on features, models, or hyperparameters

### Experiment Tracking

**Tool**: MLflow (open-source, integrates with S3 and local storage)

**Tracked Artifacts** (per training run):
- **Parameters**: Model hyperparameters, preprocessing config, random seed
- **Metrics**: F1, Precision, Recall, PR-AUC, ROC-AUC (train and val)
- **Artifacts**:
  - Trained model file (`.pkl`, `.joblib`, or `.ubj` for LightGBM)
  - Preprocessing pipeline (`.pkl`)
  - Feature importance plot
  - Confusion matrix plot
  - Precision-Recall curve
- **Metadata**:
  - Data version (SHA256 hash)
  - Training duration
  - Environment (uv.lock hash, Python version)
  - Git commit hash (code version)

**MLflow Setup**:
```bash
# Local tracking
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

# Production: PostgreSQL backend + S3 artifacts
mlflow server \
  --backend-store-uri postgresql://user:pass@host/mlflow \
  --default-artifact-root s3://card-fraud-prod/mlruns
```

**Benefits**:
- Compare experiments side-by-side
- Reproducibility: rerun any experiment with same parameters/data
- Rollback: Quickly revert to previous model version

### Model Registry

**Tool**: MLflow Model Registry or AWS SageMaker Model Registry

**Model Lifecycle Stages**:
1. **None**: Newly trained model (not yet evaluated)
2. **Staging**: Passed acceptance criteria, deployed to staging environment
3. **Production**: Validated in staging, serving live traffic
4. **Archived**: Superseded by newer model, kept for rollback

**Versioning Strategy**:
- Each model version has unique ID (e.g., `fraud_detector_v1.2.3`)
- Semantic versioning: `MAJOR.MINOR.PATCH`
  - MAJOR: Breaking changes (new features, architecture change)
  - MINOR: Performance improvement (retrained with new data)
  - PATCH: Bug fix (preprocessing correction, no retrain)

**Model Metadata** (stored with each version):
```json
{
  "model_id": "fraud_detector_v1.2.0",
  "created_at": "2025-11-09T10:30:00Z",
  "algorithm": "XGBoost",
  "data_version": "sha256:abc123...",
  "metrics": {
    "f1_score": 0.78,
    "precision": 0.82,
    "recall": 0.74,
    "pr_auc": 0.75
  },
  "hyperparameters": {...},
  "training_duration_sec": 1234,
  "git_commit": "a1b2c3d4",
  "uv_lock_hash": "sha256:def456..."
}
```

**Promotion Workflow**:
1. Model trained → MLflow logs experiment
2. If acceptance criteria met → Promote to "Staging"
3. Deploy to staging environment → Run validation tests
4. If staging tests pass → Promote to "Production"
5. Deploy to production → Monitor performance
6. If production metrics degrade → Rollback to previous "Production" model

### Reproducibility

**Key Practices**:
1. **Dependency Locking**: Commit `uv.lock` to git, use `uv sync --locked` in CI/CD
2. **Random Seeds**: Fix seeds in train scripts (numpy, sklearn, xgboost)
   ```python
   np.random.seed(42)
   random.seed(42)
   model = XGBClassifier(random_state=42)
   ```
3. **Data Versioning**: Track data hash or DVC version in MLflow
4. **Code Versioning**: Log git commit hash in MLflow
5. **Container Images**: Use Docker with pinned base images (e.g., `python:3.11.14-slim`)

---

## Deployment Architecture

### Inference Mode

**Primary Mode**: Batch Inference (as per user requirement)

**Use Case**: Process daily batches of transactions (e.g., overnight batch or hourly)

**Workflow**:
1. New transactions arrive in S3 bucket (`s3://card-fraud-prod/data/incoming/`)
2. Airflow DAG triggers batch inference job (scheduled or file-watch trigger)
3. Inference service:
   - Loads production model from S3 (cached locally)
   - Applies preprocessing pipeline
   - Predicts fraud probability for each transaction
   - Applies optimized threshold to classify (0/1)
   - Outputs predictions to S3 (`s3://card-fraud-prod/predictions/`)
4. Downstream systems consume predictions (e.g., fraud review queue)

**Optional Future Enhancement**: Real-time API for point-of-sale fraud detection
- REST API (FastAPI) behind load balancer
- Sub-300ms latency requirement
- Horizontal scaling with Kubernetes

### Infrastructure

**Development (Local + GitHub Codespaces)**:
- **Environment**: macOS M3 (16GB RAM) or Codespaces (4-core, 8GB)
- **Setup**: `uv sync` → virtual environment with locked dependencies
- **Execution**: Run scripts directly (`python train.py`, `python predict.py`)

**Production (AWS EC2)**:
- **Instance Type**: c6i.2xlarge (8 vCPU, 16 GB RAM) or t3.xlarge for cost optimization
- **OS**: Ubuntu ≥21.04 LTS
- **Deployment**: Docker containers orchestrated by systemd or lightweight Kubernetes (k3s)

**Containerization (Docker)**:

**Dockerfile**:
```dockerfile
FROM python:3.11.14-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (locked)
RUN uv sync --locked --no-dev

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Set entrypoint
ENTRYPOINT ["uv", "run", "python", "-m", "src.inference.batch"]
```

**Image Registry**: DockerHub (`username/card-fraud:v1.2.0`)

**Container Orchestration**:
- **Simple**: Docker Compose + systemd (for single-node deployment)
- **Scalable**: Kubernetes (if horizontal scaling needed)
  - Deployment with 2-3 replicas
  - ConfigMap for environment variables
  - Secret for AWS credentials (or use IAM roles)

### CI/CD Pipeline

**Tool**: GitHub Actions (or GitLab CI, Jenkins)

**Workflow Stages**:

**1. Continuous Integration (on push to `main` or PR)**:
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Install dependencies
        run: uv sync --locked
      - name: Run tests
        run: uv run pytest tests/ --cov=src
      - name: Lint code
        run: uv run ruff check src/
      - name: Type check
        run: uv run mypy src/
```

**2. Model Training (on schedule or manual trigger)**:
```yaml
# .github/workflows/train.yml
name: Train Model
on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday at 2 AM
  workflow_dispatch:  # Manual trigger
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Install dependencies
        run: uv sync --locked
      - name: Download data from S3
        run: aws s3 cp s3://card-fraud-prod/data/raw/latest.csv data/
      - name: Train model
        run: uv run python src/train.py
      - name: Upload model to S3
        run: aws s3 cp models/ s3://card-fraud-prod/models/ --recursive
```

**3. Continuous Deployment (on model promotion to Production)**:
```yaml
# .github/workflows/deploy.yml
name: Deploy Model
on:
  workflow_dispatch:
    inputs:
      model_version:
        description: 'Model version to deploy'
        required: true
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: docker build -t username/card-fraud:${{ github.event.inputs.model_version }} .
      - name: Push to DockerHub
        run: docker push username/card-fraud:${{ github.event.inputs.model_version }}
      - name: Deploy to EC2
        run: |
          ssh ec2-user@prod-server "docker pull username/card-fraud:${{ github.event.inputs.model_version }}"
          ssh ec2-user@prod-server "docker-compose up -d"
```

**Infrastructure as Code**:
- **Terraform**: Define EC2 instances, S3 buckets, IAM roles, security groups
- **Version Control**: Store `.tf` files in git
- **Apply Changes**: `terraform plan` → review → `terraform apply`

### Resilience and Scalability

**High Availability**:
- Multiple EC2 instances behind Application Load Balancer (if real-time API)
- Health checks: `/health` endpoint returns model version and status
- Auto-restart on failure: Docker `restart: always` policy or Kubernetes liveness probes

**Horizontal Scaling**:
- Batch inference: Process partitions in parallel (e.g., 10k rows per worker)
- Use AWS Batch or Kubernetes Jobs for distributed processing
- Scale based on queue depth (SQS message count) or schedule

**Rollback Strategy**:
- Keep previous model version warm in registry
- Blue-green deployment: Run old and new models in parallel, switch traffic
- Automated rollback if error rate or latency exceeds threshold

**Failure Handling**:
- Retry logic with exponential backoff (transient S3 errors)
- Dead-letter queue for failed predictions (investigate later)
- Alert on repeated failures (PagerDuty, Slack)

---

## Observability & Retraining

### Monitoring Strategy

**Infrastructure Metrics** (Prometheus + Node Exporter):
- CPU usage, memory usage, disk I/O (per EC2 instance)
- Docker container health (running, restarting, crashed)
- Network throughput

**Application Metrics** (Prometheus + custom exporters):
- **Inference Metrics**:
  - Predictions per minute (throughput)
  - Inference latency (p50, p95, p99)
  - Prediction distribution (% fraud vs. legitimate)
  - Error rate (exceptions during inference)
- **Data Quality Metrics**:
  - Input schema validation failures
  - Missing value rate (should be 0%)
  - Feature value ranges (detect out-of-distribution data)
- **Model Performance Metrics** (if ground truth available later):
  - Online F1-score (computed on labeled subset)
  - Precision and Recall (updated weekly)
  - Confusion matrix (TP, FP, TN, FN)

**Prometheus Exporters**:
```python
# Example: Custom exporter in inference service
from prometheus_client import Counter, Histogram, Gauge, start_http_server

predictions_total = Counter('fraud_predictions_total', 'Total predictions', ['label'])
inference_latency = Histogram('fraud_inference_latency_seconds', 'Inference latency')
fraud_rate = Gauge('fraud_detection_rate', 'Current fraud rate')

# In prediction loop:
with inference_latency.time():
    prediction = model.predict(features)
predictions_total.labels(label='fraud' if prediction else 'legit').inc()
fraud_rate.set(fraud_count / total_count)
```

**Grafana Dashboards**:
- **Overview Dashboard**: Prediction throughput, fraud rate trend, error rate
- **Model Performance Dashboard**: F1-score over time, precision/recall, confusion matrix heatmap
- **Data Drift Dashboard**: Feature distribution shifts (KL divergence, PSI)
- **Infrastructure Dashboard**: CPU/memory per host, container restarts

### Alerting

**Alert Rules** (Prometheus Alertmanager):
1. **Model Performance Degradation**:
   - F1-score drops below 0.70 for 7 consecutive days
   - Precision drops below 0.75 (too many false positives)
2. **Data Drift**:
   - Prediction distribution shifts >20% (sudden spike in fraud predictions)
   - Feature mean/std deviates >3 sigma from training distribution
3. **System Health**:
   - Error rate >5% for 10 minutes
   - Inference latency p95 >10 seconds (batch should be fast)
   - Container crash loop (restarted >3 times in 5 minutes)
4. **Data Quality**:
   - Schema validation failures >1% of batches
   - No new data ingested in last 48 hours (upstream issue)

**Alert Channels**:
- Slack: `#ml-alerts` channel for team notifications
- Email: On-call engineer for critical alerts
- PagerDuty: Escalation for unresolved critical alerts

### Data Drift Detection

**Method**: Compare inference data distribution to training data baseline

**Techniques**:
1. **Population Stability Index (PSI)**: Measure feature distribution shift
   - PSI < 0.1: No significant change
   - 0.1 ≤ PSI < 0.25: Small shift (monitor)
   - PSI ≥ 0.25: Major shift (retrain recommended)
2. **Kolmogorov-Smirnov Test**: Statistical test for distribution difference
3. **Feature-level Monitoring**: Track mean, std, min, max for each feature

**Implementation**:
- Compute baseline statistics on training data (save as JSON)
- In inference pipeline: Compute statistics on recent batch (e.g., last 7 days)
- Compare: Calculate PSI or KS statistic
- Trigger alert if drift detected

**Example**:
```python
# Precompute baseline (training data)
baseline_stats = {
    'Amount': {'mean': 88.35, 'std': 250.12, 'min': 0.0, 'max': 25691.16},
    'V1': {'mean': 0.0, 'std': 1.0, ...},
    # ...
}

# During inference
current_stats = compute_stats(recent_predictions_df)
psi = calculate_psi(baseline_stats['Amount'], current_stats['Amount'])
if psi > 0.25:
    send_alert("High drift detected in Amount feature")
```

### Model Retraining

**Trigger Conditions**:
1. **Scheduled**: Monthly retraining (first Sunday of each month)
2. **Performance-Based**: F1-score drops below 0.72 for 2 consecutive weeks
3. **Data Drift**: PSI > 0.25 for 3+ features
4. **Manual**: Data scientist triggers retraining after investigation

**Retraining Pipeline** (Airflow DAG):
1. **Data Collection**: Gather last 3-6 months of transactions (include recent data)
2. **Label Collection**: Retrieve ground truth labels (fraud reports from fraud team)
3. **Data Preprocessing**: Apply same pipeline as initial training
4. **Model Training**: Train on combined old + new data (or only new if concept drift)
5. **Evaluation**: 
   - Validate on recent holdout set (last 2 weeks)
   - Compare metrics to current production model
   - Require: new F1 ≥ current F1 + 0.02 (improvement threshold)
6. **Staging Deployment**: If improved, deploy to staging environment
7. **A/B Test or Shadow Mode**:
   - **Shadow Mode**: Run new model in parallel, log predictions but don't act on them
   - Compare predictions: % agreement, metric differences
   - Duration: 1 week
8. **Production Promotion**: If shadow mode successful → promote to production
9. **Archive Old Model**: Move previous production model to "Archived" stage (keep for rollback)

**Continuous Training (Future Enhancement)**:
- Automate steps 1-6 in weekly pipeline
- Auto-promote if improvement is statistically significant (A/B test)
- Human-in-the-loop for final approval (Slack notification + button)

### Logging

**Structured Logging** (JSON format):
```json
{
  "timestamp": "2025-11-09T10:30:45Z",
  "level": "INFO",
  "service": "fraud-inference",
  "model_version": "v1.2.0",
  "event": "prediction_complete",
  "transaction_id": "txn_abc123",
  "prediction": 1,
  "confidence": 0.87,
  "inference_time_ms": 12.4
}
```

**Log Levels**:
- **DEBUG**: Detailed feature values (only in dev, not prod to avoid PII logging)
- **INFO**: Prediction events, batch completion
- **WARNING**: Data quality issues (e.g., feature out of expected range)
- **ERROR**: Exceptions during inference, model loading failures
- **CRITICAL**: System failures (cannot load model, S3 unavailable)

**Centralized Logging**:
- **Tool**: AWS CloudWatch Logs or ELK Stack (Elasticsearch, Logstash, Kibana)
- **Retention**: 30 days hot storage, 1 year cold storage (S3)
- **Search**: Full-text search on transaction IDs, error messages

**Compliance Logging**:
- **Audit Trail**: Log all model deployments, retraining runs, data access
- **Anonymization**: Do NOT log raw transaction details (only anonymized IDs)
- **Access Logs**: CloudTrail for S3 and model registry access

### Feedback Loop

**Ground Truth Collection**:
- Fraud team manually reviews flagged transactions (high-confidence predictions)
- Confirmed fraud cases labeled as `1`, false alarms labeled as `0`
- Store labels in database or append to training dataset

**Model Performance Tracking**:
- Weekly: Compute F1-score on newly labeled data (past 7 days)
- Monthly: Generate model card report (metrics, confusion matrix, drift analysis)
- Quarterly: Comprehensive model audit (fairness, bias, explainability)

**User Feedback** (if applicable):
- Customer disputes: Track false positives (legitimate transactions blocked)
- Fraud reports: Track false negatives (fraud not detected)
- Incorporate into retraining data with appropriate weights

---

## Implementation Roadmap

**Phase 1: Foundation (Weeks 1-2)**
- ✅ EDA marimo notebook (completed)
- Set up MLflow tracking server (local + S3 backend)
- Implement preprocessing pipeline (scikit-learn)
- Create train/val/test splits (stratified)

**Phase 2: Baseline Model (Weeks 3-4)**
- Train Logistic Regression baseline
- Implement evaluation metrics (F1, PR-AUC, confusion matrix)
- Log experiments to MLflow
- Establish acceptance criteria

**Phase 3: Advanced Models (Weeks 5-6)**
- Train XGBoost, LightGBM, Random Forest
- Hyperparameter tuning (Optuna)
- Compare models in MLflow
- Select best model for production

**Phase 4: Deployment (Weeks 7-8)**
- Containerize inference service (Docker)
- Set up batch inference pipeline
- Deploy to AWS EC2 (single instance)
- Test end-to-end: S3 → inference → S3 predictions

**Phase 5: CI/CD (Weeks 9-10)**
- GitHub Actions workflows (CI, train, deploy)
- Automated testing (unit + integration)
- Infrastructure as Code (Terraform)

**Phase 6: Monitoring (Weeks 11-12)**
- Set up Prometheus + Grafana
- Implement custom metrics exporters
- Create dashboards and alert rules
- Test alerting (Slack integration)

**Phase 7: Production Hardening (Weeks 13-14)**
- A/B testing or shadow deployment
- Load testing (simulate production volume)
- Security audit (PCI/GDPR compliance)
- Documentation and runbooks

**Phase 8: Continuous Improvement (Ongoing)**
- Monitor performance weekly
- Retrain monthly (or as needed)
- Iterate on features and models
- Scale infrastructure based on load

---

## Risk Mitigation

**Technical Risks**:
1. **Model Overfitting**: Use stratified K-fold CV, holdout test set, regularization
2. **Data Drift**: Implement drift detection, scheduled retraining
3. **Class Imbalance**: Use class weighting, threshold optimization, PR-AUC metric
4. **Infrastructure Failure**: Multi-AZ deployment, automated backups, rollback plan

**Compliance Risks**:
1. **PCI DSS Violation**: Use only anonymized data (V1-V28), encrypt at rest/transit
2. **GDPR Non-Compliance**: Implement right to explanation (SHAP), data retention policy
3. **Audit Failures**: Maintain detailed logs, model cards, access controls

**Operational Risks**:
1. **False Positives**: Tune precision threshold, collect user feedback, improve features
2. **False Negatives**: Ensemble models, feature engineering (time-based, aggregated)
3. **Slow Inference**: Optimize model (pruning), parallelize batch processing

---

## Success Metrics Summary

**Model Performance**:
- F1-score ≥ 0.75 (primary)
- Precision ≥ 0.80, Recall ≥ 0.70
- PR-AUC ≥ 0.70

**Operational**:
- Batch inference latency < 1 hour
- System uptime ≥ 99.5%
- Model retraining duration < 4 hours

**Business**:
- Fraud loss reduction: 60-70%
- False positive rate: <20% (vs. baseline)
- Customer satisfaction: No significant complaints due to false flags

**Compliance**:
- Zero PCI/GDPR violations
- 100% audit trail coverage
- Model explainability available for all predictions

---

**End of Blueprint**

This document serves as the technical design for a production-ready ML pipeline. Implementation should follow the phased roadmap, with atomic git commits at each milestone. For questions or clarifications, consult the team lead or refer to the EDA notebook (`notebooks/eda_creditcard.py`) for data insights.
