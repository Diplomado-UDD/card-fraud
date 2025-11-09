# Monitoring Guide

## Overview

Production monitoring stack for fraud detection ML pipeline using Prometheus, Grafana, and Alertmanager.

## Architecture

```
┌─────────────────┐
│ Inference       │ ──> Prometheus metrics (:8000/metrics)
│ Service         │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Prometheus      │ ──> Scrapes metrics every 15s
│ (:9090)         │ ──> Evaluates alert rules
└─────────────────┘
         │
         ├──> Alertmanager (:9093) ──> Slack notifications
         │
         └──> Grafana (:3000) ──> Dashboards
```

## Quick Start

### 1. Start Monitoring Stack

```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

This starts:
- **Prometheus** on http://localhost:9090
- **Grafana** on http://localhost:3000 (admin/admin)
- **Alertmanager** on http://localhost:9093
- **Node Exporter** on http://localhost:9100

### 2. Access Dashboards

Open Grafana: http://localhost:3000

**Default credentials:**
- Username: `admin`
- Password: `admin` (change on first login)

**Pre-configured dashboards:**
1. Model Performance — F1-score, precision, recall, throughput
2. Data Drift — PSI values, validation errors, feature distributions

### 3. Configure Slack Alerts

Edit `monitoring/alertmanager.yml`:

```yaml
global:
  slack_api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
```

Get webhook URL from: https://api.slack.com/messaging/webhooks

## Metrics Reference

### Prediction Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `fraud_predictions_total{label}` | Counter | Total predictions by label (fraud/legitimate) |
| `fraud_inference_latency_seconds` | Histogram | Inference latency distribution |
| `fraud_detection_rate` | Gauge | Current fraud detection rate |
| `fraud_throughput_per_second` | Gauge | Predictions per second |

### Model Performance Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `fraud_model_f1_score` | Gauge | Current F1-score |
| `fraud_model_precision` | Gauge | Current precision |
| `fraud_model_recall` | Gauge | Current recall |
| `fraud_model_pr_auc` | Gauge | Precision-Recall AUC |

### Data Quality Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `fraud_feature_drift_psi{feature}` | Gauge | PSI value per feature |
| `fraud_validation_errors_total{error_type}` | Counter | Schema validation errors |
| `fraud_last_prediction_timestamp` | Gauge | Unix timestamp of last prediction |

### Error Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `fraud_inference_errors_total{error_type}` | Counter | Inference errors by type |

## Alert Rules

### Critical Alerts

1. **HighErrorRate** — Error rate > 5% for 5 minutes
2. **NoDataIngested** — No predictions in 48 hours
3. **ContainerRestartLoop** — Frequent container restarts

### Warning Alerts

1. **ModelPerformanceDegradation** — F1-score < 0.70 for 1 hour
2. **DataDriftDetected** — PSI > 0.25 for 10 minutes
3. **FraudRateSpike** — Fraud rate > 5% for 15 minutes
4. **InferenceLatencyHigh** — P95 latency > 10s for 5 minutes
5. **PrecisionBelowThreshold** — Precision < 0.75 for 1 hour
6. **RecallBelowThreshold** — Recall < 0.65 for 1 hour

## Data Drift Detection

### Running Drift Analysis

```python
from src.monitoring.drift import detect_drift, save_baseline_statistics
import pandas as pd

# Save baseline from training data
train_df = pd.read_csv('data/splits/train.csv')
features = [f'V{i}' for i in range(1, 29)] + ['Amount']
save_baseline_statistics(train_df, Path('models/baseline_stats.json'), features)

# Detect drift on new data
current_df = pd.read_csv('data/new_transactions.csv')
baseline_df = pd.read_csv('data/splits/train.csv')

drift_results = detect_drift(baseline_df, current_df, features)

# Print report
from src.monitoring.drift import print_drift_report
print_drift_report(drift_results)
```

### PSI Interpretation

- **PSI < 0.1**: No significant change — Continue using model
- **0.1 ≤ PSI < 0.25**: Small shift — Monitor closely
- **PSI ≥ 0.25**: Major shift — Retrain recommended

## Integration with Inference Service

### Adding Metrics to Batch Inference

```python
from src.monitoring import metrics, start_metrics_server
from src.inference.batch import BatchInferenceService

# Start metrics server
start_metrics_server(port=8000)

# Set model info
metrics.set_model_info(
    model_type='xgboost',
    version='v1.2.0',
    threshold=0.35
)

# In prediction loop
predictions = service.predict(df)
metrics.record_batch(
    predictions['prediction'].values,
    predictions['fraud_probability'].values,
    batch_latency=inference_time
)

# Update performance metrics (if ground truth available)
metrics.update_model_metrics(
    f1=0.78,
    precision=0.82,
    recall=0.74,
    pr_auc=0.75
)
```

## Querying Metrics

### Prometheus Query Examples

```promql
# Average fraud detection rate (last hour)
avg_over_time(fraud_detection_rate[1h])

# Prediction throughput (per minute)
rate(fraud_predictions_total[1m]) * 60

# P95 latency
histogram_quantile(0.95, rate(fraud_inference_latency_seconds_bucket[5m]))

# Error rate percentage
rate(fraud_inference_errors_total[5m]) / rate(fraud_predictions_total[5m]) * 100

# Features with drift (PSI > 0.25)
fraud_feature_drift_psi > 0.25
```

### Grafana Query Examples

Add these as panels in custom dashboards:

1. **Fraud vs Legitimate Over Time**
   ```promql
   rate(fraud_predictions_total{label="fraud"}[5m])
   rate(fraud_predictions_total{label="legitimate"}[5m])
   ```

2. **Model Health Score** (composite metric)
   ```promql
   (fraud_model_f1_score + fraud_model_precision + fraud_model_recall) / 3
   ```

3. **Time to Retrain** (based on drift)
   ```promql
   max(fraud_feature_drift_psi) < 0.25
   ```

## Troubleshooting

### Metrics Not Appearing

1. Check inference service is running and exposing metrics:
   ```bash
   curl http://localhost:8000/metrics
   ```

2. Verify Prometheus is scraping:
   ```bash
   # Visit Prometheus UI
   open http://localhost:9090/targets
   ```

3. Check Prometheus logs:
   ```bash
   docker logs fraud-prometheus
   ```

### Alerts Not Firing

1. Test alert rule syntax:
   ```bash
   promtool check rules monitoring/alert_rules.yml
   ```

2. Check Alertmanager configuration:
   ```bash
   promtool check config monitoring/alertmanager.yml
   ```

3. View active alerts:
   ```bash
   open http://localhost:9090/alerts
   ```

### Grafana Dashboard Issues

1. Check data source connection:
   - Configuration → Data Sources → Prometheus
   - URL should be: `http://prometheus:9090`

2. Import dashboards manually:
   - Dashboards → Import → Upload JSON
   - Use files from `monitoring/grafana-dashboards/`

## Production Deployment

### AWS EC2 Setup

1. **Security Groups:**
   ```
   - 9090: Prometheus (internal only)
   - 3000: Grafana (VPN or IP whitelist)
   - 8000: Metrics endpoint (internal only)
   ```

2. **Persistent Storage:**
   ```bash
   # Create EBS volumes for:
   - /var/lib/prometheus
   - /var/lib/grafana
   ```

3. **Backup Strategy:**
   ```bash
   # Daily backup of Grafana dashboards
   docker exec fraud-grafana grafana-cli admin export --homepath=/var/lib/grafana
   
   # Backup to S3
   aws s3 cp grafana-backup.json s3://card-fraud-backups/monitoring/
   ```

### High Availability

For production HA setup:

1. **Prometheus:**
   - Run 2+ instances with same config
   - Use Thanos for long-term storage and global queries

2. **Alertmanager:**
   - Cluster mode with gossip protocol
   - 3+ instances for quorum

3. **Grafana:**
   - Use PostgreSQL backend (not SQLite)
   - Multiple instances behind load balancer

## Maintenance

### Retention Policy

Edit `prometheus.yml`:

```yaml
global:
  # Data retention (default: 15 days)
  storage.tsdb.retention.time: 30d
  storage.tsdb.retention.size: 50GB
```

### Backup Dashboards

```bash
# Export Grafana dashboards
./scripts/backup_dashboards.sh

# Commit to git
git add monitoring/grafana-dashboards/
git commit -m "feat: update monitoring dashboards"
```

### Update Alert Rules

1. Edit `monitoring/alert_rules.yml`
2. Validate syntax:
   ```bash
   promtool check rules monitoring/alert_rules.yml
   ```
3. Reload Prometheus:
   ```bash
   curl -X POST http://localhost:9090/-/reload
   ```

## Cost Optimization

**Estimated AWS costs (monthly):**
- EC2 t3.medium (Prometheus): ~$30
- EC2 t3.small (Grafana): ~$15
- EBS storage (100GB): ~$10
- **Total: ~$55/month**

**Optimization tips:**
- Use Spot Instances for non-critical environments
- Enable S3 archival for old metrics (Thanos)
- Set aggressive retention policies (7-14 days)

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Alertmanager Guide](https://prometheus.io/docs/alerting/latest/alertmanager/)
- [PromQL Cheatsheet](https://promlabs.com/promql-cheat-sheet/)
