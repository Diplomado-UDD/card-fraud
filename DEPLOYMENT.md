# Deployment Guide

## Overview

Step-by-step guide to deploy the fraud detection ML pipeline to production on AWS EC2.

## Prerequisites

- AWS account with EC2, S3 access
- Docker installed locally
- AWS CLI configured
- SSH key pair for EC2 access

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         AWS Cloud                            │
│                                                              │
│  ┌─────────────┐      ┌──────────────┐      ┌────────────┐ │
│  │   S3        │      │  EC2         │      │ CloudWatch │ │
│  │  - Data     │◄────►│  - Docker    │─────►│  - Logs    │ │
│  │  - Models   │      │  - Inference │      │  - Metrics │ │
│  └─────────────┘      └──────────────┘      └────────────┘ │
│                              │                              │
│                              ▼                              │
│                       ┌──────────────┐                      │
│                       │  Prometheus  │                      │
│                       │  + Grafana   │                      │
│                       └──────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

## Phase 1: AWS Setup

### 1.1 Create S3 Buckets

```bash
# Data and model storage
aws s3 mb s3://card-fraud-prod-data
aws s3 mb s3://card-fraud-prod-models
aws s3 mb s3://card-fraud-prod-predictions

# Enable versioning for models
aws s3api put-bucket-versioning \
    --bucket card-fraud-prod-models \
    --versioning-configuration Status=Enabled

# Enable encryption
aws s3api put-bucket-encryption \
    --bucket card-fraud-prod-data \
    --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
```

### 1.2 Upload Training Data

```bash
# Upload dataset
aws s3 cp data/creditcard.csv s3://card-fraud-prod-data/raw/creditcard.csv

# Upload trained model
aws s3 cp models/xgboost_v1/ s3://card-fraud-prod-models/xgboost_v1/ --recursive
```

### 1.3 Create IAM Role for EC2

```bash
# Create role with S3 access
aws iam create-role --role-name CardFraudEC2Role \
    --assume-role-policy-document file://deployment/iam-trust-policy.json

# Attach S3 policy
aws iam attach-role-policy --role-name CardFraudEC2Role \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Create instance profile
aws iam create-instance-profile --instance-profile-name CardFraudProfile
aws iam add-role-to-instance-profile --instance-profile-name CardFraudProfile \
    --role-name CardFraudEC2Role
```

### 1.4 Launch EC2 Instance

```bash
# Launch Ubuntu instance
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type t3.xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxxx \
    --iam-instance-profile Name=CardFraudProfile \
    --block-device-mappings file://deployment/block-devices.json \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=fraud-detection-prod}]'
```

**Recommended instance types:**
- **Development**: t3.medium (2 vCPU, 4 GB RAM) — ~$30/month
- **Production**: t3.xlarge (4 vCPU, 16 GB RAM) — ~$120/month
- **High-load**: c6i.2xlarge (8 vCPU, 16 GB RAM) — ~$250/month

### 1.5 Configure Security Group

```bash
# Allow SSH (your IP only)
aws ec2 authorize-security-group-ingress --group-id sg-xxxxxxxxx \
    --protocol tcp --port 22 --cidr YOUR_IP/32

# Allow Grafana (internal VPN or whitelist)
aws ec2 authorize-security-group-ingress --group-id sg-xxxxxxxxx \
    --protocol tcp --port 3000 --cidr 10.0.0.0/16

# Allow Prometheus (internal only)
aws ec2 authorize-security-group-ingress --group-id sg-xxxxxxxxx \
    --protocol tcp --port 9090 --cidr 10.0.0.0/16
```

## Phase 2: Server Setup

### 2.1 Connect to EC2

```bash
# SSH into instance
ssh -i ~/.ssh/your-key.pem ubuntu@ec2-xx-xx-xx-xx.compute.amazonaws.com
```

### 2.2 Install Docker

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" \
    -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add ubuntu user to docker group
sudo usermod -aG docker ubuntu
newgrp docker
```

### 2.3 Install uv and Python

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv
```

### 2.4 Clone Repository

```bash
# Clone from GitHub
git clone https://github.com/your-org/card-fraud.git
cd card-fraud

# Install dependencies
uv sync --locked
```

## Phase 3: Application Deployment

### 3.1 Download Data and Models from S3

```bash
# Create directories
mkdir -p data models

# Download data
aws s3 cp s3://card-fraud-prod-data/raw/creditcard.csv data/

# Download trained model
aws s3 sync s3://card-fraud-prod-models/xgboost_v1/ models/xgboost_v1/
```

### 3.2 Build Docker Image

```bash
# Build production image
docker build -t card-fraud:latest .

# Tag for DockerHub (optional)
docker tag card-fraud:latest your-dockerhub/card-fraud:v1.0.0
docker push your-dockerhub/card-fraud:v1.0.0
```

### 3.3 Start Monitoring Stack

```bash
# Start Prometheus + Grafana
docker-compose -f docker-compose.monitoring.yml up -d

# Verify services
docker ps
curl http://localhost:9090/-/healthy
curl http://localhost:3000/api/health
```

### 3.4 Configure Systemd Service

Create `/etc/systemd/system/fraud-inference.service`:

```ini
[Unit]
Description=Fraud Detection Batch Inference
After=docker.service
Requires=docker.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/card-fraud
ExecStart=/home/ubuntu/.local/bin/uv run python -m src.inference.batch \
    --model-dir /home/ubuntu/card-fraud/models/xgboost_v1 \
    --preprocessing /home/ubuntu/card-fraud/models/xgboost_v1/preprocessing_pipeline.pkl \
    --input /home/ubuntu/card-fraud/data/incoming/transactions.csv \
    --output /home/ubuntu/card-fraud/data/predictions/predictions.csv
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable fraud-inference
sudo systemctl start fraud-inference
sudo systemctl status fraud-inference
```

## Phase 4: Scheduled Jobs

### 4.1 Daily Batch Inference

Create `/etc/cron.d/fraud-inference`:

```bash
# Run daily at 2 AM
0 2 * * * ubuntu cd /home/ubuntu/card-fraud && /home/ubuntu/.local/bin/uv run python -m src.inference.batch --model-dir models/xgboost_v1 --preprocessing models/xgboost_v1/preprocessing_pipeline.pkl --input data/incoming/daily_batch.csv --output data/predictions/daily_predictions.csv >> /var/log/fraud-inference.log 2>&1
```

### 4.2 Weekly Model Retraining

```bash
# Run every Sunday at 3 AM
0 3 * * 0 ubuntu cd /home/ubuntu/card-fraud && /home/ubuntu/.local/bin/uv run python train.py --data data/creditcard.csv --output-dir models/retrain_$(date +\%Y\%m\%d) --model-type xgboost --optimize-threshold >> /var/log/fraud-training.log 2>&1
```

### 4.3 Data Drift Monitoring

```bash
# Check drift daily at 4 AM
0 4 * * * ubuntu cd /home/ubuntu/card-fraud && /home/ubuntu/.local/bin/uv run python scripts/check_drift.py >> /var/log/fraud-drift.log 2>&1
```

## Phase 5: Monitoring & Alerts

### 5.1 Configure Grafana

1. Access Grafana: http://EC2_IP:3000
2. Login: admin/admin (change password)
3. Add Prometheus data source:
   - URL: http://prometheus:9090
   - Save & Test
4. Import dashboards from `monitoring/grafana-dashboards/`

### 5.2 Configure Slack Alerts

Edit `monitoring/alertmanager.yml`:

```yaml
global:
  slack_api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
```

Restart Alertmanager:

```bash
docker-compose -f docker-compose.monitoring.yml restart alertmanager
```

### 5.3 CloudWatch Integration (Optional)

```bash
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i -E ./amazon-cloudwatch-agent.deb

# Configure logs
sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config -m ec2 -s -c file:deployment/cloudwatch-config.json
```

## Phase 6: Testing

### 6.1 Health Checks

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job: .labels.job, health: .health}'

# Check inference service
curl http://localhost:8000/metrics | grep fraud_predictions_total

# Check Docker containers
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### 6.2 End-to-End Test

```bash
# Place test data
cp data/splits/test.csv data/incoming/test_batch.csv

# Run inference
uv run python -m src.inference.batch \
    --model-dir models/xgboost_v1 \
    --preprocessing models/xgboost_v1/preprocessing_pipeline.pkl \
    --input data/incoming/test_batch.csv \
    --output data/predictions/test_predictions.csv

# Verify output
head -5 data/predictions/test_predictions.csv
cat data/predictions/inference_stats.json | jq
```

### 6.3 Load Test

```bash
# Simulate 10,000 transactions
python scripts/generate_test_data.py --n-samples 10000 --output data/incoming/load_test.csv

# Run inference and measure time
time uv run python -m src.inference.batch \
    --model-dir models/xgboost_v1 \
    --preprocessing models/xgboost_v1/preprocessing_pipeline.pkl \
    --input data/incoming/load_test.csv \
    --output data/predictions/load_test_predictions.csv
```

## Phase 7: Backup & Recovery

### 7.1 Automated Backups

Create `/home/ubuntu/card-fraud/scripts/backup.sh`:

```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)

# Backup models
aws s3 sync models/ s3://card-fraud-prod-backups/models_$DATE/

# Backup Grafana dashboards
docker exec fraud-grafana grafana-cli admin export --homepath=/var/lib/grafana > grafana_backup_$DATE.json
aws s3 cp grafana_backup_$DATE.json s3://card-fraud-prod-backups/grafana/

# Backup Prometheus config
aws s3 cp monitoring/ s3://card-fraud-prod-backups/monitoring_$DATE/ --recursive

echo "Backup completed: $DATE"
```

Run daily:

```bash
0 5 * * * /home/ubuntu/card-fraud/scripts/backup.sh >> /var/log/backup.log 2>&1
```

### 7.2 Recovery Procedure

```bash
# Restore latest model
aws s3 sync s3://card-fraud-prod-backups/models_LATEST/ models/

# Restore Grafana
docker cp grafana_backup.json fraud-grafana:/tmp/
docker exec fraud-grafana grafana-cli admin import /tmp/grafana_backup.json

# Restart services
docker-compose -f docker-compose.monitoring.yml restart
```

## Phase 8: Cost Optimization

### 8.1 Use Spot Instances (Development)

```bash
aws ec2 request-spot-instances \
    --spot-price "0.05" \
    --instance-count 1 \
    --type "one-time" \
    --launch-specification file://deployment/spot-launch-spec.json
```

### 8.2 Auto-Scaling (Production)

- Set up Auto Scaling Group with target tracking (CPU < 70%)
- Use Application Load Balancer for multiple instances
- Configure CloudWatch alarms

### 8.3 S3 Lifecycle Policies

```bash
# Archive old predictions after 90 days
aws s3api put-bucket-lifecycle-configuration \
    --bucket card-fraud-prod-predictions \
    --lifecycle-configuration file://deployment/s3-lifecycle.json
```

## Troubleshooting

### Issue: Out of Memory

```bash
# Check memory usage
free -h
docker stats

# Solution: Increase instance size or optimize batch size
```

### Issue: Slow Inference

```bash
# Check CPU usage
top

# Solution: Use GPU instance (p3.2xlarge) for large batches
```

### Issue: Model Not Loading

```bash
# Check model files
ls -lh models/xgboost_v1/
python -c "import joblib; model = joblib.load('models/xgboost_v1/model.pkl'); print(model)"

# Solution: Re-download from S3
aws s3 sync s3://card-fraud-prod-models/xgboost_v1/ models/xgboost_v1/ --delete
```

## Security Checklist

- [ ] EC2 security group restricts SSH to specific IPs
- [ ] S3 buckets have encryption enabled
- [ ] IAM roles follow least-privilege principle
- [ ] Grafana admin password changed from default
- [ ] Prometheus/Alertmanager not exposed to public internet
- [ ] SSL/TLS enabled for Grafana (use CloudFront or ALB)
- [ ] CloudWatch logs configured for audit trail
- [ ] Regular security updates applied (`sudo apt update && sudo apt upgrade`)

## Maintenance Schedule

- **Daily**: Automated backups, drift monitoring
- **Weekly**: Model retraining (if needed)
- **Monthly**: Security updates, cost review
- **Quarterly**: Model performance audit, capacity planning

## Next Steps

1. Set up CI/CD with GitHub Actions for automated deployments
2. Implement A/B testing for model versions
3. Add API endpoint for real-time predictions
4. Integrate with MLflow for experiment tracking
5. Set up multi-region deployment for disaster recovery

## Estimated Costs (Monthly)

| Component | Resource | Cost |
|-----------|----------|------|
| Compute | t3.xlarge EC2 | $120 |
| Storage | 100 GB EBS | $10 |
| S3 | 50 GB data | $1 |
| Data Transfer | 100 GB | $9 |
| **Total** | | **~$140** |

For high-availability setup: ~$400/month (3x instances, load balancer, multi-AZ)
