"""Prometheus metrics exporter for fraud detection inference service."""

from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
import time
from typing import Optional
import numpy as np


class FraudDetectionMetrics:
    """Prometheus metrics for fraud detection service."""
    
    def __init__(self):
        # Prediction metrics
        self.predictions_total = Counter(
            'fraud_predictions_total',
            'Total number of fraud predictions',
            ['label']
        )
        
        self.inference_latency = Histogram(
            'fraud_inference_latency_seconds',
            'Inference latency in seconds',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
        )
        
        self.batch_size = Histogram(
            'fraud_batch_size',
            'Number of transactions per batch',
            buckets=[10, 50, 100, 500, 1000, 5000, 10000]
        )
        
        # Model performance metrics
        self.model_f1_score = Gauge(
            'fraud_model_f1_score',
            'Current model F1-score'
        )
        
        self.model_precision = Gauge(
            'fraud_model_precision',
            'Current model precision'
        )
        
        self.model_recall = Gauge(
            'fraud_model_recall',
            'Current model recall'
        )
        
        self.model_pr_auc = Gauge(
            'fraud_model_pr_auc',
            'Current model PR-AUC'
        )
        
        # Data quality metrics
        self.validation_errors = Counter(
            'fraud_validation_errors_total',
            'Number of schema validation errors',
            ['error_type']
        )
        
        self.feature_drift_psi = Gauge(
            'fraud_feature_drift_psi',
            'Population Stability Index for feature drift',
            ['feature']
        )
        
        # Prediction distribution
        self.fraud_rate = Gauge(
            'fraud_detection_rate',
            'Current fraud detection rate'
        )
        
        self.prediction_confidence = Histogram(
            'fraud_prediction_confidence',
            'Confidence scores of predictions',
            ['label'],
            buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
        )
        
        # Error metrics
        self.inference_errors = Counter(
            'fraud_inference_errors_total',
            'Total inference errors',
            ['error_type']
        )
        
        # Timestamps
        self.last_prediction_time = Gauge(
            'fraud_last_prediction_timestamp',
            'Timestamp of last prediction'
        )
        
        # Model info
        self.model_info = Info(
            'fraud_model_info',
            'Information about the deployed model'
        )
        
        # Throughput
        self.throughput = Gauge(
            'fraud_throughput_per_second',
            'Current throughput (predictions/second)'
        )
    
    def record_prediction(
        self,
        label: int,
        confidence: float,
        latency: float
    ) -> None:
        """Record a single prediction."""
        label_str = 'fraud' if label == 1 else 'legitimate'
        self.predictions_total.labels(label=label_str).inc()
        self.prediction_confidence.labels(label=label_str).observe(confidence)
        self.inference_latency.observe(latency)
        self.last_prediction_time.set(time.time())
    
    def record_batch(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        batch_latency: float
    ) -> None:
        """Record batch predictions."""
        batch_size = len(predictions)
        self.batch_size.observe(batch_size)
        
        fraud_count = predictions.sum()
        fraud_rate = fraud_count / batch_size
        self.fraud_rate.set(fraud_rate)
        
        throughput = batch_size / batch_latency if batch_latency > 0 else 0
        self.throughput.set(throughput)
        
        for pred, conf in zip(predictions, confidences):
            self.record_prediction(pred, conf, 0)
    
    def update_model_metrics(
        self,
        f1: float,
        precision: float,
        recall: float,
        pr_auc: float
    ) -> None:
        """Update model performance metrics."""
        self.model_f1_score.set(f1)
        self.model_precision.set(precision)
        self.model_recall.set(recall)
        self.model_pr_auc.set(pr_auc)
    
    def record_validation_error(self, error_type: str) -> None:
        """Record a validation error."""
        self.validation_errors.labels(error_type=error_type).inc()
    
    def record_inference_error(self, error_type: str) -> None:
        """Record an inference error."""
        self.inference_errors.labels(error_type=error_type).inc()
    
    def update_feature_drift(self, feature: str, psi_value: float) -> None:
        """Update feature drift metric."""
        self.feature_drift_psi.labels(feature=feature).set(psi_value)
    
    def set_model_info(
        self,
        model_type: str,
        version: str,
        threshold: float
    ) -> None:
        """Set model information."""
        self.model_info.info({
            'model_type': model_type,
            'version': version,
            'threshold': str(threshold)
        })


# Global metrics instance
metrics = FraudDetectionMetrics()


def start_metrics_server(port: int = 8000) -> None:
    """Start Prometheus metrics HTTP server."""
    start_http_server(port)
    print(f"Metrics server started on port {port}")
    print(f"Metrics available at http://localhost:{port}/metrics")
