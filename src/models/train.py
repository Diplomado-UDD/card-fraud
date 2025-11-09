"""Model training with MLflow tracking."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
from typing import Optional, Dict, Any
import json
from datetime import datetime


class FraudDetectionModel:
    """Wrapper for fraud detection models with consistent interface."""
    
    def __init__(
        self,
        model_type: str = 'logistic',
        hyperparams: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ):
        """
        Initialize fraud detection model.
        
        Args:
            model_type: Type of model ('logistic', 'xgboost', 'lightgbm', 'random_forest')
            hyperparams: Model-specific hyperparameters
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.hyperparams = hyperparams or {}
        self.model = self._create_model()
        self.optimal_threshold = 0.5
    
    def _create_model(self):
        """Create model instance based on type."""
        if self.model_type == 'logistic':
            return LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state,
                solver='saga',
                n_jobs=-1,
                **self.hyperparams
            )
        
        elif self.model_type == 'xgboost':
            default_params = {
                'scale_pos_weight': 578,  # Imbalance ratio
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'eval_metric': 'aucpr',
                'tree_method': 'hist',
                'random_state': self.random_state
            }
            default_params.update(self.hyperparams)
            return XGBClassifier(**default_params)
        
        elif self.model_type == 'lightgbm':
            default_params = {
                'is_unbalance': True,
                'metric': 'auc',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'n_estimators': 100,
                'random_state': self.random_state,
                'verbose': -1
            }
            default_params.update(self.hyperparams)
            return LGBMClassifier(**default_params)
        
        elif self.model_type == 'random_forest':
            default_params = {
                'class_weight': 'balanced_subsample',
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': self.random_state,
                'n_jobs': -1
            }
            default_params.update(self.hyperparams)
            return RandomForestClassifier(**default_params)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X, y):
        """Train the model."""
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X, threshold: Optional[float] = None):
        """Predict classes with optional threshold."""
        if threshold is None:
            threshold = self.optimal_threshold
        
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def optimize_threshold(
        self,
        X_val,
        y_val,
        metric: str = 'f1',
        min_precision: Optional[float] = None
    ) -> float:
        """
        Optimize classification threshold on validation set.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            metric: Metric to optimize ('f1', 'recall', 'precision')
            min_precision: Minimum precision constraint (for 'recall' optimization)
        
        Returns:
            Optimal threshold
        """
        y_proba = self.predict_proba(X_val)
        
        if metric == 'f1':
            thresholds = np.arange(0.1, 0.9, 0.01)
            f1_scores = []
            for thresh in thresholds:
                y_pred = (y_proba >= thresh).astype(int)
                f1_scores.append(f1_score(y_val, y_pred))
            
            optimal_idx = np.argmax(f1_scores)
            self.optimal_threshold = thresholds[optimal_idx]
        
        elif metric == 'recall' and min_precision is not None:
            thresholds = np.arange(0.1, 0.9, 0.01)
            best_recall = 0
            best_threshold = 0.5
            
            for thresh in thresholds:
                y_pred = (y_proba >= thresh).astype(int)
                prec = precision_score(y_val, y_pred, zero_division=0)
                rec = recall_score(y_val, y_pred)
                
                if prec >= min_precision and rec > best_recall:
                    best_recall = rec
                    best_threshold = thresh
            
            self.optimal_threshold = best_threshold
        
        return self.optimal_threshold


def evaluate_model(
    model: FraudDetectionModel,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X: Features
        y: True labels
        threshold: Classification threshold
    
    Returns:
        Dictionary of metrics
    """
    y_proba = model.predict_proba(X)
    y_pred = model.predict(X, threshold=threshold)
    
    precision, recall, _ = precision_recall_curve(y, y_proba)
    pr_auc = auc(recall, precision)
    
    metrics = {
        'f1_score': float(f1_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred, zero_division=0)),
        'recall': float(recall_score(y, y_pred)),
        'roc_auc': float(roc_auc_score(y, y_proba)),
        'pr_auc': float(pr_auc),
    }
    
    cm = confusion_matrix(y, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
        })
    
    return metrics


def save_model(
    model: FraudDetectionModel,
    output_dir: Path,
    metrics: Optional[Dict] = None,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save model, metrics, and metadata.
    
    Args:
        model: Trained model
        output_dir: Directory to save artifacts
        metrics: Model metrics dictionary
        metadata: Additional metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, output_dir / 'model.pkl')
    
    model_info = {
        'model_type': model.model_type,
        'optimal_threshold': float(model.optimal_threshold),
        'hyperparameters': model.hyperparams,
        'timestamp': datetime.now().isoformat(),
    }
    
    if metrics:
        model_info['metrics'] = metrics
    
    if metadata:
        model_info.update(metadata)
    
    with open(output_dir / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)


def load_model(model_dir: Path) -> FraudDetectionModel:
    """Load trained model from disk."""
    return joblib.load(model_dir / 'model.pkl')


def print_metrics(metrics: Dict[str, float], name: str = "Model") -> None:
    """Print formatted metrics."""
    print(f"\n{name} Performance:")
    print(f"  F1-Score:    {metrics['f1_score']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  PR-AUC:      {metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
    
    if 'true_positives' in metrics:
        print("\n  Confusion Matrix:")
        print(f"    TP: {metrics['true_positives']:5d}  FP: {metrics['false_positives']:5d}")
        print(f"    FN: {metrics['false_negatives']:5d}  TN: {metrics['true_negatives']:5d}")
