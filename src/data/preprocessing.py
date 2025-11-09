"""Data preprocessing pipeline with feature engineering."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Optional


class CyclicalTimeEncoder(BaseEstimator, TransformerMixin):
    """Encode time as cyclical features (sin/cos for hour of day)."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        hour_of_day = (X['Time'] % 86400) / 3600
        X['Hour_sin'] = np.sin(2 * np.pi * hour_of_day / 24)
        X['Hour_cos'] = np.cos(2 * np.pi * hour_of_day / 24)
        return X


class AmountTransformer(BaseEstimator, TransformerMixin):
    """Transform Amount feature: log1p and z-score."""
    
    def __init__(self):
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X, y=None):
        self.mean_ = X['Amount'].mean()
        self.std_ = X['Amount'].std()
        return self
    
    def transform(self, X):
        X = X.copy()
        X['log1p_Amount'] = np.log1p(X['Amount'])
        X['Amount_zscore'] = (X['Amount'] - self.mean_) / self.std_
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Select features for modeling."""
    
    def __init__(self, feature_cols: list[str]):
        self.feature_cols = feature_cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.feature_cols]


def create_preprocessing_pipeline() -> Pipeline:
    """
    Create preprocessing pipeline for fraud detection.
    
    Returns:
        Scikit-learn Pipeline with feature engineering steps
    """
    v_features = [f'V{i}' for i in range(1, 29)]
    
    feature_cols = v_features + [
        'Amount',
        'log1p_Amount',
        'Amount_zscore',
        'Hour_sin',
        'Hour_cos'
    ]
    
    pipeline = Pipeline([
        ('cyclical_time', CyclicalTimeEncoder()),
        ('amount_transform', AmountTransformer()),
        ('feature_select', FeatureSelector(feature_cols))
    ])
    
    return pipeline


def preprocess_data(
    df: pd.DataFrame,
    pipeline: Optional[Pipeline] = None,
    fit: bool = False
) -> tuple[pd.DataFrame, Pipeline]:
    """
    Preprocess fraud detection data.
    
    Args:
        df: Input DataFrame with raw features
        pipeline: Existing preprocessing pipeline (if None, creates new one)
        fit: Whether to fit the pipeline on this data
    
    Returns:
        Tuple of (processed DataFrame, fitted pipeline)
    """
    if pipeline is None:
        pipeline = create_preprocessing_pipeline()
    
    if fit:
        X_processed = pipeline.fit_transform(df)
    else:
        X_processed = pipeline.transform(df)
    
    if isinstance(X_processed, np.ndarray):
        X_processed = pd.DataFrame(
            X_processed,
            columns=pipeline.named_steps['feature_select'].feature_cols,
            index=df.index
        )
    
    return X_processed, pipeline


def save_pipeline(pipeline: Pipeline, path: Path) -> None:
    """Save preprocessing pipeline to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load_pipeline(path: Path) -> Pipeline:
    """Load preprocessing pipeline from disk."""
    return joblib.load(path)


def validate_schema(df: pd.DataFrame) -> bool:
    """
    Validate input data schema.
    
    Args:
        df: Input DataFrame
    
    Returns:
        True if schema is valid
    
    Raises:
        ValueError: If schema validation fails
    """
    required_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
    
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    if df.isnull().any().any():
        raise ValueError("Data contains missing values")
    
    if (df['Amount'] < 0).any():
        raise ValueError("Amount column contains negative values")
    
    return True
