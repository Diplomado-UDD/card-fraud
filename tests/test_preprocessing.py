"""Tests for preprocessing module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.data.preprocessing import (
    CyclicalTimeEncoder,
    AmountTransformer,
    create_preprocessing_pipeline,
    preprocess_data,
    validate_schema,
    save_pipeline,
    load_pipeline
)


@pytest.fixture
def sample_data():
    """Create sample transaction data."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'Time': np.random.uniform(0, 172800, n_samples),
        'Amount': np.random.lognormal(4, 1.5, n_samples),
        'Class': np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
    }
    
    for i in range(1, 29):
        data[f'V{i}'] = np.random.randn(n_samples)
    
    return pd.DataFrame(data)


def test_cyclical_time_encoder(sample_data):
    """Test cyclical time encoding."""
    encoder = CyclicalTimeEncoder()
    result = encoder.fit_transform(sample_data)
    
    assert 'Hour_sin' in result.columns
    assert 'Hour_cos' in result.columns
    assert result['Hour_sin'].between(-1, 1).all()
    assert result['Hour_cos'].between(-1, 1).all()


def test_amount_transformer(sample_data):
    """Test amount transformations."""
    transformer = AmountTransformer()
    result = transformer.fit_transform(sample_data)
    
    assert 'log1p_Amount' in result.columns
    assert 'Amount_zscore' in result.columns
    assert (result['log1p_Amount'] >= 0).all()
    assert np.isclose(result['Amount_zscore'].mean(), 0, atol=0.1)


def test_preprocessing_pipeline(sample_data):
    """Test full preprocessing pipeline."""
    pipeline = create_preprocessing_pipeline()
    X_processed, fitted_pipeline = preprocess_data(sample_data, pipeline, fit=True)
    
    assert X_processed.shape[0] == len(sample_data)
    assert X_processed.shape[1] == 33  # 28 V-features + 3 Amount + 2 Hour
    assert 'Hour_sin' in X_processed.columns
    assert 'log1p_Amount' in X_processed.columns


def test_validate_schema_valid(sample_data):
    """Test schema validation with valid data."""
    assert validate_schema(sample_data) is True


def test_validate_schema_missing_columns():
    """Test schema validation with missing columns."""
    df = pd.DataFrame({'Time': [1, 2, 3]})
    
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(df)


def test_validate_schema_missing_values(sample_data):
    """Test schema validation with missing values."""
    sample_data.loc[0, 'Amount'] = np.nan
    
    with pytest.raises(ValueError, match="missing values"):
        validate_schema(sample_data)


def test_validate_schema_negative_amounts(sample_data):
    """Test schema validation with negative amounts."""
    sample_data.loc[0, 'Amount'] = -10
    
    with pytest.raises(ValueError, match="negative values"):
        validate_schema(sample_data)


def test_save_load_pipeline(sample_data):
    """Test saving and loading preprocessing pipeline."""
    pipeline = create_preprocessing_pipeline()
    X_processed, fitted_pipeline = preprocess_data(sample_data, pipeline, fit=True)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline_path = Path(tmpdir) / 'pipeline.pkl'
        save_pipeline(fitted_pipeline, pipeline_path)
        
        loaded_pipeline = load_pipeline(pipeline_path)
        X_reprocessed = loaded_pipeline.transform(sample_data)
        
        pd.testing.assert_frame_equal(X_processed, X_reprocessed)
