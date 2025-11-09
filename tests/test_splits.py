"""Tests for data splits module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

from src.data.splits import (
    create_splits,
    compute_dataset_hash,
    save_splits,
    load_splits
)


@pytest.fixture
def sample_data():
    """Create sample imbalanced dataset."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Time': np.arange(n_samples),
        'Amount': np.random.lognormal(4, 1.5, n_samples),
        'Class': np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
    }
    
    for i in range(1, 29):
        data[f'V{i}'] = np.random.randn(n_samples)
    
    return pd.DataFrame(data)


def test_create_splits_sizes(sample_data):
    """Test that splits have correct sizes."""
    train_df, val_df, test_df = create_splits(sample_data)
    
    assert len(train_df) == 700
    assert len(val_df) == 150
    assert len(test_df) == 150
    assert len(train_df) + len(val_df) + len(test_df) == len(sample_data)


def test_create_splits_stratified(sample_data):
    """Test that stratified splits preserve class distribution."""
    train_df, val_df, test_df = create_splits(sample_data, stratify=True)
    
    original_fraud_rate = sample_data['Class'].mean()
    train_fraud_rate = train_df['Class'].mean()
    val_fraud_rate = val_df['Class'].mean()
    test_fraud_rate = test_df['Class'].mean()
    
    assert abs(train_fraud_rate - original_fraud_rate) < 0.02
    assert abs(val_fraud_rate - original_fraud_rate) < 0.05
    assert abs(test_fraud_rate - original_fraud_rate) < 0.05


def test_create_splits_no_overlap(sample_data):
    """Test that splits have no overlapping indices."""
    train_df, val_df, test_df = create_splits(sample_data)
    
    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    test_idx = set(test_df.index)
    
    assert len(train_idx & val_idx) == 0
    assert len(train_idx & test_idx) == 0
    assert len(val_idx & test_idx) == 0


def test_compute_dataset_hash(sample_data):
    """Test dataset hash computation."""
    hash1 = compute_dataset_hash(sample_data)
    hash2 = compute_dataset_hash(sample_data)
    
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA256 hex length
    
    modified_data = sample_data.copy()
    modified_data.loc[0, 'Amount'] += 1
    hash3 = compute_dataset_hash(modified_data)
    
    assert hash1 != hash3


def test_save_load_splits(sample_data):
    """Test saving and loading splits."""
    train_df, val_df, test_df = create_splits(sample_data)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        save_splits(train_df, val_df, test_df, output_dir)
        
        assert (output_dir / 'train.csv').exists()
        assert (output_dir / 'val.csv').exists()
        assert (output_dir / 'test.csv').exists()
        assert (output_dir / 'metadata.json').exists()
        
        loaded_train, loaded_val, loaded_test = load_splits(output_dir)
        
        pd.testing.assert_frame_equal(train_df.reset_index(drop=True), 
                                       loaded_train.reset_index(drop=True))
        pd.testing.assert_frame_equal(val_df.reset_index(drop=True),
                                       loaded_val.reset_index(drop=True))
        pd.testing.assert_frame_equal(test_df.reset_index(drop=True),
                                       loaded_test.reset_index(drop=True))
        
        with open(output_dir / 'metadata.json') as f:
            metadata = json.load(f)
        
        assert metadata['train_size'] == len(train_df)
        assert metadata['val_size'] == len(val_df)
        assert metadata['test_size'] == len(test_df)
