"""Create stratified train/validation/test splits."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Optional
import hashlib
import json


def create_splits(
    df: pd.DataFrame,
    target_col: str = 'Class',
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test splits.
    
    Args:
        df: Input DataFrame with features and target
        target_col: Name of target column
        train_size: Proportion for training set
        val_size: Proportion for validation set
        test_size: Proportion for test set
        random_state: Random seed for reproducibility
        stratify: Whether to use stratified splitting
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "Split sizes must sum to 1.0"
    
    stratify_by = df[target_col] if stratify else None
    
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        stratify=stratify_by
    )
    
    val_ratio = val_size / (val_size + test_size)
    stratify_temp = temp_df[target_col] if stratify else None
    
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio,
        random_state=random_state,
        stratify=stratify_temp
    )
    
    return train_df, val_df, test_df


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """Compute SHA256 hash of dataset for versioning."""
    df_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(df_bytes).hexdigest()


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    metadata: Optional[dict] = None
) -> None:
    """
    Save train/val/test splits to CSV files with metadata.
    
    Args:
        train_df: Training set
        val_df: Validation set
        test_df: Test set
        output_dir: Directory to save splits
        metadata: Optional metadata dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'val.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)
    
    split_metadata = {
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'train_fraud_count': int(train_df['Class'].sum()),
        'val_fraud_count': int(val_df['Class'].sum()),
        'test_fraud_count': int(test_df['Class'].sum()),
        'train_hash': compute_dataset_hash(train_df),
        'val_hash': compute_dataset_hash(val_df),
        'test_hash': compute_dataset_hash(test_df),
    }
    
    if metadata:
        split_metadata.update(metadata)
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(split_metadata, f, indent=2)


def load_splits(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test splits from CSV files."""
    train_df = pd.read_csv(data_dir / 'train.csv')
    val_df = pd.read_csv(data_dir / 'val.csv')
    test_df = pd.read_csv(data_dir / 'test.csv')
    return train_df, val_df, test_df


def print_split_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = 'Class'
) -> None:
    """Print statistics about the splits."""
    def stats(df, name):
        total = len(df)
        fraud = df[target_col].sum()
        fraud_pct = 100 * fraud / total
        print(f"{name:10s}: {total:6d} rows, {fraud:4d} fraud ({fraud_pct:.3f}%)")
    
    stats(train_df, "Train")
    stats(val_df, "Validation")
    stats(test_df, "Test")
    print(f"{'Total':10s}: {len(train_df) + len(val_df) + len(test_df):6d} rows")
