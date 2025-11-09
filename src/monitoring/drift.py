"""Data drift detection using Population Stability Index (PSI)."""

import numpy as np
import pandas as pd
from typing import Dict
import json
from pathlib import Path


def calculate_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    bins: int = 10
) -> float:
    """
    Calculate Population Stability Index (PSI).
    
    PSI measures distribution shift between baseline and current data:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.25: Small shift (monitor)
    - PSI >= 0.25: Major shift (retrain recommended)
    
    Args:
        baseline: Reference distribution (training data)
        current: Current distribution (inference data)
        bins: Number of bins for discretization
    
    Returns:
        PSI value
    """
    breakpoints = np.percentile(baseline, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    
    baseline_counts = np.histogram(baseline, bins=breakpoints)[0]
    current_counts = np.histogram(current, bins=breakpoints)[0]
    
    baseline_pct = baseline_counts / len(baseline)
    current_pct = current_counts / len(current)
    
    baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
    current_pct = np.where(current_pct == 0, 0.0001, current_pct)
    
    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    
    return float(psi)


def compute_feature_statistics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute feature statistics for drift detection.
    
    Args:
        df: DataFrame with features
    
    Returns:
        Dictionary of feature statistics
    """
    stats = {}
    
    for col in df.select_dtypes(include=[np.number]).columns:
        stats[col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'median': float(df[col].median()),
            'q25': float(df[col].quantile(0.25)),
            'q75': float(df[col].quantile(0.75)),
        }
    
    return stats


def detect_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
    psi_threshold: float = 0.25
) -> Dict[str, Dict]:
    """
    Detect data drift across features.
    
    Args:
        baseline_df: Baseline (training) data
        current_df: Current (inference) data
        features: List of features to check
        psi_threshold: PSI threshold for drift alert
    
    Returns:
        Dictionary with drift results per feature
    """
    drift_results = {}
    
    for feature in features:
        if feature not in baseline_df.columns or feature not in current_df.columns:
            continue
        
        baseline_values = baseline_df[feature].dropna().values
        current_values = current_df[feature].dropna().values
        
        if len(baseline_values) == 0 or len(current_values) == 0:
            continue
        
        psi = calculate_psi(baseline_values, current_values)
        
        baseline_mean = float(baseline_values.mean())
        current_mean = float(current_values.mean())
        mean_shift = abs(current_mean - baseline_mean) / (abs(baseline_mean) + 1e-10)
        
        baseline_std = float(baseline_values.std())
        current_std = float(current_values.std())
        
        drift_results[feature] = {
            'psi': psi,
            'drift_detected': psi >= psi_threshold,
            'baseline_mean': baseline_mean,
            'current_mean': current_mean,
            'mean_shift_pct': mean_shift * 100,
            'baseline_std': baseline_std,
            'current_std': current_std,
        }
    
    return drift_results


def save_baseline_statistics(
    df: pd.DataFrame,
    output_path: Path,
    features: list[str]
) -> None:
    """Save baseline statistics for future drift detection."""
    stats = compute_feature_statistics(df[features])
    
    baseline_data = {
        'features': features,
        'statistics': stats,
        'n_samples': len(df),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(baseline_data, f, indent=2)


def load_baseline_statistics(path: Path) -> Dict:
    """Load baseline statistics from file."""
    with open(path, 'r') as f:
        return json.load(f)


def print_drift_report(drift_results: Dict[str, Dict]) -> None:
    """Print formatted drift detection report."""
    print("\nData Drift Detection Report")
    print("=" * 80)
    
    drifted_features = [f for f, r in drift_results.items() if r['drift_detected']]
    
    if not drifted_features:
        print("No significant drift detected across all features.")
    else:
        print(f"\nWARNING: Drift detected in {len(drifted_features)} feature(s):")
        
        for feature in drifted_features:
            r = drift_results[feature]
            print(f"\n  {feature}:")
            print(f"    PSI:              {r['psi']:.4f} (threshold: 0.25)")
            print(f"    Baseline mean:    {r['baseline_mean']:.4f}")
            print(f"    Current mean:     {r['current_mean']:.4f}")
            print(f"    Mean shift:       {r['mean_shift_pct']:.2f}%")
    
    print("\nFeatures with no drift:")
    stable_features = [f for f, r in drift_results.items() if not r['drift_detected']]
    for feature in stable_features[:10]:
        r = drift_results[feature]
        print(f"  {feature}: PSI = {r['psi']:.4f}")
    
    if len(stable_features) > 10:
        print(f"  ... and {len(stable_features) - 10} more")
    
    print("=" * 80)
