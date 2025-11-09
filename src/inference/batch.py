"""Batch inference service for fraud detection."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import json
from datetime import datetime

from ..data.preprocessing import load_pipeline, validate_schema
from ..models.train import load_model


class BatchInferenceService:
    """Service for batch fraud detection predictions."""
    
    def __init__(
        self,
        model_path: Path,
        preprocessing_path: Path
    ):
        """
        Initialize inference service.
        
        Args:
            model_path: Path to trained model file
            preprocessing_path: Path to preprocessing pipeline
        """
        self.model = load_model(model_path.parent)
        self.preprocessing = load_pipeline(preprocessing_path)
        self.model_path = model_path
        self.preprocessing_path = preprocessing_path
    
    def predict(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
        include_probabilities: bool = True
    ) -> pd.DataFrame:
        """
        Generate predictions for a batch of transactions.
        
        Args:
            df: DataFrame with transaction features
            threshold: Classification threshold (uses model's optimal if None)
            include_probabilities: Whether to include probability scores
        
        Returns:
            DataFrame with predictions and optionally probabilities
        """
        validate_schema(df)
        
        X_processed = self.preprocessing.transform(df)
        
        if isinstance(X_processed, np.ndarray):
            feature_cols = self.preprocessing.named_steps['feature_select'].feature_cols
            X_processed = pd.DataFrame(
                X_processed,
                columns=feature_cols,
                index=df.index
            )
        
        y_pred = self.model.predict(X_processed, threshold=threshold)
        
        results = pd.DataFrame({
            'prediction': y_pred,
            'predicted_label': ['fraud' if p == 1 else 'legitimate' for p in y_pred]
        }, index=df.index)
        
        if include_probabilities:
            y_proba = self.model.predict_proba(X_processed)
            results['fraud_probability'] = y_proba
            results['confidence'] = np.where(
                y_pred == 1,
                y_proba,
                1 - y_proba
            )
        
        return results
    
    def predict_from_csv(
        self,
        input_path: Path,
        output_path: Path,
        threshold: Optional[float] = None
    ) -> Dict[str, any]:
        """
        Run batch inference on CSV file.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to save predictions CSV
            threshold: Classification threshold
        
        Returns:
            Dictionary with inference statistics
        """
        df = pd.read_csv(input_path)
        
        start_time = datetime.now()
        predictions = self.predict(df, threshold=threshold)
        inference_time = (datetime.now() - start_time).total_seconds()
        
        output_df = df.copy()
        for col in predictions.columns:
            output_df[col] = predictions[col]
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        
        stats = {
            'total_transactions': len(df),
            'fraud_detected': int(predictions['prediction'].sum()),
            'fraud_rate': float(predictions['prediction'].mean()),
            'inference_time_sec': inference_time,
            'throughput_per_sec': len(df) / inference_time if inference_time > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        return stats


def run_batch_inference(
    model_dir: Path,
    preprocessing_path: Path,
    input_csv: Path,
    output_csv: Path,
    threshold: Optional[float] = None
) -> None:
    """
    Run batch inference from command line.
    
    Args:
        model_dir: Directory containing trained model
        preprocessing_path: Path to preprocessing pipeline
        input_csv: Input CSV file with transactions
        output_csv: Output CSV file for predictions
        threshold: Optional classification threshold
    """
    service = BatchInferenceService(
        model_path=model_dir / 'model.pkl',
        preprocessing_path=preprocessing_path
    )
    
    stats = service.predict_from_csv(input_csv, output_csv, threshold)
    
    print(f"\nBatch Inference Complete:")
    print(f"  Transactions:   {stats['total_transactions']:,}")
    print(f"  Fraud Detected: {stats['fraud_detected']:,} ({100*stats['fraud_rate']:.2f}%)")
    print(f"  Inference Time: {stats['inference_time_sec']:.2f}s")
    print(f"  Throughput:     {stats['throughput_per_sec']:.0f} tx/sec")
    print(f"  Output:         {output_csv}")
    
    stats_path = output_csv.parent / 'inference_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved:    {stats_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch fraud detection inference')
    parser.add_argument('--model-dir', type=Path, required=True, help='Model directory')
    parser.add_argument('--preprocessing', type=Path, required=True, help='Preprocessing pipeline path')
    parser.add_argument('--input', type=Path, required=True, help='Input CSV file')
    parser.add_argument('--output', type=Path, required=True, help='Output CSV file')
    parser.add_argument('--threshold', type=float, help='Classification threshold')
    
    args = parser.parse_args()
    
    run_batch_inference(
        model_dir=args.model_dir,
        preprocessing_path=args.preprocessing,
        input_csv=args.input,
        output_csv=args.output,
        threshold=args.threshold
    )
