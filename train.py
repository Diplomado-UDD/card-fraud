"""Main training script for fraud detection models."""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from src.data.preprocessing import (
    create_preprocessing_pipeline,
    preprocess_data,
    save_pipeline,
    validate_schema
)
from src.data.splits import create_splits, save_splits, print_split_stats
from src.models.train import (
    FraudDetectionModel,
    evaluate_model,
    save_model,
    print_metrics
)


def main():
    parser = argparse.ArgumentParser(description='Train fraud detection model')
    parser.add_argument('--data', type=Path, required=True, help='Path to creditcard.csv')
    parser.add_argument('--output-dir', type=Path, default=Path('models/runs'), help='Output directory')
    parser.add_argument('--model-type', type=str, default='logistic',
                        choices=['logistic', 'xgboost', 'lightgbm', 'random_forest'],
                        help='Model type')
    parser.add_argument('--skip-splits', action='store_true', help='Use existing splits')
    parser.add_argument('--optimize-threshold', action='store_true', help='Optimize classification threshold')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print(f"Training fraud detection model...")
    print(f"  Model type: {args.model_type}")
    print(f"  Data: {args.data}")
    print(f"  Output: {args.output_dir}")
    
    np.random.seed(args.random_seed)
    
    splits_dir = Path('data/splits')
    
    if args.skip_splits and splits_dir.exists():
        print(f"\nLoading existing splits from {splits_dir}")
        train_df = pd.read_csv(splits_dir / 'train.csv')
        val_df = pd.read_csv(splits_dir / 'val.csv')
        test_df = pd.read_csv(splits_dir / 'test.csv')
    else:
        print(f"\nLoading data from {args.data}")
        df = pd.read_csv(args.data)
        validate_schema(df)
        
        print(f"Dataset: {len(df):,} transactions")
        print(f"Fraud rate: {100 * df['Class'].mean():.3f}%")
        
        print("\nCreating stratified train/val/test splits...")
        train_df, val_df, test_df = create_splits(
            df,
            target_col='Class',
            random_state=args.random_seed
        )
        
        print_split_stats(train_df, val_df, test_df)
        
        save_splits(train_df, val_df, test_df, splits_dir)
        print(f"Splits saved to {splits_dir}")
    
    print("\nPreprocessing data...")
    pipeline = create_preprocessing_pipeline()
    
    X_train, pipeline = preprocess_data(train_df, pipeline=pipeline, fit=True)
    y_train = train_df['Class']
    
    X_val, _ = preprocess_data(val_df, pipeline=pipeline, fit=False)
    y_val = val_df['Class']
    
    X_test, _ = preprocess_data(test_df, pipeline=pipeline, fit=False)
    y_test = test_df['Class']
    
    print(f"Training features: {X_train.shape}")
    print(f"Validation features: {X_val.shape}")
    print(f"Test features: {X_test.shape}")
    
    preprocessing_path = args.output_dir / 'preprocessing_pipeline.pkl'
    save_pipeline(pipeline, preprocessing_path)
    print(f"Preprocessing pipeline saved to {preprocessing_path}")
    
    print(f"\nTraining {args.model_type} model...")
    model = FraudDetectionModel(
        model_type=args.model_type,
        random_state=args.random_seed
    )
    model.fit(X_train, y_train)
    print("Training complete")
    
    print("\nEvaluating on training set...")
    train_metrics = evaluate_model(model, X_train, y_train)
    print_metrics(train_metrics, "Training")
    
    print("\nEvaluating on validation set...")
    val_metrics = evaluate_model(model, X_val, y_val)
    print_metrics(val_metrics, "Validation")
    
    if args.optimize_threshold:
        print("\nOptimizing classification threshold on validation set...")
        optimal_threshold = model.optimize_threshold(
            X_val, y_val, metric='f1'
        )
        print(f"Optimal threshold: {optimal_threshold:.4f}")
        
        val_metrics_opt = evaluate_model(model, X_val, y_val, threshold=optimal_threshold)
        print_metrics(val_metrics_opt, "Validation (optimized)")
    
    print("\nEvaluating on test set (holdout)...")
    test_metrics = evaluate_model(model, X_test, y_test)
    print_metrics(test_metrics, "Test")
    
    print("\nAcceptance criteria check:")
    passed = True
    if test_metrics['f1_score'] < 0.75:
        print(f"  FAIL: F1-score {test_metrics['f1_score']:.4f} < 0.75")
        passed = False
    else:
        print(f"  PASS: F1-score {test_metrics['f1_score']:.4f} >= 0.75")
    
    if test_metrics['precision'] < 0.80:
        print(f"  FAIL: Precision {test_metrics['precision']:.4f} < 0.80")
        passed = False
    else:
        print(f"  PASS: Precision {test_metrics['precision']:.4f} >= 0.80")
    
    if test_metrics['recall'] < 0.70:
        print(f"  FAIL: Recall {test_metrics['recall']:.4f} < 0.70")
        passed = False
    else:
        print(f"  PASS: Recall {test_metrics['recall']:.4f} >= 0.70")
    
    if test_metrics['pr_auc'] < 0.70:
        print(f"  FAIL: PR-AUC {test_metrics['pr_auc']:.4f} < 0.70")
        passed = False
    else:
        print(f"  PASS: PR-AUC {test_metrics['pr_auc']:.4f} >= 0.70")
    
    status = "PASSED" if passed else "FAILED"
    print(f"\nAcceptance criteria: {status}")
    
    metadata = {
        'model_type': args.model_type,
        'random_seed': args.random_seed,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'acceptance_status': status
    }
    
    save_model(
        model,
        args.output_dir,
        metrics={
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        },
        metadata=metadata
    )
    
    print(f"\nModel artifacts saved to {args.output_dir}")
    print(f"  - model.pkl")
    print(f"  - model_info.json")
    print(f"  - preprocessing_pipeline.pkl")


if __name__ == '__main__':
    main()
