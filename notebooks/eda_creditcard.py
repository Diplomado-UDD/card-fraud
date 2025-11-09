import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    import warnings
    warnings.filterwarnings('ignore')

    mo.md(
        """
        # Credit Card Fraud Detection - Exploratory Data Analysis

        **Objective**: Comprehensive analysis of the credit card transaction dataset to inform 
        model development and feature engineering strategies.

        **Dataset**: creditcard.csv (~151 MB, 284,807 transactions)

        **Target**: Binary classification of fraudulent transactions (Class: 0=legitimate, 1=fraud)
        """
    )
    return Path, mo, pd, plt, sns


@app.cell
def _(Path, mo):
    data_path = Path(__file__).parent.parent / "data" / "creditcard.csv"

    if not data_path.exists():
        mo.md(f"**Error**: Dataset not found at {data_path}")
    else:
        mo.md(f"**Loading dataset from**: `{data_path}`")
    return (data_path,)


@app.cell
def _(data_path, pd):
    df = pd.read_csv(data_path)
    return (df,)


@app.cell
def _(df, mo):
    mo.md(f"""
    ## Dataset Overview

    - **Total transactions**: {len(df):,}
    - **Features**: {df.shape[1]} columns
    - **Memory usage**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ### First 10 rows
    """)
    return


@app.cell
def _(df):
    df.head(10)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Dataset Info
    """)
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df, mo):
    basic_stats = df.describe()
    mo.md("### Statistical Summary")
    return (basic_stats,)


@app.cell
def _(basic_stats):
    basic_stats
    return


@app.cell
def _(df, mo):
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()

    mo.md(
        f"""
        ## Data Quality Checks

        ### Missing Values

        **Total missing values**: {total_missing}

        {f"**Columns with missing values**: {missing_values[missing_values > 0].to_dict()}" if total_missing > 0 else "**Result**: No missing values detected."}
        """
    )
    return


@app.cell
def _(df, mo):
    duplicates = df.duplicated().sum()

    mo.md(
        f"""
        ### Duplicate Rows

        **Total duplicates**: {duplicates:,}

        {f"**Action required**: Consider removing or investigating {duplicates} duplicate transactions." if duplicates > 0 else "**Result**: No duplicate rows detected."}
        """
    )
    return


@app.cell
def _(df, mo):
    class_counts = df['Class'].value_counts().sort_index()
    fraud_percentage = (class_counts[1] / len(df)) * 100
    imbalance_ratio = class_counts[0] / class_counts[1]

    mo.md(
        f"""
        ## Class Distribution Analysis

        ### Target Variable: Class

        - **Legitimate transactions (Class=0)**: {class_counts[0]:,} ({100-fraud_percentage:.2f}%)
        - **Fraudulent transactions (Class=1)**: {class_counts[1]:,} ({fraud_percentage:.3f}%)
        - **Imbalance ratio**: {imbalance_ratio:.1f}:1

        **Key insight**: This is a highly imbalanced dataset. Evaluation metrics like F1-score, 
        Precision-Recall AUC, and confusion matrix analysis are critical. ROC-AUC alone will be 
        misleading.

        **Recommendation**: 
        - Use stratified train/test splits
        - Apply class weighting in models (e.g., `class_weight='balanced'`)
        - Consider threshold tuning for precision/recall tradeoff
        - Monitor precision@k for operational scenarios
        """
    )
    return (class_counts,)


@app.cell
def _(class_counts, plt):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    class_counts.plot(kind='bar', ax=ax1, color=['#2ecc71', '#e74c3c'])
    ax1.set_title('Class Distribution (Absolute Counts)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    ax1.set_xticklabels(['Legitimate (0)', 'Fraud (1)'], rotation=0)

    class_counts.plot(kind='bar', ax=ax2, color=['#2ecc71', '#e74c3c'], logy=True)
    ax2.set_title('Class Distribution (Log Scale)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Count (log scale)')
    ax2.set_xticklabels(['Legitimate (0)', 'Fraud (1)'], rotation=0)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(df, mo):
    time_stats = df['Time'].describe()
    amount_stats = df['Amount'].describe()

    mo.md(
        f"""
        ## Feature Analysis

        ### Time Feature

        - **Min**: {time_stats['min']:.0f}s
        - **Max**: {time_stats['max']:.0f}s ({time_stats['max']/3600:.1f} hours)
        - **Mean**: {time_stats['mean']:.0f}s
        - **Interpretation**: Time represents seconds elapsed since first transaction. 
          Dataset spans approximately {time_stats['max']/3600:.1f} hours (~{time_stats['max']/86400:.1f} days).

        ### Amount Feature

        - **Min**: ${amount_stats['min']:.2f}
        - **Max**: ${amount_stats['max']:.2f}
        - **Mean**: ${amount_stats['mean']:.2f}
        - **Median**: ${amount_stats['50%']:.2f}
        - **Std**: ${amount_stats['std']:.2f}

        **Note**: Amount is right-skewed (mean > median). Consider log transformation for modeling.
        """
    )
    return


@app.cell
def _(df, mo):
    v_columns = [col for col in df.columns if col.startswith('V')]

    mo.md(
        f"""
        ### V1-V28 Features

        **Number of PCA features**: {len(v_columns)}

        These features are the result of PCA transformation applied for confidentiality. 
        They are already scaled and anonymized, which:

        - Protects sensitive cardholder information (PCI/GDPR compliance)
        - Reduces dimensionality
        - Are already normalized (no additional scaling needed for most algorithms)

        **Action**: Analyze correlation structure and feature importance post-modeling.
        """
    )
    return (v_columns,)


@app.cell
def _(df, plt):
    fig_amount, ax_amount = plt.subplots(1, 2, figsize=(14, 4))

    df[df['Class'] == 0]['Amount'].hist(bins=50, ax=ax_amount[0], color='#2ecc71', alpha=0.7, edgecolor='black')
    ax_amount[0].set_title('Amount Distribution - Legitimate Transactions', fontsize=11, fontweight='bold')
    ax_amount[0].set_xlabel('Amount ($)')
    ax_amount[0].set_ylabel('Frequency')
    ax_amount[0].set_xlim(0, 1000)

    df[df['Class'] == 1]['Amount'].hist(bins=50, ax=ax_amount[1], color='#e74c3c', alpha=0.7, edgecolor='black')
    ax_amount[1].set_title('Amount Distribution - Fraudulent Transactions', fontsize=11, fontweight='bold')
    ax_amount[1].set_xlabel('Amount ($)')
    ax_amount[1].set_ylabel('Frequency')
    ax_amount[1].set_xlim(0, 1000)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(df, mo):
    fraud_amount_stats = df[df['Class'] == 1]['Amount'].describe()
    legit_amount_stats = df[df['Class'] == 0]['Amount'].describe()

    mo.md(
        f"""
        ### Amount by Class

        | Statistic | Legitimate (Class=0) | Fraud (Class=1) |
        |-----------|---------------------|-----------------|
        | Mean      | ${legit_amount_stats['mean']:.2f} | ${fraud_amount_stats['mean']:.2f} |
        | Median    | ${legit_amount_stats['50%']:.2f} | ${fraud_amount_stats['50%']:.2f} |
        | Max       | ${legit_amount_stats['max']:.2f} | ${fraud_amount_stats['max']:.2f} |

        **Observation**: Fraudulent transactions tend to have lower amounts on average, 
        suggesting small frauds are more common. However, variance is high.
        """
    )
    return


@app.cell
def _(df, plt):
    sample_v_features = ['V1', 'V2', 'V3', 'V4', 'V12', 'V14', 'V17']

    fig_v, axes_v = plt.subplots(2, 4, figsize=(16, 8))
    axes_v = axes_v.flatten()

    for idx, feature in enumerate(sample_v_features):
        df.boxplot(column=feature, by='Class', ax=axes_v[idx])
        axes_v[idx].set_title(f'{feature} by Class')
        axes_v[idx].set_xlabel('Class')
        axes_v[idx].set_ylabel(feature)

    axes_v[-1].axis('off')

    plt.suptitle('Sample V Features Distribution by Class', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Correlation Analysis
    """)
    return


@app.cell
def _(df, v_columns):
    correlation_with_class = df[v_columns + ['Amount', 'Class']].corr()['Class'].drop('Class').sort_values(ascending=False)
    return (correlation_with_class,)


@app.cell
def _(correlation_with_class, mo):
    top_positive = correlation_with_class.head(5)
    top_negative = correlation_with_class.tail(5)

    mo.md(
        f"""
        ### Top Features Correlated with Class

        **Positive Correlations** (higher values → more likely fraud):

        {chr(10).join([f"- **{feat}**: {corr:.4f}" for feat, corr in top_positive.items()])}

        **Negative Correlations** (lower values → more likely fraud):

        {chr(10).join([f"- **{feat}**: {corr:.4f}" for feat, corr in top_negative.items()])}

        **Note**: Most correlations are weak (|r| < 0.3), typical for complex fraud patterns.
        Tree-based models and neural networks may capture non-linear interactions better.
        """
    )
    return


@app.cell
def _(correlation_with_class, plt):
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    correlation_with_class.sort_values(ascending=True).plot(kind='barh', ax=ax_corr, color='steelblue')
    ax_corr.set_title('Feature Correlation with Class (Fraud)', fontsize=13, fontweight='bold')
    ax_corr.set_xlabel('Correlation Coefficient')
    ax_corr.set_ylabel('Feature')
    ax_corr.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(df, mo, pd, plt, sns, v_columns):
    fraud_sample = df[df['Class'] == 1].sample(min(500, len(df[df['Class'] == 1])), random_state=42)
    legit_sample = df[df['Class'] == 0].sample(500, random_state=42)
    balanced_sample = pd.concat([fraud_sample, legit_sample])

    top_features_for_heatmap = v_columns[:14]

    fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        balanced_sample[top_features_for_heatmap].corr(), 
        annot=False, 
        cmap='coolwarm', 
        center=0,
        ax=ax_heatmap,
        square=True
    )
    ax_heatmap.set_title('Correlation Heatmap (V1-V14, Balanced Sample)', fontsize=13, fontweight='bold')
    plt.tight_layout()

    mo.md("### Feature Intercorrelation (Sample: V1-V14)")
    return


@app.cell
def _(plt):
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Key Findings and Recommendations

    ### Data Quality
    - No missing values detected
    - Check for duplicate transactions (if any, investigate before removal)
    - Dataset is clean and ready for modeling

    ### Class Imbalance
    - Fraud rate: ~0.17% (highly imbalanced)
    - **Critical**: Use stratified splits and class weighting
    - Evaluation must focus on Precision, Recall, F1, PR-AUC (not just accuracy)

    ### Feature Engineering Opportunities
    1. **Time-based features**:
       - Hour of day (cyclical encoding: sin/cos)
       - Day of week (if multi-day dataset)
       - Time since last transaction (if sorted by cardholder)

    2. **Amount transformations**:
       - Log1p(Amount) to handle skewness
       - Amount bins (categorical)
       - Amount_zscore (scaled)

    3. **Interaction features**:
       - V_feature * Amount
       - Ratio features

    4. **Aggregated features** (if cardholder ID available):
       - Transaction velocity
       - Average amount per cardholder
       - Deviation from typical behavior

    ### Modeling Strategy
    1. **Baseline models**:
       - Logistic Regression (with class_weight='balanced')
       - Random Forest / XGBoost with scale_pos_weight

    2. **Advanced models**:
       - LightGBM (fast, handles imbalance well)
       - Neural Networks (if dataset is large enough after augmentation)
       - Isolation Forest or Autoencoders (anomaly detection approach)

    3. **Evaluation pipeline**:
       - Stratified K-Fold cross-validation
       - Hold-out test set (stratified, time-based if applicable)
       - Metrics: F1, Precision@k, Recall@k, PR-AUC, Confusion Matrix
       - Threshold tuning for operational precision/recall balance

    4. **Resampling considerations**:
       - Start without resampling (use class weights)
       - Experiment with SMOTE or RandomUnderSampler if needed
       - Always validate on original distribution

    ### Compliance (PCI/GDPR)
    - V1-V28 are already anonymized (PCA-transformed)
    - Ensure no raw cardholder data (PAN, CVV, etc.) is logged or stored
    - Encrypt datasets at rest (local, S3) and in transit
    - Implement access controls and audit trails for model artifacts
    - Document data retention and model decision explainability (for GDPR Article 22)

    ### Next Steps
    1. Implement preprocessing pipeline (scaling Amount, time features)
    2. Create stratified train/validation/test splits
    3. Build baseline models with class weighting
    4. Tune decision threshold for business metrics (cost of false positive vs. false negative)
    5. Set up experiment tracking (MLflow)
    6. Design batch inference pipeline
    7. Implement monitoring for data drift and model performance degradation
    """)
    return


@app.cell
def _(df, mo):
    dataset_summary = {
        "total_transactions": len(df),
        "features": df.shape[1],
        "fraud_count": int(df['Class'].sum()),
        "fraud_percentage": float((df['Class'].sum() / len(df)) * 100),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "time_span_hours": float(df['Time'].max() / 3600),
        "amount_mean": float(df['Amount'].mean()),
        "amount_max": float(df['Amount'].max())
    }

    mo.md(
        f"""
        ## Summary Export

        **Dataset Metadata** (for pipeline configuration):

        ```json
        {dataset_summary}
        ```

        This summary can be saved and used for:
        - CI/CD data validation checks
        - Model card documentation
        - Monitoring baseline expectations
        """
    )
    return


if __name__ == "__main__":
    app.run()
