"""
Training Module
Implements model training with Logistic Regression
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    precision_recall_curve
)
from sklearn.pipeline import Pipeline
from typing import Tuple, Dict
import os
import sys

# Handle both relative and absolute imports
try:
    from .features import load_and_merge_data, clean_data, create_all_features
    from .preprocessing import create_preprocessor, prepare_features, get_feature_names, get_feature_columns
except ImportError:
    from features import load_and_merge_data, clean_data, create_all_features
    from preprocessing import create_preprocessor, prepare_features, get_feature_names, get_feature_columns


def train_model(
    sirh_path: str,
    eval_path: str,
    sondage_path: str,
    model_output_path: str = "models/logistic_regression_pipeline.joblib"
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Train Logistic Regression model and save the complete pipeline.
    
    Args:
        sirh_path: Path to SIRH CSV file
        eval_path: Path to evaluation CSV file
        sondage_path: Path to survey CSV file
        model_output_path: Path to save the trained pipeline
        
    Returns:
        Tuple of (trained pipeline, metrics dictionary)
    """
    print("=" * 60)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("=" * 60)
    
    # Load and merge data
    print("\n1. Loading and merging data...")
    df_central = load_and_merge_data(sirh_path, eval_path, sondage_path)
    print(f"   Loaded {len(df_central)} records")
    
    # Clean data
    print("\n2. Cleaning data...")
    df_clean = clean_data(df_central)
    print(f"   Columns after cleaning: {df_clean.shape[1]}")
    
    # Feature engineering
    print("\n3. Creating engineered features...")
    df_features = create_all_features(df_clean)
    print(f"   Columns after feature engineering: {df_features.shape[1]}")
    
    # Prepare target variable
    print("\n4. Preparing target variable...")
    y = df_features["a_quitte_l_entreprise"].map({"Non": 0, "Oui": 1})
    print(f"   Target distribution: {y.value_counts().to_dict()}")
    
    # Get feature columns for splitting
    numerical_cols, categorical_cols = get_feature_columns()
    feature_cols = numerical_cols + categorical_cols
    X = df_features[feature_cols]
    
    # Split data
    print("\n5. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Create preprocessor
    print("\n6. Creating preprocessing pipeline...")
    preprocessor = create_preprocessor()
    
    # Create and train model
    print("\n7. Training Logistic Regression model...")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=42
    )
    
    # Create complete pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    # Fit pipeline on training data (DataFrame)
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    print("\n9. Evaluating model...")
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save pipeline
    print("\n10. Saving pipeline...")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    print(f"   Pipeline saved to: {model_output_path}")
    
    # Save feature names for explainability
    feature_names = get_feature_names(preprocessor)
    feature_names_path = model_output_path.replace(".joblib", "_feature_names.joblib")
    joblib.dump(feature_names, feature_names_path)
    print(f"   Feature names saved to: {feature_names_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return pipeline, metrics


if __name__ == "__main__":
    # Update these paths to your actual data file locations
    SIRH_PATH = "../extrait_sirh.csv"
    EVAL_PATH = "../extrait_eval.csv"
    SONDAGE_PATH = "../extrait_sondage.csv"
    
    # Train the model
    pipeline, metrics = train_model(SIRH_PATH, EVAL_PATH, SONDAGE_PATH)
