"""
Prediction Module
Implements prediction functions for the trained model
"""

import pandas as pd
import numpy as np
import joblib
from typing import Dict, Tuple
import os
import streamlit as st

from features import create_all_features, map_categorical_features


@st.cache_resource
def _load_pipeline_cached(model_path: str):
    """Cache wrapper for joblib.load to avoid reloading the model."""
    return joblib.load(model_path)


@st.cache_resource
def _load_feature_names_cached(feature_names_path: str):
    """Cache wrapper for joblib.load to avoid reloading feature names."""
    return joblib.load(feature_names_path)


def load_pipeline(
    model_path: str = "models/logistic_regression_pipeline.joblib"
):
    """
    Load the trained pipeline.
    
    Args:
        model_path: Path to the saved pipeline
        
    Returns:
        Trained pipeline
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return _load_pipeline_cached(model_path)


def load_feature_names(
    feature_names_path: str = "models/logistic_regression_pipeline_feature_names.joblib"
) -> list:
    """
    Load feature names for explainability.
    
    Args:
        feature_names_path: Path to the feature names file
        
    Returns:
        List of feature names
    """
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"Feature names file not found: {feature_names_path}")
    
    return _load_feature_names_cached(feature_names_path)


def preprocess_input(input_dict: Dict) -> pd.DataFrame:
    """
    Preprocess a single input dictionary for prediction.
    
    Args:
        input_dict: Dictionary containing employee features
        
    Returns:
        Preprocessed DataFrame
    """
    # Convert to DataFrame
    df = pd.DataFrame([input_dict])
    
    # Apply feature engineering
    df = map_categorical_features(df)
    df = create_all_features(df)
    
    # Fill any NaN values with 0 (for numerical features)
    df = df.fillna(0)
    
    return df


def predict_attrition(
    input_dict: Dict,
    model_path: str = "models/logistic_regression_pipeline.joblib"
) -> Tuple[int, float, np.ndarray, list]:
    """
    Predict attrition risk for a single employee.
    
    Args:
        input_dict: Dictionary containing employee features
        model_path: Path to the trained model
        
    Returns:
        Tuple of (prediction, probability, shap_values, feature_names)
    """
    # Load pipeline
    pipeline = load_pipeline(model_path)
    
    # Preprocess input
    df = preprocess_input(input_dict)
    
    # Get feature columns
    numerical_cols, categorical_cols = pipeline.named_steps["preprocessor"].transformers_[0][2], \
                                     pipeline.named_steps["preprocessor"].transformers_[1][2]
    feature_cols = numerical_cols + categorical_cols
    
    # Select features
    X = df[feature_cols]
    
    # Make prediction
    prediction = pipeline.predict(X)[0]
    probability = pipeline.predict_proba(X)[0]
    
    # Calculate local feature contribution using model coefficients * feature values
    # For Logistic Regression, contribution = coefficient * (feature_value - mean)
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    
    # Get transformed features
    X_transformed = preprocessor.transform(X)
    
    # Get feature names after transformation
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if hasattr(transformer, 'get_feature_names_out'):
            feature_names.extend(transformer.get_feature_names_out(cols))
        else:
            feature_names.extend(cols)
    
    # Calculate contributions: coefficient * feature value
    coef = model.coef_[0]
    contributions = coef * X_transformed[0]
    
    return prediction, probability, contributions, feature_names


def get_risk_level(probability: float) -> str:
    """
    Convert probability to risk level.
    
    Args:
        probability: Probability of leaving (0-1)
        
    Returns:
        Risk level string
    """
    if probability >= 0.6:
        return "Risque élevé"
    elif probability >= 0.3:
        return "Risque modéré"
    else:
        return "Faible risque"
