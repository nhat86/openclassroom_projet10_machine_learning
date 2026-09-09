"""
Explainability Module
Implements SHAP and Permutation Importance for model explainability
"""

import pandas as pd
import numpy as np
import joblib
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from typing import Dict, Tuple
import os

from predict import load_pipeline, load_feature_names


def get_shap_explainer(
    pipeline,
    X_background: np.ndarray = None
):
    """
    Create SHAP explainer for the model.
    
    Args:
        pipeline: Trained pipeline
        X_background: Background data for SHAP explainer
        
    Returns:
        SHAP explainer
    """
    model = pipeline.named_steps["model"]
    
    # For Logistic Regression, use LinearExplainer
    if not SHAP_AVAILABLE:
        return None
    
    if X_background is not None:
        explainer = shap.LinearExplainer(model, X_background)
    else:
        # Use model coefficients directly for Logistic Regression
        explainer = None
    
    return explainer


def get_feature_importance_coefficients(
    model_path: str = "models/logistic_regression_pipeline.joblib",
    feature_names_path: str = "models/logistic_regression_pipeline_feature_names.joblib"
) -> pd.DataFrame:
    """
    Get feature importance from model coefficients.
    
    Args:
        model_path: Path to the trained model
        feature_names_path: Path to the feature names file
        
    Returns:
        DataFrame with feature names and coefficients
    """
    # Load pipeline and feature names
    pipeline = load_pipeline(model_path)
    feature_names = load_feature_names(feature_names_path)
    
    # Get coefficients
    model = pipeline.named_steps["model"]
    coef = model.coef_[0]
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coef,
        "abs_coefficient": np.abs(coef)
    })
    
    # Sort by absolute coefficient
    importance_df = importance_df.sort_values("abs_coefficient", ascending=False)
    
    return importance_df


def calculate_permutation_importance(
    pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_repeats: int = 10,
    random_state: int = 42
) -> Dict:
    """
    Calculate permutation importance.
    
    Args:
        pipeline: Trained pipeline
        X_test: Test features
        y_test: Test labels
        n_repeats: Number of permutation repeats
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with permutation importance results
    """
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )
    
    importance_dict = {
        "importances_mean": result.importances_mean,
        "importances_std": result.importances_std,
        "importances": result.importances
    }
    
    return importance_dict


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    save_path: str = None
) -> plt.Figure:
    """
    Plot feature importance bar chart.
    
    Args:
        importance_df: DataFrame with feature importance
        top_n: Number of top features to display
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get top features
    top_features = importance_df.head(top_n)
    
    # Create bar chart
    colors = ['red' if x > 0 else 'green' for x in top_features['coefficient']]
    ax.barh(top_features['feature'], top_features['coefficient'], color=colors)
    ax.set_xlabel('Coefficient Value (Impact on Prediction)')
    ax.set_ylabel('Features')
    ax.set_title('Feature Importance (Model Coefficients)')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_permutation_importance(
    importance_dict: Dict,
    feature_names: list,
    top_n: int = 15,
    save_path: str = None
) -> plt.Figure:
    """
    Plot permutation importance.
    
    Args:
        importance_dict: Dictionary with permutation importance results
        feature_names: List of feature names
        top_n: Number of top features to display
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create DataFrame
    perm_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance_dict["importances_mean"],
        "std": importance_dict["importances_std"]
    })
    
    # Sort and get top features
    perm_df = perm_df.sort_values("importance", ascending=False).head(top_n)
    
    # Create bar chart with error bars
    ax.barh(perm_df['feature'], perm_df['importance'], xerr=perm_df['std'])
    ax.set_xlabel('Permutation Importance (Decrease in Score)')
    ax.set_ylabel('Features')
    ax.set_title('Permutation Importance')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def explain_prediction(
    input_dict: Dict,
    model_path: str = "models/logistic_regression_pipeline.joblib",
    feature_names_path: str = "models/logistic_regression_pipeline_feature_names.joblib"
) -> Dict:
    """
    Explain a single prediction using model coefficients.
    
    Args:
        input_dict: Dictionary containing employee features
        model_path: Path to the trained model
        feature_names_path: Path to the feature names file
        
    Returns:
        Dictionary with explanation
    """
    # Get feature importance
    importance_df = get_feature_importance_coefficients(model_path, feature_names_path)
    
    # Get prediction
    from .predict import predict_attrition
    prediction, probability, _ = predict_attrition(input_dict, model_path)
    
    explanation = {
        "prediction": int(prediction),
        "probability": float(probability[1]),
        "top_features": importance_df.head(10).to_dict('records')
    }
    
    return explanation
