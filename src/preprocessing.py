"""
Preprocessing Module
Implements data preprocessing pipeline with ColumnTransformer
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from typing import Tuple, List


def get_feature_columns() -> Tuple[List[str], List[str]]:
    """
    Define numerical and categorical feature columns.
    
    Returns:
        Tuple of (numerical_cols, categorical_cols)
    """
    numerical_cols = [
        "age",
        "genre",
        "nombre_experiences_precedentes",
        "annee_experience_totale",
        "annees_dans_l_entreprise",
        "annees_dans_le_poste_actuel",
        "satisfaction_employee_environnement",
        "note_evaluation_precedente",
        "niveau_hierarchique_poste",
        "satisfaction_employee_nature_travail",
        "satisfaction_employee_equipe",
        "satisfaction_employee_equilibre_pro_perso",
        "note_evaluation_actuelle",
        "heure_supplementaires",
        "augementation_salaire_precedente",
        "nombre_participation_pee",
        "nb_formations_suivies",
        "distance_domicile_travail",
        "niveau_education",
        "frequence_deplacement",
        "annees_depuis_la_derniere_promotion",
        "annes_sous_responsable_actuel",
        "score_satisfaction_global",
        "taux_promotion"
    ]
    
    categorical_cols = [
        "statut_marital",
        "departement",
        "poste",
        "poste_heures_supp",
        "domaine_etude"
    ]
    
    return numerical_cols, categorical_cols


def create_preprocessor() -> ColumnTransformer:
    """
    Create preprocessing pipeline with StandardScaler and OneHotEncoder.
    
    Returns:
        ColumnTransformer with preprocessing steps
    """
    numerical_cols, categorical_cols = get_feature_columns()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="drop"
    )
    
    return preprocessor


def prepare_features(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    fit: bool = False
) -> np.ndarray:
    """
    Prepare features for model training or prediction.
    
    Args:
        df: Input DataFrame
        preprocessor: Fitted or unfitted ColumnTransformer
        fit: Whether to fit the preprocessor on the data
        
    Returns:
        Processed feature matrix
    """
    numerical_cols, categorical_cols = get_feature_columns()
    
    # Select features
    X = df[numerical_cols + categorical_cols]
    
    if fit:
        X_processed = preprocessor.fit_transform(X)
    else:
        X_processed = preprocessor.transform(X)
    
    return X_processed


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Get feature names after preprocessing.
    
    Args:
        preprocessor: Fitted ColumnTransformer
        
    Returns:
        List of feature names
    """
    numerical_cols, categorical_cols = get_feature_columns()
    
    # Get numerical feature names
    num_feature_names = numerical_cols
    
    # Get categorical feature names from OneHotEncoder
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
    
    # Combine all feature names
    all_feature_names = list(num_feature_names) + list(cat_feature_names)
    
    return all_feature_names
