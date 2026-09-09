"""
Feature Engineering Module
Implements feature creation for employee attrition prediction
"""

import pandas as pd
import numpy as np
from typing import Union


def create_satisfaction_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create global satisfaction score as the mean of 4 satisfaction variables.
    
    Args:
        df: DataFrame containing satisfaction columns
        
    Returns:
        DataFrame with new score_satisfaction_global column
    """
    df = df.copy()
    df["score_satisfaction_global"] = round((
        df["satisfaction_employee_environnement"]
        + df["satisfaction_employee_nature_travail"]
        + df["satisfaction_employee_equipe"]
        + df["satisfaction_employee_equilibre_pro_perso"]
    ) / 4, 2)
    return df


def create_promotion_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create promotion rate: years since last promotion / (years in company + 1)
    
    Args:
        df: DataFrame containing years columns
        
    Returns:
        DataFrame with new taux_promotion column
    """
    df = df.copy()
    df["taux_promotion"] = round((
        df["annees_depuis_la_derniere_promotion"]
        / (df["annees_dans_l_entreprise"] + 1)
    ), 2)
    return df


def create_poste_heures_supp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create combined feature: poste + heure_supplementaires
    
    Args:
        df: DataFrame containing poste and heure_supplementaires columns
        
    Returns:
        DataFrame with new poste_heures_supp column
    """
    df = df.copy()
    df["poste_heures_supp"] = (
        df["poste"].astype(str) + "_" + df["heure_supplementaires"].astype(str)
    )
    return df


def map_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map categorical features to numerical values.
    
    Args:
        df: DataFrame with categorical columns
        
    Returns:
        DataFrame with mapped numerical values
    """
    df = df.copy()
    
    # Binary features
    df["genre"] = df["genre"].map({"F": 0, "M": 1})
    df["heure_supplementaires"] = df["heure_supplementaires"].map({"Oui": 1, "Non": 0})
    
    # Ordinal features
    df["frequence_deplacement"] = df["frequence_deplacement"].map({
        "Aucun": 0,
        "Occasionnel": 1,
        "Frequent": 2
    })
    
    # Convert percentage to numeric
    df["augementation_salaire_precedente"] = (
        df["augementation_salaire_precedente"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.strip()
        .astype(int)
    )
    
    return df


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with all engineered features
    """
    df = df.copy()
    
    # Map categorical features first
    df = map_categorical_features(df)
    
    # Create engineered features
    df = create_satisfaction_score(df)
    df = create_promotion_rate(df)
    df = create_poste_heures_supp(df)
    
    return df


def load_and_merge_data(
    sirh_path: str,
    eval_path: str,
    sondage_path: str
) -> pd.DataFrame:
    """
    Load and merge the three data sources.
    
    Args:
        sirh_path: Path to SIRH CSV file
        eval_path: Path to evaluation CSV file
        sondage_path: Path to survey CSV file
        
    Returns:
        Merged DataFrame
    """
    df_sirh = pd.read_csv(sirh_path)
    df_eval = pd.read_csv(eval_path)
    df_sondage = pd.read_csv(sondage_path)
    
    # Extract employee ID from eval_number
    df_eval["id_employee"] = (
        df_eval["eval_number"]
        .str.replace("E_", "", regex=False)
        .astype(int)
    )
    
    # Merge dataframes
    df_central = (
        df_sirh
        .merge(df_eval, on="id_employee", how="inner", validate="one_to_one")
        .merge(df_sondage, left_on="id_employee", right_on="code_sondage",
               how="inner", validate="one_to_one")
    )
    
    return df_central


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unnecessary columns and clean data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    
    # Drop ID columns
    df = df.drop(columns=["id_employee", "code_sondage", "eval_number"])
    
    # Drop columns with single values
    df = df.drop(columns=[
        "ayant_enfants",
        "nombre_employee_sous_responsabilite",
        "nombre_heures_travailless"
    ])
    
    return df
