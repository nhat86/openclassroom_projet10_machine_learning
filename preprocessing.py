import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


def load_and_merge_data(sirh_path, eval_path, sondage_path):
    """Load and merge the three data sources"""
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


def clean_data(df):
    """Remove unnecessary columns and clean data"""
    # Drop ID columns
    df = df.drop(columns=["id_employee", "code_sondage", "eval_number"])
    
    # Drop columns with single values
    df = df.drop(columns=[
        "ayant_enfants",
        "nombre_employee_sous_responsabilite",
        "nombre_heures_travailless"
    ])
    
    return df


def map_categorical_features(df):
    """Map categorical features to numerical values"""
    df = df.copy()
    
    # Map binary features
    df["genre"] = df["genre"].map({"F": 0, "M": 1})
    df["heure_supplementaires"] = df["heure_supplementaires"].map({"Oui": 1, "Non": 0})
    
    # Map ordinal features
    df["frequence_deplacement"] = df["frequence_deplacement"].map({
        "Aucun": 0,
        "Occasionnel": 1,
        "Frequent": 2
    })
    
    # Convert percentage to numeric
    df["augementation_salaire_precedente"] = (
        df["augementation_salaire_precedente"]
        .str.replace("%", "", regex=False)
        .str.strip()
        .astype(int)
    )
    
    return df


def create_features(df):
    """Create engineered features"""
    df = df.copy()
    
    # Global satisfaction score
    df["score_satisfaction_global"] = round((
        df["satisfaction_employee_environnement"]
        + df["satisfaction_employee_nature_travail"]
        + df["satisfaction_employee_equipe"]
        + df["satisfaction_employee_equilibre_pro_perso"]
    ) / 4, 2)
    
    # Promotion rate
    df["taux_promotion"] = round((
        df["annees_depuis_la_derniere_promotion"]
        / (df["annees_dans_l_entreprise"] + 1)
    ), 2)
    
    # Post + overtime feature
    df["poste_heures_supp"] = (
        df["poste"].astype(str) + "_" + df["heure_supplementaires"].astype(str)
    )
    
    return df


def prepare_features(df, fit=False, preprocessor=None):
    """Prepare features for model training or prediction"""
    # Define categorical and numerical columns
    categorical_cols = [
        "statut_marital", "departement", "poste", "poste_heures_supp", "domaine_etude"
    ]
    
    numerical_cols = [
        "age", "genre", "revenu_mensuel", "nombre_experiences_precedentes",
        "annee_experience_totale", "annees_dans_l_entreprise", 
        "annees_dans_le_poste_actuel", "satisfaction_employee_environnement",
        "note_evaluation_precedente", "niveau_hierarchique_poste",
        "satisfaction_employee_nature_travail", "satisfaction_employee_equipe",
        "satisfaction_employee_equilibre_pro_perso", "note_evaluation_actuelle",
        "heure_supplementaires", "augementation_salaire_precedente",
        "nombre_participation_pee", "nb_formations_suivies",
        "distance_domicile_travail", "niveau_education",
        "frequence_deplacement", "annees_depuis_la_derniere_promotion",
        "annes_sous_responsable_actuel", "score_satisfaction_global", "taux_promotion"
    ]
    
    # Create preprocessor
    if preprocessor is None or fit:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", numerical_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
            ]
        )
    
    # Select features
    X = df[numerical_cols + categorical_cols]
    
    if fit:
        X_processed = preprocessor.fit_transform(X)
    else:
        X_processed = preprocessor.transform(X)
    
    return X_processed, preprocessor


def preprocess_single_input(input_dict, preprocessor):
    """Preprocess a single input dictionary for prediction"""
    # Convert to DataFrame
    df = pd.DataFrame([input_dict])
    
    # Apply same transformations
    df = map_categorical_features(df)
    df = create_features(df)
    
    # Prepare features
    X, _ = prepare_features(df, fit=False, preprocessor=preprocessor)
    
    return X


def get_feature_names(preprocessor, numerical_cols, categorical_cols):
    """Get feature names after preprocessing"""
    # Get numerical feature names
    num_feature_names = numerical_cols
    
    # Get categorical feature names from OneHotEncoder
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
    
    # Combine all feature names
    all_feature_names = list(num_feature_names) + list(cat_feature_names)
    
    return all_feature_names
