import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from preprocessing import (
    load_and_merge_data,
    clean_data,
    map_categorical_features,
    create_features,
    prepare_features
)


def train_model(sirh_path, eval_path, sondage_path, model_output_path="model.joblib", 
                preprocessor_output_path="preprocessor.joblib"):
    """
    Train the RandomForest model and save it along with the preprocessor
    """
    print("Loading and merging data...")
    df_central = load_and_merge_data(sirh_path, eval_path, sondage_path)
    
    print("Cleaning data...")
    df_clean = clean_data(df_central)
    
    print("Mapping categorical features...")
    df_mapped = map_categorical_features(df_clean)
    
    print("Creating engineered features...")
    df_features = create_features(df_mapped)
    
    # Prepare target variable
    y = df_features["a_quitte_l_entreprise"].map({"Non": 0, "Oui": 1})
    
    # Prepare features
    print("Preparing features...")
    X, preprocessor = prepare_features(df_features, fit=True)
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    print("Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and preprocessor
    print(f"\nSaving model to {model_output_path}...")
    joblib.dump(model, model_output_path)
    
    print(f"Saving preprocessor to {preprocessor_output_path}...")
    joblib.dump(preprocessor, preprocessor_output_path)
    
    print("Training complete!")
    
    return model, preprocessor


def load_model(model_path="model.joblib", preprocessor_path="preprocessor.joblib"):
    """Load the trained model and preprocessor"""
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor


if __name__ == "__main__":
    # Update these paths to your actual data file locations
    SIRH_PATH = "C:/Users/vtmnh/OneDrive/Tài liệu/openclassroom_formation IA/projet10_machine_learning/extrait_sirh.csv"
    EVAL_PATH = "C:/Users/vtmnh/OneDrive/Tài liệu/openclassroom_formation IA/projet10_machine_learning/extrait_eval.csv"
    SONDAGE_PATH = "C:/Users/vtmnh/OneDrive/Tài liệu/openclassroom_formation IA/projet10_machine_learning/extrait_sondage.csv"
    
    # Train the model
    model, preprocessor = train_model(SIRH_PATH, EVAL_PATH, SONDAGE_PATH)
