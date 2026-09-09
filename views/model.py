"""
Model Page
Displays model comparison and performance metrics
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import load_pipeline
from explainability import get_feature_importance_coefficients
from translations import get_text

def show(lang='en'):
    st.header(get_text("model_title", lang))

    # Load model
    try:
        pipeline = load_pipeline()
        model_loaded = True
    except:
        model_loaded = False
        st.error(get_text("model_not_found", lang))

    if model_loaded:
        st.markdown(f"""
        {get_text("model_subtitle", lang)}
        """)
        
        # Model comparison table
        st.subheader("📊 " + ("Comparaison des Performances des Modèles" if lang == 'fr' else "Model Performance Comparison"))
        
        # Performance data (from notebook)
        models_data = {
            "Modèle" if lang == 'fr' else "Model": ["Régression Logistique" if lang == 'fr' else "Logistic Regression", "Forêt Aléatoire" if lang == 'fr' else "Random Forest", "Classificateur Factice" if lang == 'fr' else "Dummy Classifier"],
            "Précision" if lang == 'fr' else "Accuracy": [0.765, 0.856, 0.839],
            "Précision" if lang == 'fr' else "Precision": [0.372, 0.545, 0.000],
            "Rappel" if lang == 'fr' else "Recall": [0.681, 0.274, 0.000],
            "Score F1" if lang == 'fr' else "F1-Score": [0.481, 0.364, 0.000]
        }
        
        df_comparison = pd.DataFrame(models_data)
        
        # Highlight the selected model
        def highlight_selected(row):
            model_col = 'Modèle' if lang == 'fr' else 'Model'
            selected_model = 'Régression Logistique' if lang == 'fr' else 'Logistic Regression'
            if row[model_col] == selected_model:
                return ['background-color: #d4edda'] * len(row)
            return [''] * len(row)
        
        st.dataframe(df_comparison.style.apply(highlight_selected, axis=1), use_container_width=True)
        
        if lang == 'fr':
            st.info("""
            **✅ Régression Logistique Sélectionnée**
            
            - **Meilleur Rappel (68,1%)** : Critique pour identifier les employés à risque
            - **Bon Équilibre** : Précision et Score F1 raisonnables
            - **Interprétable** : Les coefficients fournissent une importance claire des caractéristiques
            - **Efficace** : Entraînement et prédiction rapides
            """)
        else:
            st.info("""
            **✅ Logistic Regression Selected**
            
            - **Best Recall (68.1%)**: Critical for identifying at-risk employees
            - **Good Balance**: Reasonable precision and F1-score
            - **Interpretable**: Coefficients provide clear feature importance
            - **Efficient**: Fast training and prediction
            """)
        
        # Model details
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 " + ("Régression Logistique (Sélectionnée)" if st.session_state.language == 'fr' else "Logistic Regression (Selected)"))
            st.code("""
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    class_weight="balanced",
    max_iter=3000,
    random_state=42
)
            """)
            
            if lang == 'fr':
                st.markdown("""
                **Pourquoi la Régression Logistique ?**
                
                - Gère le déséquilibre des classes avec class_weight="balanced"
                - Fournit des coefficients interprétables
                - Informatiquement efficace
                - Bonne base pour la classification binaire
                - Fonctionne bien avec les relations linéaires dans les données
                """)
            else:
                st.markdown("""
                **Why Logistic Regression?**
                
                - Handles class imbalance with class_weight="balanced"
                - Provides interpretable coefficients
                - Computationally efficient
                - Good baseline for binary classification
                - Works well with linear relationships in data
                """)
        
        with col2:
            st.subheader("🌲 " + ("Forêt Aléatoire (Alternative)" if st.session_state.language == 'fr' else "Random Forest (Alternative)"))
            st.code("""
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
            """)
            
            if lang == 'fr':
                st.markdown("""
                **Pourquoi Pas la Forêt Aléatoire ?**
                
                - Précision plus élevée mais rappel plus faible (27,4%)
                - Le rappel est critique pour la prédiction du turnover
                - Moins interprétable que la régression logistique
                - Plus coûteuse informatiquement
                - Risque de surapprentissage avec un petit jeu de données
                """)
            else:
                st.markdown("""
                **Why Not Random Forest?**
                
                - Higher accuracy but lower recall (27.4%)
                - Recall is critical for attrition prediction
                - Less interpretable than Logistic Regression
                - More computationally expensive
                - Overfitting risk with small dataset
                """)
        
        # Performance metrics explanation
        st.subheader("📈 " + ("Métriques de Performance Expliquées" if lang == 'fr' else "Performance Metrics Explained"))
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            if lang == 'fr':
                st.markdown("""
                **Précision (76,5%)**
                - Correction globale des prédictions
                - Pas idéal pour les jeux de données déséquilibrés
                - Peut être trompeur lorsque les classes sont déséquilibrées
                
                **Précision (37,2%)**
                - Des départs prédits, combien sont réellement partis
                - Une précision plus faible signifie plus de faux positifs
                - Compromis acceptable pour un rappel plus élevé
                """)
            else:
                st.markdown("""
                **Accuracy (76.5%)**
                - Overall correctness of predictions
                - Not ideal for imbalanced datasets
                - Can be misleading when classes are unbalanced
                
                **Precision (37.2%)**
                - Of predicted departures, how many actually left
                - Lower precision means more false positives
                - Acceptable trade-off for higher recall
                """)
        
        with col_m2:
            if lang == 'fr':
                st.markdown("""
                **Rappel (68,1%) ⭐**
                - Des départs réels, combien ont été correctement prédits
                - **Métrique critique** pour la prédiction du turnover
                - Un rappel élevé signifie attraper la plupart des employés à risque
                - Priorisé sur la précision pour ce cas d'usage
                
                **Score F1 (48,1%)**
                - Moyenne harmonique de la précision et du rappel
                - Équilibre les deux métriques
                - Utile pour l'évaluation globale du modèle
                """)
            else:
                st.markdown("""
                **Recall (68.1%) ⭐**
                - Of actual departures, how many were correctly predicted
                - **Critical metric** for attrition prediction
                - High recall means catching most at-risk employees
                - Prioritized over precision for this use case
                
                **F1-Score (48.1%)**
                - Harmonic mean of precision and recall
                - Balances both metrics
                - Useful for overall model assessment
                """)
        
        # Feature importance comparison
        st.subheader("🔍 " + ("Importance des Caractéristiques par Modèle" if lang == 'fr' else "Feature Importance by Model"))
        
        importance_df = get_feature_importance_coefficients()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        top_features = importance_df.head(15)
        colors = ['red' if x > 0 else 'green' for x in top_features['coefficient']]
        ax.barh(top_features['feature'], top_features['coefficient'], color=colors)
        ax.set_xlabel('Valeur du Coefficient (Impact sur la Prédiction)' if lang == 'fr' else 'Coefficient Value (Impact on Prediction)')
        ax.set_ylabel('Caractéristiques' if lang == 'fr' else 'Features')
        ax.set_title('Importance des Caractéristiques - Régression Logistique' if lang == 'fr' else 'Logistic Regression Feature Importance')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Training details
        st.subheader("⚙️ " + ("Configuration de l'Entraînement" if lang == 'fr' else "Training Configuration"))
        
        if lang == 'fr':
            st.markdown("""
            **Division des Données**
            - Taille de test : 20%
            - Division stratifiée : Maintient la distribution des classes
            - État aléatoire : 42 (reproductibilité)
            
            **Pipeline de Prétraitement**
            - Caractéristiques numériques : StandardScaler
            - Caractéristiques catégorielles : OneHotEncoder(handle_unknown="ignore")
            - ColumnTransformer pour un prétraitement unifié
            
            **Ensemble de Caractéristiques**
            - Caractéristiques originales (excluant revenu_mensuel)
            - 3 caractéristiques ingénierées :
              - score_satisfaction_global
              - taux_promotion
              - poste_heures_supp
            
            **Gestion du Déséquilibre des Classes**
            - class_weight="balanced" dans la régression logistique
            - Ajuste les poids inversement proportionnel aux fréquences des classes
            - Aide le modèle à mieux apprendre la classe minoritaire
            """)
        else:
            st.markdown("""
            **Data Split**
            - Test size: 20%
            - Stratified split: Maintains class distribution
            - Random state: 42 (reproducibility)
            
            **Preprocessing Pipeline**
            - Numerical features: StandardScaler
            - Categorical features: OneHotEncoder(handle_unknown="ignore")
            - ColumnTransformer for unified preprocessing
            
            **Feature Set**
            - Original features (excluding revenu_mensuel)
            - 3 engineered features:
              - score_satisfaction_global
              - taux_promotion
              - poste_heures_supp
            
            **Class Imbalance Handling**
            - class_weight="balanced" in Logistic Regression
            - Adjusts weights inversely proportional to class frequencies
            - Helps model learn minority class better
            """)
        
        # Model selection rationale
        st.subheader("💡 " + ("Rationale de la Sélection du Modèle" if lang == 'fr' else "Model Selection Rationale"))
        
        if lang == 'fr':
            st.success("""
            **Pourquoi le Rappel est Priorisé**
            
            Dans la prédiction du turnover des employés :
            
            1. **Coût des Faux Négatifs** : Manquer un employé qui partira est coûteux
               - Perte de productivité pendant la transition
               - Coûts de recrutement et de formation
               - Défis de transfert de connaissances
               - Impact sur le moral de l'équipe
            
            2. **Coût des Faux Positifs** : Signaler un employé qui ne partira pas est moins coûteux
               - Peut être adressé par des initiatives d'engagement
               - Opportunité de développement de relations
               - Impact financier direct minimal
            
            3. **Objectif Commercial** : Rétention proactive
               - Mieux sur-identifier les employés à risque
               - Permet des stratégies d'intervention précoce
               - Soutient la planification RH et l'allocation des ressources
            """)
        else:
            st.success("""
            **Why Recall is Prioritized**
            
            In employee attrition prediction:
            
            1. **False Negative Cost**: Missing an employee who will leave is costly
               - Lost productivity during transition
               - Recruitment and training costs
               - Knowledge transfer challenges
               - Team morale impact
            
            2. **False Positive Cost**: Flagging an employee who won't leave is less costly
               - Can be addressed with engagement initiatives
               - Opportunity for relationship building
               - Minimal direct financial impact
            
            3. **Business Goal**: Proactive retention
               - Better to over-identify at-risk employees
               - Enables early intervention strategies
               - Supports HR planning and resource allocation
            """)
