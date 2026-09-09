"""
Dashboard Page
Displays KPIs and general information about the model
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import load_pipeline, load_feature_names
from explainability import get_feature_importance_coefficients, plot_feature_importance
from translations import get_text

def show(lang='en'):
    st.header(get_text("dashboard_title", lang))

    # Load model
    try:
        pipeline = load_pipeline()
        feature_names = load_feature_names()
        model_loaded = True
    except:
        model_loaded = False
        st.error(get_text("model_not_found", lang))

    if model_loaded:
        # KPIs
        st.subheader("Key Performance Indicators" if lang == 'en' else "Indicateurs Clés de Performance")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(get_text("total_employees", lang), "1,470")
        
        with col2:
            st.metric(get_text("attrition_rate", lang), "16.1%")
        
        with col3:
            st.metric("Départs" if lang == 'fr' else "Departures", "237")
        
        with col4:
            st.metric("Rappel du Modèle" if lang == 'fr' else "Model Recall", "68.1%")
        
        with col5:
            st.metric("Score F1" if lang == 'fr' else "F1-Score", "48.1%")
        
        # Model Performance
        st.subheader("🎯 " + ("Model Performance" if lang == 'en' else "Performance du Modèle"))
        
        col_perf1, col_perf2 = st.columns(2)
        
        with col_perf1:
            st.info(f"""
            **{'Modèle : Régression Logistique' if lang == 'fr' else 'Model: Logistic Regression'}**
            
            - class_weight="balanced"
            - max_iter=3000
            - random_state=42
            """)
        
        with col_perf2:
            st.info(f"""
            **{'Métriques de Performance' if lang == 'fr' else 'Performance Metrics'}**
            
            - {'Précision' if lang == 'fr' else 'Accuracy'}: 76.5%
            - {'Précision' if lang == 'fr' else 'Precision'}: 37.2%
            - {'Rappel' if lang == 'fr' else 'Recall'}: 68.1%
            - {'Score F1' if lang == 'fr' else 'F1-Score'}: 48.1%
            """)
        
        # Feature Importance
        st.subheader("🔍 " + ("Top Feature Importance" if lang == 'en' else "Importance des Caractéristiques Principales"))
        
        importance_df = get_feature_importance_coefficients()
        
        fig = plot_feature_importance(importance_df.head(10))
        st.pyplot(fig)
        plt.close()
        
        # Business Context
        st.subheader("💼 " + ("Business Context" if lang == 'en' else "Contexte Commercial"))
        
        if lang == 'fr':
            st.markdown("""
            ### Énoncé du Problème
            Prédire le turnover des employés pour permettre des stratégies de rétention proactives.
            
            ### Pourquoi C'est Important
            - **Coût du Turnover** : Remplacer un employé coûte 1,5-2x son salaire annuel
            - **Perte de Connaissances** : Les départs entraînent une perte de connaissances institutionnelles
            - **Impact sur l'Équipe** : Affecte le moral et la productivité de l'équipe
            
            ### Objectif du Modèle
            Ce modèle est un **outil d'aide à la décision** pour les RH, pas un système de décision automatisé.
            Il aide à identifier les employés à risque pour des interventions de rétention ciblées.
            
            ### Principales Observations des Données
            - Les années depuis la dernière promotion sont un prédicteur fort
            - L'expérience totale influence significativement le risque de turnover
            - Le score de satisfaction global est important
            - L'âge affecte les modèles de turnover
            - La combinaison poste + heures supplémentaires est une caractéristique clé
            """)
        else:
            st.markdown("""
            ### Problem Statement
            Predict employee attrition to enable proactive retention strategies.
            
            ### Why This Matters
            - **Cost of Turnover**: Replacing an employee costs 1.5-2x their annual salary
            - **Knowledge Loss**: Departures result in loss of institutional knowledge
            - **Team Impact**: Affects team morale and productivity
            
            ### Model Purpose
            This model is a **decision support tool** for HR, not an automated decision system.
            It helps identify at-risk employees for targeted retention interventions.
            
            ### Key Insights from Data
            - Years since last promotion is a strong predictor
            - Total experience significantly influences attrition risk
            - Global satisfaction score is important
            - Age affects attrition patterns
            - Post + overtime combination is a key feature
            """)
        
        # Data Overview
        st.subheader("📈 " + ("Data Overview" if lang == 'en' else "Aperçu des Données"))
        
        col_data1, col_data2 = st.columns(2)
        
        with col_data1:
            st.info(f"""
            **{'Sources de Données' if lang == 'fr' else 'Data Sources'}**
            
            - SIRH: {'Informations RH' if lang == 'fr' else 'HR information'} (1,470 {'employés' if lang == 'fr' else 'employees'})
            - Evaluation: {'Métriques de performance' if lang == 'fr' else 'Performance metrics'}
            - Survey: {'Satisfaction des employés' if lang == 'fr' else 'Employee satisfaction'}
            """)
        
        with col_data2:
            st.info("""
            **Feature Engineering**
            
            - score_satisfaction_global
            - taux_promotion
            - poste_heures_supp
            """)
