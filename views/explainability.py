"""
Explainability Page
Displays SHAP plots and model explainability
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
    st.header(get_text("explainability_title", lang))

    # Load model
    try:
        pipeline = load_pipeline()
        feature_names = load_feature_names()
        model_loaded = True
    except:
        model_loaded = False
        st.error(get_text("model_not_found", lang))

    if model_loaded:
        st.markdown(f"""
        {get_text("explainability_subtitle", lang)}
        """)
        
        # Feature importance
        st.subheader("📊 " + ("Importance Globale des Caractéristiques" if lang == 'fr' else "Global Feature Importance"))
        
        if lang == 'fr':
            st.markdown("""
            Ces coefficients montrent comment chaque caractéristique influence la prédiction du turnover des employés :
            - **Coefficients positifs** : Augmentent la probabilité de départ
            - **Coefficients négatifs** : Diminuent la probabilité de départ
            - **Magnitude** : Force de l'impact de la caractéristique
            """)
        else:
            st.markdown("""
            These coefficients show how each feature influences the prediction of employee attrition:
            - **Positive coefficients**: Increase the likelihood of leaving
            - **Negative coefficients**: Decrease the likelihood of leaving
            - **Magnitude**: Strength of the feature's impact
            """)
        
        importance_df = get_feature_importance_coefficients()
        
        # Display top 20 features
        st.markdown("### " + ("Top 20 des Caractéristiques les Plus Importantes" if lang == 'fr' else "Top 20 Most Important Features"))
        
        fig = plot_feature_importance(importance_df.head(20))
        st.pyplot(fig)
        plt.close()
        
        # Feature importance table
        st.markdown("### " + ("Détails de l'Importance des Caractéristiques" if lang == 'fr' else "Feature Importance Details"))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**" + ("Caractéristiques qui Augmentent le Risque de Turnover" if lang == 'fr' else "Features That Increase Attrition Risk") + "**")
            risk_increase = importance_df[importance_df['coefficient'] > 0].head(10)
            for idx, row in risk_increase.iterrows():
                st.markdown(f"- **{row['feature']}**: +{row['coefficient']:.4f}")
        
        with col2:
            st.markdown("**" + ("Caractéristiques qui Diminuent le Risque de Turnover" if lang == 'fr' else "Features That Decrease Attrition Risk") + "**")
            risk_decrease = importance_df[importance_df['coefficient'] < 0].head(10)
            for idx, row in risk_decrease.iterrows():
                st.markdown(f"- **{row['feature']}**: {row['coefficient']:.4f}")
        
        # Key insights
        st.subheader("💡 " + ("Principales Observations sur l'Importance des Caractéristiques" if lang == 'fr' else "Key Insights from Feature Importance"))
        
        col_insight1, col_insight2 = st.columns(2)
        
        with col_insight1:
            if lang == 'fr':
                st.info("""
                **Principaux Facteurs de Risque**
                
                1. **Années Depuis la Dernière Promotion**
                   - Les employés sans promotion récente sont plus susceptibles de partir
                   - Suggère le besoin d'opportunités de progression de carrière
                
                2. **Expérience Totale**
                   - Les employés plus expérimentés ont un risque de turnover plus élevé
                   - Peut indiquer des opportunités de marché pour les talents seniors
                
                3. **Combination Poste + Heures Sup**
                   - Certains postes avec heures supplémentaires montrent un risque plus élevé
                   - La charge de travail et le type de poste interagissent
                """)
            else:
                st.info("""
                **Top Risk Factors**
                
                1. **Years Since Last Promotion**
                   - Employees without recent promotions are more likely to leave
                   - Suggests need for career progression opportunities
                
                2. **Total Experience**
                   - More experienced employees have higher attrition risk
                   - May indicate market opportunities for senior talent
                
                3. **Post + Overtime Combination**
                   - Certain roles with overtime show higher risk
                   - Workload and role type interaction matters
                """)
        
        with col_insight2:
            if lang == 'fr':
                st.info("""
                **Facteurs Protecteurs**
                
                1. **Score de Satisfaction Global**
                   - Une satisfaction plus élevée réduit le risque de turnover
                   - Confirme l'importance de l'engagement des employés
                
                2. **Années dans l'Entreprise**
                   - Une ancienneté plus longue peut être protectrice
                   - Suggère que la loyauté se construit avec le temps
                
                3. **Département et Poste**
                   - Certains postes/départements ont un risque de base plus faible
                   - Le contexte compte pour la prédiction du turnover
                """)
            else:
                st.info("""
                **Protective Factors**
                
                1. **Global Satisfaction Score**
                   - Higher satisfaction reduces attrition risk
                   - Confirms importance of employee engagement
                
                2. **Years in Company**
                   - Longer tenure can be protective
                   - Suggests loyalty builds over time
                
                3. **Department and Position**
                   - Some roles/departments have lower baseline risk
                   - Context matters for attrition prediction
                """)
        
        # Engineered features explanation
        st.subheader("🔧 " + ("Caractéristiques Ingénierées" if lang == 'fr' else "Engineered Features"))
        
        if lang == 'fr':
            st.markdown("""
            ### Rationale de l'Ingénierie des Caractéristiques
            
            **1. score_satisfaction_global**
            - **Formule** : Moyenne de 4 variables de satisfaction
            - **Composants** :
              - satisfaction_employee_environnement
              - satisfaction_employee_nature_travail
              - satisfaction_employee_equipe
              - satisfaction_employee_equilibre_pro_perso
            - **Objectif** : Capture la satisfaction globale des employés en une seule métrique
            - **Impact** : Coefficient négatif fort (facteur protecteur)
            
            **2. taux_promotion**
            - **Formule** : années_depuis_la_dernière_promotion / (années_dans_l_entreprise + 1)
            - **Objectif** : Normalise le timing de promotion par l'ancienneté
            - **Interprétation** : Valeurs plus élevées = plus de temps depuis la promotion par rapport à l'ancienneté
            - **Impact** : Coefficient positif (facteur de risque)
            
            **3. poste_heures_supp**
            - **Formule** : poste + "_" + heure_supplementaires
            - **Objectif** : Capture l'interaction entre le poste et les heures supplémentaires
            - **Exemples** : "Consultant_1", "Manager_0"
            - **Impact** : Certaines combinaisons montrent un risque plus élevé
            """)
        else:
            st.markdown("""
            ### Feature Engineering Rationale
            
            **1. score_satisfaction_global**
            - **Formula**: Mean of 4 satisfaction variables
            - **Components**:
              - satisfaction_employee_environnement
              - satisfaction_employee_nature_travail
              - satisfaction_employee_equipe
              - satisfaction_employee_equilibre_pro_perso
            - **Purpose**: Captures overall employee satisfaction in a single metric
            - **Impact**: Strong negative coefficient (protective factor)
            
            **2. taux_promotion**
            - **Formula**: years_since_last_promotion / (years_in_company + 1)
            - **Purpose**: Normalizes promotion timing by tenure
            - **Interpretation**: Higher values = longer time since promotion relative to tenure
            - **Impact**: Positive coefficient (risk factor)
            
            **3. poste_heures_supp**
            - **Formula**: poste + "_" + heure_supplementaires
            - **Purpose**: Captures interaction between role and overtime
            - **Examples**: "Consultant_1", "Manager_0"
            - **Impact**: Certain combinations show higher risk
            """)
        
        # Model interpretability
        st.subheader("🎯 " + ("Interprétabilité du Modèle" if lang == 'fr' else "Model Interpretability"))
        
        if lang == 'fr':
            st.markdown("""
            ### Pourquoi la Régression Logistique est Interprétable
            
            **1. Transparence des Coefficients**
            - Chaque caractéristique a un coefficient clair
            - La direction (positif/négatif) est immédiatement claire
            - La magnitude indique l'importance relative
            
            **2. Relation Linéaire**
            - L'impact de chaque caractéristique est additif
            - Facile à comprendre les effets marginaux
            - Comportement prévisible à travers les plages de caractéristiques
            
            **3. Fondement Statistique**
            - Propriétés statistiques bien comprises
            - Intervalles de confiance disponibles
            - Tests d'hypothèse possibles
            
            ### Comparaison avec les Modèles Boîte Noire
            
            | Aspect | Régression Logistique | Forêt Aléatoire / Réseaux Neuronaux |
            |--------|----------------------|----------------------------------|
            | Interprétabilité | Élevée | Faible |
            | Importance des Caractéristiques | Coefficients clairs | Interactions complexes |
            | Explication de Prédiction | Directe | Nécessite SHAP/LIME |
            | Transparence | Élevée | Faible |
            | Conformité Réglementaire | Plus facile | Plus difficile |
            """)
        else:
            st.markdown("""
            ### Why Logistic Regression is Interpretable
            
            **1. Coefficient Transparency**
            - Each feature has a clear coefficient
            - Direction (positive/negative) is immediately clear
            - Magnitude indicates relative importance
            
            **2. Linear Relationship**
            - Impact of each feature is additive
            - Easy to understand marginal effects
            - Predictable behavior across feature ranges
            
            **3. Statistical Foundation**
            - Well-understood statistical properties
            - Confidence intervals available
            - Hypothesis testing possible
            
            ### Comparison with Black Box Models
            
            | Aspect | Logistic Regression | Random Forest / Neural Networks |
            |--------|-------------------|--------------------------------|
            | Interpretability | High | Low |
            | Feature Importance | Clear coefficients | Complex interactions |
            | Prediction Explanation | Direct | Requires SHAP/LIME |
            | Transparency | High | Low |
            | Regulatory Compliance | Easier | Harder |
            """)
        
        # Business implications
        st.subheader("💼 " + ("Implications Commerciales" if lang == 'fr' else "Business Implications"))
        
        if lang == 'fr':
            st.success("""
            **Insights Actionnables pour les RH**
            
            **1. Développement de Carrière**
            - Mettre en œuvre des cycles de promotion réguliers
            - Créer des chemins de progression de carrière clairs
            - Fournir des opportunités de croissance pour les employés expérimentés
            
            **2. Engagement des Employés**
            - Surveiller régulièrement les scores de satisfaction
            - Traiter les préoccupations d'équilibre vie pro/perso
            - Améliorer la dynamique d'équipe et l'environnement
            
            **3. Optimisation des Postes**
            - Réviser les pratiques d'heures supplémentaires dans les postes à risque élevé
            - Équilibrer la charge de travail entre les postes
            - Envisager des stratégies de rétention spécifiques par poste
            
            **4. Rétention Proactive**
            - Utiliser les prédictions du modèle pour une intervention précoce
            - Cibler les employés à risque avec des stratégies personnalisées
            - Surveiller de près les départements à risque élevé
            """)
        else:
            st.success("""
            **Actionable Insights for HR**
            
            **1. Career Development**
            - Implement regular promotion cycles
            - Create clear career progression paths
            - Provide growth opportunities for experienced employees
            
            **2. Employee Engagement**
            - Monitor satisfaction scores regularly
            - Address work-life balance concerns
            - Improve team dynamics and environment
            
            **3. Role Optimization**
            - Review overtime practices in high-risk roles
            - Balance workload across positions
            - Consider role-specific retention strategies
            
            **4. Proactive Retention**
            - Use model predictions for early intervention
            - Target at-risk employees with personalized strategies
            - Monitor high-risk departments closely
            """)
        
        # Limitations
        st.subheader("⚠️ " + ("Limitations du Modèle" if lang == 'fr' else "Model Limitations"))
        
        if lang == 'fr':
            st.warning("""
            **Considérations Importantes**
            
            **1. Corrélation vs Causalité**
            - Les caractéristiques montrent une corrélation, pas une causalité
            - Changer une caractéristique peut ne pas changer directement le risque de turnover
            - Le contexte et les circonstances individuelles comptent
            
            **2. Limitations des Données**
            - Modèle entraîné sur des données historiques
            - Peut ne pas capturer les tendances futures ou les changements
            - Limité aux caractéristiques disponibles dans le jeu de données
            
            **3. Variabilité Individuelle**
            - Le modèle fournit des insights au niveau de la population
            - Les prédictions individuelles ont une incertitude
            - Le jugement humain reste essentiel
            
            **4. Considérations Éthiques**
            - Éviter d'utiliser les prédictions pour des actions punitives
            - Assurer une application équitable et juste
            - Protéger la vie privée et la dignité des employés
            """)
        else:
            st.warning("""
            **Important Considerations**
            
            **1. Correlation vs Causation**
            - Features show correlation, not causation
            - Changing a feature may not directly change attrition risk
            - Context and individual circumstances matter
            
            **2. Data Limitations**
            - Model trained on historical data
            - May not capture future trends or changes
            - Limited to features available in dataset
            
            **3. Individual Variability**
            - Model provides population-level insights
            - Individual predictions have uncertainty
            - Human judgment remains essential
            
            **4. Ethical Considerations**
            - Avoid using predictions for punitive actions
            - Ensure fair and equitable application
            - Protect employee privacy and dignity
            """)
        
        if lang == 'fr':
            st.info("""
            **Recommandation** : Utilisez ce modèle comme un outil d'aide à la décision, pas comme un système de décision automatisé.
            Combinez toujours les prédictions du modèle avec l'expertise humaine et la compréhension contextuelle.
            """)
        else:
            st.info("""
            **Recommendation**: Use this model as a decision support tool, not an automated decision system.
            Always combine model predictions with human expertise and contextual understanding.
            """)
