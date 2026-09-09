"""
Prediction Page
Form for individual employee prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import load_pipeline, predict_attrition, get_risk_level, preprocess_input
from explainability import get_feature_importance_coefficients
from translations import get_text

def show(lang='en'):
    st.header(get_text("prediction_title", lang))

    # Load model
    try:
        pipeline = load_pipeline()
        model_loaded = True
    except:
        model_loaded = False
        st.error(get_text("model_not_found", lang))

    if model_loaded:
        if lang == 'fr':
            st.markdown("""
            Entrez les caractéristiques de l'employé pour prédire le risque de turnover.
            Cet outil aide à identifier les employés qui pourraient bénéficier de stratégies de rétention.
            """)
        else:
            st.markdown("""
            Enter employee characteristics to predict attrition risk.
            This tool helps identify employees who may benefit from retention strategies.
            """)
        
        # Create input form
        with st.form("prediction_form"):
            st.subheader("Informations sur l'Employé" if lang == 'fr' else "Employee Information")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### " + ("Informations Personnelles" if lang == 'fr' else "Personal Information"))
                age = st.number_input("Âge" if lang == 'fr' else "Age", min_value=18, max_value=70, value=35)
                genre = st.selectbox("Genre" if lang == 'fr' else "Gender", ["M", "F"])
                statut_marital = st.selectbox("Situation Matrimoniale" if lang == 'fr' else "Marital Status", ["Marié(e)", "Célibataire", "Divorcé(e)"])
                niveau_education = st.selectbox("Niveau d'Éducation" if lang == 'fr' else "Education Level", [1, 2, 3, 4, 5], 
                                                 format_func=lambda x: {1: "Inférieur au Collège" if lang == 'fr' else "Below College", 2: "Collège" if lang == 'fr' else "College", 3: "Licence" if lang == 'fr' else "Bachelor", 4: "Master", 5: "Doctorat" if lang == 'fr' else "PhD"}[x])
                domaine_etude = st.selectbox("Domaine d'Études" if lang == 'fr' else "Field of Study", ["Infra & Cloud", "Transformation Digitale", 
                                                                 "Data & IA", "Cybersécurité", "DevOps", "Autre"])
            
            with col2:
                st.markdown("### " + ("Informations Professionnelles" if lang == 'fr' else "Job Information"))
                departement = st.selectbox("Département" if lang == 'fr' else "Department", ["Consulting", "Commercial", "Support"])
                poste = st.selectbox("Poste" if lang == 'fr' else "Position", ["Consultant", "Cadre Commercial", "Manager", 
                                                  "Tech Lead", "Assistant de Direction", "Data Analyst",
                                                  "DevOps Engineer", "Cybersecurity Analyst", "Autre"])
                nombre_experiences_precedentes = st.number_input("Expériences Précédentes" if lang == 'fr' else "Previous Experiences", min_value=0, max_value=10, value=2)
                annee_experience_totale = st.number_input("Années d'Expérience Totale" if lang == 'fr' else "Total Years of Experience", min_value=0, max_value=40, value=10)
                annees_dans_l_entreprise = st.number_input("Années dans l'Entreprise" if lang == 'fr' else "Years in Company", min_value=0, max_value=40, value=5)
                annees_dans_le_poste_actuel = st.number_input("Années dans le Poste Actuel" if lang == 'fr' else "Years in Current Position", min_value=0, max_value=20, value=2)
                niveau_hierarchique_poste = st.selectbox("Niveau Hiérarchique" if lang == 'fr' else "Hierarchical Level", [1, 2, 3, 4, 5])
            
            with col3:
                st.markdown("### " + ("Performance & Satisfaction" if lang == 'fr' else "Performance & Satisfaction"))
                satisfaction_employee_environnement = st.slider("Satisfaction Environnement" if lang == 'fr' else "Environment Satisfaction", 1, 4, 3)
                satisfaction_employee_nature_travail = st.slider("Satisfaction Nature du Travail" if lang == 'fr' else "Job Nature Satisfaction", 1, 4, 3)
                satisfaction_employee_equipe = st.slider("Satisfaction Équipe" if lang == 'fr' else "Team Satisfaction", 1, 4, 3)
                satisfaction_employee_equilibre_pro_perso = st.slider("Équilibre Vie Pro/Perso" if lang == 'fr' else "Work-Life Balance", 1, 4, 3)
                note_evaluation_precedente = st.slider("Note de Performance Précédente" if lang == 'fr' else "Previous Performance Rating", 1, 4, 3)
                note_evaluation_actuelle = st.selectbox("Note de Performance Actuelle" if lang == 'fr' else "Current Performance Rating", [3, 4])
                heure_supplementaires = st.selectbox("Heures Supplémentaires" if lang == 'fr' else "Overtime", ["Non", "Oui"])
                augementation_salaire_precedente = st.slider("Augmentation Salariale Précédente (%)" if lang == 'fr' else "Previous Salary Increase (%)", 11, 25, 15)
            
            st.markdown("### " + ("Informations Additionnelles" if lang == 'fr' else "Additional Information"))
            
            col4, col5 = st.columns(2)
            
            with col4:
                nombre_participation_pee = st.number_input("Nombre de Participations PEE" if lang == 'fr' else "PEE Participation Count", min_value=0, max_value=3, value=1)
                nb_formations_suivies = st.number_input("Formations Suivies" if lang == 'fr' else "Training Courses Completed", min_value=0, max_value=6, value=3)
                distance_domicile_travail = st.number_input("Distance Domicile-Travail (km)" if lang == 'fr' else "Distance to Work (km)", min_value=1, max_value=30, value=10)
                frequence_deplacement = st.selectbox("Fréquence de Déplacement" if lang == 'fr' else "Travel Frequency", ["Aucun", "Occasionnel", "Frequent"])
            
            with col5:
                annees_depuis_la_derniere_promotion = st.number_input("Années Depuis la Dernière Promotion" if lang == 'fr' else "Years Since Last Promotion", min_value=0, max_value=15, value=2)
                annes_sous_responsable_actuel = st.number_input("Années avec le Manager Actuel" if lang == 'fr' else "Years Under Current Manager", min_value=0, max_value=18, value=3)
            
            # Submit button
            submitted = st.form_submit_button("Prédire le Risque de Turnover" if lang == 'fr' else "Predict Attrition Risk", type="primary")
        
            if submitted:
                # Create input dictionary
                input_data = {
                    "age": age,
                    "genre": genre,
                    "statut_marital": statut_marital,
                    "departement": departement,
                    "poste": poste,
                    "nombre_experiences_precedentes": nombre_experiences_precedentes,
                    "annee_experience_totale": annee_experience_totale,
                    "annees_dans_l_entreprise": annees_dans_l_entreprise,
                    "annees_dans_le_poste_actuel": annees_dans_le_poste_actuel,
                    "satisfaction_employee_environnement": satisfaction_employee_environnement,
                    "note_evaluation_precedente": note_evaluation_precedente,
                    "niveau_hierarchique_poste": niveau_hierarchique_poste,
                    "satisfaction_employee_nature_travail": satisfaction_employee_nature_travail,
                    "satisfaction_employee_equipe": satisfaction_employee_equipe,
                    "satisfaction_employee_equilibre_pro_perso": satisfaction_employee_equilibre_pro_perso,
                    "note_evaluation_actuelle": note_evaluation_actuelle,
                    "heure_supplementaires": heure_supplementaires,
                    "augementation_salaire_precedente": augementation_salaire_precedente,
                    "nombre_participation_pee": nombre_participation_pee,
                    "nb_formations_suivies": nb_formations_suivies,
                    "distance_domicile_travail": distance_domicile_travail,
                    "niveau_education": niveau_education,
                    "domaine_etude": domaine_etude,
                    "frequence_deplacement": frequence_deplacement,
                    "annees_depuis_la_derniere_promotion": annees_depuis_la_derniere_promotion,
                    "annes_sous_responsable_actuel": annes_sous_responsable_actuel
                }
                
                # Make prediction
                with st.spinner("Traitement en cours..." if lang == 'fr' else "Processing..."):
                    prediction, probability, contributions, feature_names = predict_attrition(input_data)
                    
                    churn_probability = probability[1] * 100
                    risk_level = get_risk_level(probability[1])
                    
                    # Display results
                    st.header("Résultats de la Prédiction" if lang == 'fr' else "Prediction Results")
                    
                    # Risk level display
                    if churn_probability >= 60:
                        risk_class = "high-risk"
                        risk_color = "#ff6b6b"
                        if lang == 'fr':
                            recommendation = "⚠️ Cet employé présente un risque élevé de départ. Envisagez des stratégies de rétention immédiates telles que des opportunités de développement de carrière, une révision salariale ou une amélioration de l'équilibre vie pro/perso."
                        else:
                            recommendation = "⚠️ This employee is at high risk of leaving. Consider immediate retention strategies such as career development opportunities, salary review, or work-life balance improvements."
                    elif churn_probability >= 30:
                        risk_class = "medium-risk"
                        risk_color = "#ffd43b"
                        if lang == 'fr':
                            recommendation = "⚡ Cet employé présente un risque modéré. Surveillez de près et envisagez des initiatives d'engagement comme des programmes de reconnaissance ou de développement des compétences."
                        else:
                            recommendation = "⚡ This employee shows moderate risk. Monitor closely and consider engagement initiatives like recognition programs or skill development."
                    else:
                        risk_class = "low-risk"
                        risk_color = "#51cf66"
                        if lang == 'fr':
                            recommendation = "✅ Cet employé présente un faible risque de départ. Continuez les pratiques d'engagement actuelles et les points réguliers."
                        else:
                            recommendation = "✅ This employee is at low risk of leaving. Continue current engagement practices and periodic check-ins."
                    
                    st.markdown(f"""
                    <div style="padding: 2rem; border-radius: 10px; margin-top: 2rem; background-color: {risk_color}; color: {'white' if risk_class != 'medium-risk' else 'black'};">
                        <h2 style="text-align: center; margin: 0;">{risk_level}</h2>
                        <h1 style="text-align: center; font-size: 3rem; margin: 1rem 0;">{churn_probability:.1f}%</h1>
                        <p style="text-align: center; margin: 0;">{"Probabilité de Départ" if lang == 'fr' else "Probability of Leaving"}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"### {recommendation}")
                    
                    # Probability breakdown
                    col_prob1, col_prob2 = st.columns(2)
                    with col_prob1:
                        st.metric("Probabilité de Rester" if lang == 'fr' else "Stay Probability", f"{probability[0]*100:.1f}%")
                    with col_prob2:
                        st.metric("Probabilité de Partir" if lang == 'fr' else "Leave Probability", f"{probability[1]*100:.1f}%")
                    
                    # Feature importance for this prediction (local contributions)
                    st.subheader("🔍 " + ("Analyse de l'Importance des Caractéristiques" if lang == 'fr' else "Feature Importance Analysis"))
                    if lang == 'fr':
                        st.markdown("Ceci montre comment chaque caractéristique contribue à la prédiction pour **cet employé spécifique** :")
                    else:
                        st.markdown("This shows how each feature contributes to **this specific employee's** prediction:")
                    
                    # Create DataFrame with contributions
                    contrib_df = pd.DataFrame({
                        'feature': feature_names,
                        'contribution': contributions
                    })
                    
                    # Sort by absolute contribution
                    contrib_df['abs_contribution'] = contrib_df['contribution'].abs()
                    contrib_df = contrib_df.sort_values('abs_contribution', ascending=False)
                    
                    # Create bar plot
                    fig, ax = plt.subplots(figsize=(12, 8))
                    top_features = contrib_df.head(15)
                    colors = ['red' if x > 0 else 'green' for x in top_features['contribution']]
                    ax.barh(top_features['feature'], top_features['contribution'], color=colors)
                    ax.set_xlabel('Contribution à la Prédiction (Positif = Augmente le Risque)' if lang == 'fr' else 'Contribution to Prediction (Positive = Increases Risk)')
                    ax.set_ylabel('Caractéristiques' if lang == 'fr' else 'Features')
                    ax.set_title('Importance Locale des Caractéristiques pour Cet Employé' if lang == 'fr' else 'Local Feature Importance for This Employee')
                    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Top contributing features
                    st.subheader("📊 " + ("Caractéristiques les Plus Contributives pour Cet Employé" if lang == 'fr' else "Top Contributing Features for This Employee"))
                    
                    for idx, row in top_features.head(10).iterrows():
                        if lang == 'fr':
                            direction = "↑ Augmente le Risque" if row['contribution'] > 0 else "↓ Diminue le Risque"
                        else:
                            direction = "↑ Increases Risk" if row['contribution'] > 0 else "↓ Decreases Risk"
                        st.markdown(f"**{row['feature']}**: {direction} ({row['contribution']:.4f})")
                    
                    if lang == 'fr':
                        st.warning("""
                        ⚠️ **Note Importante** : Cette prédiction est un outil d'aide à la décision, pas un système de décision automatisé.
                        Prenez toujours en compte le contexte complet et consultez les professionnels RH avant de prendre des mesures.
                        """)
                    else:
                        st.warning("""
                        ⚠️ **Important Note**: This prediction is a decision support tool, not an automated decision.
                        Always consider the full context and consult with HR professionals before taking action.
                        """)
