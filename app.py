import streamlit as st
import pandas as pd
import numpy as np
import joblib
from preprocessing import preprocess_single_input


# Page configuration
st.set_page_config(
    page_title="Employee Churn Prediction",
    page_icon="👥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
    }
    .high-risk {
        background-color: #ff6b6b;
        color: white;
    }
    .low-risk {
        background-color: #51cf66;
        color: white;
    }
    .medium-risk {
        background-color: #ffd43b;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">👥 Employee Churn Prediction</div>', unsafe_allow_html=True)

st.markdown("""
This application predicts whether an employee is likely to leave the company based on various factors.
Fill in the employee details below to get a prediction.
""")

# Load model and preprocessor
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.joblib")
        preprocessor = joblib.load("preprocessor.joblib")
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, preprocessor = load_model()

if model is None:
    st.error("Model files not found. Please train the model first by running model_training.py")
    st.stop()

# Create input form
st.header("Employee Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Information")
    age = st.number_input("Age", min_value=18, max_value=70, value=35)
    genre = st.selectbox("Gender", ["M", "F"])
    statut_marital = st.selectbox("Marital Status", ["Marié(e)", "Célibataire", "Divorcé(e)"])
    niveau_education = st.selectbox("Education Level", [1, 2, 3, 4, 5], 
                                     format_func=lambda x: {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "PhD"}[x])
    domaine_etude = st.selectbox("Field of Study", ["Infra & Cloud", "Transformation Digitale", 
                                                     "Data & IA", "Cybersécurité", "DevOps", "Autre"])

with col2:
    st.subheader("Job Information")
    departement = st.selectbox("Department", ["Consulting", "Commercial", "Support"])
    poste = st.selectbox("Position", ["Consultant", "Cadre Commercial", "Manager", 
                                      "Tech Lead", "Assistant de Direction", "Data Analyst",
                                      "DevOps Engineer", "Cybersecurity Analyst", "Autre"])
    revenu_mensuel = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000)
    nombre_experiences_precedentes = st.number_input("Previous Experiences", min_value=0, max_value=10, value=2)
    annee_experience_totale = st.number_input("Total Years of Experience", min_value=0, max_value=40, value=10)
    annees_dans_l_entreprise = st.number_input("Years in Company", min_value=0, max_value=40, value=5)
    annees_dans_le_poste_actuel = st.number_input("Years in Current Position", min_value=0, max_value=20, value=2)
    niveau_hierarchique_poste = st.selectbox("Hierarchical Level", [1, 2, 3, 4, 5])

with col3:
    st.subheader("Performance & Satisfaction")
    satisfaction_employee_environnement = st.slider("Environment Satisfaction", 1, 4, 3)
    satisfaction_employee_nature_travail = st.slider("Job Nature Satisfaction", 1, 4, 3)
    satisfaction_employee_equipe = st.slider("Team Satisfaction", 1, 4, 3)
    satisfaction_employee_equilibre_pro_perso = st.slider("Work-Life Balance", 1, 4, 3)
    note_evaluation_precedente = st.slider("Previous Performance Rating", 1, 4, 3)
    note_evaluation_actuelle = st.selectbox("Current Performance Rating", [3, 4])
    heure_supplementaires = st.selectbox("Overtime", ["Non", "Oui"])
    augementation_salaire_precedente = st.slider("Previous Salary Increase (%)", 11, 25, 15)

st.header("Additional Information")

col4, col5 = st.columns(2)

with col4:
    nombre_participation_pee = st.number_input("PEE Participation Count", min_value=0, max_value=3, value=1)
    nb_formations_suivies = st.number_input("Training Courses Completed", min_value=0, max_value=6, value=3)
    distance_domicile_travail = st.number_input("Distance to Work (km)", min_value=1, max_value=30, value=10)
    frequence_deplacement = st.selectbox("Travel Frequency", ["Aucun", "Occasionnel", "Frequent"])

with col5:
    annees_depuis_la_derniere_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=2)
    annes_sous_responsable_actuel = st.number_input("Years Under Current Manager", min_value=0, max_value=18, value=3)

# Prediction button
predict_button = st.button("Predict Churn Risk", type="primary", use_container_width=True)

if predict_button:
    # Create input dictionary
    input_data = {
        "age": age,
        "genre": genre,
        "revenu_mensuel": revenu_mensuel,
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
    
    # Preprocess input
    with st.spinner("Processing..."):
        X_processed = preprocess_single_input(input_data, preprocessor)
        
        # Make prediction
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0]
        
        # Display results
        st.header("Prediction Results")
        
        churn_probability = probability[1] * 100
        
        if churn_probability >= 60:
            risk_class = "high-risk"
            risk_text = "HIGH RISK"
            recommendation = "⚠️ This employee is at high risk of leaving. Consider immediate retention strategies."
        elif churn_probability >= 30:
            risk_class = "medium-risk"
            risk_text = "MEDIUM RISK"
            recommendation = "⚡ This employee shows moderate risk. Monitor closely and consider engagement initiatives."
        else:
            risk_class = "low-risk"
            risk_text = "LOW RISK"
            recommendation = "✅ This employee is at low risk of leaving. Continue current engagement practices."
        
        st.markdown(f"""
        <div class="prediction-box {risk_class}">
            <h2 style="text-align: center; margin: 0;">{risk_text}</h2>
            <h1 style="text-align: center; font-size: 3rem; margin: 1rem 0;">{churn_probability:.1f}%</h1>
            <p style="text-align: center; margin: 0;">Probability of Leaving</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"### {recommendation}")
        
        # Show probability breakdown
        col_prob1, col_prob2 = st.columns(2)
        with col_prob1:
            st.metric("Stay Probability", f"{probability[0]*100:.1f}%")
        with col_prob2:
            st.metric("Leave Probability", f"{probability[1]*100:.1f}%")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>Employee Churn Prediction Model | Built with RandomForest Classifier</p>
    <p>© 2024 - HR Analytics</p>
</div>
""", unsafe_allow_html=True)
