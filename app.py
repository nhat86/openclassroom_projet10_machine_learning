"""
Employee Attrition Prediction - TechNova
Main Streamlit Application
"""

import streamlit as st
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from translations import get_text

# Initialize session state for language and page
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'page_index' not in st.session_state:
    st.session_state.page_index = 0

# Page configuration
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Language selector in sidebar
st.sidebar.title(get_text("navigation", st.session_state.language))
lang = st.sidebar.radio(
    get_text("language", st.session_state.language),
    [get_text("english", st.session_state.language), get_text("french", st.session_state.language)],
    key="language_selector",
    index=0 if st.session_state.language == 'en' else 1
)
st.session_state.language = 'en' if lang == get_text("english", st.session_state.language) else 'fr'

# Page navigation (use index-based selection)
page_options = [
    get_text("dashboard", st.session_state.language), 
    get_text("prediction", st.session_state.language), 
    get_text("model", st.session_state.language), 
    get_text("explainability", st.session_state.language)
]
page = st.sidebar.radio(
    get_text("go_to", st.session_state.language),
    page_options,
    key="page_selector",
    index=st.session_state.page_index
)
st.session_state.page_index = page_options.index(page)

# Main header
st.markdown(f'<div class="main-header">👥 {get_text("app_title", st.session_state.language)}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">{get_text("app_subtitle", st.session_state.language)}</div>', unsafe_allow_html=True)

# Import and run the selected page (use index instead of text comparison)
if st.session_state.page_index == 0:
    from views import dashboard
    dashboard.show(st.session_state.language)
elif st.session_state.page_index == 1:
    from views import prediction
    prediction.show(st.session_state.language)
elif st.session_state.page_index == 2:
    from views import model
    model.show(st.session_state.language)
elif st.session_state.page_index == 3:
    from views import explainability
    explainability.show(st.session_state.language)
