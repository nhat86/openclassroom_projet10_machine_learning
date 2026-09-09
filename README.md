# Employee Attrition Prediction - TechNova

A machine learning web application for predicting employee attrition risk at TechNova. This project transforms a data science notebook into a professional, production-ready ML application with a Streamlit interface.

## 🎯 Project Overview

This application helps HR teams identify employees at risk of leaving the company, enabling proactive retention strategies. The model uses Logistic Regression with engineered features to predict attrition probability based on employee characteristics, performance metrics, and satisfaction scores.

### Key Features

- **Multi-page Streamlit Application**: Dashboard, Prediction, Model Comparison, and Explainability pages
- **Logistic Regression Model**: Interpretable model with 68.1% recall for identifying at-risk employees
- **Feature Engineering**: Three engineered features (satisfaction score, promotion rate, role-overtime combination)
- **Model Explainability**: Feature importance analysis using model coefficients
- **Decision Support Tool**: Supports HR decision-making, not automated decisions

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 76.5% |
| Precision | 37.2% |
| Recall | 68.1% |
| F1-Score | 48.1% |

**Why Recall Matters**: High recall is critical for attrition prediction to identify most at-risk employees, even if it means some false positives.

## 🏗️ Project Structure

```
employee-attrition-ml/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
├── README.md                       # This file
├── src/                            # Source code modules
│   ├── __init__.py
│   ├── features.py                 # Feature engineering functions
│   ├── preprocessing.py             # Data preprocessing pipeline
│   ├── train.py                    # Model training script
│   ├── predict.py                  # Prediction functions
│   └── explainability.py           # Model explainability
├── pages/                          # Streamlit pages
│   ├── dashboard.py                # Dashboard with KPIs
│   ├── prediction.py               # Prediction form
│   ├── model.py                    # Model comparison
│   └── explainability.py           # Feature importance
├── models/                         # Trained models
│   └── logistic_regression_pipeline.joblib
├── images/                         # Visualizations
├── data/                           # Data files (not in Git)
│   └── README.md
└── notebooks/                      # Jupyter notebooks
    └── employee_attrition_prediction.ipynb
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd employee-attrition-ml
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Prepare data:
   - Place your CSV files in the `data/` directory
   - Update file paths in `src/train.py` to match your data locations

## 📝 Usage

### Training the Model

Train the Logistic Regression model:

```bash
python -m src.train
```

This will:
- Load and merge data from three CSV files
- Apply feature engineering
- Train the model with class_weight="balanced"
- Save the complete pipeline to `models/logistic_regression_pipeline.joblib`
- Display performance metrics

### Running the Streamlit App

Start the web application:

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Application

1. **Dashboard**: View KPIs, model performance, and feature importance
2. **Prediction**: Enter employee characteristics to predict attrition risk
3. **Model**: Compare different models and view performance metrics
4. **Explainability**: Understand feature importance and model decisions

## 🔧 Feature Engineering

### Engineered Features

1. **score_satisfaction_global**
   - Mean of 4 satisfaction variables
   - Captures overall employee satisfaction
   - Strong protective factor (negative coefficient)

2. **taux_promotion**
   - Formula: years_since_last_promotion / (years_in_company + 1)
   - Normalizes promotion timing by tenure
   - Risk factor (positive coefficient)

3. **poste_heures_supp**
   - Combination of position and overtime status
   - Captures role-specific workload patterns
   - Certain combinations show higher risk

### Feature Set

The model uses 24 numerical features and 5 categorical features:
- **Excluded**: revenu_mensuel (monthly salary)
- **Included**: Original features + 3 engineered features

## 🤖 Model Details

### Logistic Regression Configuration

```python
LogisticRegression(
    class_weight="balanced",
    max_iter=3000,
    random_state=42
)
```

### Preprocessing Pipeline

- **Numerical features**: StandardScaler
- **Categorical features**: OneHotEncoder(handle_unknown="ignore")
- **ColumnTransformer**: Unified preprocessing pipeline

### Model Selection Rationale

Logistic Regression was selected over Random Forest because:
- Higher recall (68.1% vs 27.4%) - critical for attrition prediction
- Better interpretability with clear coefficients
- More efficient training and prediction
- Suitable for linear relationships in data

## 📈 Data Sources

The model uses three data sources:

1. **SIRH** (HR Information)
   - Employee demographics
   - Job information
   - Experience history

2. **Evaluation**
   - Performance ratings
   - Satisfaction scores
   - Hierarchical level

3. **Survey**
   - Employee engagement
   - Work-life balance
   - Training participation

**Total Records**: 1,470 employees
**Attrition Rate**: 16.1% (237 departures)

## ⚠️ Important Notes

### Model Limitations

- **Decision Support Tool**: This is not an automated decision system
- **Correlation vs Causation**: Features show correlation, not causation
- **Data Limitations**: Model trained on historical data
- **Individual Variability**: Predictions have uncertainty

### Ethical Considerations

- Do not use predictions for punitive actions
- Ensure fair and equitable application
- Protect employee privacy and dignity
- Combine predictions with human judgment

### Data Security

- Data files are excluded from Git (see .gitignore)
- Place sensitive data in the `data/` directory
- Do not commit employee data to version control

## 🛠️ Development

### Adding New Features

1. Add feature engineering logic to `src/features.py`
2. Update feature lists in `src/preprocessing.py`
3. Retrain the model
4. Update Streamlit pages if needed

### Extending the Application

- Add new pages in the `pages/` directory
- Update navigation in `app.py`
- Follow existing code structure and style

## 📚 References

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [SHAP Documentation](https://shap.readthedocs.io/)

## 👤 Author

This project was developed as part of the OpenClassrooms Machine Learning course.

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- OpenClassrooms for the ML course
- TechNova for the dataset (fictional)
- The open-source ML community
