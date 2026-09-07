# Employee Churn Prediction - Deployment Guide

This project deploys an employee churn prediction model as a Streamlit web application.

## Files Created

- **`preprocessing.py`** - Data loading, cleaning, feature engineering, and preprocessing functions
- **`model_training.py`** - Model training script that saves the trained model and preprocessor
- **`app.py`** - Streamlit web application for making predictions
- **`requirements.txt`** - Python dependencies
- **`README.md`** - This file

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Update the file paths in `model_training.py` to point to your actual data files:

```python
SIRH_PATH = "path/to/extrait_sirh.csv"
EVAL_PATH = "path/to/extrait_eval.csv"
SONDAGE_PATH = "path/to/extrait_sondage.csv"
```

Then run the training script:

```bash
python model_training.py
```

This will create:
- `model.joblib` - Trained RandomForest model
- `preprocessor.joblib` - Fitted preprocessing pipeline

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Deployment Options

### Local Deployment
Simply run `streamlit run app.py` from the project directory.

### Streamlit Cloud Deployment

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Click "New app" and connect your GitHub repository
4. Select the repository and specify `app.py` as the main file
5. Click "Deploy"

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t churn-prediction .
docker run -p 8501:8501 churn-prediction
```

## Model Details

- **Algorithm**: RandomForest Classifier
- **Parameters**: n_estimators=300, class_weight="balanced"
- **Features**: 28 features including engineered features (satisfaction score, promotion rate)
- **Target**: Employee churn (Yes/No)

## Application Features

- Interactive form for employee data input
- Real-time churn risk prediction
- Probability breakdown (stay vs leave)
- Risk categorization (Low/Medium/High)
- Recommendations based on risk level

## Troubleshooting

**Model files not found error**: Make sure you've run `model_training.py` first to generate `model.joblib` and `preprocessor.joblib`.

**Import errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`.

**Data path errors**: Update the file paths in `model_training.py` to match your actual data file locations.
