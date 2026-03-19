# Keratoconus Progression Predictor

A clinical decision-support web application for predicting keratoconus progression risk using a pre-trained logistic regression model with SHAP explainability.

## Features

- **3 clinical feature inputs** (BAD-D, Age, ARC 3mm) with real-time validation against training data boundaries
- **Binary classification** — predicts one-year progression risk (Class 0: No Progression, Class 1: Progression)
- **SHAP force plot** — shows how each feature pushes the prediction
- **SHAP waterfall plot** — bar-chart breakdown of individual feature contributions
- **Sparkline distributions** — inline visualization of training data distribution per feature
- **Probability display** — confidence scores for both classes

## Live Demo

Deployed on Streamlit Community Cloud.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model

- **Algorithm**: Logistic Regression (scikit-learn)
- **Training**: SMOTE-balanced (k=5, ratio=1.0) with StandardScaler preprocessing
- **Data**: 412 patients (85 progressors, 327 non-progressors), one-year prediction window
- **Features**: 3 clinical features — BAD-D, Age at Baseline, ARC (3mm Zone)
- **Coefficients**: BAD-D=+0.3225, Age=-0.7027, ARC 3mm=-0.7370

## Project Structure

```
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── logistic_model.pkl     # Trained logistic regression model
├── scaler.pkl             # StandardScaler fitted on training data
├── boundaries.pkl         # Feature validation boundaries (min/max/mean/std)
├── X_train.pkl            # Scaled training data (used for SHAP explainer)
└── .streamlit/
    └── config.toml        # Streamlit theme and server configuration
```

## Dependencies

- streamlit
- numpy, pandas
- scikit-learn, joblib
- shap
- matplotlib
- sparklines

## Disclaimer

This tool is intended for research and educational purposes only. It is not a certified medical device and should not be used as the sole basis for clinical decisions.

## License

This project is part of ongoing PhD research at Aibili.
