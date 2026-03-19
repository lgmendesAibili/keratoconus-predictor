# Keratoconus Progression Predictor

A clinical decision-support web application for predicting keratoconus progression risk using a pre-trained logistic regression model with SHAP explainability.

## Features

- **5 clinical feature inputs** with real-time validation against training data boundaries
- **Binary classification** — predicts progression risk (Class 0: No Progression, Class 1: Progression)
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
- **Training**: Repeated Stratified Group K-Fold cross-validation with SMOTENC balancing
- **Features**: 5 numerical clinical features derived from Pentacam measurements

## Project Structure

```
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── logistic_model.pkl     # Trained logistic regression model
├── boundaries.pkl         # Feature validation boundaries (min/max/mean/std)
├── X_train.pkl            # Training data (used for SHAP explainer)
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
