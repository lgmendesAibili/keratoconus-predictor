# Keratoconus 1-Year Progression Predictor

Interactive demonstration of the machine learning model described in:

> Gil P, Gil JQ, Mendes L, Alves N, Rosa A, Murta J. *"Predicting Keratoconus Progression Within 1–2 Years Using Baseline Tomography and an Explainable Open-Access Machine Learning Model."*

## Features

- **3 clinical feature inputs** (BAD-D, Age, ARC 3mm) with real-time validation against training data boundaries
- **Binary classification** — predicts one-year progression risk (Progression / Stability)
- **SHAP waterfall plot** — bar-chart breakdown of individual feature contributions
- **SHAP decision plot** — cumulative path from base value to final prediction
- **Privacy-safe** — no patient data required at runtime; SHAP uses a synthetic background derived from the StandardScaler

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
├── app.py                 # Streamlit application (privacy-safe, no patient data)
├── app_realData.py        # Reference version (requires local X_train.pkl, not included)
├── requirements.txt       # Python dependencies
├── logistic_model.pkl     # Trained logistic regression model
├── scaler.pkl             # StandardScaler fitted on training data
├── boundaries.pkl         # Feature validation boundaries (min/max/mean/std)
└── .streamlit/
    └── config.toml        # Streamlit theme and server configuration
```

## Dependencies

- streamlit
- numpy
- scikit-learn, joblib
- shap
- matplotlib

## Disclaimer

**Not for clinical use.** This application is a research demonstration intended solely to illustrate the algorithm described in the accompanying publication. It is not a certified or validated medical device and must not be used for clinical decision-making.

## License

This project is part of ongoing PhD research at Aibili.
