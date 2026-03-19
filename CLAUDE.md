# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streamlit web app demonstrating a keratoconus 1-year progression predictor (Logistic Regression, 3 Pentacam features: BAD-D, Age, ARC 3mm). Accompanies the publication by Gil P, Gil JQ, Mendes L, Alves N, Rosa A, Murta J.

**This is a research demonstration, not a clinical tool.**

## Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Architecture

- **`app.py`** — Privacy-safe version deployed to Streamlit Cloud. Uses a synthetic SHAP background (zero vector in scaled space) instead of patient data. This is the production entrypoint.
- **`app_realData.py`** — Reference version that requires `X_train.pkl` (not in repo) for sparkline distributions and real SHAP background. For local use only.
- **Model artifacts** (committed as `.pkl` files): `logistic_model.pkl` (LogisticRegression), `scaler.pkl` (StandardScaler), `boundaries.pkl` (dict with min/max/mean/std per feature).

### Prediction Pipeline

1. User enters raw feature values → validated against `boundaries.pkl` ranges
2. Raw values scaled via `scaler.transform()` (StandardScaler)
3. Scaled values passed to `model.predict()` → binary class (0=stable, 1=progression)
4. SHAP `LinearExplainer` computes feature contributions on the scaled input
5. Waterfall and decision plots rendered at 200 DPI via matplotlib

### Key Design Decisions

- **No `predict_proba`** — the model outputs binary classification only, no probabilities are shown.
- **SHAP background** — `np.zeros((1, n_features))` represents the training mean in scaled space; produces identical base values to using real training data.
- **Widget input ranges** — ±1 SD beyond training min/max, clamped to 0 to prevent negative values for features like BAD-D.
- **Force plot removed** — only waterfall and decision plots are displayed (force plot had rendering issues with individual predictions in Streamlit).
- **`X_train.pkl` purged from git history** — patient data must never be committed. The file was removed using `git filter-repo`.

## Privacy

Patient data (`X_train.pkl`) must **never** be committed to this repository. It was purged from git history. The `app.py` version is specifically designed to run without it.

## Deployment

Deployed on Streamlit Community Cloud. Pushing to `main` triggers automatic redeployment.
