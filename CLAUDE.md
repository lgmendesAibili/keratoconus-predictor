# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Streamlit web app demonstrating two keratoconus progression predictors (Logistic Regression): a 1-year model (BAD-D, Age, ARC 3mm) and a 2-year model (Kmax, Age, Pachy Min). Users enter 5 baseline Pentacam features and receive predictions from both models with SHAP explanations. Accompanies the publication by Gil P, Gil JQ, Mendes L, Alves N, Rosa A, Murta J.

**This is a research demonstration, not a clinical tool.**

## Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Architecture

- **`app.py`** — Privacy-safe dual-model version deployed to Streamlit Cloud. Uses synthetic SHAP backgrounds (zero vector in scaled space) instead of patient data. This is the production entrypoint.
- **`app_realData.py`** — Reference version that requires `X_train.pkl` (not in repo) for sparkline distributions and real SHAP background. For local use only. Only supports the 1-year model.
- **Model artifacts** (committed as `.pkl` files):
  - 1-Year: `logistic_model.pkl`, `scaler.pkl`, `boundaries.pkl`
  - 2-Year: `logistic_model_twoYear.pkl`, `scaler_twoYear.pkl`, `boundaries_twoYear.pkl`
- **`MODEL_CONFIGS`** dict in `app.py` defines both models (feature names, file paths, training info). Adding a model means adding a config entry and the corresponding pkl files.

### Prediction Pipeline

1. User enters 5 raw feature values (BAD-D, Age, ARC 3mm, Kmax, Pachy Min)
2. For each model, the relevant features are extracted and validated against that model's `boundaries.pkl`
3. Raw values scaled via each model's `scaler.transform()` (StandardScaler)
4. Scaled values passed to `model.predict()` → binary class (0=stable, 1=progression)
5. SHAP `LinearExplainer` computes feature contributions on the scaled input
6. Waterfall and decision plots rendered at 200 DPI via matplotlib, stacked (1-year then 2-year)

### Key Design Decisions

- **No `predict_proba`** — the models output binary classification only, no probabilities are shown.
- **SHAP background** — `np.zeros((1, n_features))` represents the training mean in scaled space; produces identical base values to using real training data.
- **Widget input ranges** — clamped to training data min/max (no extrapolation beyond observed range). Minimum clamped to 0 for features like BAD-D.
- **Age is integer-only** — enforced via `int` params on `st.number_input` (step=1), matching the training data where Age is always a whole number.
- **Age is shared** — both models use Age but with slightly different scaler params (trained on different SMOTE sets). The input widget uses merged boundaries (widest range); each model scales independently with its own scaler.
- **SHAP waterfall shows raw values** — the `data` field in `shap.Explanation` receives the user's original (unscaled) input so the plot displays actual clinical values (e.g., Age=24) rather than standardized z-scores.
- **Force plot removed** — only waterfall and decision plots are displayed (force plot had rendering issues with individual predictions in Streamlit).
- **`X_train.pkl` purged from git history** — patient data must never be committed. The file was removed using `git filter-repo`.

## Model Evaluation (local only)

- **`keratoconus-model-evaluation/`** — Self-contained folder for evaluating both the one-year (BAD-D, Age, ARC 3mm) and two-year (Kmax, Age, Pachy Min) logistic regression models against three datasets (one-year, two-year, all follow-up). Contains patient data CSVs, pre-trained model/scaler `.pkl` files, and an `evaluate_models.py` script that writes markdown reports to `results/`. **Git-ignored — must never be committed.**
- **`training_metrics.md`** (project root) — Generated sanity-check report. Git-ignored.

Run the evaluation:
```bash
python keratoconus-model-evaluation/evaluate_models.py
```

## Privacy

Patient data must **never** be committed to this repository. The following are git-ignored:
- `localData/` — previously held `X_train.pkl` (purged from git history via `git filter-repo`)
- `keratoconus-model-evaluation/` — contains patient CSVs and model artifacts for local evaluation
- `training_metrics.md` — generated report

The `app.py` version is specifically designed to run without any patient data.

## Deployment

Deployed on Streamlit Community Cloud. Pushing to `main` triggers automatic redeployment.
