"""
Keratoconus Progression Prediction — Streamlit Web Application (Privacy-Safe)

Interactive demonstration of two machine learning models described in:

    Gil P, Gil JQ, Mendes L, Alves N, Rosa A, Murta J.
    "Predicting Keratoconus Progression Within 1-2 Years Using Baseline
    Tomography and an Explainable Open-Access Machine Learning Model"

Users enter 5 baseline Pentacam features and receive predictions from both
a 1-year model (BAD-D, Age, ARC 3mm) and a 2-year model (Kmax, Age,
Pachy Min), each with SHAP explanations.

SHAP explanations use a synthetic background derived from the fitted
StandardScaler, requiring no patient data at runtime.

Dependencies:
    streamlit, numpy, joblib, shap, matplotlib

Usage:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# Resolve paths relative to this file
APP_DIR = Path(__file__).parent

# Page configuration
st.set_page_config(
    page_title="Keratoconus Progression Predictor",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Clinical-grade CSS
st.markdown('''
<style>
    /* Hide Streamlit branding for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Header banner */
    .clinical-header {
        background: linear-gradient(135deg, #1a5276 0%, #2980b9 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 0.75rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .clinical-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .clinical-header p {
        margin: 0.4rem 0 0 0;
        font-size: 0.95rem;
        opacity: 0.9;
    }

    /* Research notice */
    .research-notice {
        background: #eaf2f8;
        border: 1px solid #aed6f1;
        border-radius: 0.5rem;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1.5rem;
        font-size: 0.85rem;
        color: #1a5276;
        text-align: center;
    }
    .research-notice b {
        color: #c0392b;
    }

    /* Input card */
    .input-card {
        background: #ffffff;
        border: 1px solid #d5dbdb;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .input-card h3 {
        color: #1a5276;
        margin-top: 0;
        font-size: 1.1rem;
        border-bottom: 2px solid #2980b9;
        padding-bottom: 0.5rem;
    }

    /* Result cards */
    .result-card {
        border-radius: 0.75rem;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .result-progression {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        color: white;
    }
    .result-stable {
        background: linear-gradient(135deg, #27ae60 0%, #1e8449 100%);
        color: white;
    }
    .result-card h2 {
        margin: 0 0 0.3rem 0;
        font-size: 1.5rem;
    }
    .result-card .label {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    /* Feature reference table */
    .ref-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    .ref-table th {
        background: #1a5276;
        color: white;
        padding: 0.5rem 0.75rem;
        text-align: left;
        font-weight: 500;
    }
    .ref-table td {
        padding: 0.4rem 0.75rem;
        border-bottom: 1px solid #ecf0f1;
    }
    .ref-table tr:nth-child(even) {
        background: #f8f9fa;
    }

    /* Validation badge */
    .validation-ok {
        color: #27ae60;
        font-weight: 500;
    }
    .validation-warn {
        color: #e74c3c;
        font-weight: 500;
    }

    /* Section divider */
    .section-label {
        color: #1a5276;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #2980b9;
    }

    /* Disclaimer */
    .disclaimer {
        background: #fef9e7;
        border-left: 4px solid #f39c12;
        padding: 0.75rem 1rem;
        border-radius: 0 0.5rem 0.5rem 0;
        font-size: 0.8rem;
        color: #7d6608;
        margin-top: 1rem;
    }

    /* Citation box */
    .citation-box {
        background: #f4f6f9;
        border: 1px solid #d5dbdb;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        font-size: 0.78rem;
        color: #2c3e50;
        margin-top: 1rem;
        line-height: 1.5;
    }

    /* SHAP explanation text */
    .shap-guide {
        background: #f0f4f8;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        font-size: 0.82rem;
        color: #34495e;
        margin-bottom: 1rem;
        border-left: 3px solid #2980b9;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #f8f9fa;
    }
</style>
''', unsafe_allow_html=True)


# --- Model configurations ---

MODEL_CONFIGS = {
    "one_year": {
        "label": "1-Year Progression",
        "window": "1 year",
        "model_file": "logistic_model.pkl",
        "scaler_file": "scaler.pkl",
        "boundaries_file": "boundaries.pkl",
        "feature_names": ["BAD-D", "Age", "ARC 3mm"],
        "patients": 412,
    },
    "two_year": {
        "label": "2-Year Progression",
        "window": "2 years",
        "model_file": "logistic_model_twoYear.pkl",
        "scaler_file": "scaler_twoYear.pkl",
        "boundaries_file": "boundaries_twoYear.pkl",
        "feature_names": ["Kmax", "Age", "Pachy Min"],
        "patients": 412,
    },
}


@st.cache_resource
def load_model_and_data(model_key):
    """Load a trained model, scaler, and feature boundaries by config key."""
    config = MODEL_CONFIGS[model_key]
    try:
        model = joblib.load(APP_DIR / config["model_file"])
        scaler = joblib.load(APP_DIR / config["scaler_file"])
        boundaries = joblib.load(APP_DIR / config["boundaries_file"])
        return model, scaler, boundaries
    except FileNotFoundError as e:
        st.error(f"Error loading {config['label']} files: {e}")
        st.stop()


def validate_input(value, feature_name, boundaries):
    """Check whether a feature value falls within training data boundaries.

    Args:
        value: Numeric value entered by the user.
        feature_name: Name of the clinical feature (must be a key in *boundaries*).
        boundaries: dict mapping feature names to boundary dicts with 'min' and 'max'.

    Returns:
        tuple[bool, str]: (is_valid, error_message). *error_message* is empty when valid.
    """
    min_val = boundaries[feature_name]['min']
    max_val = boundaries[feature_name]['max']

    if value < min_val or value > max_val:
        return False, f"{feature_name} is outside training range [{min_val:.3f}, {max_val:.3f}]"
    return True, ""


@st.cache_resource
def get_shap_explainer(_model, _scaler):
    """Create and cache a SHAP LinearExplainer using a synthetic background.

    Constructs a background dataset from the scaler's learned mean (which maps
    to zero in scaled space). This avoids the need for actual patient training
    data while producing identical SHAP base values.

    Args:
        _model: Fitted sklearn LogisticRegression estimator.
        _scaler: Fitted sklearn StandardScaler (provides mean and scale).

    Returns:
        shap.LinearExplainer: Explainer instance ready to compute SHAP values.
    """
    background = np.zeros((1, len(_scaler.mean_)))
    return shap.LinearExplainer(_model, background)


def display_shap_guide():
    """Render the SHAP explanation guide text (call once before plots)."""
    st.markdown('''
    <div class="shap-guide">
        <b>How to read these plots:</b><br>
        <b>Waterfall plot:</b> Shows each feature's individual contribution to the prediction.
        Bars extending to the <b style="color:#c0392b">right (red)</b> push the output toward
        progression; bars to the <b style="color:#2980b9">left (blue)</b> push toward stability.
        The bar length indicates how strongly that feature influences this individual case.
        E[f(x)] is the base value (average model output across the training population).<br>
        <b>Decision plot:</b> Traces the cumulative path from the base value (vertical grey line)
        to the final model output. Each line segment shows the shift caused by one feature,
        applied sequentially from bottom to top. A final value to the right of the base value
        indicates a prediction toward progression; to the left, toward stability.
    </div>
    ''', unsafe_allow_html=True)


def display_shap_plots(model, scaler, input_data, input_raw, feature_names):
    """Render SHAP waterfall plot and decision plot for a prediction."""
    explainer = get_shap_explainer(model, scaler)
    shap_values = explainer.shap_values(input_data)

    # High DPI for crisp rendering on retina/HiDPI displays
    plot_dpi = 200
    plt.rcParams['figure.dpi'] = plot_dpi
    plt.rcParams['savefig.dpi'] = plot_dpi

    col_waterfall, col_decision = st.columns(2)

    with col_waterfall:
        st.markdown('<p class="section-label">Waterfall Plot</p>', unsafe_allow_html=True)
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_raw[0],
            feature_names=feature_names
        )
        fig_wf, ax_wf = plt.subplots(figsize=(10, 4), dpi=plot_dpi)
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(fig_wf, bbox_inches='tight', dpi=plot_dpi)
        plt.close(fig_wf)

    with col_decision:
        st.markdown('<p class="section-label">Decision Plot</p>', unsafe_allow_html=True)
        fig_dp, ax_dp = plt.subplots(figsize=(10, 4), dpi=plot_dpi)
        shap.decision_plot(
            explainer.expected_value,
            shap_values,
            feature_names=feature_names,
            show=False
        )
        st.pyplot(fig_dp, bbox_inches='tight', dpi=plot_dpi)
        plt.close(fig_dp)


# Human-readable feature descriptions for clinicians
FEATURE_LABELS = {
    "BAD-D": "BAD-D",
    "Age": "Age",
    "ARC 3mm": "ARC 3mm (Anterior Radius of Curvature)",
    "Kmax": "Kmax (Maximum Keratometry, D)",
    "Pachy Min": "Pachy Min (Minimum Pachymetry, \u03bcm)",
}

# Ordered list of all 5 input features
ALL_FEATURES = ["Age", "BAD-D", "ARC 3mm", "Kmax", "Pachy Min"]

# Which model(s) use each feature (for help text)
FEATURE_MODELS = {
    "BAD-D": "1-Year model",
    "Age": "Both models",
    "ARC 3mm": "1-Year model",
    "Kmax": "2-Year model",
    "Pachy Min": "2-Year model",
}


def _merge_bounds_for_feature(feature, all_boundaries):
    """Merge boundaries from multiple models for a shared feature (e.g. Age)."""
    sources = [b[feature] for b in all_boundaries if feature in b]
    if len(sources) == 1:
        return sources[0]
    return {
        "min": min(s["min"] for s in sources),
        "max": max(s["max"] for s in sources),
        "mean": sum(s["mean"] for s in sources) / len(sources),
        "std": sum(s["std"] for s in sources) / len(sources),
    }


def main():
    """Main Streamlit application entry point — dual-model version."""

    # Header
    st.markdown('''
    <div class="clinical-header">
        <h1>👁️ Keratoconus Progression Predictor</h1>
        <p>Interactive Demonstration — 1-Year and 2-Year Logistic Regression Models with Baseline Pentacam Features</p>
    </div>
    ''', unsafe_allow_html=True)

    # Research notice
    st.markdown('''
    <div class="research-notice">
        <b>Not for clinical use.</b> This application is a research demonstration accompanying the publication
        <i>"Predicting Keratoconus Progression Within 1–2 Years Using Baseline Tomography and an Explainable
        Open-Access Machine Learning Model"</i> by Gil P, Gil JQ, Mendes L, Alves N, Rosa A, Murta J.
        It is intended solely to illustrate the algorithm's behaviour and is not a validated medical device.
    </div>
    ''', unsafe_allow_html=True)

    # Load both models
    model_1y, scaler_1y, bounds_1y = load_model_and_data("one_year")
    model_2y, scaler_2y, bounds_2y = load_model_and_data("two_year")
    all_boundaries = [bounds_1y, bounds_2y]

    # --- Input Section: 5 features ---
    st.markdown('<p class="section-label">Baseline Pentacam Measurements</p>', unsafe_allow_html=True)

    input_cols = st.columns(len(ALL_FEATURES))
    inputs = {}

    for i, feature in enumerate(ALL_FEATURES):
        merged = _merge_bounds_for_feature(feature, all_boundaries)
        label = FEATURE_LABELS.get(feature, feature)
        used_by = FEATURE_MODELS[feature]

        with input_cols[i]:
            st.markdown(f'<div class="input-card"><h3>{label}</h3></div>', unsafe_allow_html=True)
            # Age is integer in the training data — enforce int input
            if feature == "Age":
                value = st.number_input(
                    f"{feature}",
                    min_value=int(merged['min']),
                    max_value=int(merged['max']),
                    value=round(merged['mean']),
                    step=1,
                    key=feature,
                    label_visibility="collapsed",
                    help=f"Used by: {used_by} | Range: [{int(merged['min'])} — {int(merged['max'])}] | Mean: {merged['mean']:.1f} | SD: {merged['std']:.1f}"
                )
            else:
                value = st.number_input(
                    f"{feature}",
                    min_value=float(max(0, merged['min'])),
                    max_value=float(merged['max']),
                    value=float(merged['mean']),
                    step=float(merged['std'] / 10),
                    key=feature,
                    label_visibility="collapsed",
                    help=f"Used by: {used_by} | Range: [{merged['min']:.3f} — {merged['max']:.3f}] | Mean: {merged['mean']:.3f} | SD: {merged['std']:.3f}"
                )
            inputs[feature] = value

            is_valid, _ = validate_input(value, feature, {"_": merged, feature: merged})
            if not is_valid:
                st.markdown('<span class="validation-warn">Outside training range</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="validation-ok">Within range</span>', unsafe_allow_html=True)

    # --- Predict ---
    st.markdown("")
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    with predict_col2:
        predict_clicked = st.button("Estimate Progression Risk", type="primary", use_container_width=True)

    if predict_clicked:
        # Validate per-model
        validation_errors = []
        for model_key, config in MODEL_CONFIGS.items():
            _, _, bounds = load_model_and_data(model_key)
            for feat in config["feature_names"]:
                is_valid, msg = validate_input(inputs[feat], feat, bounds)
                if not is_valid:
                    validation_errors.append(f"[{config['label']}] {msg}")

        if validation_errors:
            st.error("**Input Validation Errors**")
            for error in validation_errors:
                st.warning(error)
        else:
            try:
                # SHAP guide (once, before both models)
                st.markdown("")
                st.markdown('<p class="section-label">Model Explainability (SHAP Analysis)</p>', unsafe_allow_html=True)
                display_shap_guide()

                # Run both models
                models_data = [
                    ("one_year", model_1y, scaler_1y, bounds_1y),
                    ("two_year", model_2y, scaler_2y, bounds_2y),
                ]

                for model_key, model, scaler, bounds in models_data:
                    config = MODEL_CONFIGS[model_key]
                    feature_names = config["feature_names"]
                    window = config["window"]

                    input_raw = np.array([[inputs[f] for f in feature_names]])
                    input_scaled = scaler.transform(input_raw)
                    prediction = model.predict(input_scaled)[0]

                    # Section header
                    st.markdown(f'<p class="section-label">{config["label"]} Model</p>', unsafe_allow_html=True)

                    # Result card
                    res_col1, res_col2, res_col3 = st.columns([1, 3, 1])
                    with res_col2:
                        if prediction == 1:
                            st.markdown(f'''
                            <div class="result-card result-progression">
                                <h2>PROGRESSION RISK DETECTED</h2>
                                <div class="label">The model predicts keratoconus progression within {window}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        else:
                            st.markdown(f'''
                            <div class="result-card result-stable">
                                <h2>LOW PROGRESSION RISK</h2>
                                <div class="label">The model predicts stability within {window}</div>
                            </div>
                            ''', unsafe_allow_html=True)

                    # SHAP plots
                    display_shap_plots(model, scaler, input_scaled, input_raw, feature_names)

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.exception(e)

    # --- Sidebar ---
    with st.sidebar:
        for model_key, config in MODEL_CONFIGS.items():
            _, _, bounds = load_model_and_data(model_key)

            st.markdown(f"### {config['label']} Model")
            st.markdown(f"""
            | Property | Value |
            |----------|-------|
            | Algorithm | Logistic Regression |
            | Features | {len(config['feature_names'])} ({', '.join(config['feature_names'])}) |
            | Training set | {config['patients']} patients |
            | Balancing | SMOTE (k=5) |
            | Preprocessing | StandardScaler |
            | Prediction window | {config['window']} |
            """)

            st.markdown(f"**Feature Reference Ranges**")
            table_rows = ""
            for feature in config["feature_names"]:
                b = bounds[feature]
                table_rows += f"<tr><td><b>{feature}</b></td><td>{b['min']:.3f}</td><td>{b['max']:.3f}</td><td>{b['mean']:.3f}</td><td>{b['std']:.3f}</td></tr>"

            st.markdown(f'''
            <table class="ref-table">
                <thead><tr><th>Feature</th><th>Min</th><th>Max</th><th>Mean</th><th>SD</th></tr></thead>
                <tbody>{table_rows}</tbody>
            </table>
            ''', unsafe_allow_html=True)

            st.markdown("---")

        st.markdown("""
        <div class="citation-box">
            <b>Citation</b><br>
            Gil P, Gil JQ, Mendes L, Alves N, Rosa A, Murta J.
            <i>"Predicting Keratoconus Progression Within 1–2 Years Using
            Baseline Tomography and an Explainable Open-Access Machine
            Learning Model."</i>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
            <b>Research Demonstration Only</b><br>
            This application is provided for research and educational
            purposes to demonstrate the algorithm described in the
            accompanying publication. It is <b>not</b> a certified or
            validated medical device and must <b>not</b> be used for
            clinical decision-making. Predictions are illustrative and
            should not replace professional ophthalmological assessment.
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
