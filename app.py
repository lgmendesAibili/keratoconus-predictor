"""
Keratoconus Progression Prediction — Streamlit Web Application (Privacy-Safe)

Interactive demonstration of the machine learning model described in:

    Gil P, Gil JQ, Mendes L, Alves N, Rosa A, Murta J.
    "Predicting Keratoconus Progression Within 1-2 Years Using Baseline
    Tomography and an Explainable Open-Access Machine Learning Model"

Users enter 3 baseline Pentacam features (BAD-D, Age, ARC 3mm) and receive
a 1-year progression risk prediction with SHAP explanations showing how
each feature drives the individual prediction.

This version does NOT require patient training data (X_train.pkl). SHAP
explanations use a synthetic background derived from the fitted StandardScaler,
making it safe to deploy on public repositories without exposing patient data.

The model was trained on real keratoconus data (412 patients, one-year prediction
window) with SMOTE balancing and StandardScaler preprocessing.

Features:
    - Input validation against training data boundaries
    - SHAP waterfall plot and decision plot for interpretability
    - Probability display for both classes
    - StandardScaler preprocessing matching the training pipeline
    - No patient data required at runtime

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
    .result-card .confidence {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .result-card .label {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    /* Probability bar */
    .prob-container {
        background: #f8f9fa;
        border-radius: 0.75rem;
        padding: 1.2rem;
        border: 1px solid #d5dbdb;
    }
    .prob-bar-track {
        background: #e8e8e8;
        border-radius: 1rem;
        height: 2rem;
        position: relative;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .prob-bar-fill-stable {
        background: linear-gradient(90deg, #27ae60, #2ecc71);
        height: 100%;
        border-radius: 1rem 0 0 1rem;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
        min-width: 3rem;
    }
    .prob-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        color: #7f8c8d;
        margin-top: 0.25rem;
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


@st.cache_resource
def load_model_and_data():
    """Load the trained model, scaler, and feature boundaries.

    Returns:
        tuple: A 3-tuple of (model, scaler, boundaries) where:
            - model: sklearn LogisticRegression fitted classifier.
            - scaler: sklearn StandardScaler fitted on training data.
            - boundaries: dict mapping feature names to dicts with keys
              'min', 'max', 'mean', 'std'.

    Raises:
        SystemExit: Stops the Streamlit app if pickle files are missing.
    """
    try:
        model = joblib.load(APP_DIR / 'logistic_model.pkl')
        scaler = joblib.load(APP_DIR / 'scaler.pkl')
        boundaries = joblib.load(APP_DIR / 'boundaries.pkl')
        return model, scaler, boundaries
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}")
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


def display_shap_plots(model, scaler, input_data, feature_names):
    """Render SHAP waterfall plot and decision plot for a prediction.

    Generates two matplotlib-based visualizations (chosen for Streamlit Cloud
    compatibility over JavaScript-based alternatives):

    1. **Waterfall plot** — vertical breakdown of individual feature
       contributions ordered by magnitude.
    2. **Decision plot** — cumulative path from the base value to the final
       prediction, showing how each feature shifts the output sequentially.

    All matplotlib figures are explicitly closed after rendering to prevent
    memory leaks in long-running Streamlit sessions.

    Args:
        model: Fitted sklearn LogisticRegression estimator.
        scaler: Fitted sklearn StandardScaler.
        input_data: numpy.ndarray of shape (1, n_features) with scaled user inputs.
        feature_names: list[str] of feature names matching input_data columns.
    """
    explainer = get_shap_explainer(model, scaler)
    shap_values = explainer.shap_values(input_data)

    st.markdown('''
    <div class="shap-guide">
        <b>How to read these plots:</b> Each feature either pushes the prediction
        toward <b style="color:#c0392b">progression</b> (positive SHAP value) or
        toward <b style="color:#27ae60">stability</b> (negative SHAP value).
        The magnitude indicates how strongly that feature influences this
        individual prediction. The base value represents the average model output
        across the training population.
    </div>
    ''', unsafe_allow_html=True)

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
            data=input_data[0],
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
    "BAD-D": "BAD-D (Belin/Ambrosio Enhanced Ectasia Display)",
    "Age": "Age at Baseline (years)",
    "ARC 3mm": "ARC 3mm Zone (Anterior Radius of Curvature)",
}


def main():
    """Main Streamlit application entry point.

    Orchestrates the full prediction workflow:

    1. Loads model artifacts via ``load_model_and_data()``.
    2. Renders input fields (one per clinical feature) with defaults set to
       training-data means, step sizes of ``std / 10``, and an extended
       min/max range of +/-1 SD to allow exploratory inputs.
    3. Shows a live validation summary.
    4. On button click, runs inference and displays:
       - Predicted class with confidence percentage.
       - Class probability metrics.
       - SHAP waterfall and decision plots for interpretability.
    5. Populates the sidebar with model metadata, feature boundaries, and citation.
    """

    # Header
    st.markdown('''
    <div class="clinical-header">
        <h1>👁️ Keratoconus 1-Year Progression Predictor</h1>
        <p>Interactive Demonstration — Logistic Regression Model with 3 Baseline Pentacam Features</p>
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

    # Load model and data
    model, scaler, boundaries = load_model_and_data()
    feature_names = list(boundaries.keys())

    # --- Input Section ---
    st.markdown('<p class="section-label">Baseline Pentacam Measurements</p>', unsafe_allow_html=True)

    input_cols = st.columns(len(feature_names))
    inputs = {}
    validation_errors = []

    for i, feature in enumerate(feature_names):
        bounds = boundaries[feature]
        label = FEATURE_LABELS.get(feature, feature)

        with input_cols[i]:
            st.markdown(f'<div class="input-card"><h3>{label}</h3></div>', unsafe_allow_html=True)
            # Allow input within ±1 SD beyond training min/max, clamped to 0
            margin = bounds['std']
            value = st.number_input(
                f"{feature}",
                min_value=float(max(0, bounds['min'] - margin)),
                max_value=float(bounds['max'] + margin),
                value=float(bounds['mean']),
                step=float(bounds['std'] / 10),
                key=feature,
                label_visibility="collapsed",
                help=f"Training range: [{bounds['min']:.3f} — {bounds['max']:.3f}] | Mean: {bounds['mean']:.3f} | SD: {bounds['std']:.3f}"
            )
            inputs[feature] = value

            is_valid, error_msg = validate_input(value, feature, boundaries)
            if not is_valid:
                validation_errors.append(error_msg)
                st.markdown('<span class="validation-warn">Outside training range</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="validation-ok">Within range</span>', unsafe_allow_html=True)

    # --- Predict ---
    st.markdown("")
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    with predict_col2:
        predict_clicked = st.button("Estimate 1-Year Progression Risk", type="primary", use_container_width=True)

    if predict_clicked:
        if validation_errors:
            st.error("**Input Validation Errors**")
            for error in validation_errors:
                st.warning(error)
        else:
            input_raw = np.array([[inputs[f] for f in feature_names]])
            input_array = scaler.transform(input_raw)

            try:
                prediction = model.predict(input_array)[0]
                probability = model.predict_proba(input_array)[0]
                prob_stable = probability[0]
                prob_progression = probability[1]

                # --- Result Display ---
                st.markdown("")
                res_col1, res_col2, res_col3 = st.columns([1, 3, 1])

                with res_col2:
                    if prediction == 1:
                        st.markdown(f'''
                        <div class="result-card result-progression">
                            <h2>PROGRESSION RISK DETECTED</h2>
                            <div class="confidence">{prob_progression:.1%}</div>
                            <div class="label">Estimated probability of progression within 1 year</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                        <div class="result-card result-stable">
                            <h2>LOW PROGRESSION RISK</h2>
                            <div class="confidence">{prob_stable:.1%}</div>
                            <div class="label">Estimated probability of stability within 1 year</div>
                        </div>
                        ''', unsafe_allow_html=True)

                # --- Probability Breakdown ---
                st.markdown("")
                prob_col1, prob_col2, prob_col3 = st.columns([1, 3, 1])
                with prob_col2:
                    st.markdown(f'''
                    <div class="prob-container">
                        <div class="prob-bar-track">
                            <div class="prob-bar-fill-stable" style="width: {prob_stable*100:.1f}%">
                                Stable {prob_stable:.1%}
                            </div>
                        </div>
                        <div class="prob-labels">
                            <span>Stable (No Progression)</span>
                            <span>Progression — {prob_progression:.1%}</span>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)

                # --- SHAP Explainability ---
                st.markdown("")
                st.markdown('<p class="section-label">Model Explainability (SHAP Analysis)</p>', unsafe_allow_html=True)
                display_shap_plots(model, scaler, input_array, feature_names)

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.exception(e)

    # --- Sidebar ---
    with st.sidebar:
        st.markdown("### Model Details")
        st.markdown("""
        | Property | Value |
        |----------|-------|
        | Algorithm | Logistic Regression |
        | Features | 3 |
        | Training set | 412 patients |
        | Balancing | SMOTE (k=5) |
        | Preprocessing | StandardScaler |
        | Prediction window | 1 year |
        """)

        st.markdown("### Feature Reference Ranges")
        table_rows = ""
        for feature, bounds in boundaries.items():
            table_rows += f"<tr><td><b>{feature}</b></td><td>{bounds['min']:.3f}</td><td>{bounds['max']:.3f}</td><td>{bounds['mean']:.3f}</td><td>{bounds['std']:.3f}</td></tr>"

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
