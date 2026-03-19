"""
Keratoconus Progression Prediction — Streamlit Web Application

A clinical decision-support tool that loads a pre-trained logistic regression model
for binary classification of keratoconus progression risk. Users enter 5 numerical
clinical features and receive real-time predictions with SHAP force plot
explanations showing individual feature contributions.

Features:
    - Input validation against training data boundaries
    - Real-time sparkline distribution visualization per feature
    - SHAP force plot for model interpretability
    - Probability display for both classes

Dependencies:
    streamlit, numpy, pandas, joblib, shap, matplotlib, sparklines

Usage:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sparklines import sparklines
import streamlit.components.v1 as components
from pathlib import Path

# Resolve paths relative to this file
APP_DIR = Path(__file__).parent

# Page configuration
st.set_page_config(page_title="Binary Classification Predictor", layout="wide")

# Custom CSS for better styling
st.markdown('''
<style>
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .feature-stats {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
''', unsafe_allow_html=True)

@st.cache_resource
def load_model_and_data():
    '''Load the trained model, boundaries, and training data'''
    try:
        model = joblib.load(APP_DIR / 'logistic_model.pkl')
        boundaries = joblib.load(APP_DIR / 'boundaries.pkl')
        X_train = joblib.load(APP_DIR / 'X_train.pkl')
        return model, boundaries, X_train
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}")
        st.stop()

def create_sparkline(data):
    '''Create sparkline visualization for a feature distribution'''
    try:
        hist, _ = np.histogram(data, bins=20)
        # Normalize histogram for better visualization
        hist_normalized = (hist / hist.max() * 8).astype(int)
        spark_chars = ''.join(sparklines(hist_normalized))
        return spark_chars
    except Exception:
        return "▁▂▃▄▅▆▇█"  # Default sparkline on error

def validate_input(value, feature_name, boundaries):
    '''Validate that input is within training data boundaries'''
    min_val = boundaries[feature_name]['min']
    max_val = boundaries[feature_name]['max']

    if value < min_val or value > max_val:
        return False, f"⚠️ {feature_name} must be between {min_val:.3f} and {max_val:.3f}"
    return True, ""

@st.cache_resource
def get_shap_explainer(_model, _X_train):
    '''Create and cache the SHAP explainer'''
    return shap.LinearExplainer(_model, _X_train)

def display_shap_force_plot(model, X_train, input_data, feature_names):
    '''Display SHAP force plot for the input'''
    explainer = get_shap_explainer(model, X_train)
    shap_values = explainer.shap_values(input_data)

    # Use matplotlib-based force plot (reliable on Streamlit Cloud)
    fig = plt.figure(figsize=(12, 3))
    shap.force_plot(
        explainer.expected_value,
        shap_values,
        input_data,
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    st.pyplot(fig, bbox_inches='tight')
    plt.close(fig)

    # Also show a waterfall plot for clearer feature breakdown
    st.markdown("#### Feature Contribution Breakdown")
    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_data[0],
        feature_names=feature_names
    )
    fig_wf, ax_wf = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(explanation, show=False)
    st.pyplot(fig_wf, bbox_inches='tight')
    plt.close(fig_wf)

def main():
    """Main Streamlit application entry point.

    Renders the prediction interface with input fields, validation,
    model inference, probability display, and SHAP force plot visualization.
    """
    st.title("🔮 Binary Classification Predictor")
    st.markdown("### Logistic Regression Model with 5 Numerical Inputs")

    # Load model and data
    model, boundaries, X_train = load_model_and_data()
    feature_names = list(boundaries.keys())

    # Create input form
    st.markdown("---")
    st.subheader("📊 Input Features")

    # Create two columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Enter feature values:")

        # Collect inputs with validation
        inputs = {}
        validation_errors = []

        for feature in feature_names:
            bounds = boundaries[feature]

            # Create input with appropriate range
            col_input, col_spark = st.columns([3, 1])

            with col_input:
                # Input with default value at mean
                value = st.number_input(
                    f"{feature}",
                    min_value=float(bounds['min'] - abs(bounds['min']) * 0.5),
                    max_value=float(bounds['max'] + abs(bounds['max']) * 0.5),
                    value=float(bounds['mean']),
                    step=float(bounds['std'] / 10),
                    key=feature,
                    help=f"Valid range: [{bounds['min']:.3f}, {bounds['max']:.3f}]"
                )
                inputs[feature] = value

                # Validate input
                is_valid, error_msg = validate_input(value, feature, boundaries)
                if not is_valid:
                    validation_errors.append(error_msg)

            with col_spark:
                # Get training data for this feature
                feature_idx = feature_names.index(feature)
                feature_data = X_train[:, feature_idx]
                sparkline = create_sparkline(feature_data)
                st.markdown(f"<div class='feature-stats'><small>Distribution:</small><br>{sparkline}</div>",
                           unsafe_allow_html=True)

    with col2:
        st.markdown("#### Current Input Values:")
        for feature, value in inputs.items():
            bounds = boundaries[feature]
            in_range = bounds['min'] <= value <= bounds['max']
            icon = "✅" if in_range else "⚠️"
            st.markdown(f"{icon} **{feature}**: {value:.3f}")

    st.markdown("---")

    # Prediction button
    if st.button("🎯 Make Prediction", type="primary", use_container_width=True):
        # Check for validation errors
        if validation_errors:
            st.error("### ❌ Input Validation Errors")
            for error in validation_errors:
                st.error(error)
            st.warning("Please adjust your inputs to be within the training data range.")
        else:
            # Create input array
            input_array = np.array([[inputs[f] for f in feature_names]])

            # Make prediction
            try:
                prediction = model.predict(input_array)[0]
                probability = model.predict_proba(input_array)[0]

                # Display result
                st.markdown("---")
                st.subheader("🎯 Prediction Result")

                result_col1, result_col2, result_col3 = st.columns([1, 2, 1])

                with result_col2:
                    if prediction == 1:
                        st.error("### ⚠️ WARNING")
                        st.markdown(f"**Class: 1** (Positive)")
                        st.markdown(f"**Confidence:** {probability[1]:.1%}")
                    else:
                        st.success("### ✅ OK")
                        st.markdown(f"**Class: 0** (Negative)")
                        st.markdown(f"**Confidence:** {probability[0]:.1%}")

                # Display probabilities
                st.markdown("---")
                st.subheader("📈 Class Probabilities")
                prob_col1, prob_col2 = st.columns(2)

                with prob_col1:
                    st.metric("Class 0 (OK)", f"{probability[0]:.1%}")
                with prob_col2:
                    st.metric("Class 1 (Warning)", f"{probability[1]:.1%}")

                # Display SHAP force plot
                st.markdown("---")
                st.subheader("🔍 Feature Importance (SHAP Analysis)")
                st.markdown("This plot shows how each feature contributes to the prediction:")
                display_shap_force_plot(model, X_train, input_array, feature_names)

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.exception(e)

    # Display model info in sidebar
    with st.sidebar:
        st.markdown("### 📋 Model Information")
        st.info(f"**Model Type:** Logistic Regression\n**Features:** {len(feature_names)}\n**Training Samples:** {len(X_train)}")

        st.markdown("### 📊 Feature Boundaries")
        for feature, bounds in boundaries.items():
            st.markdown(f"**{feature}**")
            st.markdown(f"- Min: {bounds['min']:.3f}")
            st.markdown(f"- Max: {bounds['max']:.3f}")
            st.markdown(f"- Mean: {bounds['mean']:.3f}")

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("This app uses a trained logistic regression model for binary classification. "
                   "All inputs are validated against training data boundaries to ensure reliable predictions.")

if __name__ == "__main__":
    main()
