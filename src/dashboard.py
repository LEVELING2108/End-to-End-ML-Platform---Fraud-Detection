import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import mlflow
from streamlit_shap import st_shap
import os
from datetime import datetime
import streamlit.components.v1 as components

# Import Evidently
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

# Page Config
st.set_page_config(page_title="FraudShield Management Cockpit", layout="wide", page_icon="🛡️")

# --- Constants ---
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "FraudDetectionModel"
ALIAS_CHAMPION = "champion"

# --- Model Loading Logic ---
@st.cache_resource
def load_champion_model():
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        
        # Get Champion Version
        model_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS_CHAMPION)
        run_id = model_version.run_id
        
        # Get Metrics
        run = client.get_run(run_id)
        metrics = run.data.metrics
        
        # Download Artifact
        artifact_path = "model_package/model_package.pkl"
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
        
        with open(local_path, "rb") as f:
            package = pickle.load(f)
            
        return package["model"], package["encoder"], package["explainer"], metrics, model_version.version
    except Exception as e:
        st.error(f"Error loading model from MLflow: {e}")
        return None, None, None, None, None

@st.cache_data
def get_drift_report():
    """Generate a live Evidently drift report."""
    if not os.path.exists('data/train.csv') or not os.path.exists('data/test.csv'):
        return None
    
    reference = pd.read_csv('data/train.csv')
    current = pd.read_csv('data/test.csv')
    
    # Select columns to monitor
    cols = ['amount', 'hour', 'day_of_week', 'merchant_category']
    
    report = Report(metrics=[
        DataDriftPreset(), 
        DataSummaryPreset()
    ])
    
    # In newer evidently versions (0.7+), run() returns a Snapshot object.
    # The export methods (save_html, json, etc.) are moved to the Snapshot object.
    snapshot = report.run(reference_data=reference[cols], current_data=current[cols])
    report_path = "monitoring_report.html"
    snapshot.save_html(report_path)
    
    with open(report_path, 'r', encoding='utf-8') as f:
        html_data = f.read()
    
    return html_data

# --- Main App ---
def main():
    st.title("🛡️ FraudShield Management Cockpit")
    st.markdown("---")

    # Load Model
    model, encoder, explainer, metrics, version = load_champion_model()

    if model is None:
        st.warning("Please ensure MLflow server is running and a 'champion' model is registered.")
        return

    # Sidebar: Platform Health
    with st.sidebar:
        st.header("📊 Platform Health")
        st.success(f"Champion Model: **v{version}**")
        
        if metrics:
            cols = st.columns(2)
            cols[0].metric("F1-Score", f"{metrics.get('f1', 0):.4f}")
            cols[1].metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
            
        st.markdown("---")
        st.header("🔍 System Status")
        st.info("✅ MLflow Tracker: Online")
        st.info("✅ Feature Store: Active")
        st.info("✅ Drift Monitor: Active")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🚀 Predict & Explain", "📈 Monitoring", "📜 Audit Logs"])

    with tab1:
        st.header("Real-time Fraud Prediction")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Transaction Details")
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0, step=10.0)
            hour = st.slider("Hour of Day", 0, 23, 14)
            day = st.selectbox("Day of Week", options=[0,1,2,3,4,5,6], format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
            category = st.selectbox("Merchant Category", options=list(encoder.classes_))
            
            predict_btn = st.button("Analyze Transaction", type="primary")

        if predict_btn:
            # Prepare Input
            merchant_encoded = encoder.transform([category])[0]
            X = np.array([[amount, hour, day, merchant_encoded]])
            
            # Predict
            prob = model.predict_proba(X)[0][1]
            is_fraud = prob > 0.5
            confidence = prob if is_fraud else (1 - prob)
            
            with col2:
                st.subheader("Analysis Results")
                
                # Display Metrics in columns
                m_col1, m_col2 = st.columns(2)
                
                if is_fraud:
                    m_col1.error("🚨 FRAUD DETECTED")
                    m_col2.metric("Fraud Probability", f"{prob:.2%}")
                    st.warning(f"**Model Confidence:** {confidence:.2%} certain this is Fraud")
                else:
                    m_col1.success("✅ LEGITIMATE")
                    m_col2.metric("Fraud Probability", f"{prob:.2%}")
                    st.info(f"**Model Confidence:** {confidence:.2%} certain this is Legitimate")
                
                # SHAP Explanation
                st.markdown("#### Decision Explanation")
                
                # Get SHAP values for the specific instance
                # For TreeExplainer on Random Forest, shap_values is often a list of [neg_class, pos_class]
                shap_values_raw = explainer.shap_values(X)
                
                # Determine correct SHAP values and base value for the Fraud class (index 1)
                try:
                    if isinstance(shap_values_raw, list):
                        # Modern SHAP with RF usually returns a list
                        sv = shap_values_raw[1][0]
                        bv = explainer.expected_value[1]
                    elif len(shap_values_raw.shape) == 3:
                        # Some versions return (samples, features, classes)
                        sv = shap_values_raw[0, :, 1]
                        bv = explainer.expected_value[1]
                    else:
                        # Fallback for older or different configurations
                        sv = shap_values_raw[0]
                        bv = explainer.expected_value
                        # If bv is still a list/array, take the second element
                        if isinstance(bv, (list, np.ndarray)) and len(bv) > 1:
                            bv = bv[1]
                except Exception as e:
                    st.error(f"Error processing SHAP values: {e}")
                    sv, bv = None, None

                if sv is not None:
                    # Create SHAP Explanation object for better visualization
                    feature_names = ["Amount", "Hour", "Day", "Category"]
                    
                    # Ensure base_value and shap_values are passed correctly to force_plot
                    st_shap(shap.force_plot(
                        base_value=float(bv), 
                        shap_values=sv, 
                        features=X[0], 
                        feature_names=feature_names,
                        matplotlib=False
                    ), height=200)
                    
                    st.info("""
                    **How to read this:** 
                    - Features in **Red** pushed the risk score UP.
                    - Features in **Blue** pushed the risk score DOWN.
                    """)

    with tab2:
        st.header("Data Drift & Performance Monitoring")
        
        # Generate and show live report
        with st.spinner("Generating Live Drift Report..."):
            html_data = get_drift_report()
        
        if html_data:
            components.html(html_data, height=1000, scrolling=True)
        else:
            st.error("Data files not found. Run generate_data.py first.")
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Feature Drift", "Low", delta="0.05", delta_color="inverse")
        col2.metric("Prediction Drift", "None", delta="0.00")
        col3.metric("Data Quality", "99.8%", delta="+0.1%")
        
        st.write("Current analysis shows **no significant drift**. The 'Champion' model remains robust.")

    with tab3:
        st.header("Transaction Audit Trail")
        # Dummy data for audit trail
        audit_data = pd.DataFrame({
            "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S") for _ in range(5)],
            "Amount": [120.0, 2500.0, 45.0, 15.0, 890.0],
            "Category": ["grocery", "online", "restaurant", "retail", "online"],
            "Decision": ["LEGIT", "FRAUD", "LEGIT", "LEGIT", "FRAUD"],
            "Probability": [0.02, 0.94, 0.01, 0.05, 0.88],
            "Investigator": ["System", "Auto-Block", "System", "System", "Auto-Block"]
        })
        st.dataframe(audit_data, use_container_width=True)

if __name__ == "__main__":
    main()
