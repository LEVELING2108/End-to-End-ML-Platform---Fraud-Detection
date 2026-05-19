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
import matplotlib.pyplot as plt
from fpdf import FPDF
import io

# Import Evidently
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

# Page Config
st.set_page_config(page_title="FraudShield High-Dimensional Cockpit", layout="wide", page_icon="🛡️")

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
        model_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS_CHAMPION)
        run_id = model_version.run_id
        run = client.get_run(run_id)
        metrics = run.data.metrics
        artifact_path = "model_package/model_package.pkl"
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
        with open(local_path, "rb") as f:
            package = pickle.load(f)
        return package["model"], package["explainer"], metrics, model_version.version
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

@st.cache_data
def get_drift_report():
    if not os.path.exists('data/train.csv') or not os.path.exists('data/test.csv'):
        return None
    reference = pd.read_csv('data/train.csv')
    current = pd.read_csv('data/test.csv')
    report = Report(metrics=[DataDriftPreset(), DataSummaryPreset()])
    snapshot = report.run(reference_data=reference.drop('Class', axis=1), current_data=current.drop('Class', axis=1))
    report_path = "monitoring_report.html"
    snapshot.save_html(report_path)
    with open(report_path, 'r', encoding='utf-8') as f:
        html_data = f.read()
    return html_data

def create_shap_bar_chart(feature_names, shap_values):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#ff0051' if v > 0 else '#008bfb' for v in shap_values]
    ax.barh(feature_names, shap_values, color=colors)
    ax.set_xlabel('Impact on Fraud Risk')
    plt.axvline(0, color='black', linewidth=0.8)
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    plt.close()
    return img_buf

class FraudReportPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'FraudShield Enterprise Report', 0, 1, 'C')
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(tx_data, result, metrics, version, shap_img):
    pdf = FraudReportPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '1. Transaction & Result', 0, 1)
    pdf.set_font('Arial', '', 10)
    for k, v in tx_data.items(): pdf.cell(0, 7, f"- {k}: {v}", 0, 1)
    pdf.ln(5)
    pdf.cell(0, 7, f"DECISION: {result['decision']} ({result['prob']:.2%})", 0, 1)
    pdf.ln(5)
    pdf.cell(0, 10, '2. Visual Explanation', 0, 1)
    pdf.image(shap_img, x=10, w=150)
    return pdf.output()

def main():
    if 'audit_log' not in st.session_state: st.session_state.audit_log = []
    st.title("🛡️ FraudShield: High-Dimensional Dashboard")
    model, explainer, metrics, version = load_champion_model()
    if model is None: return

    with st.sidebar:
        st.header("📊 Model Metrics")
        st.success(f"Champion: v{version}")
        if metrics:
            st.metric("AUPRC (Fraud)", f"{metrics.get('auprc', 0):.4f}")
            st.metric("F1-Score", f"{metrics.get('f1', 0):.4f}")
        st.info("Imbalance Handling: SMOTE Enabled")

    tab1, tab2, tab3 = st.tabs(["🚀 Predict", "📈 Monitoring", "📜 Audit"])

    with tab1:
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.subheader("Features")
            amount = st.number_input("Amount", value=100.0)
            time = st.number_input("Time (sec)", value=1000.0)
            v_inputs = {}
            for i in range(1, 11):
                v_inputs[f'V{i}'] = st.slider(f"V{i}", -5.0, 5.0, 0.0)
            predict_btn = st.button("Analyze", type="primary")

        if predict_btn:
            input_dict = {"Time": time}
            input_dict.update(v_inputs)
            input_dict["Amount"] = amount
            X = np.array([list(input_dict.values())])
            
            prob = model.predict_proba(X)[0][1]
            is_fraud = prob > 0.5
            confidence = prob if is_fraud else (1 - prob)
            
            st.session_state.audit_log.insert(0, {
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Amount": amount,
                "Decision": "FRAUD" if is_fraud else "LEGIT",
                "Prob": round(prob, 4)
            })
            
            with col2:
                st.subheader("Results")
                if is_fraud: st.error(f"🚨 FRAUD DETECTED ({prob:.2%})")
                else: st.success(f"✅ LEGITIMATE ({prob:.2%})")
                
                shap_values_raw = explainer.shap_values(X)
                sv = shap_values_raw[1][0] if isinstance(shap_values_raw, list) else shap_values_raw[0, :, 1]
                bv = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
                
                feature_names = list(input_dict.keys())
                st_shap(shap.force_plot(float(bv), sv, X[0], feature_names=feature_names), height=200)
                
                if st.button("Generate PDF"):
                    shap_img = create_shap_bar_chart(feature_names, sv)
                    pdf_data = generate_pdf_report(input_dict, {"decision": "FRAUD" if is_fraud else "LEGIT", "prob": prob}, metrics, version, shap_img)
                    st.download_button("Download PDF", data=bytes(pdf_data), file_name="report.pdf", mime="application/pdf")

    with tab2:
        st.header("Drift Report")
        html = get_drift_report()
        if html: components.html(html, height=800, scrolling=True)

    with tab3:
        st.header("Session Audit")
        if st.session_state.audit_log: st.dataframe(pd.DataFrame(st.session_state.audit_log), use_container_width=True)

if __name__ == "__main__":
    main()
