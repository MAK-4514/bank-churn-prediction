"""
streamlit_app.py
----------------
Bank Customer Churn Risk Scoring – Interactive Streamlit Application (v2.0.0).

Features
--------
• Loads the trained GBC pipeline from models/model.pkl (Defensive checks included)
• Feature Auditing: Ensures inference columns match training exactly
• Tab 1: Customer Prediction UI
• Tab 2: What-If Simulator (Synced with Tab 1)
• Tab 3: SHAP Model Explainability
• Audit Logging: Record all predictions to logs/predictions.csv
"""

import sys
import json
import csv
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
APP_DIR      = Path(__file__).resolve().parent       # app/
PROJECT_ROOT = APP_DIR.parent                        # project root
LOG_DIR      = PROJECT_ROOT / "logs"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Internal imports
from utils import assign_risk_band, assign_risk_band_optimized  # noqa: E402
from version import __version__ as APP_VERSION

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Churn Risk Scorer",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS – premium dark UI
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
:root {
  --bg-primary:    #0f172a;
  --bg-secondary:  #1e293b;
  --bg-card:       #1e293b;
  --accent-blue:   #3b82f6;
  --accent-purple: #8b5cf6;
  --accent-green:  #10b981;
  --accent-yellow: #f59e0b;
  --accent-red:    #ef4444;
  --text-primary:  #f1f5f9;
  --text-muted:    #94a3b8;
  --border:        #334155;
}
.stApp {
  background: var(--bg-primary) !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', 'Segoe UI', sans-serif;
}
section[data-testid="stSidebar"] {
  background: var(--bg-secondary) !important;
  border-right: 1px solid var(--border);
}
div[data-testid="metric-container"] {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 20px;
}
.stButton > button {
  background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
  color: white !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  padding: 0.6rem 2.2rem !important;
}
.badge {
  display: inline-block;
  padding: 6px 18px;
  border-radius: 50px;
  font-weight: 700;
}
.badge-low    { background: #064e3b; color: #34d399; border: 1px solid #34d399; }
.badge-medium { background: #78350f; color: #fbbf24; border: 1px solid #fbbf24; }
.badge-high   { background: #7f1d1d; color: #f87171; border: 1px solid #f87171; }
.info-box {
  background: #1e293b;
  border-left: 4px solid var(--accent-blue);
  padding: 12px 16px;
  border-radius: 6px;
  color: var(--text-muted);
}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    model_path = PROJECT_ROOT / "models" / "model.pkl"
    threshold_path = PROJECT_ROOT / "models" / "best_threshold.json"

    if not model_path.exists():
        return None, 0.50, 0.50, f"Model file not found at {model_path}"

    try:
        pipeline = joblib.load(model_path)
    except Exception as e:
        return None, 0.50, 0.50, f"Error loading model: {e}"

    if not hasattr(pipeline, "predict_proba"):
        return "NO_PROBA", 0.50, 0.50, "Loaded pipeline does not support predict_proba."

    best_f1, best_recall = 0.50, 0.50
    if threshold_path.exists():
        try:
            with open(threshold_path) as f:
                cfg = json.load(f)
            best_f1 = float(cfg.get("best_threshold_f1", 0.50))
            best_recall = float(cfg.get("best_threshold_recall", 0.50))
        except Exception:
            pass

    return pipeline, best_f1, best_recall, None

pipeline, BEST_THRESH_F1, BEST_THRESH_RECALL, load_error = load_model()

# ──────────────────────────────────────────────────────────────────────────────
# Feature Audit & Scoring
# ──────────────────────────────────────────────────────────────────────────────
def score_customer(data: dict, threshold: float, mode: str) -> dict:
    if pipeline is None or pipeline == "NO_PROBA":
        return {"prob": 0.0, "band": "Unknown", "flag": 0}

    # 1. Build DataFrame
    row = pd.DataFrame([data])
    
    # 2. Audit Features
    if hasattr(pipeline, "feature_names_in_"):
        expected = list(pipeline.feature_names_in_)
        missing = set(expected) - set(row.columns)
        extra = set(row.columns) - set(expected)
        
        if missing or extra:
            st.error(f"⚠️ **Feature Schema Mismatch!**\n\nMissing: `{missing}`\nExtra: `{extra}`")
            st.stop()
        
        row = row[expected] # Reorder
    
    # 3. Predict
    prob = float(pipeline.predict_proba(row)[0, 1])
    
    # 4. Banding
    if "Optimized" in mode:
        band = assign_risk_band_optimized(np.array([prob]), threshold)[0]
    else:
        band = assign_risk_band(np.array([prob]))[0]
        
    return {"prob": prob, "band": band, "flag": int(prob >= threshold)}

def render_risk_badge(band: str) -> str:
    cls = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high"}.get(band, "badge-medium")
    return f'<span class="badge {cls}">{band.upper()} RISK</span>'

# ──────────────────────────────────────────────────────────────────────────────
# App Header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style='text-align:center; padding-bottom: 20px;'>
      <h1 style='background:linear-gradient(135deg,#3b82f6,#8b5cf6); -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-size:2.5rem;'>
        🏦 Bank Customer Churn Risk Scorer
      </h1>
      <p style='color:#94a3b8;'>v{APP_VERSION} &nbsp;·&nbsp; Enterprise ML Dashboard</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Fail fast checks
if load_error:
    st.error(f"🚫 {load_error}")
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar Settings
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Settings")
    st.markdown("---")
    
    threshold_mode = st.radio("Decision Threshold", ["F1-Optimized", "Recall-Priority", "Custom"])
    if threshold_mode == "F1-Optimized":
        active_threshold = BEST_THRESH_F1
    elif threshold_mode == "Recall-Priority":
        active_threshold = BEST_THRESH_RECALL
    else:
        active_threshold = st.slider("Custom Threshold", 0.0, 1.0, 0.50, 0.01)

    risk_band_mode = st.selectbox("Risk Banding Mode", ["Fixed (0.40/0.70)", "Optimized (Uses Threshold)"])
    
    st.markdown("---")
    st.markdown("### 📝 Audit Control")
    enable_logging = st.checkbox("Enable audit logging", value=True)
    
    log_file = LOG_DIR / "predictions.csv"
    if log_file.exists():
        with open(log_file, "rb") as f:
            st.download_button("📥 Download Logs (CSV)", data=f, file_name="churn_audit_logs.csv", mime="text/csv")

# ──────────────────────────────────────────────────────────────────────────────
# Main UI
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "🔄 What-If Simulator", "🧠 Explainability"])

with tab1:
    st.markdown("### 👤 Customer Profile")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Core Info**")
        cs = st.number_input("Credit Score", 300, 850, 650, key="cs_main")
        geo = st.selectbox("Geography", ["France", "Germany", "Spain"], key="geo_main")
        gen = st.selectbox("Gender", ["Female", "Male"], key="gen_main")
        age = st.number_input("Age", 18, 100, 40, key="age_main")

    with col2:
        st.markdown("**Activity**")
        ten = st.number_input("Tenure (Years)", 0, 20, 5, key="ten_main")
        nop = st.number_input("Number of Products", 1, 4, 1, key="nop_main")
        hcc = st.selectbox("Has Credit Card", [0, 1], format_func=lambda x: "Yes" if x else "No", key="hcc_main")
        iam = st.selectbox("Is Active Member", [0, 1], format_func=lambda x: "Yes" if x else "No", key="iam_main")

    with col3:
        st.markdown("**Financials**")
        bal = st.number_input("Balance (€)", 0.0, 300000.0, 75000.0, step=500.0, key="bal_main")
        sal = st.number_input("Estimated Salary (€)", 0.0, 300000.0, 100000.0, step=500.0, key="sal_main")

    predict_btn = st.button("🚀 Run Prediction", use_container_width=True)

    if predict_btn:
        input_data = {
            "CreditScore": cs, "Geography": geo, "Gender": gen, "Age": age,
            "Tenure": ten, "Balance": bal, "NumOfProducts": nop,
            "HasCrCard": int(hcc), "IsActiveMember": int(iam), "EstimatedSalary": sal
        }
        res = score_customer(input_data, active_threshold, risk_band_mode)
        st.session_state["last_res"] = res
        st.session_state["last_input"] = input_data

        # Results Display
        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        r1.metric("Churn Probability", f"{res['prob']*100:.1f}%")
        r2.metric("Decision", "⚠️ CHURN" if res['flag'] else "✅ RETAIN")
        r3.metric("Risk Band", res['band'])
        
        st.markdown(f"<div style='text-align:center;'>{render_risk_badge(res['band'])}</div>", unsafe_allow_html=True)

        if enable_logging:
            LOG_DIR.mkdir(exist_ok=True)
            exists = log_file.exists()
            with open(log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not exists:
                    writer.writerow(["Timestamp"] + list(input_data.keys()) + ["Threshold", "BandingMode", "Prob", "Band", "Flag"])
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S")] + list(input_data.values()) + [active_threshold, risk_band_mode, res['prob'], res['band'], res['flag']])

with tab2:
    st.markdown("### 🔄 What-If Simulator")
    st.info("Baseline profile is synced from Tab 1. Adjust levers below to see the impact.")
    
    # Baseline from Tab 1 keys
    base_iam = st.session_state.get("iam_main", 1)
    base_nop = st.session_state.get("nop_main", 1)
    base_bal = st.session_state.get("bal_main", 75000.0)

    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("**Intervention Levers**")
        sim_iam = st.selectbox("Simulated Active Status", [0, 1], index=int(base_iam), format_func=lambda x: "Active" if x else "Inactive", key="sim_iam")
        sim_nop = st.number_input("Simulated Products", 1, 4, int(base_nop), key="sim_nop")
        sim_bal = st.number_input("Simulated Balance (€)", 0.0, 300000.0, float(base_bal), step=1000.0, key="sim_bal")

    with sc2:
        st.markdown("**Intervention Impact**")
        # Build base and sim data
        base_data = {
            "CreditScore": st.session_state.get("cs_main", 650), "Geography": st.session_state.get("geo_main", "France"),
            "Gender": st.session_state.get("gen_main", "Female"), "Age": st.session_state.get("age_main", 40),
            "Tenure": st.session_state.get("ten_main", 5), "Balance": base_bal, "NumOfProducts": base_nop,
            "HasCrCard": st.session_state.get("hcc_main", 1), "IsActiveMember": base_iam, "EstimatedSalary": st.session_state.get("sal_main", 100000.0)
        }
        sim_data = base_data.copy()
        sim_data.update({"IsActiveMember": sim_iam, "NumOfProducts": sim_nop, "Balance": sim_bal})

        if st.button("⚡ Calculate Delta", use_container_width=True):
            r_base = score_customer(base_data, active_threshold, risk_band_mode)
            r_sim  = score_customer(sim_data, active_threshold, risk_band_mode)
            
            delta = (r_sim['prob'] - r_base['prob']) * 100
            st.metric("New Probability", f"{r_sim['prob']*100:.1f}%", delta=f"{delta:+.1f}%", delta_color="inverse")
            st.write(f"Risk Band: **{r_base['band']}** → **{r_sim['band']}**")

with tab3:
    st.markdown("### 🧠 Model Explainability")
    if "last_input" not in st.session_state:
        st.info("Run a prediction in Tab 1 first to see explanations.")
    else:
        try:
            import shap
            
            # Extract models
            preprocessor = pipeline.named_steps["preprocessor"]
            classifier = pipeline.named_steps["classifier"]
            
            # Prepare row
            row_df = pd.DataFrame([st.session_state["last_input"]])
            if hasattr(pipeline, "feature_names_in_"):
                row_df = row_df[list(pipeline.feature_names_in_)]
            
            X_tx = preprocessor.transform(row_df)
            feat_names = preprocessor.get_feature_names_out()
            
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_tx)
            
            # Handle binary output shape
            sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
            
            fig, ax = plt.subplots(figsize=(8, 5))
            idx = np.argsort(np.abs(sv))
            colors = ['#ef4444' if v > 0 else '#10b981' for v in sv[idx]]
            ax.barh(feat_names[idx], sv[idx], color=colors)
            ax.set_title("Feature Contribution to Churn Risk (SHAP)")
            ax.set_xlabel("Impact (SHAP Value)")
            
            # Dark theme styling
            fig.patch.set_facecolor('#1e293b')
            ax.set_facecolor('#1e293b')
            ax.xaxis.label.set_color('#f1f5f9'); ax.yaxis.label.set_color('#f1f5f9')
            ax.title.set_color('#f1f5f9'); ax.tick_params(colors='#94a3b8')
            for s in ax.spines.values(): s.set_color('#334155')
            
            st.pyplot(fig)
            st.caption("Red bars increase churn risk, green bars decrease it.")
            
        except ImportError:
            st.warning("⚠️ SHAP library not found. Install it via `pip install shap` to see feature explanations.")
        except Exception as e:
            st.info(f"Local explanations via SHAP are currently optimized for tree-based models. ({e})")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='text-align:center; color:#475569;'>Bank Churn Scorer v2.0.0 · Built with Scikit-Learn & Streamlit</p>", unsafe_allow_html=True)
