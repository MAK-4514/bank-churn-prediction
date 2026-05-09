"""
streamlit_app.py
----------------
Bank Customer Churn Risk Scoring – Interactive Streamlit Application.

Features
--------
• Loads the trained GBC pipeline from models/model.pkl
• Interactive input panel for all 10 customer features
• Displays churn probability, risk band, and churn decision
• Gauge / progress bar visualisation of risk
• What-If Simulator: adjust 3 key levers and see probability change instantly
• Comparison table between original and what-if scenario

Run
---
streamlit run app/streamlit_app.py
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
APP_DIR     = Path(__file__).resolve().parent       # app/
PROJECT_ROOT = APP_DIR.parent                        # d:/project/
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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
/* ── Root palette ───────────────────────────────────── */
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

/* ── Global overrides ───────────────────────────────── */
.stApp {
  background: var(--bg-primary) !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: var(--bg-secondary) !important;
  border-right: 1px solid var(--border);
}

/* Metric cards */
div[data-testid="metric-container"] {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 20px;
}

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
  color: white !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-size: 1rem !important;
  padding: 0.6rem 2.2rem !important;
  transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Dividers */
hr { border-color: var(--border) !important; }

/* Risk band badges */
.badge {
  display: inline-block;
  padding: 6px 18px;
  border-radius: 50px;
  font-weight: 700;
  font-size: 1.1rem;
  letter-spacing: 0.05em;
}
.badge-low    { background: #064e3b; color: #34d399; border: 1px solid #34d399; }
.badge-medium { background: #78350f; color: #fbbf24; border: 1px solid #fbbf24; }
.badge-high   { background: #7f1d1d; color: #f87171; border: 1px solid #f87171; }

/* Section header */
.section-header {
  font-size: 1.15rem;
  font-weight: 700;
  color: var(--accent-blue);
  letter-spacing: 0.04em;
  margin-bottom: 6px;
  border-bottom: 2px solid var(--accent-blue);
  padding-bottom: 4px;
}

/* Probability bar container */
.prob-bar-outer {
  background: #334155;
  border-radius: 8px;
  height: 22px;
  width: 100%;
  overflow: hidden;
  margin-top: 8px;
}
.prob-bar-inner {
  height: 100%;
  border-radius: 8px;
  transition: width 0.5s ease;
}

/* Info callout box */
.info-box {
  background: #1e293b;
  border-left: 4px solid var(--accent-blue);
  border-radius: 6px;
  padding: 12px 16px;
  margin: 10px 0;
  font-size: 0.9rem;
  color: var(--text-muted);
}
</style>
""",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Model loading (cached so it only loads once)  ✅ SAFE VERSION
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model…")
def load_model():
    model_path = PROJECT_ROOT / "models" / "model.pkl"
    threshold_path = PROJECT_ROOT / "models" / "best_threshold.json"

    if not model_path.exists():
        return None, None, None

    pipeline = joblib.load(model_path)
    
    if not hasattr(pipeline, "predict_proba"):
        return "NO_PROBA", None, None

    best_f1 = 0.50
    best_recall = 0.50

    if threshold_path.exists():
        try:
            with open(threshold_path) as f:
                cfg = json.load(f)

            if cfg.get("best_threshold_f1") is not None:
                best_f1 = float(cfg["best_threshold_f1"])
            if cfg.get("best_threshold_recall") is not None:
                best_recall = float(cfg["best_threshold_recall"])
        except Exception:
            pass

    return pipeline, best_f1, best_recall


pipeline, BEST_THRESH_F1, BEST_THRESH_RECALL = load_model()

# ──────────────────────────────────────────────────────────────────────────────
# Scoring helper
# ──────────────────────────────────────────────────────────────────────────────

def score_customer(
    credit_score, geography, gender, age, tenure,
    balance, num_products, has_cr_card, is_active, salary,
    threshold: float = None,
    risk_band_mode: str = "Fixed (0.40 / 0.70)",
) -> dict:
    """Build feature dict, run through pipeline, return result dict."""
    if pipeline is None or pipeline == "NO_PROBA" or not hasattr(pipeline, "predict_proba"):
        return {"prob": 0.0, "band": "Unknown", "flag": 0}

    if threshold is None:
        threshold = BEST_THRESH_F1

    row = pd.DataFrame([{
        "CreditScore":      credit_score,
        "Geography":        geography,
        "Gender":           gender,
        "Age":              age,
        "Tenure":           tenure,
        "Balance":          balance,
        "NumOfProducts":    num_products,
        "HasCrCard":        int(has_cr_card),
        "IsActiveMember":   int(is_active),
        "EstimatedSalary":  salary,
    }])

    prob  = float(pipeline.predict_proba(row)[0, 1])
    
    if risk_band_mode == "Optimized (Uses Threshold)":
        band = str(assign_risk_band_optimized(np.array([prob]), threshold)[0])
    else:
        band = str(assign_risk_band(np.array([prob]))[0])
        
    flag  = int(prob >= threshold)

    return {"prob": prob, "band": band, "flag": flag}


def render_risk_badge(band: str) -> str:
    css = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high"}
    cls = css.get(band, "badge-medium")
    return f'<span class="badge {cls}">{band.upper()} RISK</span>'


def render_prob_bar(prob: float) -> str:
    pct = max(0.0, min(100.0, prob * 100.0))
    if prob < 0.40:
        colour = "#10b981"
    elif prob < 0.70:
        colour = "#f59e0b"
    else:
        colour = "#ef4444"

    return f"""
    <div class="prob-bar-outer">
      <div class="prob-bar-inner" style="width:{pct:.1f}%; background:{colour};"></div>
    </div>
    """


# ──────────────────────────────────────────────────────────────────────────────
# App layout
# ──────────────────────────────────────────────────────────────────────────────

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div style='text-align:center; padding: 24px 0 8px 0;'>
      <h1 style='font-size:2.4rem; font-weight:800;
                 background:linear-gradient(135deg,#3b82f6,#8b5cf6);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        🏦 Bank Customer Churn Risk Scorer
      </h1>
      <p style='color:#94a3b8; font-size:1rem; margin-top:-6px;'>
        v{APP_VERSION} &nbsp;·&nbsp; Powered by Gradient Boosting &nbsp;·&nbsp; Predictive ML Pipeline
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Model not loaded guard
if pipeline is None:
    st.error(
        "⚠️  Model not found at **models/model.pkl**. "
        "Please run `python src/train.py` first.",
        icon="🚫",
    )
    st.stop()
elif pipeline == "NO_PROBA":
    st.error(
        "⚠️  Loaded model does not support probability predictions (`predict_proba`). "
        "Please ensure the pipeline includes a supported classifier.",
        icon="🚫",
    )
    st.stop()

# ── Sidebar – threshold picker ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    threshold_mode = st.radio(
        "Decision Threshold",
        ["F1-Optimized", "Recall-Priority (≥0.80)", "Custom"],
        help="Choose which threshold to use for the churn flag.",
    )
    if threshold_mode == "F1-Optimized":
        active_threshold = BEST_THRESH_F1
    elif threshold_mode == "Recall-Priority (≥0.80)":
        active_threshold = BEST_THRESH_RECALL
    else:
        active_threshold = st.slider(
            "Custom Threshold", 0.10, 0.90, float(BEST_THRESH_F1), 0.01
        )

    st.markdown(
        f'<div class="info-box">Active threshold: <b>{active_threshold:.3f}</b></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("### 📊 Threshold Info")
    st.markdown(
        f"- **F1-Optimal**: `{BEST_THRESH_F1:.3f}`\n"
        f"- **Recall ≥ 0.80**: `{BEST_THRESH_RECALL:.3f}`"
    )
    st.markdown("---")
    st.markdown("### 🚦 Risk Banding")
    risk_band_mode = st.radio(
        "Risk Band Mode",
        ["Fixed (0.40 / 0.70)", "Optimized (Uses Threshold)"],
        help="Fixed uses static 0.40 and 0.70 cutoffs. Optimized aligns the 'High' band precisely with your chosen threshold.",
    )
    if risk_band_mode == "Fixed (0.40 / 0.70)":
        st.markdown(
            "| Band | Probability |\n"
            "|------|-------------|\n"
            "| 🟢 Low    | < 0.40 |\n"
            "| 🟡 Medium | 0.40 – 0.70 |\n"
            "| 🔴 High   | ≥ 0.70 |"
        )
    else:
        st.markdown(
            "| Band | Probability |\n"
            "|------|-------------|\n"
            f"| 🟢 Low    | < {active_threshold/2:.2f} |\n"
            f"| 🟡 Medium | {active_threshold/2:.2f} – {active_threshold:.2f} |\n"
            f"| 🔴 High   | ≥ {active_threshold:.2f} |"
        )

# ── Main panel ───────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍  Customer Prediction", "🔄  What-If Simulator", "🧠  Model Explainability"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 – Customer Prediction
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<p class="section-header">Enter Customer Details</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📋 Account Information**")
        credit_score = st.number_input(
            "Credit Score", min_value=300, max_value=850, value=650, step=1,
            help="Customer credit score (300–850)",
            key="cs_main",
        )
        geography = st.selectbox(
            "Geography", ["France", "Spain", "Germany"], key="geo_main"
        )
        gender = st.selectbox(
            "Gender", ["Female", "Male"], key="gender_main"
        )

    with col2:
        st.markdown("**📅 Customer Profile**")
        age = st.number_input(
            "Age", min_value=18, max_value=100, value=40, step=1, key="age_main"
        )
        tenure = st.number_input(
            "Tenure (years)", min_value=0, max_value=20, value=5, step=1, key="tenure_main"
        )
        num_products = st.number_input(
            "Number of Products", min_value=1, max_value=4, value=1, step=1, key="nop_main"
        )

    with col3:
        st.markdown("**💰 Financial Details**")
        balance = st.number_input(
            "Account Balance (€)", min_value=0.0, max_value=300_000.0,
            value=75_000.0, step=500.0, format="%.2f", key="bal_main"
        )
        salary = st.number_input(
            "Estimated Salary (€)", min_value=0.0, max_value=300_000.0,
            value=100_000.0, step=500.0, format="%.2f", key="sal_main"
        )
        has_cr_card  = st.selectbox(
            "Has Credit Card", [0, 1],
            format_func=lambda x: "✅ Yes" if x else "❌ No", key="hcc_main"
        )
        is_active    = st.selectbox(
            "Is Active Member", [0, 1],
            format_func=lambda x: "✅ Yes" if x else "❌ No", key="iam_main"
        )

    st.markdown("---")
    
    # Input validation / sanity hint
    if balance > 150_000 and salary < 30_000:
        st.warning("⚠️ **Data Sanity Hint:** The account balance is unusually high compared to the estimated salary. Please verify the inputs.")
        
    predict_btn = st.button("🚀  Predict Churn Risk", use_container_width=True)

    if predict_btn:
        result = score_customer(
            credit_score, geography, gender, age, tenure,
            balance, num_products, has_cr_card, is_active, salary,
            threshold=active_threshold,
            risk_band_mode=risk_band_mode,
        )
        prob = result["prob"]
        band = result["band"]
        flag = result["flag"]

        st.markdown("---")
        st.markdown("### 📈 Prediction Results")

        r1, r2, r3 = st.columns(3)
        r1.metric("Churn Probability", f"{prob*100:.1f}%")
        r2.metric("Churn Decision",    "⚠️ CHURN" if flag else "✅ RETAIN")
        r3.metric("Threshold Used",    f"{active_threshold:.3f}")

        # Risk badge
        st.markdown(
            f"**Risk Classification:** {render_risk_badge(band)}",
            unsafe_allow_html=True,
        )

        # Probability bar
        st.markdown(
            f"**Churn Probability Bar**{render_prob_bar(prob)}",
            unsafe_allow_html=True,
        )

        # Narrative explanation
        st.markdown("---")
        st.markdown("#### 🔍 Interpretation")
        if band == "Low":
            narrative = (
                f"This customer has a **{prob*100:.1f}%** churn probability — "
                "classified as **Low Risk**. Standard engagement is appropriate. "
                "No immediate retention action required."
            )
        elif band == "Medium":
            narrative = (
                f"This customer has a **{prob*100:.1f}%** churn probability — "
                "classified as **Medium Risk**. Consider proactive outreach: "
                "personalised offers, loyalty programme enrolment, or a check-in call."
            )
        else:
            narrative = (
                f"This customer has a **{prob*100:.1f}%** churn probability — "
                "classified as **High Risk** ⚠️. Immediate intervention recommended: "
                "dedicated retention specialist contact, premium product upgrade offer, "
                "or fee waiver for the next billing period."
            )
        st.info(narrative)

        # Export result
        st.markdown("---")
        export_data = {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active,
            "EstimatedSalary": salary,
            "ChurnProbability": round(prob, 4),
            "RiskBand": band,
            "ChurnFlag": bool(flag)
        }
        st.download_button(
            label="📥 Download Executive Summary (JSON)",
            data=json.dumps(export_data, indent=2),
            file_name="churn_prediction_summary.json",
            mime="application/json",
            use_container_width=True
        )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 – What-If Simulator
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🔄 What-If Simulator")
    st.markdown(
        '<div class="info-box">'
        "Adjust key levers to instantly see how interventions affect churn probability. "
        "The baseline is automatically synced with your inputs from the Prediction tab."
        "</div>",
        unsafe_allow_html=True,
    )

    wif_col1, wif_col2 = st.columns([1, 1])

    with wif_col1:
        st.markdown("**📋 Baseline Profile (from Prediction tab)**")
        # Display baseline values for context
        st.markdown(f"- **Geography:** {geography}")
        st.markdown(f"- **Age:** {age}")
        st.markdown(f"- **Credit Score:** {credit_score}")
        st.markdown(f"- **Salary:** €{salary:,.2f}")
        st.markdown("---")
        st.markdown("**Current State of Levers:**")
        st.markdown(f"- **Active Member:** {'✅ Yes' if is_active else '❌ No'}")
        st.markdown(f"- **Number of Products:** {num_products}")
        st.markdown(f"- **Account Balance:** €{balance:,.2f}")

    with wif_col2:
        st.markdown("**🎚️ Simulate Interventions**")
        st.markdown(
            "_Adjust the 3 most actionable levers below and compare the outcome:_"
        )

        wif_active_sim = st.selectbox(
            "Is Active Member (after intervention)", [0, 1],
            format_func=lambda x: "✅ Yes (Active)" if x else "❌ No (Inactive)",
            index=is_active,       
            key="iam_wif_sim",
            help="Activating a member is the single biggest lever in most churn models.",
        )
        wif_nop_sim = st.number_input(
            "Num of Products (after cross-sell)", 1, 4, 
            value=num_products,
            key="nop_wif_sim",
            help="Customers with 2 products churn far less.",
        )
        wif_balance_sim = st.number_input(
            "Balance (€) after deposit / offer", 0.0, 300_000.0,
            value=balance,
            step=500.0, format="%.2f",
            key="bal_wif_sim",
        )

        st.markdown("---")
        if st.button("⚡  Run Simulation", use_container_width=True):
            # Score baseline using Prediction tab values
            base_res = score_customer(
                credit_score, geography, gender, age, tenure,
                balance, num_products, has_cr_card, is_active, salary,
                threshold=active_threshold,
                risk_band_mode=risk_band_mode,
            )
            # Score simulated using adjusted values
            sim_res = score_customer(
                credit_score, geography, gender, age, tenure,
                wif_balance_sim, wif_nop_sim, has_cr_card, wif_active_sim, salary,
                threshold=active_threshold,
                risk_band_mode=risk_band_mode,
            )

            delta_prob = sim_res["prob"] - base_res["prob"]
            delta_pct  = delta_prob * 100

            st.markdown("#### 📊 Simulation Results")

            c1, c2, c3 = st.columns(3)
            c1.metric("Baseline Probability", f"{base_res['prob']*100:.1f}%")
            c2.metric(
                "Simulated Probability",
                f"{sim_res['prob']*100:.1f}%",
                delta=f"{delta_pct:+.1f}%",
                delta_color="inverse",
            )
            c3.metric(
                "Risk Band Change",
                f"{base_res['band']} → {sim_res['band']}",
            )

            # Comparison table
            comparison = pd.DataFrame({
                "Scenario":            ["Baseline", "Simulated"],
                "IsActiveMember":      [is_active, wif_active_sim],
                "NumOfProducts":       [num_products, wif_nop_sim],
                "Balance (€)":         [balance, wif_balance_sim],
                "Churn Probability":   [
                    f"{base_res['prob']*100:.1f}%",
                    f"{sim_res['prob']*100:.1f}%",
                ],
                "Risk Band":           [base_res["band"], sim_res["band"]],
                "Churn Flag":          [
                    "Churn" if base_res["flag"] else "Retain",
                    "Churn" if sim_res["flag"]  else "Retain",
                ],
            })
            st.table(comparison.set_index("Scenario"))

            # Narrative
            if delta_pct < -5:
                st.success(
                    f"✅ The simulated interventions reduce churn probability by "
                    f"**{abs(delta_pct):.1f} percentage points** — "
                    "a meaningful improvement in retention likelihood."
                )
            elif delta_pct > 5:
                st.warning(
                    f"⚠️ The simulated changes **increase** churn probability by "
                    f"**{delta_pct:.1f}pp**. Review the adjustments."
                )
            else:
                st.info(
                    "ℹ️ The interventions have a modest effect (<5pp change). "
                    "Consider stronger or combined actions."
                )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 – Model Explainability
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🧠 Model Explainability")
    st.markdown(
        '<div class="info-box">'
        "Understand which factors drive the model's predictions globally, and locally for the current customer."
        "</div>",
        unsafe_allow_html=True,
    )
    
    st.markdown("#### 🌍 Global Feature Importance")
    fi_path = PROJECT_ROOT / "outputs" / "feature_importance.png"
    if fi_path.exists():
        st.image(str(fi_path), caption="Global Feature Importance (Gradient Boosting)", use_container_width=True)
    else:
        st.info("Feature importance plot not found. Ensure models have been trained and evaluated.")

    st.markdown("---")
    st.markdown("#### 👤 Local Explanation (Current Customer)")
    
    if pipeline is not None and pipeline != "NO_PROBA":
        try:
            import shap
            
            # The current state of the widgets from Tab 1
            current_row = pd.DataFrame([{
                "CreditScore":      credit_score,
                "Geography":        geography,
                "Gender":           gender,
                "Age":              age,
                "Tenure":           tenure,
                "Balance":          balance,
                "NumOfProducts":    num_products,
                "HasCrCard":        int(has_cr_card),
                "IsActiveMember":   int(is_active),
                "EstimatedSalary":  salary,
            }])
            
            preprocessor = pipeline.named_steps["preprocessor"]
            classifier = pipeline.named_steps["classifier"]
            X_transformed = preprocessor.transform(current_row)
            
            tree_models = ("GradientBoostingClassifier", "RandomForestClassifier", "DecisionTreeClassifier", "XGBClassifier", "LGBMClassifier")
            model_type = type(classifier).__name__
            
            if model_type in tree_models:
                # TreeExplainer is fast for tree-based models
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(X_transformed)
                
                # Handle binary classification lists
                if isinstance(shap_values, list):
                    sv = shap_values[1][0]
                else:
                    sv = shap_values[0]
                    
                # Get feature names
                if hasattr(preprocessor, "get_feature_names_out"):
                    feature_names = preprocessor.get_feature_names_out()
                else:
                    feature_names = [f"Feature {i}" for i in range(X_transformed.shape[1])]
                
                # Build a robust dark-themed horizontal bar plot
                fig, ax = plt.subplots(figsize=(8, 5))
                idx = np.argsort(np.abs(sv))
                colors = ['#ef4444' if val > 0 else '#10b981' for val in sv[idx]]
                
                ax.barh(np.array(feature_names)[idx], sv[idx], color=colors)
                ax.set_xlabel("SHAP Value (Impact on Churn Probability Log-Odds)")
                ax.set_title(f"Why did this customer get this score? ({model_type})")
                
                # Apply dark theme formatting
                fig.patch.set_facecolor('#1e293b')
                ax.set_facecolor('#1e293b')
                ax.xaxis.label.set_color('#f1f5f9')
                ax.yaxis.label.set_color('#f1f5f9')
                ax.title.set_color('#f1f5f9')
                ax.tick_params(colors='#94a3b8')
                for spine in ax.spines.values():
                    spine.set_color('#334155')
                    
                st.pyplot(fig)
                plt.close(fig)
                
                st.markdown(
                    "<small><i>Red bars increase churn risk, green bars decrease churn risk. "
                    "Longer bars have a stronger impact.</i></small>",
                    unsafe_allow_html=True
                )
            else:
                st.info(f"Local explanations via SHAP are currently optimized for tree-based models. "
                        f"Current model: `{model_type}`.")
                
        except ImportError:
            st.warning("⚠️ `shap` library not found. Please add `shap` to your requirements to see local explanations.")
        except Exception as e:
            st.warning(f"⚠️ Could not generate local SHAP explanation: {e}")
    else:
        st.warning("Model not loaded properly.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#475569;font-size:0.8rem;'>"
    "Bank Customer Churn Risk Scorer · Gradient Boosting Pipeline · "
    "Built with scikit-learn & Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
