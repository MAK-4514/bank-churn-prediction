# 🏦 Predictive Modeling and Risk Scoring for Bank Customer Churn

> End-to-end ML pipeline using scikit-learn, pandas, and Streamlit.

---

## 📁 Project Structure

```
project/
├── data/
│   └── European_Bank.csv          ← Place your dataset here
├── src/
│   ├── utils.py                   ← Shared helpers, constants, risk-banding
│   ├── train.py                   ← Training pipeline (LR + GBC + GridSearch)
│   ├── evaluate.py                ← Standalone evaluation & plots
│   └── predict.py                 ← Batch / single-row inference
├── app/
│   └── streamlit_app.py           ← Interactive web application
├── models/                        ← Auto-created; stores .pkl and threshold config
├── outputs/                       ← Auto-created; metrics, plots, CSVs
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Create & activate a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Place your dataset
Copy `European_Bank.csv` into the `data/` folder:
```
data/European_Bank.csv
```

---

## 🚀 Running the Pipeline

### Step 1 – Train models
```bash
python src/train.py
# Optional: keep the Year column
python src/train.py --keep-year
# Optional: custom data path
python src/train.py --data path/to/your/file.csv
```

**What this does:**
- Loads and preprocesses the dataset  
- Trains a **Logistic Regression** baseline (class_weight="balanced")  
- Tunes a **Gradient Boosting Classifier** via GridSearchCV (3-fold stratified CV)  
- Finds the best decision threshold by **F1** and by **Recall ≥ 0.80**  
- Saves `models/model.pkl`, `models/lr_model.pkl`, `models/best_threshold.json`  
- Saves `outputs/metrics.json`, `outputs/feature_importance.csv`, and all plots  

### Step 2 – Evaluate (optional standalone)
```bash
python src/evaluate.py
```

Generates:
- Threshold comparison table (0.50 vs F1-optimal vs Recall-priority)
- `outputs/risk_scores.csv` — every test row with probability + risk band
- `outputs/precision_recall_curve.png`
- `outputs/f1_vs_threshold.png`

### Step 3 – Batch scoring new customers
```bash
python src/predict.py --input new_customers.csv --output outputs/predictions.csv
```

### Step 4 – Launch the Streamlit app
```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## 🖥️ Streamlit App Features

| Feature | Description |
|---------|-------------|
| **Customer Prediction** | Enter all 10 features → instant churn probability + risk band |
| **Risk badge** | Colour-coded Low / Medium / High label |
| **Probability bar** | Visual gauge from green → amber → red |
| **Threshold selector** | Switch between F1-optimized, Recall-priority, or custom |
| **What-If Simulator** | Adjust IsActiveMember, NumOfProducts, Balance → compare probabilities |
| **Narrative insight** | Auto-generated retention recommendation per risk band |

---

## 📊 Output Files

| File | Description |
|------|-------------|
| `models/model.pkl` | Fitted GBC pipeline (preprocessor + classifier) |
| `models/lr_model.pkl` | Fitted Logistic Regression pipeline |
| `models/best_threshold.json` | Optimised thresholds + best hyperparameters |
| `outputs/metrics.json` | All evaluation metrics for both models |
| `outputs/feature_importance.csv` | GBC feature importances ranked |
| `outputs/lr_coefficients.csv` | LR coefficients with feature names |
| `outputs/risk_scores.csv` | Test-set predictions with risk bands |
| `outputs/threshold_comparison.csv` | Metrics at each threshold |
| `outputs/roc_curve_*.png` | ROC curve plots |
| `outputs/confusion_matrix_*.png` | Confusion matrix plots |
| `outputs/feature_importance.png` | Feature importance chart |
| `outputs/precision_recall_curve.png` | Precision-Recall curve |
| `outputs/f1_vs_threshold.png` | F1 score across all thresholds |

---

## 🔬 Modelling Decisions

### Why drop Year?
The `Year` column records when a row was entered, not a behavioural signal. Including it risks introducing temporal data leakage and makes the model fragile to new years at inference time.

### Why class_weight="balanced" for Logistic Regression?
Churn datasets are typically imbalanced (~20% positive class). Balanced weights up-weight minority class samples during optimisation, improving recall without requiring explicit over/undersampling.

### Why optimise the threshold?
At 0.5 (the sklearn default), the model is often conservative about labelling customers as churned. Since the cost of missing a churning customer (false negative) is higher than the cost of a false positive, we find the threshold that maximises F1 and separately show the Recall ≥ 0.80 threshold.

### Risk bands
| Band | Probability | Recommended Action |
|------|-------------|-------------------|
| Low | < 0.40 | Standard engagement |
| Medium | 0.40 – 0.70 | Proactive outreach / personalised offers |
| High | ≥ 0.70 | Immediate retention specialist intervention |

---

## 🔁 Reproducibility

All stochastic steps use `random_state=42`:
- `train_test_split`
- `StratifiedKFold`
- `LogisticRegression`
- `GradientBoostingClassifier`
- `GridSearchCV`

---

## 📦 Dependencies

```
scikit-learn >= 1.3.0
pandas       >= 2.0.0
numpy        >= 1.24.0
matplotlib   >= 3.7.0
joblib       >= 1.3.0
streamlit    >= 1.32.0
```
