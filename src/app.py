import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import matplotlib.pyplot as plt

# (preprocess_features / scale_features not called at inference — artifacts loaded from disk)

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="🔐 Network Intrusion Detection Dashboard",
    layout="wide"
)

st.title("🔐 Network Intrusion Detection System")
st.caption("Model Used: XGBoost | Dataset: UNSW-NB15")

st.markdown("""
### 🔎 Overview

This dashboard detects malicious network traffic using a trained **Machine Learning model**.

The model analyzes network flow features and classifies traffic as:

- 🟢 **Normal Traffic**
- 🔴 **Attack Traffic**

📌 **How to use this dashboard**

1. Upload a **network traffic CSV file**
2. The model will classify each record
3. View prediction summaries and dataset insights
""")

# -----------------------------
# Sidebar Upload
# -----------------------------
st.sidebar.header("📤 Upload Network Traffic Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# -----------------------------
# Load Model & Pipeline Artifacts
# Fix Step 10: Load preprocessor, scaler, pca saved at train time.
# At inference we transform-only — never fit on live data.
# -----------------------------
model       = joblib.load("models/best_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")
scaler      = joblib.load("models/scaler.pkl")
pca         = joblib.load("models/pca.pkl")

# -----------------------------
# Prediction Section
# -----------------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df.head())

    # X = preprocess_features_for_prediction(df)
    # X = apply_pca_for_prediction(X)

   # If label exists, separate it
    if " label " in df.columns:
        X = df.drop(" label ", axis=1)
        y = df[" label "]
    else:
        X = df

    # Fix Step 10: Drop columns that must not be present at inference
    # (id = row index with zero predictive value; stcpb/dtcpb = TCP seq numbers)
    COLS_TO_DROP = [col for col in ["id", "stcpb", "dtcpb"] if col in X.columns]
    if COLS_TO_DROP:
        X = X.drop(COLS_TO_DROP, axis=1)

    try:
        # -----------------------------
        # Preprocessing — transform only (preprocessor fitted at train time)
        # -----------------------------
        X_processed = preprocessor.transform(X)

        # -----------------------------
        # Scaling — transform only (scaler fitted at train time)
        # -----------------------------
        X_scaled = scaler.transform(X_processed)

        # -----------------------------
        # PCA — transform only (pca fitted at train time)
        # -----------------------------
        X_pca = pca.transform(X_scaled)

        # -----------------------------
        # Prediction
        # -----------------------------
        predictions = model.predict(X_pca)

        results = df.copy()
        results["Prediction"] = predictions
        results["Prediction"] = results["Prediction"].map(
            {0: 0, 1: 1}
        )

        st.subheader("🧠 Prediction Results")
        st.dataframe(results)

        st.caption("* 0 = Normal, 1 = Attack")

        # Summary
        attack_count = (results["Prediction"] == 1).sum()
        normal_count = (results["Prediction"] == 0).sum()

        col1, col2 = st.columns(2)

        col1.metric("Normal Traffic", normal_count)
        col2.metric("Attack Traffic", attack_count)

        # -----------------------------
        # Prediction Pie Chart
        # -----------------------------
        st.subheader("Traffic Distribution")

        fig, ax = plt.subplots()

        ax.pie(
            [normal_count, attack_count],
            labels=["Normal", "Attack"],
            autopct="%1.1f%%",
            colors=["green", "red"]
        )

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction Error: {e}")


# -----------------------------
# Dataset Analysis
# -----------------------------
st.markdown("---")
st.header("📊 Dataset & Model Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Class Distribution Before SMOTE")
    st.image("outputs/before_smote.png")

with col2:
    st.subheader("Class Distribution After SMOTE")
    st.image("outputs/after_smote.png")

st.subheader("Correlation Heatmap")
st.image("outputs/correlation_heatmap.png")

st.subheader("Feature Importance")
st.image("outputs/feature_importance.png")

# -----------------------------
# Model Evaluation
# -----------------------------
st.subheader("📊 Model Performance Comparison")

st.markdown("""
The chart below compares different machine learning models used for
network intrusion detection.

Metrics used:

- **Accuracy** — overall correct classifications
- **Macro F1** — equal weight to both Normal and Attack classes
- **Attack Recall** — fraction of actual attacks correctly detected (minimises missed attacks)
- **ROC-AUC** — overall discrimination ability

📌 Higher values indicate better performance. XGBoost ranked best overall.
""")

comparison_img = Image.open("outputs/model_comparison.png")

st.image(
    comparison_img,
    caption="Model Comparison (F1 Score)",
    width="stretch"
)

# -----------------------------
# Best Model Selected
# -----------------------------
st.subheader("🏆 Best Model Selected")

st.markdown("""
Based on the evaluation of multiple machine learning models using **Macro F1 Score** and **Attack Recall**,  
the model with the best performance on the dataset is:

### 🧠 **XGBoost (Extreme Gradient Boosting)**

This model achieved the **highest Macro F1 Score**, providing the best balance between detecting attacks and avoiding false alarms.

| Metric | Score |
|---|---|
| Accuracy | 88.28% |
| Macro F1 | 87.99% |
| Attack Recall | 94.08% |
| ROC-AUC | 97.34% |

📌 **Why XGBoost works well for Network Intrusion Detection:**
- Handles **complex non-linear patterns** in network traffic
- Robust to **imbalanced and noisy data**
- Excellent **generalisation** — highest CV accuracy (94.15%) among all models
- Faster and more scalable than kernel-based methods like SVM

Therefore, the deployed model used in this dashboard is:

### 🚀 **XGBoost**
""")

st.success("✔ Active Detection Model: XGBoost")