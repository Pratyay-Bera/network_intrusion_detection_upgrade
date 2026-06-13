import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Improve CPU parallel usage
os.environ["OMP_NUM_THREADS"] = "8"

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from data_preprocessing import *
from feature_engineering import *
from evaluate import *

# -----------------------
# Create folders
# -----------------------
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -----------------------
# Load Data
# -----------------------
train_df, test_df = load_data(
    "data/UNSW_NB15_training-set_updated.csv",
    "data/UNSW_NB15_testing-set_updated.csv"
)

# Separate features and labels
# Fix Step 2: Drop id (row index) and stcpb/dtcpb (TCP seq numbers — not flow features)
COLS_TO_DROP = [col for col in ["id", "stcpb", "dtcpb"] if col in train_df.columns]

X_train_raw = train_df.drop(["label"] + COLS_TO_DROP, axis=1)
y_train = train_df["label"]

X_test_raw = test_df.drop(["label"] + COLS_TO_DROP, axis=1)
y_test = test_df["label"]

# -----------------------
# Class distribution BEFORE SMOTE
# -----------------------
plot_class_distribution(y_train, "outputs/before_smote.png")

# -----------------------
# Correlation Heatmap
# -----------------------
correlation_heatmap(train_df)

# -----------------------
# Preprocessing
# -----------------------
X_train, X_test, preprocessor = preprocess_features(
    X_train_raw,
    X_test_raw
)

# -----------------------
# SMOTE
# -----------------------
# Save original y_train (pre-SMOTE) for honest feature importance
y_train_orig = y_train.copy()

X_train, y_train = apply_smote(X_train, y_train)

plot_class_distribution(y_train, "outputs/after_smote.png")

# -----------------------
# Scaling
# -----------------------
X_train, X_test, scaler = scale_features(X_train, X_test)

# -----------------------
# Feature Importance
# Fix Step 6: Compute on original (pre-SMOTE) scaled data for honest ranking
# -----------------------
X_train_orig_processed, _, _ = preprocess_features(X_train_raw, X_test_raw)
X_train_orig_scaled, _, _ = scale_features(X_train_orig_processed, X_train_orig_processed)

importance = feature_importance(X_train_orig_scaled, y_train_orig)

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Convert to numpy array
feature_names = np.array(feature_names)

# Get top 20 most important features
indices = np.argsort(importance)[-20:]

plt.figure(figsize=(12,6))
plt.barh(feature_names[indices], importance[indices])

plt.xlabel("Importance (%)")
plt.title("Top 20 Most Important Features")

plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()

# -----------------------
# PCA
# -----------------------
X_train_pca, X_test_pca, pca = apply_pca(
    X_train,
    X_test,
    n_components=20
)

# -----------------------
# Models
# -----------------------
models = {

    "LightGBM": (
        LGBMClassifier(n_jobs=-1, random_state=42, verbose=-1),
        {
            "n_estimators":[100,200],
            "learning_rate":[0.05,0.1]
        }
    ),

    "XGBoost": (
        XGBClassifier(n_jobs=-1, random_state=42, eval_metric="logloss"),
        {
            "n_estimators":[100,200],
            "learning_rate":[0.05,0.1]
        }
    ),

    "KNN": (
        KNeighborsClassifier(),
        {"n_neighbors":[3,5]}
    ),

    "DecisionTree": (
        DecisionTreeClassifier(),
        {"max_depth":[10,20]}
    ),

    "RandomForest": (
        RandomForestClassifier(n_jobs=-1),
        {
            "n_estimators":[100,200],
            "max_depth":[5,10]
        }
    ),

    "Logistic": (
        LogisticRegression(max_iter=2000, n_jobs=-1),
        {"C":[0.1,1]}
    ),

    "NaiveBayes": (
        GaussianNB(),
        {
            "var_smoothing":[1e-12,1e-10]
        }
    ),

    "SVM_Linear": (
        CalibratedClassifierCV(
            LinearSVC(max_iter=2000, random_state=42),
            cv=3
        ),
        {"estimator__C": [0.01, 0.1, 1]}
    )

}

results = []

best_f1 = 0
best_recall = 0
best_accuracy = 0
best_model = None
best_model_name = ""

# -----------------------
# Train Models
# -----------------------
for name,(model,param_grid) in models.items():

    print(f"\n{name} Model")

    # 10 Fold Cross Validation
    cv_scores = cross_val_score(
        model,
        X_train_pca,
        y_train,
        cv=10,
        n_jobs=-1
    )

    # GridSearch
    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train_pca, y_train)

    tuned_model = grid.best_estimator_

    # Prediction
    y_pred = tuned_model.predict(X_test_pca)

    # ROC-AUC: use predict_proba when available, decision_function otherwise
    y_prob = None
    if hasattr(tuned_model, "predict_proba"):
        y_prob = tuned_model.predict_proba(X_test_pca)[:, 1]
    elif hasattr(tuned_model, "decision_function"):
        y_prob = tuned_model.decision_function(X_test_pca)

    # Fix Step 9: use enhanced evaluate_model returning macro F1, attack recall, ROC-AUC, CM, report
    acc, prec, rec, f1, f1_macro, recall_attack, roc_auc, cm, report = evaluate_model(
        y_test, y_pred, y_prob
    )

    print("10 Fold CV Accuracy:", cv_scores.mean())
    print("Best Parameters:", grid.best_params_)
    print("Accuracy:", acc)
    print("Precision (weighted):", prec)
    print("Recall (weighted):", rec)
    print("F1 (weighted):", f1)
    print("F1 (macro):", f1_macro)
    print("Recall — Attack class:", recall_attack)
    print("ROC-AUC:", roc_auc)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)

    results.append([name, cv_scores.mean(), acc, prec, rec, f1, f1_macro, recall_attack, roc_auc])

    # Fix Step 9: best model selection weights attack recall most heavily (missed attacks > false alarms)
    # Primary: macro F1  Secondary: attack-class recall  Tertiary: accuracy
    if (
        f1_macro > best_f1 or
        (f1_macro == best_f1 and recall_attack > best_recall) or
        (f1_macro == best_f1 and recall_attack == best_recall and acc > best_accuracy)
    ):
        best_f1 = f1_macro
        best_recall = recall_attack
        best_accuracy = acc
        best_model = tuned_model
        best_model_name = name


# -----------------------
# Save comparison
# -----------------------
results_df = pd.DataFrame(
    results,
    columns=["Model", "CV Accuracy", "Accuracy", "Precision", "Recall",
             "F1 (weighted)", "F1 (macro)", "Recall Attack", "ROC-AUC"]
)

results_df.to_csv("data/model_comparison.csv",index=False)

# -----------------------
# Model Comparison Bar Graph
# -----------------------
plt.figure(figsize=(12,6))

x = range(len(results_df["Model"]))

plt.bar(x, results_df["Accuracy"], width=0.2, label="Accuracy")
plt.bar([i + 0.2 for i in x], results_df["Precision"], width=0.2, label="Precision")
plt.bar([i + 0.4 for i in x], results_df["Recall"], width=0.2, label="Recall")
plt.bar([i + 0.6 for i in x], results_df["F1 (weighted)"], width=0.2, label="F1 (weighted)")

plt.xticks([i + 0.3 for i in x], results_df["Model"], rotation=45)
plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.legend()

plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.close()

# -----------------------
# Save Best Model using Joblib
# Fix Step 10: Also save preprocessor, scaler, pca so app.py can transform-only at inference
# -----------------------
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(preprocessor, "models/preprocessor.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(pca, "models/pca.pkl")
print(f"\nBest model selected: {best_model_name}")
print("Model saved successfully as models/best_model.pkl")