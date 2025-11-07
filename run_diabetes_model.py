#!/usr/bin/env python3
"""
Day 11 — Diabetes Prediction Model (robust)
Builds and evaluates ML models on the Pima Indians Diabetes dataset.
Saves visuals and models into ./outputs.
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
)

# ---------- Config ----------
DATA_PATH = "diabetes.csv"
OUT_DIR = "outputs"
RANDOM_STATE = 42

# ---------- Load ----------
print("📥 Loading dataset...")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at '{DATA_PATH}'. Put diabetes.csv in this folder.")

df = pd.read_csv(DATA_PATH)
print("✅ Dataset loaded.")
print(f"Shape: {df.shape}")
print(df.head(), "\n")

# ---------- Prepare features & target ----------
# find target column (many variants exist)
if "Outcome" in df.columns:
    target_col = "Outcome"
elif "class" in df.columns:
    target_col = "class"
elif "target" in df.columns:
    target_col = "target"
else:
    # fallback: assume last column is target
    target_col = df.columns[-1]

print(f"Using target column: '{target_col}'")

X = df.drop(columns=[target_col])
y = df[target_col].copy()

# If y is non-numeric (strings), map to numeric labels (0/1, etc.)
if not np.issubdtype(y.dtype, np.number):
    uniques = list(pd.Series(y).unique())
    # common mapping for Pima CSVs
    if set(["tested_positive", "tested_negative"]).issubset(set(uniques)):
        mapping = {"tested_negative": 0, "tested_positive": 1}
    elif set(["positive", "negative"]).issubset(set(uniques)):
        mapping = {"negative": 0, "positive": 1}
    else:
        # stable deterministic mapping: sort then enumerate
        uniques_sorted = sorted(uniques)
        mapping = {label: idx for idx, label in enumerate(uniques_sorted)}
    print("Label mapping:", mapping)
    y = y.map(mapping)

# Check final classes
class_values = sorted(pd.Series(y).unique())
print("Final numeric labels:", class_values)
if len(class_values) > 2:
    print("Warning: more than 2 classes found. Metrics will run but interpret results accordingly.\n")

# ---------- Missing values handling ----------
print("\n🔹 Preprocessing data...")
# replace zeros in columns where zero is invalid (common Pima handling)
# typical Pima columns where 0 is invalid: plas (glucose), pres (blood pressure), skin, insu, mass (BMI)
cols_zero_invalid = []
for col in ["plas", "pres", "skin", "insu", "mass", "glucose", "BloodPressure", "BMI"]:
    if col in X.columns:
        cols_zero_invalid.append(col)

# replace 0 with NaN for those columns to impute median
if cols_zero_invalid:
    X[cols_zero_invalid] = X[cols_zero_invalid].replace(0, np.nan)

X = X.fillna(X.median())

# Train/test split (stratify if binary)
stratify_arg = y if len(class_values) == 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_arg
)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Preprocessing complete.\n")

# ---------- Models ----------
print("🧠 Training models...")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

results = {}
os.makedirs(OUT_DIR, exist_ok=True)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # try predict_proba for ROC-AUC if available, else fallback to decision_function or 0/1 proba
    try:
        y_proba = model.predict_proba(X_test_scaled)[:, 1] if model.predict_proba(X_test_scaled).shape[1] > 1 else model.predict_proba(X_test_scaled)[:, 0]
    except Exception:
        try:
            y_proba = model.decision_function(X_test_scaled)
        except Exception:
            y_proba = None

    # metrics: use zero_division=0 to avoid exceptions if a class isn't predicted
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba) if (y_proba is not None and len(class_values) == 2) else None,
    }
    results[name] = metrics

    print(f"\n🔸 {name} Results:")
    for k, v in metrics.items():
        if v is None:
            print(f"   {k}: n/a")
        else:
            print(f"   {k}: {v:.4f}")

# ---------- Visualizations ----------
print("\n📊 Generating visualizations...")
# ROC (use Random Forest if possible)
rf = models.get("Random Forest")
if rf is not None:
    try:
        RocCurveDisplay.from_estimator(rf, X_test_scaled, y_test)
        plt.title("ROC Curve - Random Forest")
        plt.savefig(os.path.join(OUT_DIR, "roc_curve.png"))
        plt.close()
    except Exception:
        pass

# Feature importance (Random Forest)
if rf is not None:
    try:
        importances = rf.feature_importances_
        features = X.columns
        importance_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)
        plt.figure(figsize=(8, 5))
        sns.barplot(data=importance_df, x="Importance", y="Feature")
        plt.title("Feature Importance - Random Forest")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"))
        plt.close()
    except Exception:
        pass

# Confusion matrix (Random Forest)
try:
    cm = confusion_matrix(y_test, rf.predict(X_test_scaled))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Random Forest")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"))
    plt.close()
except Exception:
    pass

# ---------- Save models & scaler ----------
print("💾 Saving models and scaler...")
joblib.dump(models["Logistic Regression"], os.path.join(OUT_DIR, "logistic.joblib"))
joblib.dump(models["Random Forest"], os.path.join(OUT_DIR, "random_forest.joblib"))
joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))

print("\n✅ All done. Outputs saved to:", OUT_DIR)