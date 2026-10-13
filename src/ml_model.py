"""
src/ml_model.py
End-to-End Machine Learning Pipeline for Supply Chain Delivery Delay Prediction.
Trains, evaluates, and persists a Random Forest classification model with feature importances
and prescriptive operational risk mitigation strategies.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from src.utils import get_csv_path
except ImportError:
    from utils import get_csv_path


MODEL_FILE = Path(__file__).resolve().parent / "model.pkl"

CATEGORICAL_FEATURES = ["ProductCategory", "ShippingMode", "Region"]
NUMERICAL_FEATURES = ["Quantity", "Discount", "UnitPrice", "InventoryLevel", "PromisedLeadDays"]
FEATURE_COLS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

def train_and_save_model(data_path: Path = None, save_path: Path = None) -> Dict[str, Any]:
    """Trains the delivery delay prediction model on historical orders and saves pipeline artifact."""
    if data_path is None:
        try:
            from src.utils import ensure_dataset_ready
            csv_path, _ = ensure_dataset_ready()
            data_path = csv_path
        except Exception:
            data_path = get_csv_path()
    if save_path is None:
        save_path = MODEL_FILE

    print(f"[*] Loading training dataset from {data_path}...", flush=True)
    df = pd.read_csv(data_path)

    # Filter out cancelled orders for binary classification (Delayed vs On-Time)
    df_clean = df[df["DeliveryStatus"].isin(["Delayed", "On-Time"])].copy()
    
    # Target: 1 if Delayed, 0 if On-Time
    df_clean["IsDelayed"] = (df_clean["DeliveryStatus"] == "Delayed").astype(int)

    X = df_clean[FEATURE_COLS]
    y = df_clean["IsDelayed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERICAL_FEATURES)
        ]
    )

    # Model Pipeline (Optimized depth & estimators for snappy performance)
    clf = RandomForestClassifier(
        n_estimators=45,
        max_depth=7,
        min_samples_split=6,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    print("[*] Fitting Random Forest model pipeline...", flush=True)

    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_curve": {"fpr": [round(float(x), 4) for x in fpr], "tpr": [round(float(x), 4) for x in tpr]},
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }


    # Extract human-readable feature importances
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feature_names = cat_feature_names + NUMERICAL_FEATURES

    raw_importances = pipeline.named_steps["classifier"].feature_importances_
    importance_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Importance": raw_importances
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    # Clean label representation for chart display
    importance_df["DisplayFeature"] = (
        importance_df["Feature"]
        .str.replace("cat__", "")
        .str.replace("num__", "")
        .str.replace("_", ": ")
    )

    # Cache metadata ranges for UI input defaults
    categorical_options = {
        col: sorted(df_clean[col].unique().tolist()) for col in CATEGORICAL_FEATURES
    }
    numerical_ranges = {
        col: {
            "min": float(df_clean[col].min()),
            "max": float(df_clean[col].max()),
            "mean": float(df_clean[col].mean()),
            "median": float(df_clean[col].median())
        } for col in NUMERICAL_FEATURES
    }

    artifact = {
        "pipeline": pipeline,
        "metrics": metrics,
        "feature_importances": importance_df,
        "categorical_options": categorical_options,
        "numerical_ranges": numerical_ranges
    }

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, save_path)
    print(f"[+] Model artifact successfully saved to: {save_path}")
    print(f"    - Accuracy: {metrics['accuracy'] * 100:.2f}% | ROC-AUC: {metrics['roc_auc']:.3f} | F1: {metrics['f1_score']:.3f}")
    return artifact

def load_trained_model(model_path: Path = None) -> Dict[str, Any]:
    """Loads cached model pipeline artifact or retrains if not found or version incompatible."""
    if model_path is None:
        model_path = MODEL_FILE
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception as e:
            print(f"[!] Warning: Could not unpickle {model_path} ({e}). Retraining fresh model pipeline...", flush=True)
            return train_and_save_model(save_path=model_path)
    return train_and_save_model(save_path=model_path)

def predict_order_delay_risk(
    artifact: Dict[str, Any],
    order_input: Dict[str, Any]
) -> Tuple[float, str, List[str]]:
    """
    Computes delay risk probability, classification risk tier, and prescriptive mitigation actions.
    
    Returns:
        (delay_prob_pct, risk_level, recommendations)
    """
    pipeline = artifact["pipeline"]
    input_df = pd.DataFrame([order_input])

    prob_delayed = float(pipeline.predict_proba(input_df)[0, 1]) * 100.0

    # Risk Tier assignment
    if prob_delayed >= 60.0:
        risk_level = "High Risk"
    elif prob_delayed >= 35.0:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    # Prescriptive operational recommendations
    recs = []
    ship_mode = order_input.get("ShippingMode", "")
    lead_days = order_input.get("PromisedLeadDays", 5)
    inv = order_input.get("InventoryLevel", 100)
    region = order_input.get("Region", "")

    if prob_delayed >= 60.0:
        if ship_mode == "Standard":
            recs.append("🚀 Upgrade freight tier from Standard to Express to compress in-transit variance.")
        if lead_days < 4:
            recs.append("⏱️ Re-negotiate customer delivery promise window: extend promised SLA by +1 to +2 business days.")
        if inv < 40:
            recs.append("⚠️ Stockout Trigger: Local inventory is critically depleted (<40 units); re-route order allocation to regional hub.")
        if region in ["South", "East"]:
            recs.append("🗺️ High-congestion destination corridor: utilize regional dedicated carrier rather than national consolidator.")
    elif prob_delayed >= 35.0:
        if inv < 60:
            recs.append("📦 Safety stock caution: reserve buffer stock at dispatch warehouse prior to batch release.")
        recs.append("🔔 Flag for Priority Handling in Warehouse Management System (WMS) pick-and-pack queue.")
        if ship_mode == "Same Day":
            recs.append("⚡ Ensure local dispatch before 11:00 AM local time cutoff to guarantee same-day SLA.")
    else:
        recs.append("✅ Fulfillment trajectory is optimal. Standard dispatch schedule approved.")
        recs.append("📈 Low logistical risk profile: eligible for automated bulk pick batching.")

    return round(prob_delayed, 1), risk_level, recs

if __name__ == "__main__":
    train_and_save_model()
