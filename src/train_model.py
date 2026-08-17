"""
train_model.py
---------------
Trains and compares Logistic Regression, Decision Tree, Random Forest, and
Gradient Boosting churn models. XGBoost is skipped (not installed in this
environment) in favor of sklearn's GradientBoostingClassifier, per the
project's own fallback instruction. Saves the best model + metadata to
models/churn_model.pkl.
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)

sys.path.insert(0, '/home/claude/airtel-customer-churn/src')
from feature_engineering import prepare_model_matrix, TARGET
from data_preprocessing import run_full_validation

DATA_PATH = "/home/claude/airtel-customer-churn/data/airtel_enterprise_churn.csv"
MODEL_OUT = "/home/claude/airtel-customer-churn/models/churn_model.pkl"
METRICS_OUT = "/home/claude/airtel-customer-churn/models/model_comparison.json"

RANDOM_STATE = 42


def main():
    df = run_full_validation(DATA_PATH)
    X, y, encoder_categories = prepare_model_matrix(df)
    X = X.astype(np.float64)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train churn rate: {y_train.mean()*100:.2f}%  |  Test churn rate: {y_test.mean()*100:.2f}%")

    # Scale for Logistic Regression only (tree models don't need it)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    fitted_models = {}

    # ---------------- Logistic Regression (baseline, interpretable) ----------------
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)
    fitted_models["Logistic Regression"] = lr
    results["Logistic Regression"] = evaluate(lr, X_test_scaled, y_test)

    # ---------------- Decision Tree ----------------
    dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=25,
                                 class_weight="balanced", random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    fitted_models["Decision Tree"] = dt
    results["Decision Tree"] = evaluate(dt, X_test, y_test)

    # ---------------- Random Forest ----------------
    # Light manual hyperparameter comparison on a validation slice carved out of
    # the training set (full GridSearchCV was too slow for this environment).
    X_tr2, X_val, y_tr2, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                                    random_state=RANDOM_STATE, stratify=y_train)
    rf_configs = [
        {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 5},
        {"n_estimators": 300, "max_depth": 14, "min_samples_leaf": 4},
        {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 3},
    ]
    best_auc, best_cfg = -1, None
    for cfg in rf_configs:
        m = RandomForestClassifier(**cfg, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(X_tr2, y_tr2)
        auc = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
        print(f"RF config {cfg} -> val ROC-AUC {auc:.4f}")
        if auc > best_auc:
            best_auc, best_cfg = auc, cfg
    print(f"Best RF config: {best_cfg} (val ROC-AUC {best_auc:.4f})")

    rf = RandomForestClassifier(**best_cfg, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    fitted_models["Random Forest"] = rf
    results["Random Forest"] = evaluate(rf, X_test, y_test)

    # ---------------- Gradient Boosting ----------------
    gb = GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.08,
                                     random_state=RANDOM_STATE)
    gb.fit(X_train, y_train)
    fitted_models["Gradient Boosting"] = gb
    results["Gradient Boosting"] = evaluate(gb, X_test, y_test)

    # ---------------- Comparison table ----------------
    comparison = pd.DataFrame(results).T
    comparison = comparison[["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]]
    comparison = comparison.round(4).sort_values("ROC_AUC", ascending=False)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(comparison.to_string())

    # Business-driven model selection: for churn, missing an at-risk customer
    # (false negative) costs more than a wasted retention call (false positive),
    # so we prioritize ROC-AUC and Recall over raw Accuracy.
    best_name = comparison["ROC_AUC"].idxmax()
    best_model = fitted_models[best_name]
    print(f"\nSelected model: {best_name} (highest ROC-AUC: {comparison.loc[best_name, 'ROC_AUC']})")
    print("Rationale: churn prediction is imbalanced (~21% positive class) and the business cost of\n"
          "missing a churner (lost revenue) generally exceeds the cost of an unnecessary retention\n"
          "outreach, so ROC-AUC and Recall are weighted above raw Accuracy when selecting the model.")

    # ---------------- Feature importance (matched to whichever model actually won) ----------------
    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_names)
    else:
        # Logistic Regression: coefficients are on standardized features, so their
        # absolute magnitude is directly comparable and interpretable as importance.
        importances = pd.Series(np.abs(best_model.coef_[0]), index=feature_names)
    top_features = importances.sort_values(ascending=False).head(15)
    print("\nTop 15 features by importance:")
    print(top_features.to_string())

    # ---------------- Save artifacts ----------------
    artifact = {
        "model": best_model,
        "model_name": best_name,
        "scaler": scaler if best_name == "Logistic Regression" else None,
        "feature_names": feature_names,
        "encoder_categories": encoder_categories,
        "uses_scaling": best_name == "Logistic Regression",
        "top_features": top_features.to_dict(),
    }
    joblib.dump(artifact, MODEL_OUT)
    print(f"\nSaved best model ({best_name}) -> {MODEL_OUT}")

    with open(METRICS_OUT, "w") as f:
        json.dump({
            "comparison": comparison.to_dict(orient="index"),
            "best_model": best_name,
            "top_features": top_features.round(4).to_dict(),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }, f, indent=2)
    print(f"Saved metrics -> {METRICS_OUT}")

    return artifact, comparison, X_test, y_test


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
    }


if __name__ == "__main__":
    main()
