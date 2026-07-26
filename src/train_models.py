"""
Step 6: Train Models
=======================
Trains several classification algorithms per disease and keeps the best
one (by ROC-AUC on a held-out validation split) for deployment in the UI.

Algorithms compared:
  - Logistic Regression   (interpretable clinical baseline)
  - Random Forest         (handles non-linear feature interactions)
  - Gradient Boosting     (usually strongest tabular performance)
  - Support Vector Machine (RBF kernel)
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def get_candidate_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }


def train_and_select_best(X: pd.DataFrame, y: pd.Series, disease_name: str):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    results = {}
    fitted = {}
    for name, model in get_candidate_models().items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, proba)
        results[name] = auc
        fitted[name] = model
        print(f"[{disease_name}] {name}: validation ROC-AUC = {auc:.4f}")

    best_name = max(results, key=results.get)
    best_model = fitted[best_name]
    print(f"[{disease_name}] Best model: {best_name} (ROC-AUC={results[best_name]:.4f})")

    # Refit best model on ALL available data before saving for deployment
    best_model.fit(X, y)

    path = os.path.join(MODEL_DIR, f"{disease_name}_best_model.joblib")
    joblib.dump({"model": best_model, "model_name": best_name, "features": list(X.columns)}, path)
    print(f"[{disease_name}] Saved to {path}")

    return best_model, best_name, results


if __name__ == "__main__":
    from data_loader import load_diabetes_data, load_heart_data
    from preprocessing import preprocess_diabetes, preprocess_heart

    Xd, yd, scaler_d, cols_d = preprocess_diabetes(load_diabetes_data(verbose=False))
    joblib.dump({"scaler": scaler_d, "features": cols_d}, os.path.join(MODEL_DIR, "diabetes_scaler.joblib"))
    train_and_select_best(Xd, yd, "diabetes")

    print()
    Xh, yh, scaler_h, cols_h = preprocess_heart(load_heart_data(verbose=False))
    joblib.dump({"scaler": scaler_h, "features": cols_h}, os.path.join(MODEL_DIR, "heart_scaler.joblib"))
    train_and_select_best(Xh, yh, "heart")
