"""
Step 7: Evaluate Performance (Medical Metrics)
=================================================
In a clinical screening context, accuracy alone is misleading (e.g. a
model that always predicts "healthy" on an imbalanced dataset can still
score high accuracy while missing every real case). We therefore report:

  - Sensitivity / Recall  : of all truly diseased patients, how many did
                             we correctly flag? (missing a case is the
                             costlier error in screening)
  - Specificity           : of all truly healthy patients, how many did we
                             correctly clear? (false alarms drive
                             unnecessary follow-up testing and anxiety)
  - Precision              : of all patients flagged positive, how many
                             truly have the disease?
  - F1-score               : harmonic mean of precision and recall
  - ROC-AUC                 : ranking quality across all thresholds
  - Confusion matrix        : full breakdown of TP/TN/FP/FN
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay
)

RANDOM_STATE = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, disease_name: str, plot: bool = True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    # Evaluate on a fresh held-out split for an honest read (the saved
    # production model was refit on all data; this eval uses its own split)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    }

    print(f"\n=== {disease_name.title()} — Held-out Test Set Metrics ===")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:20s}: {v:.4f}")
    print(f"  confusion_matrix    : {metrics['confusion_matrix']}")

    if plot:
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title(f"{disease_name.title()} ROC Curve")
        plt.tight_layout()
        path = os.path.join(OUT_DIR, f"{disease_name}_roc_curve.png")
        plt.savefig(path, dpi=130)
        plt.close()
        print(f"  ROC curve saved to  : {path}")

    return metrics


if __name__ == "__main__":
    from data_loader import load_diabetes_data, load_heart_data
    from preprocessing import preprocess_diabetes, preprocess_heart
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    Xd, yd, _, _ = preprocess_diabetes(load_diabetes_data(verbose=False))
    evaluate_model(GradientBoostingClassifier(random_state=RANDOM_STATE), Xd, yd, "diabetes")

    Xh, yh, _, _ = preprocess_heart(load_heart_data(verbose=False))
    evaluate_model(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), Xh, yh, "heart")
