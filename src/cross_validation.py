import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

RANDOM_STATE = 42


def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, disease_name: str, n_splits: int = 5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = cross_validate(model, X, y, cv=cv, scoring=scoring)

    print(f"\n=== {disease_name.title()} — {n_splits}-Fold Stratified Cross-Validation ===")
    summary = {}
    for metric in scoring:
        scores = results[f"test_{metric}"]
        summary[metric] = (scores.mean(), scores.std())
        print(f"  {metric:10s}: {scores.mean():.4f} +/- {scores.std():.4f}  (folds: {[round(s,3) for s in scores]})")

    return summary


if __name__ == "__main__":
    from data_loader import load_diabetes_data, load_heart_data
    from preprocessing import preprocess_diabetes, preprocess_heart
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    Xd, yd, _, _ = preprocess_diabetes(load_diabetes_data(verbose=False))
    cross_validate_model(GradientBoostingClassifier(random_state=RANDOM_STATE), Xd, yd, "diabetes")

    Xh, yh, _, _ = preprocess_heart(load_heart_data(verbose=False))
    cross_validate_model(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), Xh, yh, "heart")
