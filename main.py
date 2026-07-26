import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_diabetes_data, load_heart_data
from preprocessing import preprocess_diabetes, preprocess_heart
from feature_selection import rank_features_anova, rank_features_importance, select_top_features
from eda import (
    plot_correlation_heatmap, plot_class_conditional_distributions,
)
from train_models import train_and_select_best
from evaluate import evaluate_model
from cross_validation import cross_validate_model
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def run_pipeline(disease_key: str, load_fn, preprocess_fn, target_col: str, label: str):
    print("\n" + "=" * 70)
    print(f"  {label.upper()} PIPELINE")
    print("=" * 70)

    raw_df = load_fn()

    X, y, scaler, feature_cols = preprocess_fn(raw_df)
    joblib.dump({"scaler": scaler, "features": feature_cols},
                os.path.join(MODEL_DIR, f"{disease_key}_scaler.joblib"))

    print(f"\n--- Step 4: Feature Selection ({label}) ---")
    anova_scores = rank_features_anova(X, y)
    rf_scores = rank_features_importance(X, y)
    top_features = select_top_features(X, y, k=6)
    print("Top features (ANOVA):", anova_scores.head(6).index.tolist())
    print("Top features (RF importance):", rf_scores.head(6).index.tolist())
    print("Selected (union):", top_features)

    print(f"\n--- Step 5: Exploratory Analysis ({label}) ---")
    raw_labeled = raw_df.copy()
    heat_path = plot_correlation_heatmap(
        raw_df, target_col, f"{label}: Correlation Heatmap", f"{disease_key}_correlation_heatmap.png"
    )
    dist_path = plot_class_conditional_distributions(
        raw_labeled, target_col, anova_scores.head(6).index.tolist(),
        f"{label}: Behavior Patterns", f"{disease_key}_behavior_patterns.png"
    )
    print(f"Saved: {heat_path}\nSaved: {dist_path}")

    print(f"\n--- Step 6: Train Models ({label}) ---")
    best_model, best_name, results = train_and_select_best(X, y, disease_key)

    print(f"\n--- Step 7: Evaluate Performance ({label}) ---")
    fresh_model = type(best_model)(**best_model.get_params())
    evaluate_model(fresh_model, X, y, disease_key)

    print(f"\n--- Step 8: Cross-Validation ({label}) ---")
    fresh_model_cv = type(best_model)(**best_model.get_params())
    cross_validate_model(fresh_model_cv, X, y, disease_key)

    print(f"\n{label} pipeline complete. Best model: {best_name}")


if __name__ == "__main__":
    run_pipeline("diabetes", load_diabetes_data, preprocess_diabetes, "Outcome", "Diabetes")
    run_pipeline("heart", load_heart_data, preprocess_heart, "target", "Heart Disease")

    print("\n" + "=" * 70)
    print("  ALL DONE")
    print("=" * 70)
    print("Step 9 (Develop UI): run `streamlit run app.py`")
    print("Step 10 (Document Ethics): see README.md")
