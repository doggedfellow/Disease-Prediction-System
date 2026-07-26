"""
Step 4: Feature Selection
============================
Ranks features by two complementary methods and returns the union of the
top-k from each, so the model keeps whatever a single method might miss:

  1. ANOVA F-test (SelectKBest with f_classif) -- captures linear
     separability between classes for each feature independently.
  2. Random Forest feature importance -- captures non-linear / interaction
     effects a linear test can miss.

Normalization itself is handled upstream in preprocessing.py; this module
is purely about choosing which already-normalized columns to keep.
"""

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42


def rank_features_anova(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    selector = SelectKBest(score_func=f_classif, k="all")
    selector.fit(X, y)
    return pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)


def rank_features_importance(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)


def select_top_features(X: pd.DataFrame, y: pd.Series, k: int = 6) -> list:
    anova_top = set(rank_features_anova(X, y).head(k).index)
    rf_top = set(rank_features_importance(X, y).head(k).index)
    selected = list(anova_top | rf_top)
    return selected


if __name__ == "__main__":
    from data_loader import load_diabetes_data, load_heart_data
    from preprocessing import preprocess_diabetes, preprocess_heart

    Xd, yd, _, _ = preprocess_diabetes(load_diabetes_data(verbose=False))
    print("Diabetes ANOVA F-scores:\n", rank_features_anova(Xd, yd).round(2))
    print("\nDiabetes RF importances:\n", rank_features_importance(Xd, yd).round(3))
    print("\nSelected features (union top-6):", select_top_features(Xd, yd, k=6))

    Xh, yh, _, _ = preprocess_heart(load_heart_data(verbose=False))
    print("\n\nHeart ANOVA F-scores:\n", rank_features_anova(Xh, yh).round(2))
    print("\nHeart RF importances:\n", rank_features_importance(Xh, yh).round(3))
    print("\nSelected features (union top-6):", select_top_features(Xh, yh, k=6))
