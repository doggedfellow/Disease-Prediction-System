"""
Step 3: Preprocess Data
=========================
Handles missing values and outliers, then normalizes features.

Diabetes dataset quirk: Glucose, BloodPressure, SkinThickness, Insulin and
BMI use 0 as a sentinel for "not recorded" (a value of 0 is not
physiologically possible for these). We treat those zeros as missing,
impute with the median (grouped by Outcome so imputation doesn't leak the
label's signal into a single global number), then cap outliers with the
IQR method before scaling.

Heart dataset: no sentinel-missing values in the standard release, but we
still run the same missing-value and outlier-capping logic defensively so
the pipeline is safe against dirtier real-world data dumps.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ZERO_AS_MISSING_DIABETES = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def _impute_group_median(df: pd.DataFrame, cols, group_col: str) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df.groupby(group_col)[col].transform(lambda s: s.fillna(s.median()))
        # fallback in case a whole group is NaN
        df[col] = df[col].fillna(df[col].median())
    return df


def _cap_outliers_iqr(df: pd.DataFrame, cols, k: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        df[col] = df[col].clip(lower, upper)
    return df


def preprocess_diabetes(df: pd.DataFrame):
    df = df.copy()
    df = _impute_group_median(df, ZERO_AS_MISSING_DIABETES, "Outcome")

    feature_cols = [c for c in df.columns if c != "Outcome"]
    df = _cap_outliers_iqr(df, feature_cols)

    X = df[feature_cols]
    y = df["Outcome"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)
    return X_scaled, y, scaler, feature_cols


def preprocess_heart(df: pd.DataFrame):
    df = df.copy()
    df = df.fillna(df.median(numeric_only=True))

    feature_cols = [c for c in df.columns if c != "target"]
    numeric_continuous = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    df = _cap_outliers_iqr(df, numeric_continuous)

    X = df[feature_cols]
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)
    return X_scaled, y, scaler, feature_cols


if __name__ == "__main__":
    from data_loader import load_diabetes_data, load_heart_data

    Xd, yd, _, cols_d = preprocess_diabetes(load_diabetes_data(verbose=False))
    print("Diabetes preprocessed:", Xd.shape, "features:", cols_d)
    print(Xd.describe().loc[["mean", "std"]].round(2))

    Xh, yh, _, cols_h = preprocess_heart(load_heart_data(verbose=False))
    print("\nHeart preprocessed:", Xh.shape, "features:", cols_h)
    print(Xh.describe().loc[["mean", "std"]].round(2))
