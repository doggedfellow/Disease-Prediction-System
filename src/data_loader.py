import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

DIABETES_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

KAGGLE_DATASETS = {
    "diabetes.csv": "uciml/pima-indians-diabetes-database",
    "heart.csv": "johnsmith88/heart-disease-dataset",
}


def _try_kaggle_download(filename: str, verbose: bool = True) -> bool:
    """Attempt to auto-download the real dataset from Kaggle into DATA_DIR.

    Returns True if the file exists at DATA_DIR/filename after this call
    (whether it was already there or just downloaded), False if the
    download could not be completed (missing package, missing/invalid
    token, network issue, etc.) so callers can fall back to synthetic data.
    """
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        return True

    ref = KAGGLE_DATASETS.get(filename)
    if ref is None:
        return False

    try:
        import kaggle
    except ImportError:
        if verbose:
            print(f"[data_loader] 'kaggle' package not installed -> "
                  f"run 'pip install kaggle' to enable auto-download.")
        return False

    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        kaggle.api.authenticate()
        if verbose:
            print(f"[data_loader] Downloading {ref} from Kaggle...")
        kaggle.api.dataset_download_files(ref, path=DATA_DIR, unzip=True)
    except Exception as e:
        if verbose:
            print(f"[data_loader] Kaggle auto-download failed ({e}). "
                  f"Place kaggle.json in ~/.kaggle/ or download {filename} "
                  f"manually into {DATA_DIR}.")
        return False

    return os.path.exists(path)


def _synthetic_diabetes(n=768, seed=RANDOM_STATE) -> pd.DataFrame:
    """Generate a synthetic stand-in with the same 8 features + Outcome
    label as the real PIMA dataset (n=768 matches the original size)."""
    rng = np.random.default_rng(seed)

    outcome = rng.binomial(1, 0.35, n)  

    glucose = np.where(
        outcome == 1,
        rng.normal(142, 30, n),
        rng.normal(110, 24, n),
    ).clip(0, 250)

    bmi = np.where(
        outcome == 1,
        rng.normal(35, 7, n),
        rng.normal(30, 6.5, n),
    ).clip(0, 67)

    age = np.where(
        outcome == 1,
        rng.normal(37, 11, n),
        rng.normal(31, 10, n),
    ).clip(21, 81).round().astype(int)

    pedigree = np.where(
        outcome == 1,
        rng.gamma(2.0, 0.28, n),
        rng.gamma(1.6, 0.25, n),
    ).clip(0.078, 2.42)

    pregnancies = rng.poisson(np.where(outcome == 1, 4.5, 3.0), n).clip(0, 17)
    blood_pressure = rng.normal(72, 12, n).clip(0, 122)
    skin_thickness = rng.normal(21, 10, n).clip(0, 99)
    insulin = np.where(
        outcome == 1,
        rng.normal(120, 90, n),
        rng.normal(70, 70, n),
    ).clip(0, 846)

    for col, arr, missing_rate in [
        ("Glucose", glucose, 0.007),
        ("BloodPressure", blood_pressure, 0.045),
        ("SkinThickness", skin_thickness, 0.30),
        ("Insulin", insulin, 0.49),
        ("BMI", bmi, 0.014),
    ]:
        mask = rng.random(n) < missing_rate
        arr[mask] = 0

    df = pd.DataFrame({
        "Pregnancies": pregnancies,
        "Glucose": glucose.round(1),
        "BloodPressure": blood_pressure.round(1),
        "SkinThickness": skin_thickness.round(1),
        "Insulin": insulin.round(1),
        "BMI": bmi.round(1),
        "DiabetesPedigreeFunction": pedigree.round(3),
        "Age": age,
        "Outcome": outcome,
    })
    return df[DIABETES_COLUMNS]


def _synthetic_heart(n=1025, seed=RANDOM_STATE) -> pd.DataFrame:
    """Generate a synthetic stand-in with the same 13 features + target
    label as the real UCI Heart Disease dataset."""
    rng = np.random.default_rng(seed)

    target = rng.binomial(1, 0.51, n) 

    age = np.where(
        target == 1, rng.normal(56, 8, n), rng.normal(53, 9, n)
    ).clip(29, 77).round().astype(int)
    sex = rng.binomial(1, 0.68, n)
    cp = np.where(target == 1,
                  rng.choice([0, 1, 2, 3], n, p=[0.55, 0.20, 0.15, 0.10]),
                  rng.choice([0, 1, 2, 3], n, p=[0.20, 0.30, 0.30, 0.20]))
    trestbps = rng.normal(131, 17, n).clip(94, 200).round().astype(int)
    chol = rng.normal(246, 51, n).clip(126, 564).round().astype(int)
    fbs = rng.binomial(1, 0.15, n)
    restecg = rng.choice([0, 1, 2], n, p=[0.48, 0.50, 0.02])
    thalach = np.where(
        target == 1, rng.normal(139, 22, n), rng.normal(158, 19, n)
    ).clip(71, 202).round().astype(int)
    exang = np.where(target == 1,
                      rng.binomial(1, 0.55, n),
                      rng.binomial(1, 0.15, n))
    oldpeak = np.where(
        target == 1, rng.gamma(2.0, 0.9, n), rng.gamma(1.1, 0.6, n)
    ).clip(0, 6.2).round(1)
    slope = rng.choice([0, 1, 2], n, p=[0.07, 0.46, 0.47])
    ca = np.where(target == 1,
                   rng.choice([0, 1, 2, 3, 4], n, p=[0.45, 0.25, 0.15, 0.10, 0.05]),
                   rng.choice([0, 1, 2, 3, 4], n, p=[0.75, 0.15, 0.06, 0.03, 0.01]))
    thal = rng.choice([0, 1, 2, 3], n, p=[0.02, 0.06, 0.55, 0.37])

    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
        "target": target,
    })
    return df[HEART_COLUMNS]


def load_diabetes_data(verbose: bool = True) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "diabetes.csv")
    if _try_kaggle_download("diabetes.csv", verbose=verbose):
        df = pd.read_csv(path)
        if verbose:
            print(f"[data_loader] Loaded REAL PIMA diabetes data from {path} ({len(df)} rows)")
        return df
    if verbose:
        print("[data_loader] Could not obtain data/diabetes.csv -> using synthetic "
              "stand-in data. Download the real file from "
              "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database "
              "and place it at data/diabetes.csv to use real data.")
    return _synthetic_diabetes()


def load_heart_data(verbose: bool = True) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "heart.csv")
    if _try_kaggle_download("heart.csv", verbose=verbose):
        df = pd.read_csv(path)
        if verbose:
            print(f"[data_loader] Loaded REAL heart disease data from {path} ({len(df)} rows)")
        return df
    if verbose:
        print("[data_loader] Could not obtain data/heart.csv -> using synthetic "
              "stand-in data. Download the real file from "
              "https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset "
              "and place it at data/heart.csv to use real data.")
    return _synthetic_heart()


if __name__ == "__main__":
    d = load_diabetes_data()
    h = load_heart_data()
    print("\nDiabetes shape:", d.shape, "| Outcome balance:\n", d["Outcome"].value_counts(normalize=True))
    print("\nHeart shape:", h.shape, "| target balance:\n", h["target"].value_counts(normalize=True))