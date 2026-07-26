# Major Project: Disease Prediction System (Diabetes / Heart)

A machine-learning system that predicts a patient's risk of **Diabetes**
and **Heart Disease** from clinical measurements, built following the
project's 10-step guidance.

---

## Step 1 — Define Medical Problem

**Target diseases:** Diabetes and Heart Disease — two of the highest-burden
chronic conditions globally, both diagnosable earlier and more cheaply from
routine clinical measurements than from symptoms alone.

**Problem framing:** two independent **binary classification** tasks —

| Task | Positive class | Input |
|---|---|---|
| Diabetes screening | Patient has diabetes | 8 clinical measurements (glucose, BMI, age, etc.) |
| Heart disease screening | Patient has heart disease | 13 clinical measurements (chest pain type, cholesterol, ECG results, etc.) |

**Intended use:** an educational decision-support prototype that flags
higher-risk patients for **further clinical testing** — not a diagnostic
replacement for a physician (see Ethics section below).

---

## Step 2 — Data

| | Diabetes | Heart Disease |
|---|---|---|
| **Dataset name** | PIMA Indians Diabetes Dataset | Heart Disease Dataset |
| **Source** | UCI Machine Learning Repository | UCI Machine Learning Repository |
| **Location** | Kaggle | Kaggle |
| **Link** | https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database | https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset |
| **Rows** | 768 patients | 1,025 patients |
| **Features** | 8 (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age) | 13 (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal) |
| **Label** | `Outcome` (0/1) | `target` (0/1) |

**Getting the real data:** download `diabetes.csv` and `heart.csv` from
the Kaggle links above and place them in `data/`. `src/data_loader.py`
loads them automatically if present.

**No internet access in this build environment:** since this project was
assembled without web access, `src/data_loader.py` falls back to a
**statistically-realistic synthetic dataset** matching each real dataset's
schema, feature ranges, missing-value pattern, and class balance, so every
step of the pipeline is fully runnable and testable right now. **Drop the
real Kaggle CSVs into `data/` at any time — no other code changes
needed** — and the pipeline will automatically switch to real data.

---

## Step 3–8 — Pipeline

Run the whole thing in order with:

```bash
pip install -r requirements.txt
python main.py
```

This executes, for **both** diseases:

3. **Preprocess Data** (`src/preprocessing.py`) — imputes missing values
   (the PIMA dataset encodes missing Glucose/BloodPressure/SkinThickness/
   Insulin/BMI as biologically-impossible zeros; these are treated as NaN
   and median-imputed per outcome group), caps outliers with the IQR
   method, then standardizes all features.
4. **Feature Selection** (`src/feature_selection.py`) — ranks features
   with an ANOVA F-test and Random Forest importance, and keeps the union
   of the top predictors from each.
5. **Model Patterns / EDA** (`src/eda.py`) — correlation heatmaps and
   class-conditional distribution plots of the top features, saved to
   `outputs/`, to see how patient health-behavior patterns differ between
   diseased and healthy groups.
6. **Train Models** (`src/train_models.py`) — trains Logistic Regression,
   Random Forest, Gradient Boosting, and SVM classifiers; keeps whichever
   has the best validation ROC-AUC; saves it to `models/`.
7. **Evaluate Performance** (`src/evaluate.py`) — reports accuracy,
   precision, **sensitivity (recall)**, **specificity**, F1, ROC-AUC, and
   the confusion matrix on a held-out test set, plus a saved ROC curve.
   Sensitivity/specificity are reported explicitly because, in a screening
   context, missing a true case (false negative) and raising a false
   alarm (false positive) carry very different costs — accuracy alone
   hides that tradeoff.
8. **Cross-Validation** (`src/cross_validation.py`) — 5-fold **stratified**
   cross-validation (stratified to preserve the disease-positive ratio in
   every fold) reporting mean ± standard deviation for each metric, to
   confirm results aren't an artifact of one lucky train/test split.

## Step 9 — Deploy UI

```bash
streamlit run app.py
```

A Streamlit form lets a user enter patient values for either disease and
returns a prediction with a probability score, using the models saved by
`main.py` / `src/train_models.py`.

---

## Step 10 — Document System Design, Ethics & Limitations

### System design summary
- Two independent pipelines (diabetes, heart) sharing the same
  architecture: load → preprocess → select features → explore →
  train/select best model → evaluate → cross-validate → serve via UI.
- Best model is chosen per-disease by validation ROC-AUC among four
  candidate algorithms, then refit on all available data before being
  saved for deployment.
- All artifacts (scalers, trained models, plots) are versioned to disk
  under `models/` and `outputs/` so the UI and evaluation stay consistent
  with what was trained.

### Ethical considerations
- **Not a diagnostic device.** This is an educational prototype. A
  positive/negative prediction must never be treated as a diagnosis — it
  should, at most, prompt a conversation with a qualified clinician and
  proper diagnostic testing.
- **Training data limitations.** The PIMA dataset only includes female
  patients of Pima Indian heritage aged 21+, and the heart disease dataset
  reflects the demographics of the clinics that originally collected it.
  A model trained on either does not necessarily generalize to other
  sexes, ages, or populations — deploying it outside its training
  population risks systematically biased predictions.
- **Class imbalance and error costs.** Both datasets are imbalanced
  (more healthy than diseased patients). In screening, a false negative
  (telling a sick patient they're healthy) is typically far more harmful
  than a false positive (recommending an unnecessary follow-up test) —
  which is why this project reports sensitivity/specificity, not just
  accuracy, and should be tuned/thresholded with that asymmetry in mind
  before any real-world use.
- **Privacy.** Any real deployment must handle patient data under
  applicable health-privacy regulations (e.g. HIPAA/GDPR) — encrypted at
  rest and in transit, with access controls and audit logging. This
  prototype has none of that and should not be given real patient data.
- **Synthetic-data caveat.** As noted in Step 2, the datasets bundled by
  default in this environment are synthetic stand-ins with no real
  patients. All reported metrics are for demonstration of the pipeline,
  not a claim about real-world clinical accuracy. Metrics must be
  re-measured on the real Kaggle datasets before drawing any conclusions.

### Known limitations
- Small dataset sizes (hundreds to low thousands of rows) limit how well
  any model generalizes; a production system would need substantially
  more, more diverse data.
- No temporal/longitudinal patient data — each row is a single snapshot,
  so the models cannot account for how a patient's health is trending.
- No model explainability layer (e.g. SHAP) is included yet — clinicians
  would reasonably want per-prediction feature attributions before
  trusting a flagged case.
- The UI performs no input validation against physiologically implausible
  combinations of values.

---

## Tools & Platforms

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit / Flask (UI)

## Reference Links

- Medical ML GitHub: https://github.com/krishnaik06/Disease-Prediction-ML
- Healthcare Analytics GitHub: https://github.com/anishghose/Diabetes-Prediction
- ML Healthcare GitHub: https://github.com/IBM/ai-healthcare-projects

## Project Structure

```
disease_prediction_system/
├── data/                    # place diabetes.csv / heart.csv here (optional)
├── models/                  # saved scalers + trained models (generated)
├── outputs/                 # saved plots: heatmaps, distributions, ROC curves (generated)
├── src/
│   ├── data_loader.py       # Step 2
│   ├── preprocessing.py     # Step 3
│   ├── feature_selection.py # Step 4
│   ├── eda.py                # Step 5
│   ├── train_models.py      # Step 6
│   ├── evaluate.py          # Step 7
│   └── cross_validation.py  # Step 8
├── app.py                   # Step 9 (Streamlit UI)
├── main.py                  # runs steps 2-8 end to end
├── requirements.txt
└── README.md                 # Steps 1 & 10
```
