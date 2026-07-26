# Disease Prediction System

A machine learning-based application that predicts the risk of **Diabetes** and **Heart Disease** using clinical and patient health information.

This project was developed as a major project to explore the use of machine learning in healthcare for early risk assessment. Separate machine learning pipelines are used for each disease, with predictions provided through an interactive Streamlit web application.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a medical diagnostic tool and should not replace professional medical advice or clinical testing.

---

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Datasets](#datasets)
- [Setup](#setup)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Project Structure](#project-structure)
- [Ethical Considerations](#ethical-considerations)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [References](#references)
- [Disclaimer](#disclaimer)

---

## Features

- Diabetes risk prediction
- Heart disease risk prediction
- Separate ML models for each disease
- Automatic dataset download using the Kaggle API
- Data preprocessing and missing-value handling
- Feature selection
- Exploratory data analysis (EDA)
- Comparison of multiple machine learning algorithms
- Model evaluation using multiple metrics
- 5-fold stratified cross-validation
- Interactive Streamlit web interface
- Prediction probability / risk score
- Evaluation visualizations

---

## Technologies Used

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn, Plotly |
| Web App | Streamlit |
| Data Access | Kaggle API |

---

## Datasets

The project uses two separate datasets, both downloaded automatically via the Kaggle API when the training pipeline is executed.

### Diabetes Dataset

**Source:** PIMA Indians Diabetes Dataset

| Property | Value |
|---|---|
| Rows | 768 |
| Features | 8 |
| Target | `Outcome` |
| Classes | 0 = No Diabetes, 1 = Diabetes |

**Features:** `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`

### Heart Disease Dataset

**Source:** Heart Disease Dataset

| Property | Value |
|---|---|
| Rows | 1,025 |
| Features | 13 |
| Target | `target` |
| Classes | 0 = No Heart Disease, 1 = Heart Disease |

**Features:** `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Kaggle API

Get your Kaggle API token from **Kaggle → Settings → API → Create New Token**.

Download the `kaggle.json` file and save it at:

```text
~/.kaggle/kaggle.json
```

On Windows, the file is usually stored at:

```text
C:\Users\<your-username>\.kaggle\kaggle.json
```

Make sure the Kaggle API credentials are configured correctly before running the training pipeline.

### 3. Download Datasets and Train Models

```bash
python main.py
```

This runs the complete machine learning pipeline. It:

- Downloads the required datasets
- Loads and preprocesses the data
- Performs feature selection
- Generates exploratory data analysis visualizations
- Trains multiple machine learning models
- Selects the best-performing model
- Evaluates model performance
- Performs 5-fold stratified cross-validation
- Saves trained models and preprocessing artifacts

Trained models are saved in `models/`. Generated plots and evaluation results are saved in `outputs/`.

### 4. Launch the Dashboard

```bash
streamlit run app.py
```

This launches the interactive Streamlit dashboard, where users can enter patient information and receive predictions for diabetes or heart disease.

> **Note:** `main.py` must be run at least once before `app.py`, since the dashboard loads trained model files from `models/` that only exist after the training pipeline has completed.

---

## Machine Learning Pipeline

The project follows a complete machine learning workflow for both diseases.

### 1. Data Preprocessing
Raw datasets are cleaned and prepared before training, including:
- Handling missing and invalid values
- Replacing biologically unrealistic zero values where applicable
- Median imputation
- Outlier handling using the IQR method
- Feature standardization

*Implemented in `src/preprocessing.py`*

### 2. Feature Selection
Relevant features are identified using:
- ANOVA F-test
- Random Forest feature importance

*Implemented in `src/feature_selection.py`*

### 3. Exploratory Data Analysis
EDA is performed to understand the data and identify patterns between different patient groups, producing:
- Correlation heatmaps
- Feature distribution plots
- Class-based comparisons

Visualizations are saved in the `outputs/` directory.
*Implemented in `src/eda.py`*

### 4. Model Training
The following algorithms are trained and compared:
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

The best-performing model for each disease is selected based on validation ROC-AUC and saved for later use.
*Implemented in `src/train_models.py`*

### 5. Model Evaluation
Trained models are evaluated using:
- Accuracy
- Precision
- Recall / Sensitivity
- Specificity
- F1 Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

Sensitivity and specificity are included because both are important when evaluating models for health-related screening.
*Implemented in `src/evaluate.py`*

### 6. Cross-Validation
The project uses **5-fold stratified cross-validation** to evaluate model consistency across different subsets of the data. Results include the mean and standard deviation for key performance metrics.
*Implemented in `src/cross_validation.py`*

---

## Project Structure

```text
disease_prediction_system/
│
├── data/                     # Auto-downloaded datasets (via Kaggle API)
├── models/                   # Trained models and preprocessing artifacts
├── outputs/                  # EDA plots and evaluation results
│
├── src/
│   ├── data_loader.py        # Dataset loading / auto-download
│   ├── preprocessing.py      # Cleaning, imputation, scaling
│   ├── feature_selection.py  # ANOVA + Random Forest feature ranking
│   ├── eda.py                # Correlation heatmaps, distribution plots
│   ├── train_models.py       # Model training and selection
│   ├── evaluate.py           # Performance metrics and plots
│   └── cross_validation.py   # 5-fold stratified CV
│
├── app.py                    # Streamlit dashboard
├── main.py                   # End-to-end pipeline entry point
├── requirements.txt
└── README.md
```

---

## Ethical Considerations

**Not a Diagnostic Tool**
Predictions generated by this application should not be considered a medical diagnosis. The system is an educational machine learning project demonstrating preliminary risk assessment.

**Dataset Limitations**
The datasets have limitations in terms of size, demographics, and diversity. A model trained on this data may not perform equally well across every population.

**False Positives and False Negatives**
Both types of prediction errors matter in healthcare applications — a false negative may cause a serious case to be missed, while a false positive may lead to unnecessary testing. For this reason, the project reports sensitivity and specificity alongside accuracy.

**Privacy**
Any real-world healthcare application would require appropriate security measures to protect patient information, including secure storage, access control, and compliance with applicable data protection regulations. This academic project should not be used with real patient data without proper authorization and security measures.

---

## Limitations

- The datasets are relatively small compared with what a production healthcare system would require
- Training data may not represent all populations and demographic groups
- The model uses a single set of patient measurements rather than historical health records
- The project does not currently include explainability methods such as SHAP or LIME
- The application does not fully validate whether every combination of user-entered values is physiologically realistic
- Model performance depends heavily on the quality and representativeness of the training data
- The system has not been clinically validated and should not be used for real medical decision-making

---

## Future Improvements

- Use larger and more diverse healthcare datasets
- Add explainable AI techniques such as SHAP or LIME
- Improve input validation
- Test additional machine learning and ensemble models
- Perform hyperparameter optimization
- Add model performance comparison charts
- Improve the user interface and prediction visualizations
- Add secure database integration
- Perform external validation using independent datasets

---

## References

- PIMA Indians Diabetes Dataset — UCI Machine Learning Repository / Kaggle
- Heart Disease Dataset — UCI Machine Learning Repository / Kaggle
- Scikit-learn Documentation
- Streamlit Documentation
- Plotly Documentation
- Kaggle API Documentation

---

## Disclaimer

This project is developed for **educational and academic purposes**. The predictions generated by the system are not intended to diagnose, treat, or prevent any disease. Users should always consult qualified healthcare professionals for medical advice and diagnosis.