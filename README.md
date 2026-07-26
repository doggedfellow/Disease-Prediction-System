# Disease Prediction System

A machine learning-based application that predicts the risk of **Diabetes** and **Heart Disease** using clinical and patient health information.

This project was developed as a major project to explore the use of machine learning in healthcare for early risk assessment. Separate machine learning pipelines are used for each disease, with predictions provided through a Streamlit web application.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a medical diagnostic tool and should not replace professional medical advice or clinical testing.

---

## Features

* Diabetes risk prediction
* Heart disease risk prediction
* Separate ML models for each disease
* Data preprocessing and missing-value handling
* Feature selection
* Exploratory data analysis (EDA)
* Comparison of multiple ML algorithms
* Model evaluation using multiple metrics
* 5-fold stratified cross-validation
* Interactive Streamlit web interface
* Prediction probability/risk score
* Evaluation visualizations

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit

---

## Datasets

The project uses two separate datasets.

### Diabetes Dataset

**Dataset:** PIMA Indians Diabetes Dataset

* Rows: 768
* Features: 8
* Target: `Outcome`
* Classes: 0 (No Diabetes), 1 (Diabetes)

Features:

`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`

### Heart Disease Dataset

**Dataset:** Heart Disease Dataset

* Rows: 1,025
* Features: 13
* Target: `target`
* Classes: 0 (No Heart Disease), 1 (Heart Disease)

Features:

`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`

### Data Setup

Place the datasets inside the `data/` directory:

```text
data/
├── diabetes.csv
└── heart.csv
```

The data loader automatically reads the files from this location.

For development and testing, the project can also work with synthetic data that follows the structure of the original datasets. When using the real datasets, place the CSV files in the `data/` directory before training the models.

---

## Machine Learning Pipeline

The project follows a complete machine learning workflow for both diseases.

### 1. Data Preprocessing

The raw datasets are cleaned and prepared before training. This includes:

* Handling missing and invalid values
* Replacing biologically unrealistic zero values where applicable
* Median imputation
* Outlier handling using the IQR method
* Feature standardization

Implemented in:

```text
src/preprocessing.py
```

### 2. Feature Selection

Relevant features are identified using:

* ANOVA F-test
* Random Forest feature importance

Implemented in:

```text
src/feature_selection.py
```

### 3. Exploratory Data Analysis

EDA is performed to understand the data and identify patterns between different patient groups.

The project generates:

* Correlation heatmaps
* Feature distribution plots
* Class-based comparisons

Visualizations are saved in the `outputs/` directory.

Implemented in:

```text
src/eda.py
```

### 4. Model Training

The following algorithms are trained and compared:

* Logistic Regression
* Random Forest
* Gradient Boosting
* Support Vector Machine (SVM)

The best-performing model for each disease is selected based on validation ROC-AUC and saved for later use.

Implemented in:

```text
src/train_models.py
```

### 5. Model Evaluation

The trained models are evaluated using:

* Accuracy
* Precision
* Recall / Sensitivity
* Specificity
* F1 Score
* ROC-AUC
* Confusion Matrix
* ROC Curve

Sensitivity and specificity are included because both are important when evaluating models for health-related screening.

Implemented in:

```text
src/evaluate.py
```

### 6. Cross-Validation

The project uses **5-fold stratified cross-validation** to evaluate model consistency across different subsets of the data.

The results include the mean and standard deviation for important performance metrics.

Implemented in:

```text
src/cross_validation.py
```

---

## Running the Project

### 1. Clone the Repository

```bash
git clone <repository-url>
cd disease_prediction_system
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Datasets

Place the required CSV files inside the `data/` folder:

```text
data/
├── diabetes.csv
└── heart.csv
```

### 5. Train the Models

Run:

```bash
python main.py
```

This runs the preprocessing, feature selection, EDA, model training, evaluation, and cross-validation pipeline.

Trained models are saved in:

```text
models/
```

Generated plots and evaluation results are saved in:

```text
outputs/
```

### 6. Run the Web Application

After training the models, start the Streamlit application:

```bash
streamlit run app.py
```

The application allows users to enter patient information and receive a prediction for diabetes or heart disease.

---

## Project Structure

```text
disease_prediction_system/
│
├── data/
│   ├── diabetes.csv
│   └── heart.csv
│
├── models/
├── outputs/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── eda.py
│   ├── train_models.py
│   ├── evaluate.py
│   └── cross_validation.py
│
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Ethical Considerations

### Not a Diagnostic Tool

The predictions generated by this application should not be considered a medical diagnosis. The system is designed as an educational machine learning project for demonstrating preliminary risk assessment.

### Dataset Limitations

The datasets have limitations in terms of size, demographics, and diversity. A model trained on these datasets may not perform equally well for every population.

### False Positives and False Negatives

Both types of prediction errors are important in healthcare applications. A false negative may cause a potentially serious case to be missed, while a false positive may lead to unnecessary medical testing.

For this reason, the project reports sensitivity and specificity in addition to accuracy.

### Privacy

Any real-world healthcare application would require appropriate security measures to protect patient information, including secure data storage, access control, and compliance with applicable data protection regulations.

This academic project should not be used with real patient data without appropriate authorization and security measures.

---

## Limitations

* The datasets are relatively small compared with the data required for a production healthcare system.
* The training data may not represent all populations and demographic groups.
* The model uses a single set of patient measurements rather than historical health records.
* The project does not currently include explainability methods such as SHAP or LIME.
* The application does not fully validate whether every combination of user-entered values is physiologically realistic.
* Model performance depends heavily on the quality and representativeness of the training data.
* The system has not been clinically validated and should not be used for real medical decision-making.

---

## Future Improvements

* Use larger and more diverse healthcare datasets
* Add explainable AI techniques such as SHAP or LIME
* Improve input validation
* Test additional machine learning and ensemble models
* Perform hyperparameter optimization
* Add model performance comparison charts
* Improve the user interface and prediction visualizations
* Add secure database integration
* Perform external validation using independent datasets

---

## References

* PIMA Indians Diabetes Dataset — UCI Machine Learning Repository / Kaggle
* Heart Disease Dataset — UCI Machine Learning Repository / Kaggle
* Scikit-learn Documentation
* Streamlit Documentation

---

## Disclaimer

This project is developed for **educational and academic purposes**. The predictions generated by the system are not intended to diagnose, treat, or prevent any disease. Users should always consult qualified healthcare professionals for medical advice and diagnosis.
