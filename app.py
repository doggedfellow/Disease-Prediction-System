import os
import math

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Theme
# --------------------------------------------------

if "theme" not in st.session_state:
    st.session_state.theme = "Light"

LIGHT = """
--bg:#ffffff;
--surface:#ffffff;
--card:#fafafa;
--text:#111827;
--muted:#6b7280;
--border:#e5e7eb;
--accent:#2563eb;
--success:#16a34a;
--danger:#dc2626;
"""

DARK = """
--bg:#0f172a;
--surface:#111827;
--card:#1f2937;
--text:#f9fafb;
--muted:#9ca3af;
--border:#374151;
--accent:#60a5fa;
--success:#22c55e;
--danger:#ef4444;
"""

theme_css = LIGHT if st.session_state.theme == "Light" else DARK

st.markdown(
f"""
<style>

:root{{
{theme_css}
}}

html,body,.stApp{{
background:var(--bg);
color:var(--text);
font-family:Inter,sans-serif;
}}

#MainMenu,
footer,
header{{
visibility:hidden;
}}

.block-container{{
padding-top:2rem;
max-width:1200px;
}}

.metric-card{{
background:var(--surface);
border:1px solid var(--border);
border-radius:18px;
padding:25px;
box-shadow:0 4px 18px rgba(0,0,0,.05);
}}

.result-card{{
background:var(--surface);
border:1px solid var(--border);
border-radius:18px;
padding:25px;
box-shadow:0 6px 20px rgba(0,0,0,.06);
}}

.big-title{{
font-size:34px;
font-weight:700;
}}

.subtitle{{
color:var(--muted);
font-size:15px;
margin-bottom:25px;
}}

.status-high{{
color:var(--danger);
font-size:28px;
font-weight:700;
}}

.status-low{{
color:var(--success);
font-size:28px;
font-weight:700;
}}

.small-label{{
font-size:12px;
text-transform:uppercase;
letter-spacing:.08em;
color:var(--muted);
}}

.big-number{{
font-size:24px;
font-weight:700;
}}

.stButton>button{{
width:100%;
border-radius:12px;
height:48px;
font-weight:600;
}}

</style>
""",
unsafe_allow_html=True
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("🩺 MedScreen")

    st.caption("Disease Prediction Dashboard")

    disease = st.radio(
        "Prediction Model",
        ["Diabetes","Heart Disease"]
    )

    st.divider()

    theme = st.selectbox(
        "Appearance",
        ["Light","Dark"],
        index=0 if st.session_state.theme=="Light" else 1
    )

    if theme != st.session_state.theme:
        st.session_state.theme = theme
        st.rerun()

    st.divider()

    st.info(
        "This tool provides screening predictions using trained "
        "machine learning models. "
        "It should not replace medical diagnosis."
    )

# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
"""
<div class="big-title">
Disease Prediction System
</div>
""",
unsafe_allow_html=True,
)

st.markdown(
"""
<div class="subtitle">
Predict Diabetes and Heart Disease risk using trained machine learning models.
</div>
""",
unsafe_allow_html=True,
)

# --------------------------------------------------
# Load Models
# --------------------------------------------------

@st.cache_resource
def load_artifacts(name):

    model = joblib.load(
        os.path.join(MODEL_DIR,f"{name}_best_model.joblib")
    )

    scaler = joblib.load(
        os.path.join(MODEL_DIR,f"{name}_scaler.joblib")
    )

    return model,scaler


def predict(name,values):

    model_bundle,scaler_bundle = load_artifacts(name)

    features = scaler_bundle["features"]

    df = pd.DataFrame(
        [[values[f] for f in features]],
        columns=features,
    )

    scaled = scaler_bundle["scaler"].transform(df)

    scaled = pd.DataFrame(
        scaled,
        columns=features
    )

    model = model_bundle["model"]

    pred = model.predict(scaled)[0]

    proba = model.predict_proba(scaled)[0,1]

    return pred,proba,model_bundle["model_name"]
    # --------------------------------------------------
# Plotly Gauge
# --------------------------------------------------

def radial_gauge_plot(probability, high_risk):

    value = probability * 100

    color = "#ef4444" if high_risk else "#22c55e"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={
                "suffix": "%",
                "font": {"size": 34}
            },
            gauge={
                "shape": "angular",
                "axis": {
                    "range": [0,100],
                    "tickwidth": 0,
                    "tickcolor": "rgba(0,0,0,0)"
                },
                "bar": {
                    "color": color,
                    "thickness": 0.28
                },
                "borderwidth": 0,
                "bgcolor": "rgba(0,0,0,0)",
                "steps":[
                    {
                        "range":[0,100],
                        "color":"#d1d5db"
                    }
                ]
            }
        )
    )

    fig.update_layout(
        margin=dict(l=10,r=10,t=10,b=10),
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig


# --------------------------------------------------
# Recommendation
# --------------------------------------------------

def recommendation(pred):

    if pred == 0:

        return (
            "🟢 Low Risk",
            """
Continue maintaining a healthy lifestyle.

• Exercise regularly

• Maintain a balanced diet

• Stay hydrated

• Schedule regular health checkups
"""
        )

    return (
        "🔴 Elevated Risk",
        """
Please consult a qualified healthcare professional.

Recommended:

• Clinical evaluation

• Blood investigations

• Lifestyle modification

• Follow physician advice
"""
    )


# --------------------------------------------------
# Result Dashboard
# --------------------------------------------------

def render_result(pred, probability, model_name, disease):

    high = pred == 1

    status = "Elevated Risk" if high else "Low Risk"

    status_class = "status-high" if high else "status-low"

    title, advice = recommendation(pred)

    left,right = st.columns([1.2,1.8],gap="large")

    # ---------------- LEFT ----------------

    with left:

        st.plotly_chart(
            radial_gauge_plot(probability,high),
            use_container_width=True,
            config={
                "displayModeBar":False
            }
        )

    # ---------------- RIGHT ----------------

    with right:

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="{status_class}">{status}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"<p>{disease} Screening Result</p>",
            unsafe_allow_html=True
        )

        st.divider()

        a,b = st.columns(2)

        with a:

            st.markdown(
                '<div class="small-label">Probability</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="big-number">{probability:.1%}</div>',
                unsafe_allow_html=True
            )

        with b:

            st.markdown(
                '<div class="small-label">Model Used</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="big-number">{model_name}</div>',
                unsafe_allow_html=True
            )

        st.progress(float(probability))

        st.markdown("</div>",unsafe_allow_html=True)

    st.markdown("### Recommendation")

    st.success(title)

    st.info(advice)

    st.caption(
        "This prediction is intended only as a screening aid and "
        "should not be considered a medical diagnosis."
    )
    # ==================================================
# DIABETES PAGE
# ==================================================

if disease == "Diabetes":

    st.subheader("🩸 Diabetes Risk Assessment")

    st.write(
        "Enter the patient's clinical information below and click "
        "**Predict Risk**."
    )

    left, right = st.columns(2, gap="large")

    with left:

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=1,
        )

        glucose = st.number_input(
            "Glucose",
            min_value=0,
            max_value=300,
            value=120,
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0,
            max_value=200,
            value=70,
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0,
            max_value=100,
            value=20,
        )

    with right:

        insulin = st.number_input(
            "Insulin",
            min_value=0,
            max_value=900,
            value=80,
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=70.0,
            value=28.0,
        )

        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=35,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Predict Diabetes Risk", use_container_width=True):

        values = {

            "Pregnancies": pregnancies,

            "Glucose": glucose,

            "BloodPressure": blood_pressure,

            "SkinThickness": skin_thickness,

            "Insulin": insulin,

            "BMI": bmi,

            "DiabetesPedigreeFunction": diabetes_pedigree,

            "Age": age,

        }

        prediction, probability, model_name = predict(
            "diabetes",
            values,
        )

        st.divider()

        render_result(
            prediction,
            probability,
            model_name,
            "Diabetes",
        )
        # ==================================================
# HEART DISEASE PAGE
# ==================================================

elif disease == "Heart Disease":

    st.subheader("❤️ Heart Disease Risk Assessment")

    st.write(
        "Enter the patient's clinical measurements below."
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:

        age = st.number_input(
            "Age",
            1,
            120,
            45
        )

        sex = st.selectbox(
            "Sex",
            ["Male", "Female"]
        )

        cp = st.selectbox(
            "Chest Pain Type",
            [0,1,2,3]
        )

        trestbps = st.number_input(
            "Resting Blood Pressure",
            50,
            250,
            120
        )

        chol = st.number_input(
            "Serum Cholesterol",
            100,
            700,
            220
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar >120 mg/dl",
            ["No","Yes"]
        )

        restecg = st.selectbox(
            "Resting ECG",
            [0,1,2]
        )

    with col2:

        thalach = st.number_input(
            "Maximum Heart Rate",
            50,
            250,
            150
        )

        exang = st.selectbox(
            "Exercise Induced Angina",
            ["No","Yes"]
        )

        oldpeak = st.number_input(
            "Old Peak",
            0.0,
            10.0,
            1.0
        )

        slope = st.selectbox(
            "Slope",
            [0,1,2]
        )

        ca = st.selectbox(
            "Major Vessels",
            [0,1,2,3,4]
        )

        thal = st.selectbox(
            "Thalassemia",
            [0,1,2,3]
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Predict Heart Disease Risk", use_container_width=True):

        values = {

            "age": age,

            "sex": 1 if sex=="Male" else 0,

            "cp": cp,

            "trestbps": trestbps,

            "chol": chol,

            "fbs": 1 if fbs=="Yes" else 0,

            "restecg": restecg,

            "thalach": thalach,

            "exang": 1 if exang=="Yes" else 0,

            "oldpeak": oldpeak,

            "slope": slope,

            "ca": ca,

            "thal": thal

        }

        prediction, probability, model_name = predict(
            "heart",
            values,
        )

        st.divider()

        render_result(
            prediction,
            probability,
            model_name,
            "Heart Disease",
        )


# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "Disease Prediction System • Powered by Scikit-learn • Streamlit • Plotly"
)

st.caption(
    "⚠ This application is intended for educational and screening purposes only "
    "and must not be used as a substitute for professional medical advice."
)