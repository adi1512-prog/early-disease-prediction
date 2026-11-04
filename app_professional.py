import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Early Disease Prediction",
    page_icon="🩺",
    layout="wide"
)

# =========================================================
# SESSION STATE
# =========================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

.stApp {
    background: #f5f8fc;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* HERO */

.hero {
    background: linear-gradient(135deg, #0b4f78, #087ea4, #18a6a6);
    padding: 38px 42px;
    border-radius: 24px;
    margin-bottom: 25px;
    box-shadow: 0 15px 35px rgba(10, 80, 120, 0.18);
}

.hero-title {
    color: white;
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1px;
}

.hero-subtitle {
    color: rgba(255,255,255,0.9);
    font-size: 17px;
    margin-top: 8px;
}

.hero-badge {
    display: inline-block;
    margin-top: 18px;
    padding: 7px 14px;
    border-radius: 20px;
    background: rgba(255,255,255,0.16);
    color: white;
    font-size: 13px;
}

/* DASHBOARD CARDS */

.dashboard-card {
    background: white;
    border: 1px solid #e3ebf2;
    border-radius: 17px;
    padding: 20px;
    min-height: 115px;
    box-shadow: 0 6px 20px rgba(25,55,80,0.06);
    transition: 0.2s ease;
}

.dashboard-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(25,55,80,0.10);
}

.card-icon {
    font-size: 25px;
}

.card-title {
    color: #6a7f8e;
    font-size: 13px;
    margin-top: 5px;
}

.card-value {
    color: #123c57;
    font-size: 21px;
    font-weight: 750;
    margin-top: 3px;
}

/* SECTION */

.section-title {
    color: #143c57;
    font-size: 28px;
    font-weight: 750;
    margin-top: 30px;
    margin-bottom: 20px;
}

.section-label {
    color: #557184;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* SELECTOR */

.selector-card {
    background: white;
    padding: 20px 24px 12px 24px;
    border-radius: 17px;
    border: 1px solid #e2ebf2;
    box-shadow: 0 6px 20px rgba(30,55,75,0.06);
    margin-bottom: 25px;
}

/* INPUTS */

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border-color: #d7e2ea !important;
    background: white !important;
}

div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: #1593a4 !important;
    box-shadow: 0 0 0 1px #1593a4 !important;
}

label {
    color: #334f61 !important;
    font-weight: 600 !important;
}

/* BUTTON */

.stButton > button {
    border: none;
    border-radius: 12px;
    min-height: 50px;
    background: linear-gradient(135deg, #096fa6, #159da2);
    color: white;
    font-size: 16px;
    font-weight: 700;
    box-shadow: 0 8px 20px rgba(9,111,166,0.20);
    transition: 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 25px rgba(9,111,166,0.28);
}

/* RESULT */

.result-card {
    background: white;
    border-radius: 20px;
    padding: 25px;
    margin-top: 25px;
    border: 1px solid #e2ebf2;
    box-shadow: 0 8px 25px rgba(25,55,80,0.08);
}

.result-title {
    color: #123c57;
    font-size: 23px;
    font-weight: 750;
}

.result-number {
    color: #087da2;
    font-size: 42px;
    font-weight: 800;
    margin-top: 5px;
}

.result-description {
    color: #6d7f8d;
    font-size: 14px;
}

/* INFO */

.info-card {
    background: linear-gradient(135deg, #ffffff, #f1f8fb);
    border: 1px solid #dceaf1;
    border-radius: 18px;
    padding: 22px;
    margin-top: 25px;
}

.info-title {
    color: #17445d;
    font-size: 19px;
    font-weight: 700;
}

.info-text {
    color: #627786;
    line-height: 1.6;
    font-size: 14px;
}

/* EXPANDER */

div[data-testid="stExpander"] {
    background: white;
    border: 1px solid #e2ebf2;
    border-radius: 15px;
}

/* ALERT */

div[data-testid="stAlert"] {
    border-radius: 14px;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #8193a1;
    font-size: 13px;
    padding: 30px 0 10px;
}

@media (max-width: 768px) {

    .hero {
        padding: 28px 22px;
    }

    .hero-title {
        font-size: 30px;
    }

    .hero-subtitle {
        font-size: 14px;
    }

    .section-title {
        font-size: 24px;
    }

}

</style>
""",
    unsafe_allow_html=True
)

# =========================================================
# HERO
# =========================================================

st.markdown(
    """
<div class="hero">
<div class="hero-title">🩺 Early Disease Prediction</div>
<div class="hero-subtitle">Machine Learning Based Health Risk Prediction System</div>
<div class="hero-badge">🤖 Powered by Machine Learning • Random Forest</div>
</div>
""",
    unsafe_allow_html=True
)

# =========================================================
# DASHBOARD CARDS
# =========================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        """
<div class="dashboard-card">
<div class="card-icon">🩺</div>
<div class="card-title">Disease Modules</div>
<div class="card-value">4</div>
</div>
""",
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        """
<div class="dashboard-card">
<div class="card-icon">🤖</div>
<div class="card-title">ML Engine</div>
<div class="card-value">Random Forest</div>
</div>
""",
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        """
<div class="dashboard-card">
<div class="card-icon">📊</div>
<div class="card-title">Prediction Type</div>
<div class="card-value">Risk Analysis</div>
</div>
""",
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        """
<div class="dashboard-card">
<div class="card-icon">⚡</div>
<div class="card-title">Prediction</div>
<div class="card-value">Real-Time</div>
</div>
""",
        unsafe_allow_html=True
    )

# =========================================================
# HOW IT WORKS
# =========================================================

with st.expander("🔍 How does this system work?"):

    st.markdown(
        """
### Machine Learning Workflow

**1. Patient Data**

User enters relevant health parameters.

**2. Data Preprocessing**

Input values are transformed using the trained scaler.

**3. Machine Learning Model**

The trained Random Forest model analyzes the input features.

**4. Risk Prediction**

The system calculates the predicted disease risk.

**5. Feature Importance**

Important factors used by the model can be visualized.
"""
    )

# =========================================================
# DISEASE SELECTOR
# =========================================================

st.markdown(
    """
<div class="selector-card">
<div class="section-label">Prediction Module</div>
</div>
""",
    unsafe_allow_html=True
)

disease = st.selectbox(
    "Choose the disease you want to predict",
    [
        "Diabetes",
        "Heart Disease",
        "Liver Disease",
        "Kidney Disease"
    ]
)

# =========================================================
# RESULT FUNCTION
# =========================================================

def show_result(
    disease_name,
    prediction,
    probability,
    feature_file=None
):

    st.markdown("---")

    if prediction == 1:
        result_text = f"⚠️ Higher Predicted {disease_name} Risk"
    else:
        result_text = f"✅ Lower Predicted {disease_name} Risk"

    st.markdown(
        f"""
<div class="result-card">
<div class="result-title">{result_text}</div>
<div class="result-number">{probability * 100:.2f}%</div>
<div class="result-description">Model predicted risk probability</div>
</div>
""",
        unsafe_allow_html=True
    )

    st.progress(
        float(probability),
        text=f"Predicted Risk: {probability * 100:.2f}%"
    )

    if prediction == 1:
        st.warning(
            "The model has identified a higher predicted risk based on the entered values."
        )
    else:
        st.success(
            "The model has identified a lower predicted risk based on the entered values."
        )

    # Prediction history

    st.session_state.prediction_history.insert(
        0,
        {
            "Disease": disease_name,
            "Risk": f"{probability * 100:.2f}%",
            "Result": "Higher Risk" if prediction == 1 else "Lower Risk",
            "Time": datetime.now().strftime("%H:%M:%S")
        }
    )

    st.session_state.prediction_history = \
        st.session_state.prediction_history[:5]

    # Feature importance

    if feature_file:

        try:

            feature_importance = pd.read_csv(
                feature_file
            )

            feature_importance = feature_importance.sort_values(
                "Importance",
                ascending=False
            )

            st.markdown(
                f"""
<div class="section-title">
📊 {disease_name} Feature Importance
</div>
""",
                unsafe_allow_html=True
            )

            st.bar_chart(
                feature_importance.set_index("Feature")
            )

        except Exception:
            pass


# =========================================================
# DIABETES
# =========================================================

if disease == "Diabetes":

    st.markdown(
        '<div class="section-title">🩸 Diabetes Risk Prediction</div>',
        unsafe_allow_html=True
    )

    model = joblib.load(
        "model/random_forest_model.pkl"
    )

    scaler = joblib.load(
        "model/scaler.pkl"
    )

    col1, col2 = st.columns(2)

    with col1:

        pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0,
            max_value=20,
            value=1,
            help="Total number of pregnancies"
        )

        if pregnancies > 0:

            pregnancy_months = st.number_input(
                "Current Pregnancy Duration (Months)",
                min_value=0,
                max_value=9,
                value=0,
                help="Current pregnancy duration in months"
            )

        else:

            pregnancy_months = 0

        glucose = st.number_input(
            "Glucose",
            min_value=0,
            max_value=300,
            value=120
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0,
            max_value=200,
            value=70
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0,
            max_value=100,
            value=20
        )

    with col2:

        insulin = st.number_input(
            "Insulin",
            min_value=0,
            max_value=900,
            value=80
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=70.0,
            value=25.0
        )

        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30
        )

    if st.button(
        "🔮 Predict Diabetes Risk",
        use_container_width=True
    ):

        # Pregnancy months is displayed in the UI,
        # but is NOT passed to the existing ML model.

        input_data = pd.DataFrame(
            [[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                age
            ]],
            columns=[
                "Pregnancies",
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI",
                "DiabetesPedigreeFunction",
                "Age"
            ]
        )

        input_scaled = scaler.transform(
            input_data
        )

        prediction = model.predict(
            input_scaled
        )[0]

        probability = model.predict_proba(
            input_scaled
        )[0][1]

        show_result(
            "Diabetes",
            prediction,
            probability,
            "model/feature_importance.csv"
        )


# =========================================================
# HEART DISEASE
# =========================================================

elif disease == "Heart Disease":

    st.markdown(
        '<div class="section-title">❤️ Heart Disease Risk Prediction</div>',
        unsafe_allow_html=True
    )

    model = joblib.load(
        "model/heart/random_forest_model.pkl"
    )

    scaler = joblib.load(
        "model/heart/scaler.pkl"
    )

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50
        )

        sex = st.selectbox(
            "Sex",
            ["Male", "Female"]
        )

        cp = st.selectbox(
            "Chest Pain Type",
            [
                "Typical Angina",
                "Atypical Angina",
                "Non-anginal Pain",
                "Asymptomatic"
            ]
        )

        trestbps = st.number_input(
            "Resting Blood Pressure",
            min_value=50,
            max_value=250,
            value=120
        )

        chol = st.number_input(
            "Cholesterol",
            min_value=50,
            max_value=600,
            value=200
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            ["No", "Yes"]
        )

        restecg = st.selectbox(
            "Resting ECG",
            [
                "Normal",
                "ST-T Wave Abnormality",
                "Left Ventricular Hypertrophy"
            ]
        )

    with col2:

        thalach = st.number_input(
            "Maximum Heart Rate",
            min_value=50,
            max_value=250,
            value=150
        )

        exang = st.selectbox(
            "Exercise Induced Angina",
            ["No", "Yes"]
        )

        oldpeak = st.number_input(
            "ST Depression",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

        slope = st.selectbox(
            "Slope",
            [
                "Upsloping",
                "Flat",
                "Downsloping"
            ]
        )

        ca = st.number_input(
            "Number of Major Vessels",
            min_value=0,
            max_value=4,
            value=0
        )

        thal = st.selectbox(
            "Thalassemia",
            [
                "Normal",
                "Fixed Defect",
                "Reversible Defect"
            ]
        )

    sex_value = 1 if sex == "Male" else 0

    cp_value = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }[cp]

    fbs_value = 1 if fbs == "Yes" else 0

    restecg_value = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }[restecg]

    exang_value = 1 if exang == "Yes" else 0

    slope_value = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }[slope]

    thal_value = {
        "Normal": 1,
        "Fixed Defect": 2,
        "Reversible Defect": 3
    }[thal]

    if st.button(
        "🔮 Predict Heart Disease Risk",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            [[
                age,
                sex_value,
                cp_value,
                trestbps,
                chol,
                fbs_value,
                restecg_value,
                thalach,
                exang_value,
                oldpeak,
                slope_value,
                ca,
                thal_value
            ]],
            columns=[
                "age",
                "sex",
                "cp",
                "trestbps",
                "chol",
                "fbs",
                "restecg",
                "thalach",
                "exang",
                "oldpeak",
                "slope",
                "ca",
                "thal"
            ]
        )

        input_scaled = scaler.transform(
            input_data
        )

        prediction = model.predict(
            input_scaled
        )[0]

        probability = model.predict_proba(
            input_scaled
        )[0][1]

        show_result(
            "Heart Disease",
            prediction,
            probability,
            "model/heart/feature_importance.csv"
        )


# =========================================================
# LIVER DISEASE
# =========================================================

elif disease == "Liver Disease":

    st.markdown(
        '<div class="section-title">🧬 Liver Disease Risk Prediction</div>',
        unsafe_allow_html=True
    )

    model = joblib.load(
        "model/liver/random_forest_model.pkl"
    )

    scaler = joblib.load(
        "model/liver/scaler.pkl"
    )

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=40
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        total_bilirubin = st.number_input(
            "Total Bilirubin",
            min_value=0.0,
            max_value=50.0,
            value=1.0,
            step=0.1
        )

        direct_bilirubin = st.number_input(
            "Direct Bilirubin",
            min_value=0.0,
            max_value=20.0,
            value=0.3,
            step=0.1
        )

        alkaline_phosphotase = st.number_input(
            "Alkaline Phosphotase",
            min_value=0,
            max_value=2000,
            value=200
        )

    with col2:

        alamine_aminotransferase = st.number_input(
            "Alamine Aminotransferase",
            min_value=0,
            max_value=2000,
            value=30
        )

        aspartate_aminotransferase = st.number_input(
            "Aspartate Aminotransferase",
            min_value=0,
            max_value=2000,
            value=30
        )

        total_protiens = st.number_input(
            "Total Proteins",
            min_value=0.0,
            max_value=15.0,
            value=6.5,
            step=0.1
        )

        albumin = st.number_input(
            "Albumin",
            min_value=0.0,
            max_value=10.0,
            value=3.5,
            step=0.1
        )

        albumin_globulin_ratio = st.number_input(
            "Albumin and Globulin Ratio",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.1
        )

    gender_value = 1 if gender == "Male" else 0

    if st.button(
        "🔮 Predict Liver Disease Risk",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            [[
                age,
                gender_value,
                total_bilirubin,
                direct_bilirubin,
                alkaline_phosphotase,
                alamine_aminotransferase,
                aspartate_aminotransferase,
                total_protiens,
                albumin,
                albumin_globulin_ratio
            ]],
            columns=[
                "Age",
                "Gender",
                "Total_Bilirubin",
                "Direct_Bilirubin",
                "Alkaline_Phosphotase",
                "Alamine_Aminotransferase",
                "Aspartate_Aminotransferase",
                "Total_Protiens",
                "Albumin",
                "Albumin_and_Globulin_Ratio"
            ]
        )

        input_scaled = scaler.transform(
            input_data
        )

        prediction = model.predict(
            input_scaled
        )[0]

        probability = model.predict_proba(
            input_scaled
        )[0][1]

        show_result(
            "Liver Disease",
            prediction,
            probability,
            "model/liver/feature_importance.csv"
        )


# =========================================================
# KIDNEY DISEASE
# =========================================================

elif disease == "Kidney Disease":

    st.markdown(
        '<div class="section-title">🫘 Kidney Disease Risk Prediction</div>',
        unsafe_allow_html=True
    )

    model = joblib.load(
        "model/kidney/random_forest_model.pkl"
    )

    scaler = joblib.load(
        "model/kidney/scaler.pkl"
    )

    st.markdown(
        '<div class="section-label">Patient Information</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=45
        )

        bp = st.number_input(
            "Blood Pressure",
            min_value=50,
            max_value=200,
            value=80
        )

        sg = st.number_input(
            "Specific Gravity",
            min_value=1.000,
            max_value=1.030,
            value=1.020,
            step=0.005
        )

        al = st.number_input(
            "Albumin",
            min_value=0,
            max_value=5,
            value=0
        )

        su = st.number_input(
            "Sugar",
            min_value=0,
            max_value=5,
            value=0
        )

        bgr = st.number_input(
            "Blood Glucose Random",
            min_value=50,
            max_value=500,
            value=120
        )

        bu = st.number_input(
            "Blood Urea",
            min_value=1,
            max_value=300,
            value=40
        )

    with col2:

        sc = st.number_input(
            "Serum Creatinine",
            min_value=0.1,
            max_value=20.0,
            value=1.2,
            step=0.1
        )

        sod = st.number_input(
            "Sodium",
            min_value=80,
            max_value=200,
            value=140
        )

        pot = st.number_input(
            "Potassium",
            min_value=2.0,
            max_value=10.0,
            value=4.5,
            step=0.1
        )

        hemo = st.number_input(
            "Hemoglobin",
            min_value=3.0,
            max_value=20.0,
            value=13.0,
            step=0.1
        )

        pcv = st.number_input(
            "Packed Cell Volume",
            min_value=10,
            max_value=60,
            value=40
        )

        wc = st.number_input(
            "White Blood Cell Count",
            min_value=2000,
            max_value=30000,
            value=8000
        )

        rc = st.number_input(
            "Red Blood Cell Count",
            min_value=2.0,
            max_value=8.0,
            value=5.0,
            step=0.1
        )

    st.markdown(
        '<div class="section-label">Health Information</div>',
        unsafe_allow_html=True
    )

    col3, col4, col5 = st.columns(3)

    with col3:

        rbc = st.selectbox(
            "Red Blood Cells",
            ["Normal", "Abnormal"]
        )

        pc = st.selectbox(
            "Pus Cell",
            ["Normal", "Abnormal"]
        )

        pcc = st.selectbox(
            "Pus Cell Clumps",
            ["Not Present", "Present"]
        )

        ba = st.selectbox(
            "Bacteria",
            ["Not Present", "Present"]
        )

    with col4:

        htn = st.selectbox(
            "Hypertension",
            ["No", "Yes"]
        )

        dm = st.selectbox(
            "Diabetes Mellitus",
            ["No", "Yes"]
        )

        cad = st.selectbox(
            "Coronary Artery Disease",
            ["No", "Yes"]
        )

        appet = st.selectbox(
            "Appetite",
            ["Good", "Poor"]
        )

    with col5:

        pe = st.selectbox(
            "Pedal Edema",
            ["No", "Yes"]
        )

        ane = st.selectbox(
            "Anemia",
            ["No", "Yes"]
        )

    rbc_value = 0 if rbc == "Normal" else 1
    pc_value = 0 if pc == "Normal" else 1
    pcc_value = 0 if pcc == "Not Present" else 1
    ba_value = 0 if ba == "Not Present" else 1

    htn_value = 0 if htn == "No" else 1
    dm_value = 0 if dm == "No" else 1
    cad_value = 0 if cad == "No" else 1

    appet_value = 0 if appet == "Good" else 1

    pe_value = 0 if pe == "No" else 1
    ane_value = 0 if ane == "No" else 1

    if st.button(
        "🔮 Predict Kidney Disease Risk",
        use_container_width=True
    ):

        input_data = pd.DataFrame(
            [[
                age,
                bp,
                sg,
                al,
                su,
                rbc_value,
                pc_value,
                pcc_value,
                ba_value,
                bgr,
                bu,
                sc,
                sod,
                pot,
                hemo,
                pcv,
                wc,
                rc,
                htn_value,
                dm_value,
                cad_value,
                appet_value,
                pe_value,
                ane_value
            ]],
            columns=[
                "age",
                "bp",
                "sg",
                "al",
                "su",
                "rbc",
                "pc",
                "pcc",
                "ba",
                "bgr",
                "bu",
                "sc",
                "sod",
                "pot",
                "hemo",
                "pcv",
                "wc",
                "rc",
                "htn",
                "dm",
                "cad",
                "appet",
                "pe",
                "ane"
            ]
        )

        input_scaled = scaler.transform(
            input_data
        )

        prediction = model.predict(
            input_scaled
        )[0]

        probability = model.predict_proba(
            input_scaled
        )[0][1]

        show_result(
            "Kidney Disease",
            prediction,
            probability,
            "model/kidney/feature_importance.csv"
        )


# =========================================================
# RECENT PREDICTIONS
# =========================================================

if st.session_state.prediction_history:

    st.markdown(
        '<div class="section-title">🕘 Recent Predictions</div>',
        unsafe_allow_html=True
    )

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# PROJECT INFORMATION
# =========================================================

st.markdown(
    """
<div class="info-card">
<div class="info-title">🧠 Machine Learning Pipeline</div>

<div class="info-text">
The system uses trained machine learning models to analyze
health-related input features and generate a predicted risk
probability for the selected disease.
<br><br>
The application includes separate trained models for
Diabetes, Heart Disease, Liver Disease and Kidney Disease.
</div>

</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<div class="footer">
Early Disease Prediction
&nbsp; • &nbsp;
Machine Learning Health Risk Prediction System
&nbsp; • &nbsp;
B.Tech CSE Project
</div>
""",
    unsafe_allow_html=True
)