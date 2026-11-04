import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Early Disease Prediction",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.stApp { background: #f6f9fc; }

.block-container {
    max-width: 1180px;
    padding-top: 2.2rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(135deg, #0f4c81 0%, #1677a8 55%, #20a6a8 100%);
    padding: 34px 38px;
    border-radius: 22px;
    margin-bottom: 24px;
    box-shadow: 0 12px 30px rgba(15, 76, 129, 0.16);
}

.hero-title {
    color: white;
    font-size: 40px;
    font-weight: 750;
    letter-spacing: -0.8px;
    margin: 0;
}

.hero-subtitle {
    color: rgba(255,255,255,0.88);
    font-size: 16px;
    margin-top: 7px;
}

.selector-card {
    background: white;
    padding: 18px 22px 8px 22px;
    border-radius: 16px;
    border: 1px solid #e5edf4;
    box-shadow: 0 5px 18px rgba(31, 52, 73, 0.06);
    margin-bottom: 24px;
}

.section-title {
    color: #12344d;
    font-size: 27px;
    font-weight: 700;
    margin: 12px 0 20px 0;
}

.section-label {
    color: #5b7183;
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border-color: #d9e4ec !important;
    background: white !important;
}

div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: #1b8ca6 !important;
    box-shadow: 0 0 0 1px #1b8ca6 !important;
}

label {
    color: #334e60 !important;
    font-weight: 550 !important;
}

.stButton > button {
    border: none;
    border-radius: 11px;
    min-height: 48px;
    background: linear-gradient(135deg, #0f6fa8, #149aa0);
    color: white;
    font-size: 16px;
    font-weight: 650;
    box-shadow: 0 7px 18px rgba(15, 111, 168, 0.18);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 22px rgba(15, 111, 168, 0.25);
}

div[data-testid="stAlert"] {
    border-radius: 14px;
    margin-top: 18px;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e3ebf1;
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: 0 5px 18px rgba(31, 52, 73, 0.06);
}

hr {
    border: none;
    border-top: 1px solid #e3ebf1;
    margin: 28px 0;
}

h2, h3 {
    color: #173b53 !important;
}

.footer {
    text-align: center;
    color: #8193a1;
    font-size: 13px;
    padding: 12px 0 4px;
}

@media (max-width: 768px) {
    .hero {
        padding: 25px 22px;
        border-radius: 18px;
    }
    .hero-title { font-size: 30px; }
    .hero-subtitle { font-size: 14px; }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '''
    <div class="hero">
        <div class="hero-title">🩺 Early Disease Prediction</div>
        <div class="hero-subtitle">
            Machine Learning Based Health Risk Prediction System
        </div>
    </div>
    ''',
    unsafe_allow_html=True
)


# ============================================================
# DISEASE SELECTION
# ============================================================

st.markdown('<div class="selector-card">', unsafe_allow_html=True)

st.markdown(
    '<div class="section-label">Prediction Module</div>',
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

st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# DIABETES
# ============================================================

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
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=1
        )

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

        input_data = pd.DataFrame([[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]], columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        probability = model.predict_proba(
            input_scaled
        )[0][1]

        st.markdown("---")

        if prediction == 1:

            st.error(
                "⚠️ Higher Predicted Diabetes Risk"
            )

        else:

            st.success(
                "✅ Lower Predicted Diabetes Risk"
            )

        st.metric(
            "Predicted Risk",
            f"{probability * 100:.2f}%"
        )


# ============================================================
# HEART DISEASE
# ============================================================

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

    # Encoding

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

        input_data = pd.DataFrame([[
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
        ]], columns=[
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
        ])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        probability = model.predict_proba(
            input_scaled
        )[0][1]

        st.markdown("---")

        if prediction == 1:

            st.error(
                "⚠️ Higher Predicted Heart Disease Risk"
            )

        else:

            st.success(
                "✅ Lower Predicted Heart Disease Risk"
            )

        st.metric(
            "Predicted Risk",
            f"{probability * 100:.2f}%"
        )


# ============================================================
# LIVER DISEASE
# ============================================================

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

        input_data = pd.DataFrame([[
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
        ]], columns=[
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
        ])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        probability = model.predict_proba(
            input_scaled
        )[0][1]

        st.markdown("---")

        if prediction == 1:

            st.error(
                "⚠️ Higher Predicted Liver Disease Risk"
            )

        else:

            st.success(
                "✅ Lower Predicted Liver Disease Risk"
            )

        st.metric(
            "Predicted Risk",
            f"{probability * 100:.2f}%"
        )


# ============================================================
# KIDNEY DISEASE
# ============================================================

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

    # --------------------------------------------------------
    # NUMERICAL INPUTS
    # --------------------------------------------------------

    st.subheader("Patient Information")

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

    # --------------------------------------------------------
    # CATEGORICAL INPUTS
    # --------------------------------------------------------

    st.subheader("Health Information")

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

    # --------------------------------------------------------
    # ENCODING
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Kidney Disease Risk",
        use_container_width=True
    ):

        input_data = pd.DataFrame([[
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
        ]], columns=[
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
        ])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        probability = model.predict_proba(
            input_scaled
        )[0][1]

        st.markdown("---")

        if prediction == 1:

            st.error(
                "⚠️ Higher Predicted Kidney Disease Risk"
            )

        else:

            st.success(
                "✅ Lower Predicted Kidney Disease Risk"
            )

        st.metric(
            "Predicted Risk",
            f"{probability * 100:.2f}%"
        )

        # Feature Importance

        st.subheader(
            "📊 Kidney Disease Feature Importance"
        )

        feature_importance = pd.read_csv(
            "model/kidney/feature_importance.csv"
        )

        feature_importance = feature_importance.sort_values(
            "Importance",
            ascending=False
        )

        st.bar_chart(
            feature_importance.set_index("Feature")
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '''
    <div class="footer">
        Early Disease Prediction&nbsp;&nbsp;•&nbsp;&nbsp;Machine Learning Health Risk Prediction System
    </div>
    ''',
    unsafe_allow_html=True
)