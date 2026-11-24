
import streamlit as st
import pandas as pd
import joblib
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #101827;
    color: #ffffff;
}

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: #38bdf8;
    padding: 15px;
}

.sub-title {
    text-align: center;
    font-size: 17px;
    color: #cbd5e1;
    margin-bottom: 25px;
}

.section-title {
    color: #38bdf8;
    font-size: 24px;
    font-weight: bold;
}

div.stButton > button {
    background-color: #0284c7;
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    border: none;
}

div.stButton > button:hover {
    background-color: #0369a1;
    color: white;
}

.result-card {
    padding: 20px;
    border-radius: 12px;
    background-color: #1e293b;
    text-align: center;
    margin-top: 15px;
}

.footer {
    text-align: center;
    color: #94a3b8;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MACHINE LEARNING MODEL
# ==========================================

@st.cache_resource
def load_model():
    model_path = "models/best_heart_model.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)


try:
    model = load_model()
except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">❤️ Heart Disease Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'AI Powered Heart Health Risk Assessment'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("❤️ Heart Health")

    st.write("Machine Learning Based Prediction")

    st.divider()

    st.subheader("Model Information")

    st.success("Random Forest Model Loaded")

    st.write("Model: Random Forest")
    st.write("Training Accuracy: 90.16%")
    st.write("Training F1 Score: 90.00%")

    st.divider()

    st.info(
        "This application is developed "
        "for educational purposes."
    )


# ==========================================
# PATIENT INFORMATION
# ==========================================

st.markdown(
    '<div class="section-title">👤 Patient Information</div>',
    unsafe_allow_html=True
)

st.write("Enter the patient's health parameters below.")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=45
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    sex = 1 if gender == "Male" else 0

    cp = st.selectbox(
        "Chest Pain Type",
        [1, 2, 3, 4]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=80,
        max_value=250,
        value=120
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=100,
        max_value=600,
        value=200
    )


with col2:

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120",
        [0, 1]
    )

    restecg = st.selectbox(
        "Resting ECG",
        [0, 1, 2]
    )

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=60,
        max_value=250,
        value=150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        [0, 1]
    )


with col3:

    oldpeak = st.number_input(
        "ST Depression",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    slope = st.selectbox(
        "ST Slope",
        [1, 2, 3]
    )

    ca = st.selectbox(
        "Major Vessels",
        [0, 1, 2, 3]
    )

    thal = st.selectbox(
        "Thalassemia",
        [3, 6, 7]
    )


st.divider()

# ==========================================
# PREDICTION SECTION
# ==========================================

st.markdown(
    '<div class="section-title">🔍 Heart Disease Prediction</div>',
    unsafe_allow_html=True
)

if st.button(
    "Predict Heart Disease",
    use_container_width=True
):

    # Create Input DataFrame

    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    try:

        # Model Prediction

        prediction = model.predict(input_data)[0]

        st.divider()

        st.subheader("📊 Prediction Result")

        if prediction == 1:

            st.error(
                "⚠️ Model indicates a possible "
                "heart disease risk."
            )

        else:

            st.success(
                "✅ Model indicates no heart "
                "disease detected."
            )

        # Probability Calculation

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]

            classes = list(model.classes_)

            positive_index = classes.index(1)

            probability = probabilities[positive_index]

            st.subheader("Risk Probability")

            st.metric(
                "Model Estimated Positive Probability",
                f"{probability * 100:.2f}%"
            )

            st.progress(float(probability))

        # Display Patient Data

        with st.expander("View Patient Information"):

            st.dataframe(
                input_data,
                use_container_width=True
            )

        st.warning(
            "Important: This is an educational "
            "machine learning prediction, not a "
            "medical diagnosis. Consult a qualified "
            "healthcare professional for medical advice."
        )

    except Exception as e:

        st.error(f"Prediction Error: {e}")


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    '<div class="footer">'
    '❤️ Heart Disease Prediction System<br>'
    'Developed using Python, Machine Learning & Streamlit'
    '</div>',
    unsafe_allow_html=True
)
