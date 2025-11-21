import streamlit as st
import pandas as pd
import joblib

# -----------------------------------------------------------
# CONFIG PAGE
# -----------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Prediction App")
st.write("""
Aplikasi ini memprediksi *kemungkinan diabetes* berdasarkan data medis pasien.
Model yang digunakan:
- Logistic Regression  
- Random Forest  
""")

# -----------------------------------------------------------
# LOAD MODEL & FEATURE NAMES
# -----------------------------------------------------------
@st.cache_resource
def load_models():
    logreg = joblib.load("model_logreg_diabetes.pkl")
    rf = joblib.load("model_rf_diabetes.pkl")
    features = joblib.load("feature_names_diabetes.pkl")
    return logreg, rf, features

logreg_model, rf_model, feature_names = load_models()

# -----------------------------------------------------------
# SIDEBAR PENGATURAN
# -----------------------------------------------------------
st.sidebar.header("🔧 Pilih Model")
chosen_model = st.sidebar.selectbox(
    "Model Prediksi",
    ("Logistic Regression", "Random Forest")
)

st.sidebar.info("Gunakan input form di bawah untuk memprediksi diabetes.")

# -----------------------------------------------------------
# INPUT FORM
# -----------------------------------------------------------
st.subheader("📝 Input Data Pasien")

inputs = {}

for col in feature_names:
    inputs[col] = st.number_input(
        col,
        min_value=0.0,
        max_value=300.0,
        value=0.0
    )

# Convert to DataFrame
X_new = pd.DataFrame([inputs])

# -----------------------------------------------------------
# PREDIKSI
# -----------------------------------------------------------
if st.button("🔍 Prediksi Diabetes"):
    if chosen_model == "Logistic Regression":
        model = logreg_model
    else:
        model = rf_model

    pred = model.predict(X_new)[0]
    prob = model.predict_proba(X_new)[0][1]

    if pred == 1:
        st.error(f"⚠ Hasil: *Berisiko Diabetes* (Confidence {prob*100:.2f}%)")
    else:
        st.success(f"✅ Hasil: *Tidak Diabetes* (Confidence {(1-prob)*100:.2f}%)")

    st.write("### 📌 Data yang Dimasukkan")
    st.table(X_new)
