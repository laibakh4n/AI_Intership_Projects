import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Electricity Cost Predictor",
    page_icon="⚡",
    layout="wide"
)

# ---------------- DARK MODE + UI SCALE ----------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }

    h2 {
        font-size: 2rem !important;
        color: #f1f3f5;
    }

    h3 {
        font-size: 1.6rem !important;
        color: #dee2e6;
    }

    label, p, span {
        font-size: 1.1rem !important;
        color: #ced4da;
    }

    div.stButton > button {
        font-size: 1.2rem;
        padding: 0.7rem 1.2rem;
        border-radius: 12px;
        background-color: #ff4b4b;
        color: white;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #ff6b6b;
        transform: scale(1.03);
    }

    .prediction-card {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(135deg, #1f2933, #111827);
        text-align: center;
        box-shadow: 0 0 25px rgba(255, 75, 75, 0.3);
    }
    /* ---- GLOBAL UI TEXT SCALE ---- */

     /* Subheaders like "Model Selection", "Input Features" */
     div[data-testid="stSubheader"] {
        font-size: 1.6rem !important;}

     /* Selectbox + Slider labels */
    div[data-testid="stWidgetLabel"] label {
        font-size: 1.25rem !important;
}

   /* Slider values (numbers on the right) */
   div[data-testid="stSlider"] span {
        font-size: 1.15rem !important;
}

    /* Selectbox selected value */
    div[data-testid="stSelectbox"] div {
        font-size: 1.15rem !important;
}

/* Button text */
div.stButton > button {
    font-size: 1.3rem !important;
}

/* General markdown text */
p {
    font-size: 1.2rem !important;
}




    
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <div style="
        text-align: center;
        font-size: 4.2rem;
        font-weight: 800;
        color: #f8f9fa;
        letter-spacing: 1.5px;
        line-height: 1.2;
        margin-top: 10px;
        margin-bottom: 5px;
    ">
        ⚡ Electricity Cost Predictor
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <p style="
        text-align: center;
        font-size: 1.3rem;
        color: #ced4da;
        margin-bottom: 30px;
    ">
        Compare multiple regression models in real time
    </p>
    """,
    unsafe_allow_html=True
)


# ---------------- LOAD MODELS ----------------
models = {
    "Linear Regression": joblib.load("linear_regression_model.pkl"),
    "Ridge Regression": joblib.load("ridge_model.pkl"),
    "Lasso Regression": joblib.load("lasso_model.pkl"),
    "ElasticNet Regression": joblib.load("elasticnet_model.pkl"),
    "Gradient Boosting": joblib.load("gradient_boosting_model.pkl"),
}

# ---------------- MODEL SELECTION ----------------
st.subheader("🔍 Model Selection")
model_name = st.selectbox(
    "Choose a regression model",
    list(models.keys())
)
model = models[model_name]

st.markdown("---")

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([2, 1])

# ---------------- INPUTS ----------------
with col1:
    st.subheader("🎚️ Input Features")

    site_area = st.slider("Site Area", 0.0, 10000.0, 500.0, 10.0)
    water_consumption = st.slider("Water Consumption", 0.0, 5000.0, 200.0, 10.0)
    utilisation_rate = st.slider("Utilisation Rate", 0.0, 1.0, 0.5, 0.01)
    resident_count = st.slider("Resident Count", 0, 1000, 50, 1)

    structure_type = st.selectbox(
        "Structure Type",
        ["Industrial", "Mixed-Use", "Residential"]
    )

    structure_industrial = 1 if structure_type == "Industrial" else 0
    structure_mixed = 1 if structure_type == "Mixed-Use" else 0
    structure_residential = 1 if structure_type == "Residential" else 0

# ---------------- PREDICTION CARD ----------------
with col2:
    st.subheader("📈 Prediction")

    if st.button("🔮 Predict Electricity Cost", use_container_width=True):
        input_data = np.array([
            site_area,
            water_consumption,
            utilisation_rate,
            resident_count,
            structure_industrial,
            structure_mixed,
            structure_residential
        ]).reshape(1, -1)

        prediction = model.predict(input_data)

        st.markdown(
            f"""
            <div class="prediction-card">
                <h2>💡 Estimated Cost</h2>
                <h1 style="color:#ff4b4b; font-size:3.2rem;">
                    {prediction[0]:.2f}
                </h1>
                <p style="font-size:1.2rem;">
                    <b>Model:</b> {model_name}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-size:1.05rem;'>📊 Machine Learning Project • Regression Models Comparison</p>",
    unsafe_allow_html=True
)
