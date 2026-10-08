"""
Heart Disease Risk Screening - Streamlit app
Loads two models trained on the CDC BRFSS "Personal Key Indicators of Heart Disease" data:
  - Logistic Regression  (models/heart_logreg_model.pkl)
  - LightGBM             (models/heart_lgbm_model.pkl)
Educational project. NOT a medical device and NOT medical advice.
"""
import pickle
import warnings
from pathlib import Path

import pandas as pd
import sklearn
import streamlit as st

# LightGBM prints a harmless "X does not have valid feature names" warning; hide it
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Heart Disease Risk Screening", page_icon="❤️", layout="wide")

# ----------------------------------------------------------------------------
# Model files (looked up in ./models first, then in the repo root)
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_FILES = {
    "Logistic Regression": "heart_logreg_model.pkl",
    "LightGBM": "heart_lgbm_model.pkl",
}


def find_model_file(filename):
    for folder in (BASE_DIR / "models", BASE_DIR):
        path = folder / filename
        if path.exists():
            return path
    return None


@st.cache_resource(show_spinner="Loading model...")
def load_artifact(path_str):
    # Only load pickle files that YOU created. Never unpickle files from strangers.
    with open(path_str, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------
# Allowed values (exactly the spellings used in the training data)
# ----------------------------------------------------------------------------
AGE_CATEGORIES = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54",
                  "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"]
GEN_HEALTH = ["Poor", "Fair", "Good", "Very good", "Excellent"]
RACES = ["White", "Black", "Asian", "Hispanic", "American Indian/Alaskan Native", "Other"]
DIABETIC = ["No", "No, borderline diabetes", "Yes", "Yes (during pregnancy)"]
YES_NO = ["No", "Yes"]


def predict(artifact, row):
    """Return (risk_score, is_higher_risk) for one patient dictionary."""
    frame = pd.DataFrame([row])[artifact["feature_columns"]]   # same column order as training
    score = float(artifact["pipeline"].predict_proba(frame)[0, 1])
    return score, score >= float(artifact["threshold"])


# ----------------------------------------------------------------------------
# Sidebar: choose model
# ----------------------------------------------------------------------------
available = {name: find_model_file(fn) for name, fn in MODEL_FILES.items()}
loaded = {}
for name, path in available.items():
    if path is not None:
        try:
            loaded[name] = load_artifact(str(path))
        except Exception as exc:                       # wrong library version, corrupt file, ...
            st.sidebar.error(f"Could not load {name}: {exc}")

st.sidebar.title("⚙️ Model")
if not loaded:
    st.title("❤️ Heart Disease Risk Screening")
    st.error(
        "No model files found. Put `heart_logreg_model.pkl` and/or `heart_lgbm_model.pkl` "
        "inside the `models/` folder of your repository and redeploy.")
    st.stop()

options = list(loaded.keys())
if len(loaded) == 2:
    options.append("Compare both")
choice = st.sidebar.radio("Which model?", options)

missing = [name for name in MODEL_FILES if name not in loaded]
if missing:
    st.sidebar.warning("Not available: " + ", ".join(missing))

for name, art in loaded.items():
    saved = art.get("versions", {}).get("sklearn")
    if saved and saved != sklearn.__version__:
        st.sidebar.warning(
            f"{name} was saved with scikit-learn {saved}, but this app runs {sklearn.__version__}. "
            "Results may be wrong. Pin the same version in requirements.txt.")

st.sidebar.markdown("---")
st.sidebar.caption("Educational project built on a self-reported CDC survey. "
                   "Not a medical device and not medical advice.")

# ----------------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------------
st.title("❤️ Heart Disease Risk Screening")
st.caption("Enter your information and the model returns a heart-disease **risk score**. "
           "This is a learning project, not a diagnosis.")

tab_predict, tab_about = st.tabs(["🩺 Predict", "ℹ️ About the models"])

with tab_predict:
    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("About you")
            sex = st.selectbox("Sex", ["Female", "Male"])
            age = st.selectbox("Age group", AGE_CATEGORIES, index=7)
            race = st.selectbox("Race / ethnicity", RACES)
            height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=170.0, step=1.0)
            weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=75.0, step=1.0)
            sleep = st.slider("Average sleep per night (hours)", 1, 24, 7)

        with c2:
            st.subheader("Health status")
            gen_health = st.select_slider("General health", options=GEN_HEALTH, value="Good")
            phys_days = st.slider("Days of poor PHYSICAL health (last 30 days)", 0, 30, 0)
            ment_days = st.slider("Days of poor MENTAL health (last 30 days)", 0, 30, 0)
            diabetic = st.selectbox("Diabetes", DIABETIC)
            diff_walking = st.radio("Serious difficulty walking or climbing stairs?", YES_NO, horizontal=True)
            kidney = st.radio("Kidney disease?", YES_NO, horizontal=True)

        with c3:
            st.subheader("History and lifestyle")
            stroke = st.radio("Ever had a stroke?", YES_NO, horizontal=True)
            asthma = st.radio("Asthma?", YES_NO, horizontal=True)
            skin = st.radio("Skin cancer?", YES_NO, horizontal=True)
            smoking = st.radio("Smoked at least 100 cigarettes in your life?", YES_NO, horizontal=True)
            alcohol = st.radio("Heavy alcohol drinking?", YES_NO, horizontal=True)
            activity = st.radio("Physical activity in the last 30 days (outside work)?", YES_NO, index=1, horizontal=True)

        submitted = st.form_submit_button("Calculate risk", type="primary", use_container_width=True)

    bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
    st.caption(f"Calculated BMI: **{bmi}**")

    if submitted:
        patient = {
            "BMI": bmi, "Smoking": smoking, "AlcoholDrinking": alcohol, "Stroke": stroke,
            "PhysicalHealth": phys_days, "MentalHealth": ment_days, "DiffWalking": diff_walking,
            "Sex": sex, "AgeCategory": age, "Race": race, "Diabetic": diabetic,
            "PhysicalActivity": activity, "GenHealth": gen_health, "SleepTime": sleep,
            "Asthma": asthma, "KidneyDisease": kidney, "SkinCancer": skin,
        }
        to_show = list(loaded.keys()) if choice == "Compare both" else [choice]

        st.markdown("### Result")
        cols = st.columns(len(to_show))
        results = {}
        for col, name in zip(cols, to_show):
            art = loaded[name]
            try:
                score, higher = predict(art, patient)
            except Exception as exc:
                col.error(f"{name} failed: {exc}")
                continue
            results[name] = higher
            with col:
                st.markdown(f"**{name}**")
                st.metric("Risk score", f"{score:.1%}")
                st.progress(min(max(score, 0.0), 1.0))
                st.caption(f"Decision threshold: {float(art['threshold']):.1%}")
                if higher:
                    st.error("⚠️ **Higher risk** - the score is at or above the threshold.")
                else:
                    st.success("✅ **Lower risk** - the score is below the threshold.")

        if len(results) == 2 and len(set(results.values())) == 2:
            st.info("The two models disagree for this person. Treat the result as uncertain.")

        st.warning(
            "**Please read:** this is a screening-style estimate from a self-reported survey, not a diagnosis. "
            "The score is NOT a literal probability (the models were trained to catch most sick people, "
            "so many healthy people also score high). Talk to a doctor about any health concern.")

with tab_about:
    st.subheader("Data")
    st.write("CDC BRFSS 2020 survey, *Personal Key Indicators of Heart Disease* (Kaggle). "
             "About 320,000 respondents, 9% with heart disease. After removing duplicates, "
             "the models were trained on about 300,000 rows.")

    st.subheader("Models")
    rows = []
    for name, art in loaded.items():
        rows.append({
            "Model": name,
            "Test ROC-AUC": round(float(art.get("test_roc_auc", float("nan"))), 4),
            "Decision threshold": round(float(art["threshold"]), 3),
            "scikit-learn used for training": art.get("versions", {}).get("sklearn", "?"),
        })
    st.table(pd.DataFrame(rows))
    st.markdown(
        "- **Logistic Regression:** simple and explainable; ranks people nearly as well as LightGBM.\n"
        "- **LightGBM:** gradient-boosted trees; slightly higher ROC-AUC.\n"
        "- **ROC-AUC** is the chance the model gives a sick person a higher score than a healthy person "
        "(0.5 = coin flip, 1.0 = perfect).\n"
        "- **Threshold:** the score above which a person is labelled *higher risk*. It was tuned to favour "
        "catching sick people over avoiding false alarms.")

    st.subheader("Limits")
    st.markdown(
        "- The data is **self-reported**; it has no blood pressure, cholesterol, ECG or lab results.\n"
        "- Even good models on this data reach only about 0.84 ROC-AUC, so mistakes are common.\n"
        "- It describes **associations**, not causes.\n"
        "- Not for diagnosis or treatment decisions.")
