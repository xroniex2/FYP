import streamlit as st
import pandas as pd
import numpy as np
import urllib.request
import os
import joblib

# === Set up multi-page ===
st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾")

# === Sidebar navigation ===
page = st.sidebar.selectbox("Navigate", ["Prediction", "About"])

# === Load model ===
@st.cache_resource
def load_model():
    model_path = "random_forest_model.joblib"
    if not os.path.exists(model_path):
        url = "https://www.dropbox.com/scl/fi/4bnty5g7s2ix3iw7p06t0/random_forest_model.joblib?rlkey=o8n7pzrj1g65cznolynqfkgad&st=off6994f&dl=1"
        urllib.request.urlretrieve(url, model_path)
    return joblib.load(model_path)

model = load_model()

# === Load cleaned crop info ===
@st.cache_data
def load_crop_info():
    df = pd.read_csv("Crop_production.csv")
    df = df.drop_duplicates(subset=["Crop"]).sort_values("Crop").reset_index(drop=True)
    return df[["Crop", "Crop_Type"]]

crop_info = load_crop_info()
crop_list = crop_info["Crop"].tolist()

# === About Page ===
if page == "About":
    st.title("🌾 About This Application")
    st.markdown("""
        This application predicts crop production (in tons) based on user input for environmental and soil conditions.

        Built using a machine learning model (Random Forest) trained on a cleaned dataset, the app helps users estimate expected crop yield by inputting realistic values for nitrogen, phosphorus, potassium, pH, rainfall, temperature, and area.

        Please ensure the values you enter are within recommended agricultural ranges for reliable results.
    """)

# === Prediction Page ===
elif page == "Prediction":
    st.title("🌿 Crop Yield Predictor")
    st.markdown("Enter the details below to estimate the expected **crop yield in tons**.")

    selected_crop = st.selectbox("Select a Crop", crop_list)
    crop_type = crop_info[crop_info["Crop"] == selected_crop]["Crop_Type"].values[0]
    st.caption(f"📌 Crop Type: **{crop_type.capitalize()}**")

    st.subheader("🌿 Soil & Climate Conditions")

    N = st.number_input("Nitrogen (N) [Recommended: 20–150 kg/ha]", min_value=0, max_value=200, value=70)
    if N < 20:
        st.warning("🔸 Nitrogen is too low. This may lead to poor plant growth.")
    elif N > 150:
        st.warning("🔸 Excess nitrogen can cause excessive foliage and poor yield.")

    P = st.number_input("Phosphorus (P) [Recommended: 15–100 kg/ha]", min_value=0, max_value=150, value=40)
    if P < 15:
        st.warning("🔸 Low phosphorus may cause stunted root growth.")
    elif P > 100:
        st.warning("🔸 High phosphorus can lead to nutrient imbalance in soil.")

    K = st.number_input("Potassium (K) [Recommended: 20–120 kg/ha]", min_value=0, max_value=200, value=40)
    if K < 20:
        st.warning("🔸 Potassium deficiency affects water regulation and crop resilience.")
    elif K > 120:
        st.warning("🔸 Too much potassium may block other nutrients like calcium and magnesium.")

    pH = st.number_input("Soil pH [Recommended: 5.5 – 7.5]", min_value=3.0, max_value=9.0, value=6.5, step=0.01)
    if pH < 5.5:
        st.warning("🔸 Soil is too acidic. Consider liming to raise pH.")
    elif pH > 7.5:
        st.warning("🔸 Soil is too alkaline. Apply sulfur or organic matter to lower pH.")

    rainfall = st.number_input("Rainfall (mm) [Recommended: 300–1500 mm]", min_value=0.0, max_value=3500.0, value=700.0)
    if rainfall < 300:
        st.warning("🔸 Low rainfall may not support healthy crop growth.")
    elif rainfall > 1500:
        st.warning("🔸 High rainfall can cause waterlogging or nutrient leaching.")

    temperature = st.number_input("Temperature (°C) [Recommended: 15–35°C]", min_value=0.0, max_value=50.0, value=26.0)
    if temperature < 15:
        st.warning("🔸 Temperature is too low for optimal crop development.")
    elif temperature > 35:
        st.warning("🔸 High temperature may reduce flowering and grain filling.")

    area = st.number_input("Area in hectares", min_value=0.1, max_value=100000.0, value=1000.0)
    if area < 1.0:
        st.info("ℹ️ Small-scale farming area. Ensure efficient land use.")
    elif area > 10000:
        st.info("ℹ️ Large-scale operation detected. Consider precision agriculture tools.")

    if st.button("Predict Crop Yield"):
        crop_dummy = pd.get_dummies(pd.Series(selected_crop), prefix="Crop")
        dummy_template = pd.get_dummies(crop_info["Crop"], prefix="Crop")
        crop_vector = pd.DataFrame(columns=dummy_template.columns).fillna(0)
        for col in crop_dummy.columns:
            if col in crop_vector.columns:
                crop_vector.loc[0, col] = 1

        input_features = np.concatenate((
            np.array([[N, P, K, pH, rainfall, temperature, area]]),
            crop_vector.to_numpy()
        ), axis=1)

        prediction = model.predict(input_features)
        st.success(f"🌾 Estimated Crop Yield: **{prediction[0]:,.2f} tons**")
