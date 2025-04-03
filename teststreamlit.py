import streamlit as st
import numpy as np
import pandas as pd
import joblib
import urllib.request
import os
import unicodedata


st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾")

@st.cache_resource
def load_model():
    model_path = "random_forest_model.joblib"
    if not os.path.exists(model_path):
        url = "https://www.dropbox.com/scl/fi/4bnty5g7s2ix3iw7p06t0/random_forest_model.joblib?rlkey=o8n7pzrj1g65cznolynqfkgad&st=off6994f&dl=1"
        urllib.request.urlretrieve(url, model_path)
    return joblib.load(model_path)  # returns dict with 'model' and 'columns'



model = load_model()


# === Load crop list (cleaned) ===
@st.cache_data
def load_crop_info():
    df = pd.read_csv("Crop_production.csv")

    def clean_crop_name(name):
        name = str(name)
        name = unicodedata.normalize("NFKD", name)
        name = name.strip().replace("\n", "").replace("\t", "")
        name = " ".join(name.split())
        name = name.title()
        return name

    df["Crop"] = df["Crop"].apply(clean_crop_name)
    df["Crop_Type"] = df["Crop_Type"].astype(str).str.strip().str.lower()

    # Keep unique crops only
    df = df.drop_duplicates(subset=["Crop"]).sort_values("Crop").reset_index(drop=True)
    return df[["Crop", "Crop_Type"]]

crop_info = load_crop_info()
crop_list = crop_info["Crop"].tolist()

# === UI Layout ===
st.title("🌾 Crop Yield Predictor")
st.markdown("Enter the details below to estimate the expected **crop yield in tons**.")

# Crop Type and Crop selection logic
st.subheader("🌾 Crop Selection")

# Get unique crop types
crop_types = sorted(crop_info["Crop_Type"].unique())

# Choose how to filter
selection_mode = st.radio("How would you like to select a crop?", ["Select Crop Type First", "Select Crop First"])

if selection_mode == "Select Crop Type First":
    selected_type = st.selectbox("Select Crop Type", crop_types)
    filtered_crops = crop_info[crop_info["Crop_Type"] == selected_type]["Crop"].tolist()
    selected_crop = st.selectbox("Select Crop", filtered_crops)
    st.caption(f"📌 Selected Crop Type: **{selected_type.capitalize()}**")

else:  # Select Crop First
    selected_crop = st.selectbox("Select Crop", crop_list)
    selected_type = crop_info[crop_info["Crop"] == selected_crop]["Crop_Type"].values[0]
    st.caption(f"📌 Detected Crop Type: **{selected_type.capitalize()}**")


st.subheader("🌿 Soil & Climate Conditions")

def is_input_valid(N, P, K, pH, rainfall, temperature, area):
    issues = []

    if temperature < 10:
        issues.append("🌡️ Temperature is too low for any crop to grow.")
    elif temperature > 45:
        issues.append("🌡️ Temperature is extremely high and may kill crops.")

    if pH < 4.5 or pH > 9:
        issues.append("🧪 Soil pH is beyond survivable range for most crops.")

    if rainfall < 50:
        issues.append("☔ Rainfall is far too low — drought likely.")
    elif rainfall > 2500:
        issues.append("☔ Rainfall is excessive — likely to cause flooding.")

    if N < 10 or P < 5 or K < 5:
        issues.append("🧬 One or more nutrients (N, P, K) are critically low.")

    if area < 0.1:
        issues.append("📏 Area is too small to produce measurable yield.")

    return issues

# Input fields with validations
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

# Predict button
if st.button("Predict Crop Yield"):
    issues = is_input_valid(N, P, K, pH, rainfall, temperature, area)

    if issues:
        st.error("❌ One or more values are not realistic for crop production:")
        for i in issues:
            st.markdown(f"- {i}")
    else:
        # [Keep your existing prediction code below]
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
