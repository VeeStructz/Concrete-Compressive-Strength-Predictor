# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io
from io import BytesIO

# Set page configuration
st.set_page_config(page_title="Concrete Compressive Strength Predictor", page_icon="🧱", layout="wide")

# Load models and scaler
catboost_model = joblib.load('catboost_model.pkl')
xgboost_model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define model dictionary
model_dict = {
    "CatBoost (Tuned)": catboost_model,
    "XGBoost (Tuned)": xgboost_model,
}

# App title
st.title("🧱 Concrete Compressive Strength Predictor")
st.markdown("Enhancing Concrete Strength Prediction using Machine Learning")

# Sidebar: model selection
st.sidebar.header("📊 Model Selector")
selected_model_name = st.sidebar.selectbox("Choose a Model", list(model_dict.keys()))
selected_model = model_dict[selected_model_name]

# Sidebar: input parameters
st.sidebar.header("🧪 Mix Properties")
cement = st.sidebar.slider('Cement (kg/m³)', 100, 600, 300)
slag = st.sidebar.slider('Slag (kg/m³)', 0, 300, 0)
fly_ash = st.sidebar.slider('Fly Ash (kg/m³)', 0, 300, 0)
water = st.sidebar.slider('Water (kg/m³)', 100, 300, 180)
superplasticizer = st.sidebar.slider('Superplasticizer (kg/m³)', 0, 30, 5)
coarse_agg = st.sidebar.slider('Coarse Aggregate (kg/m³)', 800, 1200, 1000)
fine_agg = st.sidebar.slider('Fine Aggregate (kg/m³)', 600, 1000, 800)
age = st.sidebar.slider('Curing Age (days)', 1, 365, 28)

# Derived parameter
w_c_ratio = round(water / cement, 3) if cement != 0 else 0

# Combine input
input_data = pd.DataFrame([{
    'cement': cement,
    'slag': slag,
    'fly_ash': fly_ash,
    'water': water,
    'superplasticizer': superplasticizer,
    'coarse_aggregate': coarse_agg,
    'fine_aggregate': fine_agg,
    'age': age,
    'water_cement_ratio': w_c_ratio
}])

# Scale input
scaled_input = scaler.transform(input_data)

# Make prediction
predicted_strength = selected_model.predict(scaled_input)[0]

# Display Results
st.subheader("🔍 Prediction Result")
st.success(f"**Predicted Compressive Strength:** {predicted_strength:.2f} MPa")

# Show heatmap (feature importance)
st.subheader("📌 Feature Importance")
try:
    importances = selected_model.feature_importances_
    features = input_data.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax1, palette='viridis')
    ax1.set_title("Feature Importance")
    st.pyplot(fig1)
except:
    st.info("Feature importance not available for the selected model.")

# Stress-Strain Curve (Simulated)
st.subheader("📈 Stress-Strain Curve")

# Simulate curve for selected model
def generate_stress_strain_curve(strength):
    strain = np.linspace(0, 0.005, 100)
    stress = strength * (strain / 0.002) * np.exp(1 - (strain / 0.002))
    return strain, stress

strain, stress = generate_stress_strain_curve(predicted_strength)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(strain, stress, color='crimson', linewidth=2)
ax2.set_xlabel("Strain")
ax2.set_ylabel("Stress (MPa)")
ax2.set_title(f"Stress-Strain Curve — {selected_model_name}")
st.pyplot(fig2)

# Download Excel
st.subheader("📥 Export Results")

result_df = input_data.copy()
result_df["Predicted_Strength (MPa)"] = predicted_strength

excel_buffer = BytesIO()
with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
    result_df.to_excel(writer, index=False, sheet_name="Prediction")
st.download_button(
    label="⬇️ Download Excel",
    data=excel_buffer.getvalue(),
    file_name="concrete_prediction.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Footer
st.markdown("---")
st.markdown("|| Project by Ranti-Owoeye Victor ||")
