import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Load models and scaler
catboost_model = joblib.load('catboost_model.pkl')
xgboost_model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')

# App title and styling
st.set_page_config(page_title="Concrete Strength Predictor", page_icon="🧱")
st.markdown("<h1 style='text-align: left; color: #4CAF50;'>Concrete Compressive Strength Predictor</h1>", unsafe_allow_html=True)
st.markdown("""This app predicts the **compressive strength** of concrete based on its mix composition. Adjust the values below to simulate different mix designs.""")
st.markdown("---")

# Sidebar model selector
model_dict = {
    "CatBoost Model": catboost_model,
    "XGBoost Model": xgboost_model
}
selected_model_name = st.sidebar.selectbox("🔎 Choose a Model", list(model_dict.keys()))
selected_model = model_dict[selected_model_name]

# --- Input Section ---
st.subheader("Input Concrete Mix Details")

cement = st.number_input("Cement (kg/m³)", 0.0, 600.0, 300.0, step=10.0)
slag = st.number_input("Blast Furnace Slag (kg/m³)", 0.0, 300.0, step=10.0)
fly_ash = st.number_input("Fly Ash (kg/m³)", 0.0, 200.0, step=10.0)
water = st.number_input("Water (kg/m³)", 100.0, 300.0, step=5.0)
superplasticizer = st.number_input("Superplasticizer (kg/m³)", 0.0, 30.0, step=1.0)
coarse_agg = st.number_input("Coarse Aggregate (kg/m³)", 800.0, 1200.0, step=10.0)
fine_agg = st.number_input("Fine Aggregate (kg/m³)", 600.0, 1000.0, step=10.0)
age = st.number_input("Age (days)", 1, 365, step=1)
Water_Cement_Ratio = st.number_input("Water_Cement_Ratio")

# Collect input into dictionary
user_input = {
    'Cement': cement,
    'Blast Furnace Slag': slag,
    'Fly Ash': fly_ash,
    'Water': water,
    'Superplasticizer': superplasticizer,
    'Coarse Aggregate': coarse_agg,
    'Fine Aggregate': fine_agg,
    'Age': age
}

# Add Water-Cement Ratio
user_input["Water_Cement_Ratio"] = user_input["Water"] / user_input["Cement"]

# Convert to DataFrame with proper order
input_df = pd.DataFrame([user_input])[[
    "Cement", "Blast Furnace Slag", "Fly Ash", "Water",
    "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age",
    "Water_Cement_Ratio"
]]

feature_names = list(user_input.keys())
input_df = pd.DataFrame([user_input], columns=feature_names)
input_scaled = scaler.transform(input_df)

# --- Prediction ---
if st.button("Predict Compressive Strength"):
    prediction = selected_model.predict(input_scaled)[0]
    st.success(f"Predicted Compressive Strength: **{prediction:.2f} MPa**")

    # --- Stress-Strain Curve ---
    st.subheader("📈 Stress–Strain Curve (Dynamic)")

    # Strain range (0% to 0.5%)
    strain = np.linspace(0, 0.005, 100)

    # Dynamic modulus based on prediction
    E = 5000 * np.sqrt(prediction)  # MPa

    # Simplified parabolic curve
    stress = E * strain * (1 - strain / 0.0035)

    # No negative stress values
    stress = np.maximum(stress, 0)

    # Plot
    fig, ax = plt.subplots()
    ax.plot(strain * 1000, stress, color='orange', linewidth=2)
    ax.set_title('Stress–Strain Curve')
    ax.set_xlabel('Strain')
    ax.set_ylabel('Stress')
    ax.grid(True)
    st.pyplot(fig)



    # --- Export Section ---
    st.subheader("Export Results")

    export_df = input_df.copy()
    export_df['Predicted Strength (MPa)'] = prediction

    # Export to Excel
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        export_df.to_excel(writer, index=False, sheet_name='Prediction')
    excel_data = excel_buffer.getvalue()
    st.download_button(
    label="Download",
    data=excel_data,
    file_name="Concrete_Prediction_Results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Footer
st.markdown("---")

st.markdown("<div style='text-align: center;'>© 2025 Ranti-Owoeye Victor | Powered by Machine Learning & Streamlit 🚀</div>", unsafe_allow_html=True)
