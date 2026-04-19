import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="HealthGuard AI", layout="centered", initial_sidebar_state="auto", page_icon="🛡️")
st.title("🛡️ HealthGuard AI - Health Risk Detection from Industrial Gas Exposure")

@st.cache_resource(show_spinner=True)
def load_model():
    with open("model.pkl", "rb") as f:
        model_data = pickle.load(f)
    return model_data

model_data = load_model()
feature_columns = model_data['feature_columns']
encoders = model_data.get('encoders', {})

st.header("Enter Exposure & Gas Details")

pm25 = st.number_input("PM2.5 (μg/m³)", min_value=0.0, max_value=220.0, value=25.0)
exposure_hours = st.number_input("Exposure Hours/Day", min_value=1, max_value=12, value=8)
exposure_years = st.number_input("Exposure Years", min_value=0, max_value=30, value=5)

industry_type = st.selectbox("Industry Type", encoders['Industry_Type'].classes_)
unit = st.selectbox("Unit", encoders['Unit'].classes_)
so2 = st.number_input("SO₂ (ppm)", min_value=0.0, max_value=75.0, value=20.0, step=1.0)
nox = st.number_input("NOₓ (ppm)", min_value=0.0, max_value=55.0, value=30.0, step=1.0)
co = st.number_input("CO (ppm)", min_value=0.0, max_value=120.0, value=5.0, step=1.0)
h2s = st.number_input("H₂S (ppm)", min_value=0.0, max_value=20.0, value=1.0, step=1.0)
voc = st.number_input("VOC (ppm)", min_value=0.0, max_value=200.0, value=10.0, step=1.0)
benzene = st.number_input("Benzene (ppm)", min_value=0.0, max_value=10.0, value=1.0, step=1.0)
toluene = st.number_input("Toluene (ppm)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
pm25 = st.number_input("PM2.5 (μg/m³)", min_value=0.0, max_value=220.0, value=25.0, step=1.0)
exposure_hours = st.number_input("Exposure Hours/Day", min_value=1, max_value=12, value=8, step=1)
exposure_years = st.number_input("Exposure Years", min_value=0, max_value=30, value=5, step=1)

inputs = {
    'Industry_Type': industry_type,
    'Unit': unit,
    'SO2_ppm': so2,
    'NOx_ppm': nox,
    'CO_ppm': co,
    'H2S_ppm': h2s,
    'VOC_ppm': voc,
    'Benzene_ppm': benzene,
    'Toluene_ppm': toluene,
    'PM2_5_ugm3': pm25,
    'Exposure_Hours_Day': exposure_hours,
    'Exposure_Years': exposure_years
}

def predict_disease(model_data, inputs):
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoder = model_data['label_encoder']
    feature_columns = model_data['feature_columns']
    encoders = model_data.get('encoders', {})

    # Define which columns are categorical and which are numeric (must match training)
    categorical_cols = list(encoders.keys())
    numeric_cols = [col for col in feature_columns if col not in categorical_cols]

    # Prepare input row in correct order
    input_row = []
    cat_row = []
    num_row = []
    for col in feature_columns:
        val = inputs[col]
        if col in categorical_cols:
            val_enc = encoders[col].transform([val])[0]
            cat_row.append(val_enc)
            input_row.append(val_enc)
        else:
            num_row.append(val)
            input_row.append(val)

    # Scale only numeric features
    num_row_np = np.array([num_row])
    num_scaled = scaler.transform(num_row_np)

    # Reconstruct the final input in the same order as feature_columns
    final_input = []
    num_idx = 0
    cat_idx = 0
    for col in feature_columns:
        if col in categorical_cols:
            final_input.append(cat_row[cat_idx])
            cat_idx += 1
        else:
            final_input.append(num_scaled[0, num_idx])
            num_idx += 1

    input_array = np.array([final_input])
    prediction = model.predict(input_array)[0]
    predicted_disease = label_encoder.inverse_transform([prediction])[0]
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(input_array)[0]
        confidence = probabilities[prediction]
        all_probs = {label_encoder.classes_[i]: prob for i, prob in enumerate(probabilities)}
    else:
        confidence = None
        all_probs = None
    return predicted_disease, confidence, all_probs

if st.button("ANALYZE HEALTH RISK"):
    with st.spinner("Analyzing health risk..."):
        predicted_disease, confidence, all_probs = predict_disease(model_data, inputs)
    st.success(f"Predicted Health Condition: {predicted_disease}")
    if confidence is not None:
        st.info(f"Confidence: {confidence*100:.2f}%")
    if all_probs is not None:
        st.subheader("Probability Distribution:")
        st.bar_chart(all_probs)
    st.subheader("Recommendations:")
    if predicted_disease == "Healthy":
        st.write("✅ No significant health risk detected. Continue regular monitoring.")
    else:
        st.write("⚠️ Please consult a healthcare professional for further assessment.")