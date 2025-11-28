import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Page configuration
st.set_page_config(page_title="Bank Churn Prediction", page_icon="💳", layout="wide")

# Load model and scaler
@st.cache_resource
def load_models():
    try:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        feature_names = pd.read_csv("friance new.csv").drop("Exited", axis=1).columns.tolist()
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading files: {e}")
        st.stop()

model, scaler, feature_names = load_models()

# Title and description
st.title("💳 Bank Customer Churn Prediction System")
st.write("Fill in the customer details below to predict whether they will exit the bank.")
st.markdown("---")

# Create columns for better layout
col1, col2 = st.columns(2)

input_data = {}

# Distribute inputs across two columns
for idx, feature in enumerate(feature_names):
    # Alternate between columns
    with col1 if idx % 2 == 0 else col2:
        # Add better labels and reasonable defaults
        if "Credit" in feature or "Balance" in feature or "Salary" in feature:
            value = st.number_input(f"{feature}", value=0.0, step=1000.0, format="%.2f")
        elif "Age" in feature:
            value = st.number_input(f"{feature}", value=30.0, min_value=18.0, max_value=100.0, step=1.0)
        elif "Tenure" in feature:
            value = st.number_input(f"{feature}", value=0.0, min_value=0.0, max_value=10.0, step=1.0)
        elif "NumOfProducts" in feature:
            value = st.number_input(f"{feature}", value=1.0, min_value=1.0, max_value=4.0, step=1.0)
        else:
            value = st.number_input(f"{feature}", value=0.0)
        
        input_data[feature] = value

st.markdown("---")

# Prediction Button with better styling
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    predict_button = st.button("🔮 Predict Churn", use_container_width=True, type="primary")

if predict_button:
    try:
        # Convert input to array in correct order
        user_array = np.array([[input_data[feature] for feature in feature_names]])
        
        # Scale and predict
        user_scaled = scaler.transform(user_array)
        prediction = model.predict(user_scaled)[0]
        prediction_proba = model.predict_proba(user_scaled)[0]
        
        st.markdown("---")
        st.subheader("Prediction Result:")
        
        # Display results
        if prediction == 1:
            st.error("**The customer is likely to EXIT / CHURN**")
            st.metric("Churn Probability", f"{prediction_proba[1]*100:.2f}%")
        else:
            st.success("**The customer will STAY / NOT CHURN**")
            st.metric("Retention Probability", f"{prediction_proba[0]*100:.2f}%")
        
        # Show confidence
        st.info(f"**Model Confidence:** {max(prediction_proba)*100:.2f}%")
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Bank Customer Churn Prediction System")