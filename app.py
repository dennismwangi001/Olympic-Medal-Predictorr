import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the saved models, scaler, and the dataset
@st.cache_resource
def load_models():
    lr_model = joblib.load('linear_model.pkl')
    xgb_model = joblib.load('xgb_model.pkl')
    svm_model = joblib.load('svm_model.pkl')
    knn_model = joblib.load('knn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return lr_model, xgb_model, svm_model, knn_model, scaler

@st.cache_data
def load_data():
    # Load the dataset to enable country lookup
    return pd.read_csv('teams.csv')

lr_model, xgb_model, svm_model, knn_model, scaler = load_models()
teams_df = load_data()

# 2. Set up the Streamlit UI
st.set_page_config(page_title="Olympic Medal Predictor", page_icon="🏅", layout="centered")

st.title("🏅 Olympic Medal Predictor")
st.write("Select a country to auto-fill their latest stats, or choose 'Custom Input' to enter your own values.")

st.markdown("---")

# 3. Country Selection Dropdown
unique_countries = sorted(teams_df['country'].dropna().unique())
selected_country = st.selectbox("Select a Country", ["-- Custom Input --"] + unique_countries)

# 4. Determine default values based on selection
default_athletes = 50
default_prev_medals = 0.0

if selected_country != "-- Custom Input --":
    # Filter for the selected country and get the most recent year's data
    country_data = teams_df[teams_df['country'] == selected_country].sort_values('year', ascending=False)
    
    if not country_data.empty:
        latest = country_data.iloc[0]
        default_athletes = int(latest['athletes'])
        default_prev_medals = float(latest['prev_medals'])
        
        st.info(f"📊 Auto-filled with **{selected_country}**'s latest available data (Year: {int(latest['year'])}). You can still adjust these values below!")

# 5. User Inputs
st.markdown("### Input Parameters")
col1, col2 = st.columns(2)
with col1:
    athletes = st.number_input("Number of Athletes", min_value=1, max_value=2000, value=default_athletes, step=1)
with col2:
    prev_medals = st.number_input("Medals in Previous Olympics", min_value=0.0, max_value=500.0, value=default_prev_medals, step=0.5)

# 6. Prediction Button
st.markdown("---")
if st.button("🔮 Predict Medals", type="primary"):
    with st.spinner("Calculating predictions..."):
        # Create a dataframe for the input
        input_data = pd.DataFrame({
            'athletes': [athletes],
            'prev_medals': [prev_medals]
        })
        
        # Scale the data ONLY for SVM and KNN (Linear Regression and XGBoost don't require scaling)
        input_scaled = scaler.transform(input_data)
        
        # Make predictions
        pred_lr = lr_model.predict(input_data)[0]
        pred_xgb = xgb_model.predict(input_data)[0]
        pred_svm = svm_model.predict(input_scaled)[0]
        pred_knn = knn_model.predict(input_scaled)[0]
        
        # Clip negative predictions to 0 (a country can't win negative medals)
        pred_lr = max(0, pred_lr)
        pred_xgb = max(0, pred_xgb)
        pred_svm = max(0, pred_svm)
        pred_knn = max(0, pred_knn)
        
        # 7. Display Results (4 columns now)
        st.success("Predictions Complete!")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Linear Regression", f"{pred_lr:.1f} 🥇")
        col_b.metric("XGBoost", f"{pred_xgb:.1f} 🥈")
        col_c.metric("SVM", f"{pred_svm:.1f} 🥉")
        col_d.metric("KNN", f"{pred_knn:.1f} 🏅")
        
        st.info("💡 *Note: Predictions are rounded to the nearest whole number. Negative predictions are automatically set to 0.*")