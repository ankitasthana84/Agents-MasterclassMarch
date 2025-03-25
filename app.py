import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
import os
from dotenv import load_dotenv

# Load API key securely (if needed for future integrations)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.warning("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file if integrating external APIs.")

# Streamlit UI Styling
st.set_page_config(page_title="AI Revenue Forecaster", page_icon="📈", layout="wide")
st.title("📊 AI Revenue Forecasting with Prophet")

# File Upload Section
st.sidebar.header("Upload Your Excel File")
uploaded_file = st.sidebar.file_uploader("Upload an Excel file with 'Date' and 'Revenue' columns", type=["xlsx", "csv"])

if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith("xlsx") else pd.read_csv(uploaded_file)
    
    # Data Preprocessing
    df.columns = df.columns.str.strip().str.lower()
    if 'date' not in df.columns or 'revenue' not in df.columns:
        st.error("The file must contain 'Date' and 'Revenue' columns.")
        st.stop()
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.rename(columns={'date': 'ds', 'revenue': 'y'}, inplace=True)
    
    # Prophet Model Initialization
    model = Prophet()
    model.fit(df)
    
    # Forecasting Future Data
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    # Display Data and Forecasting Results
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df.tail())
    
    st.subheader("📈 Forecast Results")
    fig, ax = plt.subplots()
    model.plot(forecast, ax=ax)
    st.pyplot(fig)
    
    st.subheader("📉 Forecast Components")
    fig_components = model.plot_components(forecast)
    st.pyplot(fig_components)
    
    st.success("✅ Forecasting completed! Adjust parameters as needed.")
