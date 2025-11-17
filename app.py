# app.py
import streamlit as st

st.set_page_config(page_title="WFO Forecasting System", layout="wide")

st.title("🧠 WFO Forecasting System")
st.write("Use the left Pages sidebar to navigate:")
st.write("- **1_Future_Flags** → Configure upcoming hiring, exits, holidays, events")
st.write("- **2_Run_Forecast** → Run the forecasting models")
st.write("- **EDA** → Explore EDA for WFO Count (Chennai / Bangalore)")