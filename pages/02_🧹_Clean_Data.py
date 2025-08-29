import streamlit as st
import pandas as pd

from chatbot import chatbot_sidebar

st.session_state["page_name"] = "Clean"

st.title("🧹 Data Cleaning")

# Check if dataset exists in session_state
if "dataset" not in st.session_state:
    st.warning("⚠️ Please upload a dataset first in the Upload & Schema page.")
    st.stop()

# Load dataset
df = st.session_state["dataset"]

st.subheader("Current Data Preview")
st.write(df.head())

# -------------------------
# Cleaning Options
# -------------------------
st.subheader("Cleaning Options")

if st.checkbox("Remove Missing Values"):
    df = df.dropna()

if st.checkbox("Remove Duplicates"):
    df = df.drop_duplicates()

if st.checkbox("Standardize Column Names"):
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# -------------------------
# Save Cleaned Data
# -------------------------
if st.button("💾 Save Cleaned Dataset"):
    st.session_state["dataset"] = df  # replace original dataset
    st.success("✅ Cleaned dataset saved! This version will be used in the next steps.")

st.subheader("Preview of Cleaned Data")
st.write(df.head())

# -------------------------
# Download Option
# -------------------------
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Download as CSV",
    data=csv,
    file_name="cleaned_dataset.csv",
    mime="text/csv",
)


chatbot_sidebar()
