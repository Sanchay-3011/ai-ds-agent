import streamlit as st
import pandas as pd

from chatbot import chatbot_sidebar

st.title("📂 Upload & Schema")

uploaded_file = st.file_uploader("Upload a CSV file and open the sidebar from the Top-Left corner (>>) to interact with the specialized AI Data Scientist!", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Store in session state
    st.session_state["dataset"] = df
    st.session_state["uploaded_filename"] = uploaded_file.name

    st.success(f"✅ Uploaded: {uploaded_file.name}")
    st.dataframe(df.head())


chatbot_sidebar()
