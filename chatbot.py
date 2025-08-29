import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

def chatbot_sidebar():
    st.sidebar.markdown("## 🤖 Chat with AI Data Scientist!")

    if "dataset" not in st.session_state:
        st.sidebar.warning("⚠️ Please upload a dataset first.")
        return

    df = st.session_state["dataset"]

    # Use the working model
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    user_input = st.sidebar.text_area("💬 Ask me about your dataset:")
    if user_input:
        try:
            prompt = f"""
            You are a professional data scientist. Analyze the DataFrame `df` below:

            Preview:
            {df.head(5).to_string()}

            Schema:
            {df.dtypes.to_string()}

            Now answer the user’s question in plain English, based on the full dataset—not code experiments.

            User: {user_input}
            """

            response = llm.invoke(prompt)
            st.sidebar.write("🤖:", response.content.strip())

        except Exception as e:
            st.sidebar.error(f"⚠️ Error: {e}")
