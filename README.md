---
title: AI Data Scientist Agent
emoji: 📊
sdk: streamlit
app_file: app.py
pinned: false
license: mit
sdk_version: 1.49.1
---

# 🤖 AI Data Scientist Agent

An end-to-end AI-powered data scientist app built with **Streamlit**, **LangChain**, and **Groq LLMs**.  
It helps you upload datasets, clean data, visualize insights, build ML models, and generate PDF reports — all in one place.  

---

## ✨ Features
- 📂 **Upload & Explore** – Upload datasets (CSV) and view schema.
- 🧹 **Data Cleaning** – Handle missing values, duplicates, outliers.
- 📊 **Visualization** – Interactive charts & correlation heatmaps.
- 🤖 **Modeling** – Automatically detects regression/classification and evaluates multiple ML models.
- 📑 **Report Generation** – Exports clean **PDF reports** with descriptive statistics, heatmaps, and model results.
- 💬 **AI Chatbot Assistant** – Ask dataset-related questions in natural language.

---

## 🚀 Demo
🔗 Try it live on Hugging Face Spaces:  
👉 [AI Data Scientist Agent](https://huggingface.co/spaces/Sanchay3011/ai-ds-agent)

---


🛠️ Tech Stack

Streamlit
 – App framework

LangChain
 – Agent & LLM integration

Groq
 – Fast inference LLMs

scikit-learn
 – ML models

pandas
 – Data handling

seaborn
 / matplotlib
 – Visualizations

ReportLab
 – PDF report generation

 ---

 📌 Future Improvements

📈 Support for time series forecasting

🧠 Add AutoML pipeline with hyperparameter tuning

☁️ Cloud dataset storage

---


👩‍💻 Author

Developed with ❤️ by Sanchay Roy

--

⭐ Contribute

Pull requests are welcome! Feel free to fork the repo and submit improvements.

If you find this project useful, don’t forget to star ⭐ the repo.

---

## ⚡ Installation (Local Setup)
Clone this repo and install dependencies:
```bash
git clone https://github.com/Sanchay-3011/ai-ds-agent.git
cd ai-ds-agent
pip install -r requirements.txt
streamlit run Home.py
