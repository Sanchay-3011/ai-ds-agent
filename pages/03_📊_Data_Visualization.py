import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from chatbot import chatbot_sidebar

st.session_state["page_name"] = "Data_Visualisation"

st.title("📊 Exploratory Data Analysis (EDA)")

# -------------------------
# Load Dataset
# -------------------------
if "dataset" not in st.session_state:
    st.warning("⚠️ Please upload and clean your dataset first.")
    st.stop()

df = st.session_state["dataset"]

st.subheader("Data Preview")
st.write(df.head())

# -------------------------
# Summary Statistics
# -------------------------
st.subheader("📌 Summary Statistics")
st.write(df.describe(include="all"))

# -------------------------
# 1. Histogram
# -------------------------
st.subheader("📈 Histogram")
column = st.selectbox("Select a column", df.columns, key="hist")
if column:
    fig, ax = plt.subplots()
    df[column].hist(ax=ax, bins=20, color="skyblue", edgecolor="black")
    ax.set_title(f"Histogram of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# -------------------------
# 2. Boxplot
# -------------------------
st.subheader("📦 Boxplot (Detect Outliers)")
box_col = st.selectbox("Select numeric column", df.select_dtypes(include="number").columns, key="box")
if box_col:
    fig, ax = plt.subplots()
    ax.boxplot(df[box_col].dropna())
    ax.set_title(f"Boxplot of {box_col}")
    ax.set_ylabel(box_col)
    st.pyplot(fig)

# -------------------------
# 3. Scatter Plot
# -------------------------
st.subheader("⚖️ Scatter Plot (Relationship)")
col_x = st.selectbox("X-axis (Numeric)", df.select_dtypes(include="number").columns, key="scatter_x")
col_y = st.selectbox("Y-axis (Numeric)", df.select_dtypes(include="number").columns, key="scatter_y")
if col_x and col_y:
    fig, ax = plt.subplots()
    ax.scatter(df[col_x], df[col_y], alpha=0.6, color="purple")
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)
    ax.set_title(f"{col_x} vs {col_y}")
    st.pyplot(fig)

# -------------------------
# 4. Bar Chart (Categorical Count)
# -------------------------
st.subheader("📊 Bar Chart (Category Counts)")
cat_col = st.selectbox("Select categorical column", df.select_dtypes(exclude="number").columns, key="bar")
if cat_col:
    counts = df[cat_col].value_counts()
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax, color="orange", edgecolor="black")
    ax.set_title(f"Count of {cat_col}")
    ax.set_xlabel(cat_col)
    ax.set_ylabel("Count")
    st.pyplot(fig)

# -------------------------
# 5. Correlation Heatmap
# -------------------------
st.subheader("🔥 Correlation Heatmap")
if len(df.select_dtypes(include="number").columns) > 1:
    fig, ax = plt.subplots(figsize=(6, 4))
    corr = df.corr(numeric_only=True)
    im = ax.imshow(corr, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
else:
    st.info("Need at least 2 numeric columns for correlation heatmap.")


chatbot_sidebar()
