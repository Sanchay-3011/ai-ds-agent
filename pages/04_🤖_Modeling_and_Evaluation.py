import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

from chatbot import chatbot_sidebar

st.session_state["page_name"] = "Modeling and Evaluation"

st.title("🤖 Modeling & Evaluation")

# -------------------------
# Load dataset
# -------------------------
if "dataset" not in st.session_state:
    st.warning("⚠️ Please upload a dataset first.")
    st.stop()

df = st.session_state["dataset"]

# -------------------------
# Target Column Selection
# -------------------------
st.markdown("### 🎯 Select Target Column")

target_col = st.selectbox("Choose the target column:", df.columns, key="target_select")

if st.button("Confirm Target"):
    st.session_state["target_col"] = target_col
    st.session_state["run_modeling"] = False  # reset before running
    st.success(f"✅ Target column set to **{target_col}**")

# -------------------------
# Run Modeling Button
# -------------------------
if "target_col" in st.session_state:
    if st.button("🚀 Run Modeling"):
        st.session_state["run_modeling"] = True

# -------------------------
# Modeling Logic
# -------------------------
if st.session_state.get("run_modeling", False):

    with st.spinner("⏳ Training models... Please wait."):
        target_col = st.session_state["target_col"]

        # Split X and y
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode categorical features
        X = pd.get_dummies(X, drop_first=True)

        # Encode target if categorical
        if y.dtype == "object" or y.dtype.name == "category":
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        results = []

        # Classification vs Regression detection
        problem_type = "classification" if len(pd.Series(y).unique()) <= 10 else "regression"

        if problem_type == "classification":
            try:
                clf = LogisticRegression(max_iter=1000)
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                acc = accuracy_score(y_test, preds)
                results.append(("Logistic Regression", acc))
            except Exception as e:
                st.error(f"❌ Logistic Regression failed: {e}")

            try:
                rf_clf = RandomForestClassifier()
                rf_clf.fit(X_train, y_train)
                preds = rf_clf.predict(X_test)
                acc = accuracy_score(y_test, preds)
                results.append(("Random Forest Classifier", acc))
            except Exception as e:
                st.error(f"❌ Random Forest failed: {e}")

        else:  # Regression
            try:
                lr = LinearRegression()
                lr.fit(X_train, y_train)
                preds = lr.predict(X_test)
                mse = mean_squared_error(y_test, preds)
                results.append(("Linear Regression", mse))
            except Exception as e:
                st.error(f"❌ Linear Regression failed: {e}")

            try:
                rf_reg = RandomForestRegressor()
                rf_reg.fit(X_train, y_train)
                preds = rf_reg.predict(X_test)
                mse = mean_squared_error(y_test, preds)
                results.append(("Random Forest Regressor", mse))
            except Exception as e:
                st.error(f"❌ Random Forest failed: {e}")

        # -------------------------
        # Show Results
        # -------------------------
        if results:
            st.markdown("### 📊 Model Results")

            for model, score in results:
                if problem_type == "classification":
                    st.write(f"✅ **{model} Accuracy:** {score:.4f}")
                else:
                    st.write(f"✅ **{model} MSE:** {score:.2f}")

            # Pick best model
            if problem_type == "classification":
                best_model = max(results, key=lambda x: x[1])
                st.success(f"🏆 Best Model: **{best_model[0]}** with Accuracy = {best_model[1]:.4f}")
            else:
                best_model = min(results, key=lambda x: x[1])
                st.success(f"🏆 Best Model: **{best_model[0]}** with MSE = {best_model[1]:.2f}")

            # Save best model to session
            st.session_state["best_model_name"] = best_model[0]
            st.session_state["best_score"] = best_model[1]
            st.session_state["problem_type"] = problem_type

        else:
            st.error("❌ No models could be trained. Please check your dataset.")

chatbot_sidebar()
