import streamlit as st
import pandas as pd

from utils.eda import run_eda
from utils.modeling import run_modeling
from utils.graphing import plot_histograms, plot_boxplots, plot_pairplot
from utils.preprocessing import auto_preprocess
from utils.predictor import run_prediction

st.set_page_config(page_title="AutoML Dashboard", layout="wide")
st.title("🤖 AutoML System with EDA, Modeling & Prediction")

# Sidebar for upload
st.sidebar.header("📁 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# If file uploaded
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ File uploaded successfully")

        # Select target
        target = st.sidebar.selectbox("🎯 Select Target Variable", df.columns)

        # Preprocess
        df = auto_preprocess(df)

        # Create 4 tabs
        tabs = st.tabs(["📊 EDA", "📈 Graphs", "⚙️ Modeling", "📤 Predict"])

        # --- Tab 1: EDA ---
        with tabs[0]:
            try:
                run_eda(df, target)
            except Exception as e:
                st.error(f"❌ EDA Error: {e}")

        # --- Tab 2: Graphs ---
        with tabs[1]:
            try:
                plot_histograms(df)
                plot_boxplots(df)
                plot_pairplot(df, target)
            except Exception as e:
                st.error(f"❌ Graphing Error: {e}")

        # --- Tab 3: Modeling ---
        with tabs[2]:
            try:
                best_model, used_features = run_modeling(df, target)
                st.session_state["best_model"] = best_model
                st.session_state["used_features"] = used_features
            except Exception as e:
                st.error(f"❌ Modeling Error: {e}")

        # --- Tab 4: Prediction ---
        with tabs[3]:
            try:
                run_prediction(
                    st.session_state.get("best_model"),
                    st.session_state.get("used_features")
                )
            except Exception as e:
                st.error(f"❌ Prediction Error: {e}")

    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
else:
    st.warning("📤 Please upload a CSV file to begin.")
