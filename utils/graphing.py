import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def plot_histograms(df):
    st.markdown("### 📊 Histograms")
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

def plot_boxplots(df):
    st.markdown("### 📦 Boxplots")
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Boxplot of {col}")
        st.pyplot(fig)

def plot_pairplot(df, target):
    st.markdown("### 🔗 Pairplot")
    try:
        fig = sns.pairplot(df, hue=target)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not generate pairplot: {e}")
