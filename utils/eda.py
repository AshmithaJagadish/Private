import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def run_eda(df: pd.DataFrame, target: str):
    st.markdown("## 🧠 Enhanced EDA & Auto-Cleaning")

    # 1. Dataset overview
    st.write("### 📌 Shape and Columns")
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())

    # 2. Data types
    st.write("### 🔍 Data Types")
    st.dataframe(df.dtypes)

    # 3. Constant columns removal
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        df.drop(columns=constant_cols, inplace=True)
        st.warning(f"Removed constant columns: {constant_cols}")

    # 4. Remove duplicates
    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    after = df.shape[0]
    if before != after:
        st.info(f"Removed {before - after} duplicate rows.")

    # 5. Handle missing values
    st.write("### 🧼 Missing Values")
    null_df = df.isnull().sum().reset_index()
    null_df.columns = ["Column", "Missing Values"]
    null_df["% Missing"] = round(null_df["Missing Values"] / len(df) * 100, 2)
    st.dataframe(null_df[null_df["Missing Values"] > 0])

    # Auto impute missing
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["int64", "float64"]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    # 6. Descriptive Stats
    st.write("### 📊 Descriptive Stats")
    st.dataframe(df.describe(include='all').transpose())

    # 7. Target Variable Distribution
    st.write("### 🎯 Target Variable Distribution")
    if target in df.columns:
        st.write(df[target].value_counts(normalize=True) if df[target].dtype == "object" else df[target].describe())

    # 8. Outlier detection (IQR)
    numeric_cols = df.select_dtypes(include=np.number).columns
    st.write("### 🚨 Outlier Detection (IQR method)")
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outliers > 0:
            st.write(f"{col}: {outliers} potential outliers")

    # 9. Correlation heatmap
    st.write("### 🔗 Correlation Heatmap")
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # 10. Class imbalance
    st.write("### ⚖️ Class Balance (if classification)")
    if target in df.columns and df[target].nunique() <= 10:
        st.bar_chart(df[target].value_counts())
