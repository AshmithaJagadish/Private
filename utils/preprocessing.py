import pandas as pd
import numpy as np
import streamlit as st

def auto_preprocess(df: pd.DataFrame):
    st.markdown("## 🧹 Auto Data Preprocessing")

    try:
        # Drop constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            df.drop(columns=constant_cols, inplace=True)
            st.warning(f"Removed constant columns: {constant_cols}")

        # Remove duplicates
        before = df.shape[0]
        df.drop_duplicates(inplace=True)
        after = df.shape[0]
        if before != after:
            st.info(f"Removed {before - after} duplicate rows")

        # Fill missing values
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ["int64", "float64"]:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)

        return df

    except Exception as e:
        st.error(f"❌ Preprocessing failed: {e}")
        return df
