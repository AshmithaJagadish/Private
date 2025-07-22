import streamlit as st
import pandas as pd
import numpy as np

def run_prediction(best_model, used_features):
    st.markdown("## 📤 Make Predictions with Trained Model")

    if not best_model or not used_features:
        st.error("❌ No trained model found. Please run modeling first.")
        return

    input_mode = st.radio("Select Input Mode", ["Manual Input", "Upload CSV"])

    if input_mode == "Manual Input":
        st.markdown("### 📝 Enter values manually")
        user_input = {}
        for feature in used_features:
            user_input[feature] = st.text_input(f"{feature}", value="0")

        if st.button("Predict"):
            try:
                input_df = pd.DataFrame([user_input])
                input_df = input_df.astype(float)
                prediction = best_model.predict(input_df)[0]
                st.success(f"✅ Prediction: {prediction}")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

    elif input_mode == "Upload CSV":
        st.markdown("### 📂 Upload a CSV with same features")
        uploaded_csv = st.file_uploader("Choose CSV", type=["csv"])

        if uploaded_csv:
            try:
                df_input = pd.read_csv(uploaded_csv)
                missing_cols = [col for col in used_features if col not in df_input.columns]
                if missing_cols:
                    st.error(f"❌ Missing required columns: {missing_cols}")
                else:
                    df_input = df_input[used_features]
                    df_input = df_input.astype(float)
                    preds = best_model.predict(df_input)
                    df_input["Prediction"] = preds
                    st.success("✅ Predictions completed!")
                    st.dataframe(df_input)
                    st.download_button("Download Predictions", df_input.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
