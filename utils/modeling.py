import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def run_modeling(df, target):
    st.markdown("## 🤖 Model Training & Evaluation")
    best_model = None
    best_score = -np.inf
    best_model_name = ""
    used_features = []

    try:
        X = df.drop(columns=[target])
        y = df[target]

        # Encode categorical
        X = pd.get_dummies(X)

        # Detect problem type
        problem_type = 'classification' if y.nunique() <= 10 and y.dtype != 'float64' else 'regression'
        st.info(f"Detected problem type: **{problem_type}**")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Define models
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000) if problem_type == "classification" else LinearRegression(),
            "Random Forest": RandomForestClassifier() if problem_type == "classification" else RandomForestRegressor(),
            "SVM": SVC(probability=True) if problem_type == "classification" else SVR(),
            "XGBoost": xgb.XGBClassifier(eval_metric='logloss') if problem_type == "classification" else xgb.XGBRegressor()
        }

        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                st.subheader(f"📌 {name} Results")
                if problem_type == "classification":
                    acc = accuracy_score(y_test, y_pred)
                    st.write("Accuracy:", acc)
                    st.text(classification_report(y_test, y_pred))
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                    st.pyplot(fig)
                    if hasattr(model, "predict_proba"):
                        y_prob = model.predict_proba(X_test_scaled)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_prob)
                        roc_auc = auc(fpr, tpr)
                        fig2, ax2 = plt.subplots()
                        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                        ax2.plot([0, 1], [0, 1], 'k--')
                        ax2.legend()
                        st.pyplot(fig2)
                    if acc > best_score:
                        best_score = acc
                        best_model = model
                        best_model_name = name
                        used_features = X.columns.tolist()
                else:
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)
                    st.write("RMSE:", rmse)
                    st.write("R²:", r2)
                    if r2 > best_score:
                        best_score = r2
                        best_model = model
                        best_model_name = name
                        used_features = X.columns.tolist()

                # Feature importance
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    fig3, ax3 = plt.subplots()
                    sns.barplot(x=importances[:15], y=X.columns[:15], ax=ax3)
                    st.pyplot(fig3)

            except Exception as model_error:
                st.warning(f"{name} failed: {model_error}")
                continue

        if best_model:
            st.success(f"✅ Best model: {best_model_name}")
        else:
            st.error("❌ All models failed. Please check your dataset.")

        return best_model, used_features

    except Exception as e:
        st.error(f"❌ Modeling error: {e}")
        return None, []
