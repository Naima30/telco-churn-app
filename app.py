# app.py
"""
Customer Churn Prediction Dashboard 
- Auto-loads dataset from default paths (no upload needed)
- Multi-page dashboard: Overview, EDA, Modeling, Prediction, About
- Models: Logistic Regression, Random Forest, optional XGBoost
- Auto-encoding + cleaning
- Works in Jupyter and local Windows
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings("ignore")
sns.set_style("darkgrid")


# ------------------------
#  DEFAULT LOAD PATHS
# ------------------------

DEFAULT_PATHS = [
    Path("/mnt/data/Telco_customer_churn.xlsx"),  # Jupyter path
    Path(r"C:/Users/naima/OneDrive/ドキュメント/GT/Telco_customer_churn.xlsx")  # Windows path
]


# ------------------------
#  Auto-load dataset
# ------------------------
def load_dataset():
    for p in DEFAULT_PATHS:
        if p.exists():
            try:
                if p.suffix == ".csv":
                    return pd.read_csv(p)
                else:
                    return pd.read_excel(p)
            except Exception as e:
                st.error(f"❌ Error reading dataset: {e}")
                st.stop()

    st.error("❌ Dataset not found. Please place Telco_customer_churn.xlsx in the default path.")
    st.stop()


# ------------------------
#  Cleaning Utilities
# ------------------------
def replace_blank_with_nan(df):
    return df.replace(r'^\s*$', np.nan, regex=True)


def clean_numeric_like(df):
    df = df.copy()
    df = replace_blank_with_nan(df)

    numeric_cols = ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV", "Churn Score"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=['object', 'category', 'bool']).columns:
        if df[col].isnull().any():
            try:
                df[col] = df[col].fillna(df[col].mode()[0])
            except:
                df[col] = df[col].fillna("Unknown")

    return df


def auto_encode_features(X):
    cat = X.select_dtypes(include=['object', 'category', 'bool']).columns
    return pd.get_dummies(X, columns=cat, drop_first=True)


def classification_metrics(y_true, y_pred):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0))
    }


# ------------------------
#  Streamlit Styling
# ------------------------
def apply_css():
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(180deg,#08151d 0%, #071419 100%); color: #e6fff5; }
    .card { background: rgba(255,255,255,0.07); padding:15px; border-radius:12px; margin-bottom:15px; }
    </style>
    """, unsafe_allow_html=True)


# ------------------------
#  Dashboard Pages
# ------------------------
def overview_page(df):
    st.header("📊 Dataset Overview")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    st.metric("Rows", df.shape[0])
    st.metric("Columns", df.shape[1])
    st.write("Missing Values:")
    st.dataframe(df.isnull().sum().to_frame("Missing"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def eda_page(df):
    st.header("🔍 Full EDA")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    # Histogram
    st.subheader("Distribution")
    if len(numeric_cols) > 0:
        col = st.selectbox("Select numerical column", numeric_cols)
        fig, ax = plt.subplots()
        ax.hist(df[col], bins=30, color="#0b6b48")
        st.pyplot(fig)

    # Categorical
    st.subheader("Categorical Distribution")
    if len(cat_cols) > 0:
        col = st.selectbox("Select categorical column", cat_cols)
        vc = df[col].value_counts()
        st.bar_chart(vc)

    # Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(9,6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="YlGn", ax=ax)
    st.pyplot(fig)


def modeling_page(df):
    st.header("🤖 Modeling & Comparison")

    # Target
    target = st.selectbox(
        "Select target:",
        ["Churn Label", "Churn Value"] if "Churn Label" in df.columns else df.columns
    )

    features = st.multiselect(
        "Select features:",
        [c for c in df.columns if c != target],
        default=["Tenure Months", "Monthly Charges", "Total Charges", "Contract", "Internet Service"]
    )

    if st.button("Train Models"):
        X = df[features].copy()
        y = df[target].copy()

        # Encode target
        lbl = LabelEncoder()
        y_enc = lbl.fit_transform(y.astype(str))

        if len(np.unique(y_enc)) < 2:
            st.error("❌ Target has only one class. Choose a different target.")
            return

        X = clean_numeric_like(X)
        X = auto_encode_features(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )

        models = {}

        # Logistic Regression
        log = LogisticRegression(max_iter=2000)
        log.fit(X_train, y_train)
        pred_log = log.predict(X_test)
        models["Logistic Regression"] = (log, pred_log)

        # Random Forest
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        pred_rf = rf.predict(X_test)
        models["Random Forest"] = (rf, pred_rf)

        # Display metrics
        st.subheader("Results")
        results = []
        for name, (mdl, pred) in models.items():
            results.append({"Model": name, **classification_metrics(y_test, pred)})
            st.session_state[f"model_{name}"] = mdl

        st.dataframe(pd.DataFrame(results).set_index("Model"))

        # Save metadata
        st.session_state["meta"] = {
            "features": features,
            "encoder": lbl,
            "feature_columns": X.columns
        }


def prediction_page(df):
    st.header("🎯 Prediction Playground")

    if "meta" not in st.session_state:
        st.warning("Train a model first in the Modeling page.")
        return

    meta = st.session_state["meta"]
    features = meta["features"]

    inputs = {}
    cols = st.columns(2)

    for i, f in enumerate(features):
        if pd.api.types.is_numeric_dtype(df[f]):
            inputs[f] = cols[i%2].number_input(f, value=float(df[f].median()))
        else:
            options = df[f].unique().tolist()
            inputs[f] = cols[i%2].selectbox(f, options)

    mdl_name = st.selectbox("Choose model", [k.replace("model_", "") for k in st.session_state if k.startswith("model_")])
    mdl = st.session_state[f"model_{mdl_name}"]

    if st.button("Predict"):
        Xnew = pd.DataFrame([inputs])
        Xnew = auto_encode_features(Xnew)

        # align columns
        for col in meta["feature_columns"]:
            if col not in Xnew.columns:
                Xnew[col] = 0

        Xnew = Xnew[meta["feature_columns"]]

        pred = mdl.predict(Xnew)[0]
        label = meta["encoder"].inverse_transform([pred])[0]

        st.success(f"Prediction: **{label}**")


def about_page():
    st.header("ℹ️ About")
    st.write("This is a complete Customer Churn Prediction dashboard built using Streamlit.")
    st.write("Includes: EDA, ML Modeling, Prediction UI, Auto-Encoding, Dark Theme.")


# ------------------------
# MAIN APP
# ------------------------
def main():
    st.set_page_config(page_title="Churn Dashboard", layout="wide")
    apply_css()
    st.title("☁️ Customer Churn Prediction Dashboard")

    df = load_dataset()
    df = clean_numeric_like(df)

    page = st.sidebar.radio("Navigate", ["Overview", "Full EDA", "Modeling", "Prediction Playground", "About"])

    if page == "Overview":
        overview_page(df)
    elif page == "Full EDA":
        eda_page(df)
    elif page == "Modeling":
        modeling_page(df)
    elif page == "Prediction Playground":
        prediction_page(df)
    else:
        about_page()


if __name__ == "__main__":
    main()
