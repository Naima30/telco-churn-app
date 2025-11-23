# app.py
"""
Customer Churn Prediction
- Full multi-page dashboard (Overview, Full EDA, Modeling & Comparison,
  Prediction Playground, About)
- Models: Logistic Regression, Random Forest (optional XGBoost if installed)
- Auto-encoding of categorical features, robust cleaning, session-state model storage
- Trains inside the app (fixed 80/20 split). Safe checks to avoid single-class training errors.
- Default dataset path set to the user's Windows path (change if needed).
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tempfile

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder

import joblib

# Optional XGBoost if installed
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

sns.set_style("darkgrid")

# -------------------------
# DEFAULT DATA PATH (Windows)
# -------------------------
DEFAULT_DATA_PATH = r"C:/Users/naima/OneDrive/ドキュメント/GT/Telco_customer_churn.xlsx"
MODEL_SAVE_DIR = Path("/mnt/data")


# -------------------------
# Styling
# -------------------------
def apply_css():
    css = """
    <style>
      .stApp { background: linear-gradient(180deg,#07121a 0%, #041018 100%); color: #e6fff5; }
      .card { background: rgba(255,255,255,0.02); padding:14px; border-radius:10px; margin-bottom:12px; }
      .muted { color: rgba(230,255,245,0.85); }
      .small { font-size:13px; color: rgba(230,255,245,0.75); }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# -------------------------
# Data utilities
# -------------------------
@st.cache_data(ttl=3600)
def load_dataset(uploaded_file):
    """
    Priority: uploaded file -> DEFAULT_DATA_PATH -> error
    """
    if uploaded_file is not None:
        try:
            if str(uploaded_file.name).lower().endswith(".csv"):
                return pd.read_csv(uploaded_file)
            else:
                return pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
            return None
    # try default path
    p = Path(DEFAULT_DATA_PATH)
    if p.exists():
        try:
            return pd.read_excel(p)
        except Exception:
            try:
                return pd.read_csv(p)
            except Exception as e:
                st.error(f"Failed to read dataset at default path: {e}")
                return None
    st.error(f"No dataset found. Upload a file via the sidebar or place it at: {DEFAULT_DATA_PATH}")
    return None

def replace_blank_with_nan(df: pd.DataFrame):
    return df.replace(r'^\s*$', np.nan, regex=True)

def clean_numeric_like(df: pd.DataFrame, numeric_candidates=None):
    df = df.copy()
    df = replace_blank_with_nan(df)
    if numeric_candidates is None:
        for col in df.columns:
            if df[col].dtype == object:
                s = df[col].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
                s = s.str.replace(r"[^\d\.\-]", "", regex=True)
                coerced = pd.to_numeric(s, errors="coerce")
                if coerced.notna().sum() >= 0.5 * len(coerced):
                    df[col] = coerced
    else:
        for col in numeric_candidates:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    # fill numeric NaNs with median
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    return df

def auto_encode_features(X: pd.DataFrame):
    X = X.copy()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    return X

def classification_metrics(y_true, y_pred):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0))
    }

def save_model(obj, name="model.pkl"):
    path = MODEL_SAVE_DIR / name
    joblib.dump(obj, path)
    return path

# -------------------------
# Page components
# -------------------------
def overview_page(df):
    st.header("Dataset Overview")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Preview")
    st.dataframe(df.head(8), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Numeric cols", len(df.select_dtypes(include=[np.number]).columns.tolist()))
    st.write("Missing values (top 10):")
    st.dataframe(df.isnull().sum().sort_values(ascending=False).head(10).to_frame("missing"))
    st.markdown("</div>", unsafe_allow_html=True)

def eda_page(df):
    st.header("Exploratory Data Analysis — Full")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object','category','bool']).columns.tolist()

    # Distribution + boxplot
    st.subheader("Distribution & Outliers")
    if numeric_cols:
        dist_col = st.selectbox("Choose numeric column for distribution/boxplot:", numeric_cols, index=0)
        fig, axes = plt.subplots(1,2, figsize=(12,3.5))
        axes[0].hist(df[dist_col].dropna(), bins=30, color="#0b6b48", edgecolor="k")
        axes[0].set_title(f"Distribution of {dist_col}")
        axes[0].set_xlabel(dist_col)
        axes[0].set_ylabel("Frequency")
        sns.boxplot(x=df[dist_col].dropna(), ax=axes[1], color="#cfead8")
        axes[1].set_title(f"Boxplot of {dist_col}")
        st.pyplot(fig)
        st.write("Explanation: Distribution shows central tendency & skew; boxplot highlights IQR & outliers.")
    else:
        st.info("No numeric columns available.")

    st.markdown("---")

    # Scatter
    st.subheader("Scatter Plot (Relationship)")
    if len(numeric_cols) >= 2:
        x_var = st.selectbox("X-axis (numeric)", numeric_cols, index=0, key="sc_x")
        y_var = st.selectbox("Y-axis (numeric)", numeric_cols, index=1, key="sc_y")
        fig, ax = plt.subplots(figsize=(7,4))
        ax.scatter(df[x_var], df[y_var], alpha=0.6, s=40, edgecolors="w", color="#ffd59e")
        ax.set_xlabel(x_var); ax.set_ylabel(y_var); ax.set_title(f"{y_var} vs {x_var}")
        st.pyplot(fig)
    else:
        st.info("At least two numeric columns are required for scatter plot.")

    st.markdown("---")

    # Categorical distribution
    st.subheader("Categorical Distribution (Pie & Bar)")
    if cat_cols:
        cat = st.selectbox("Choose categorical column:", cat_cols, index=0, key="cat_pie")
        counts = df[cat].value_counts().head(8)
        fig, axes = plt.subplots(1,2, figsize=(12,4))
        axes[0].pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140, colors=sns.color_palette("Greens"))
        axes[0].set_title(f"Top categories — {cat}")
        counts.plot.bar(ax=axes[1], color="#0b6b48")
        axes[1].set_ylabel("Count"); axes[1].set_title(f"Counts — {cat}")
        st.pyplot(fig)
    else:
        st.info("No categorical columns detected for pie/bar charts.")

    st.markdown("---")

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(9,6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="YlGn", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
    else:
        st.info("Need at least two numeric columns for heatmap.")

def modeling_comparison_page(df):
    st.header("Modeling & Comparison (80/20 fixed split)")
    # Suggest targets
    suggested_targets = [t for t in ["Churn Label", "Churn Value", "Churn", "churn_label", "churn_value"] if t in df.columns]
    if not suggested_targets:
        suggested_targets = df.select_dtypes(include=['object','category','bool']).columns.tolist()
    if not suggested_targets:
        st.error("No categorical target found. Upload a dataset with a churn-like target (e.g., 'Churn Label').")
        return

    # Choose target
    default_target = suggested_targets[0] if suggested_targets else df.columns[0]
    target = st.selectbox("Select target (what to predict):", options=list(df.columns), index=list(df.columns).index(default_target) if default_target in df.columns else 0)
    st.write("Target dtype:", df[target].dtype)

    # Choose features
    prefer = ["Tenure Months", "Monthly Charges", "Total Charges", "Contract", "Internet Service", "Payment Method", "Online Security", "Tech Support"]
    default_features = [p for p in prefer if p in df.columns]
    if not default_features:
        default_features = [c for c in df.columns if c != target][:6]

    features = st.multiselect("Select features to use (categorical allowed):", options=[c for c in df.columns if c != target], default=default_features)

    if not features:
        st.warning("Select at least one feature.")
        return

    do_xgboost = XGBOOST_AVAILABLE and st.checkbox("Include XGBoost (if available)", value=False)

    if st.button("Train & Compare Models"):
        X = df[features].copy()
        y = df[target].copy()

        # Cleaning & target encoding
        X = clean_numeric_like(X)
        if y.dtype == object or y.dtype.name == "category" or y.dtype == bool:
            lbl = LabelEncoder()
            y_enc = lbl.fit_transform(y.astype(str))
        else:
            if y.nunique() <= 10 and not np.issubdtype(y.dtype, np.floating):
                lbl = LabelEncoder()
                y_enc = lbl.fit_transform(y.astype(str))
            else:
                st.error("Selected target appears continuous numeric. This page is for classification. Choose a categorical target.")
                return

        # Ensure >= 2 classes
        if len(np.unique(y_enc)) < 2:
            st.error(f"Target '{target}' contains only one class after encoding. Choose another target with >= 2 classes.")
            return

        X_enc = auto_encode_features(X)
        # remove constant columns if any
        if X_enc.shape[0] > 1:
            X_enc = X_enc.loc[:, (X_enc != X_enc.iloc[0]).any()]

        # Split (try stratify)
        try:
            X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=0.2, random_state=42, stratify=None)

        trained = {}
        # Logistic Regression
        try:
            log = LogisticRegression(max_iter=2000)
            log.fit(X_train, y_train)
            pred_log = log.predict(X_test)
            trained["Logistic Regression"] = {"model": log, "pred": pred_log}
        except Exception as e:
            st.warning(f"Logistic Regression training failed: {e}")

        # Random Forest
        try:
            rfc = RandomForestClassifier(n_estimators=200, random_state=42)
            rfc.fit(X_train, y_train)
            pred_rfc = rfc.predict(X_test)
            trained["Random Forest"] = {"model": rfc, "pred": pred_rfc}
        except Exception as e:
            st.warning(f"Random Forest training failed: {e}")

        # XGBoost (optional)
        if do_xgboost:
            try:
                xclf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
                xclf.fit(X_train, y_train)
                pred_xgb = xclf.predict(X_test)
                trained["XGBoost"] = {"model": xclf, "pred": pred_xgb}
            except Exception as e:
                st.warning(f"XGBoost training failed: {e}")

        if not trained:
            st.error("No models trained successfully.")
            return

        rows = []
        for name, info in trained.items():
            preds = info["pred"]
            mets = classification_metrics(y_test, preds)
            rows.append({"Model": name, **mets})
            st.session_state[f"model_{name}"] = info["model"]

        metrics_df = pd.DataFrame(rows).set_index("Model")
        st.subheader("Model Comparison (classification metrics)")
        st.dataframe(metrics_df.style.format("{:.4f}"))

        # Visuals and reports
        for name, info in trained.items():
            st.markdown(f"---\n### {name}")
            preds = info["pred"]
            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", cmap="Greens", ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig)
            st.text("Classification Report:\n" + classification_report(y_test, preds, zero_division=0))

            mdl = info["model"]
            if hasattr(mdl, "feature_importances_"):
                try:
                    fi = mdl.feature_importances_
                    feat_names = X_enc.columns.tolist()
                    fi_df = pd.DataFrame({"feature": feat_names, "importance": fi}).sort_values("importance", ascending=False).head(20)
                    st.subheader("Top feature importances (Random Forest)")
                    st.table(fi_df.reset_index(drop=True))
                except Exception:
                    pass

        st.session_state["last_train"] = {
            "features": features,
            "features_encoded": X_enc.columns.tolist(),
            "X_test": X_test,
            "y_test": y_test,
            "label_encoder": lbl if 'lbl' in locals() else None,
            "target": target
        }
        # Optionally persist models
        if st.checkbox("Save trained models to /mnt/data", value=False):
            for name, info in trained.items():
                try:
                    path = save_model(info["model"], name=f"churn_{name.replace(' ','_')}.pkl")
                    st.write(f"Saved {name} -> {path}")
                except Exception as e:
                    st.warning(f"Failed saving {name}: {e}")

        st.success("Training complete. Models saved to session state.")

def prediction_playground_page(df):
    st.header("Prediction Playground")
    if "last_train" not in st.session_state:
        st.warning("No trained models found. Train in 'Modeling & Comparison' first.")
        return

    meta = st.session_state["last_train"]
    # build model list from session state
    available_models = {k.replace("model_", ""): st.session_state[k] for k in st.session_state if k.startswith("model_")}
    if not available_models:
        st.warning("No models found in session state.")
        return

    model_choice = st.selectbox("Pick model:", options=list(available_models.keys()), index=0)
    st.write("Enter feature values (numeric defaults = medians):")
    features = meta["features"]
    cols = st.columns(2)
    user_vals = {}
    for i,f in enumerate(features):
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f]):
            default = float(df[f].median())
            user_vals[f] = cols[i%2].number_input(f, value=default, format="%.3f")
        else:
            if f in df.columns:
                opts = df[f].dropna().unique().tolist()
                if opts and len(opts) <= 15:
                    user_vals[f] = cols[i%2].selectbox(f, options=opts)
                else:
                    user_vals[f] = cols[i%2].text_input(f, value=str(opts[0]) if opts else "")
            else:
                user_vals[f] = cols[i%2].text_input(f, value="")

    if st.button("Predict"):
        Xnew = pd.DataFrame([user_vals])
        Xnew_enc = auto_encode_features(Xnew)
        train_cols = meta["features_encoded"]
        for c in train_cols:
            if c not in Xnew_enc.columns:
                Xnew_enc[c] = 0
        Xnew_enc = Xnew_enc[train_cols]
        mdl = available_models[model_choice]
        try:
            pred = mdl.predict(Xnew_enc)[0]
            lbl = meta.get("label_encoder", None)
            if lbl is not None:
                try:
                    pred_label = lbl.inverse_transform([pred])[0]
                except Exception:
                    pred_label = str(pred)
            else:
                pred_label = str(pred)
            st.success(f"Predicted: {pred_label}")
            if hasattr(mdl, "predict_proba"):
                probs = mdl.predict_proba(Xnew_enc)[0]
                classes = mdl.classes_
                proba_df = pd.DataFrame({"class": classes, "probability": probs}).sort_values("probability", ascending=False)
                st.table(proba_df.reset_index(drop=True))
        except Exception as e:
            st.error(f"Prediction failed: {e}")

def about_page():
    st.header("About — Customer Churn Prediction")
    st.markdown("""
    Customer Churn Prediction.
    - Pages: Overview, Full EDA, Modeling & Comparison, Prediction Playground, About.
    - Models: Logistic Regression, Random Forest, optional XGBoost.
    - Automatic one-hot encoding of categorical features.
    - Fixed 80/20 train-test split. Trained models stored in session_state and optionally saved to disk.
    """)

# -------------------------
# Main
# -------------------------
def main():
    st.set_page_config(page_title="Customer Churn Pro", layout="wide", initial_sidebar_state="expanded")
    apply_css()
    st.markdown("<h1 style='font-weight:700'>☁️ Customer Churn Prediction</h1>", unsafe_allow_html=True)
    st.write("End-to-end Telco churn prediction dashboard. Upload dataset or use default path if present.")

    # Sidebar controls
    st.sidebar.header("Data & Controls")
    uploaded_file = st.sidebar.file_uploader("Upload Telco dataset (CSV or XLSX)", type=["csv", "xlsx"])
    st.sidebar.markdown(f"Default path: `{DEFAULT_DATA_PATH}` (used if you don't upload)")
    st.sidebar.markdown("---")
    st.sidebar.markdown("Tip: Train models in 'Modeling & Comparison', then use 'Prediction Playground'.")

    df = load_dataset(uploaded_file)
    if df is None:
        return

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Initial cleaning
    df = clean_numeric_like(df)

    # Navigation
    page = st.sidebar.radio("Navigate", ["Overview", "Full EDA", "Modeling & Comparison", "Prediction Playground", "About"])

    if page == "Overview":
        overview_page(df)
    elif page == "Full EDA":
        eda_page(df)
    elif page == "Modeling & Comparison":
        modeling_comparison_page(df)
    elif page == "Prediction Playground":
        prediction_playground_page(df)
    else:
        about_page()

if __name__ == "__main__":
    main()