"""
app.py — Streamlit Dashboard for CommissionLens.
Launch: streamlit run app.py
"""

import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="CommissionLens", page_icon="🔍", layout="wide")

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")


@st.cache_data
def load_data():
    p = DATA_DIR / "fund_dataset.csv"
    return pd.read_csv(p, parse_dates=["quarter_end"]) if p.exists() else None


@st.cache_resource
def load_model():
    try:
        models = joblib.load(MODEL_DIR / "trained_classifiers.pkl")
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        meta = joblib.load(MODEL_DIR / "model_meta.pkl")
        return models, scaler, meta
    except FileNotFoundError:
        return None, None, None


@st.cache_data
def load_metrics():
    p = REPORT_DIR / "metrics.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def get_ensemble_prob(models, X):
    probs = [m.predict_proba(X)[:, 1] for m in models.values()]
    return np.mean(probs, axis=0)


def main():
    st.title("🔍 CommissionLens")
    st.markdown("**Commission-Adjusted Alpha Prediction for Indian Mutual Funds**")
    st.markdown("---")

    df = load_data()
    models, scaler, meta = load_model()
    metrics = load_metrics()

    if df is None or models is None:
        st.error("⚠️ Run `python run_pipeline.py` first.")
        return

    feat_cols = meta["feature_cols"]
    threshold = meta.get("optimal_threshold", 0.5)

    st.sidebar.header("Select a Fund")
    fund_names = sorted(df["fund_name"].unique())
    selected = st.sidebar.selectbox("Fund Name", fund_names)

    fund_df = df[df["fund_name"] == selected].sort_values("quarter_end")
    if fund_df.empty:
        st.warning("No data.")
        return

    latest = fund_df.iloc[-1]
    available = [c for c in feat_cols if c in fund_df.columns]
    X = pd.DataFrame([latest[available].values], columns=available).fillna(0)
    X_scaled = scaler.transform(X)
    prob = get_ensemble_prob(models, X_scaled)[0]
    score = prob * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Justification Score", f"{score:.1f}%",
              delta="Justified" if prob > threshold else "Unjustified")
    eg = latest.get("expense_gap_annualised", 0)
    c2.metric("Expense Gap", f"{eg*100:.2f}%" if pd.notna(eg) else "N/A")
    al = latest.get("rolling_alpha", 0)
    c3.metric("Rolling Alpha", f"{al*100:.2f}%" if pd.notna(al) else "N/A")

    st.markdown("---")
    t1, t2, t3 = st.tabs(["📈 Net Alpha", "📊 Features", "🏦 Metrics"])

    with t1:
        if "net_alpha" in fund_df.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(fund_df["quarter_end"], fund_df["net_alpha"] * 100,
                   color=["#2ECC71" if v > 0 else "#E74C3C" for v in fund_df["net_alpha"]])
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.set_ylabel("Net Alpha (%)"); plt.xticks(rotation=45)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with t2:
        feat = st.selectbox("Feature", ["rolling_sharpe", "rolling_beta", "rolling_alpha",
            "information_ratio", "volatility", "max_drawdown", "alpha_consistency"])
        if feat in fund_df.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(fund_df["quarter_end"], fund_df[feat], marker="o")
            ax.set_ylabel(feat.replace("_", " ").title()); plt.xticks(rotation=45)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with t3:
        if metrics:
            st.json(metrics)

    st.markdown("---")
    if (REPORT_DIR / "shap_summary.png").exists():
        st.subheader("🔬 SHAP Explainability")
        st.image(str(REPORT_DIR / "shap_summary.png"))
    if (REPORT_DIR / "sip_comparison.png").exists():
        st.subheader("💰 SIP Back-Validation")
        st.image(str(REPORT_DIR / "sip_comparison.png"))


if __name__ == "__main__":
    main()
