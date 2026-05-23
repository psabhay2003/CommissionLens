"""
app.py — Streamlit Dashboard for CommissionLens.

Launch:
    streamlit run app.py
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
        model = joblib.load(MODEL_DIR / "ensemble_classifier.pkl")
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        meta = joblib.load(MODEL_DIR / "model_meta.pkl")
        return model, scaler, meta
    except FileNotFoundError:
        return None, None, None


@st.cache_data
def load_metrics():
    p = REPORT_DIR / "metrics.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def main():
    st.title("🔍 CommissionLens")
    st.markdown("**Commission-Adjusted Alpha Prediction for Indian Mutual Funds**")
    st.markdown("---")

    df = load_data()
    model, scaler, meta = load_model()
    metrics = load_metrics()

    if df is None or model is None:
        st.error("⚠️ Data or model not found. Run `python run_pipeline.py` first.")
        return

    feat_cols = meta["feature_cols"]
    threshold = meta.get("optimal_threshold", 0.5)

    st.sidebar.header("Select a Fund")
    fund_names = sorted(df["fund_name"].unique())
    selected = st.sidebar.selectbox("Fund Name", fund_names)

    fund_df = df[df["fund_name"] == selected].sort_values("quarter_end")
    if fund_df.empty:
        st.warning("No data for this fund.")
        return

    latest = fund_df.iloc[-1]
    available = [c for c in feat_cols if c in fund_df.columns]
    X = pd.DataFrame([latest[available].values], columns=available).fillna(0)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=available)

    prob = model.predict_proba(X_scaled)[0][1]
    score = prob * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Justification Score", f"{score:.1f}%",
              delta="Justified" if prob > threshold else "Unjustified")
    eg = latest.get("expense_gap_annualised", 0)
    c2.metric("Annualised Expense Gap",
              f"{eg*100:.2f}%" if pd.notna(eg) else "N/A")
    al = latest.get("rolling_alpha", 0)
    c3.metric("Rolling Alpha (1Y)",
              f"{al*100:.2f}%" if pd.notna(al) else "N/A")

    st.markdown("---")
    t1, t2, t3 = st.tabs(["📈 Net Alpha", "📊 Features", "🏦 Metrics"])

    with t1:
        if "net_alpha" in fund_df.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(fund_df["quarter_end"], fund_df["net_alpha"] * 100,
                   color=["#2ECC71" if v > 0 else "#E74C3C"
                          for v in fund_df["net_alpha"]])
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.set_ylabel("Net Alpha (%)"); plt.xticks(rotation=45)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with t2:
        feat = st.selectbox("Feature", [
            "rolling_sharpe", "rolling_beta", "rolling_alpha",
            "information_ratio", "volatility", "max_drawdown"])
        if feat in fund_df.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(fund_df["quarter_end"], fund_df[feat], marker="o")
            ax.set_ylabel(feat.replace("_", " ").title()); plt.xticks(rotation=45)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with t3:
        if metrics:
            st.json(metrics)

    st.markdown("---")
    st.subheader("🔬 SHAP Explainability")
    if (REPORT_DIR / "shap_summary.png").exists():
        st.image(str(REPORT_DIR / "shap_summary.png"))
    if (REPORT_DIR / "sip_comparison.png").exists():
        st.subheader("💰 SIP Back-Validation")
        st.image(str(REPORT_DIR / "sip_comparison.png"))


if __name__ == "__main__":
    main()
