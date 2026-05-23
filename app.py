"""
app.py — Streamlit Dashboard for CommissionLens.

A user inputs a fund name and gets:
  1. Commission-justification score (0–100%)
  2. Historical net alpha trend
  3. Rolling feature explorer
  4. Model metrics & SHAP visuals

Launch:
    streamlit run app.py
"""

import json
import numpy as np
import pandas as pd
import joblib
import torch
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.model_training import CommissionNet

st.set_page_config(page_title="CommissionLens", page_icon="🔍", layout="wide")

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

FEATURE_COLS = [
    "return_regular_lag1", "return_direct_lag1",
    "expense_gap_quarterly_lag1",
    "rolling_alpha_lag1", "rolling_beta_lag1",
    "rolling_sharpe_lag1", "information_ratio_lag1",
    "volatility_lag1",
    "repo_rate", "cpi_inflation", "yield_curve_slope_bps",
    "fii_net_flow_cr", "dii_net_flow_cr",
    "max_drawdown", "rolling_sortino",
    "expense_gap_annualised",
]


@st.cache_data
def load_data():
    p = DATA_DIR / "fund_dataset.csv"
    return pd.read_csv(p, parse_dates=["quarter_end"]) if p.exists() else None


@st.cache_resource
def load_model():
    try:
        device = torch.device("cpu")
        arch = joblib.load(MODEL_DIR / "dnn_arch.pkl")
        scaler = joblib.load(MODEL_DIR / "dnn_scaler.pkl")
        model = CommissionNet(
            arch["input_dim"], arch["hidden_layers"], arch["dropout"]
        )
        model.load_state_dict(torch.load(
            MODEL_DIR / "dnn_model.pt", map_location=device, weights_only=True
        ))
        model.eval()
        return model, scaler
    except FileNotFoundError:
        return None, None


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
    result = load_model()
    metrics = load_metrics()

    if df is None or result[0] is None:
        st.error("⚠️ Data or model not found. Run `python run_pipeline.py` first.")
        return

    model, scaler = result

    st.sidebar.header("Select a Fund")
    fund_names = sorted(df["fund_name"].unique())
    selected = st.sidebar.selectbox("Fund Name", fund_names)

    fund_df = df[df["fund_name"] == selected].sort_values("quarter_end")
    if fund_df.empty:
        st.warning("No data for this fund.")
        return

    latest = fund_df.iloc[-1]
    available = [c for c in FEATURE_COLS if c in fund_df.columns]
    X = pd.DataFrame([latest[available].values], columns=available).fillna(0)
    X_scaled = scaler.transform(X)

    with torch.no_grad():
        _, logit = model(torch.FloatTensor(X_scaled))
        prob = torch.sigmoid(logit).item()
    score = prob * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Justification Score", f"{score:.1f}%",
              delta="Justified" if score > 50 else "Unjustified")
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
