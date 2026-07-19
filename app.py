"""Streamlit dashboard: pick a fund, see its commission-justification score."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from config import load_config
from features import FEATURE_COLUMNS
from model import TrainedModel

st.set_page_config(page_title="CommissionLens", layout="centered")


@st.cache_resource
def _load():
    config = load_config()
    panel = pd.read_parquet(config.data_path / "panel.parquet")
    model = TrainedModel.load(config.artifacts_path / "commissionnet.pt")
    return config, panel, model


def main() -> None:
    st.title("CommissionLens")
    st.caption("Commission-adjusted alpha prediction for Indian equity mutual funds")

    try:
        config, panel, model = _load()
    except FileNotFoundError:
        st.error("Artifacts missing. Run `py main.py build` and `py main.py train` first.")
        return

    fund_name = st.selectbox("Select a fund", sorted(panel["fund_name"].unique()))
    fund = panel[panel["fund_name"] == fund_name].sort_values("quarter_end")

    if len(fund) < config.model.sequence_length:
        st.warning("Not enough quarterly history for this fund to score.")
        return

    window = fund[FEATURE_COLUMNS].to_numpy(dtype=np.float32)[-config.model.sequence_length:]
    net_alpha, probability = model.predict(window[np.newaxis, :, :])
    score, alpha = float(probability[0]), float(net_alpha[0])
    latest_quarter = pd.Timestamp(fund["quarter_end"].max()).date()

    left, right = st.columns(2)
    left.metric("Commission-justification score", f"{score:.0%}")
    right.metric("Predicted next-quarter net alpha", f"{alpha * 100:.2f}%")

    verdict = "likely to justify its commission" if score >= model.threshold else "unlikely to justify its commission"
    st.write(f"As of quarter ending **{latest_quarter}**, this fund is **{verdict}**.")
    st.progress(min(max(score, 0.0), 1.0))

    st.subheader("Latest fund-quarter features")
    st.dataframe(fund.iloc[-1][FEATURE_COLUMNS].rename("value").to_frame())

    st.subheader("Rolling alpha vs expense gap")
    st.line_chart(fund.set_index("quarter_end")[["alpha_annual", "expense_gap"]])


if __name__ == "__main__":
    main()
