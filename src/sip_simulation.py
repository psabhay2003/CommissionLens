"""
sip_simulation.py — SIP back-validation analysis.

Compares three strategies over 2018–2023:
  A) Model-guided: Each quarter, pick top-K funds the DNN predicts
     will have justified commissions.  SIP into those.
  B) Naive baseline: SIP into ALL regular-plan funds equally.
  C) Direct-plan baseline: SIP into direct plans (no commission).

Run standalone:
    python -m src.sip_simulation
"""

import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (DATA_DIR, MODEL_DIR, REPORT_DIR,
                    SIP_MONTHLY_AMOUNT, TOP_K_FUNDS, TEST_QUARTERS, DNN_PARAMS)
from src.utils import xirr
from src.model_training import CommissionNet, FEATURE_COLS


def simulate_sip(nav_series, dates, monthly=5000):
    """Simulate monthly SIP on a single fund, return XIRR & final value."""
    df = pd.DataFrame({"date": dates, "nav": nav_series}).sort_values("date")
    df = df.set_index("date")
    monthly_nav = df.resample("MS").first().dropna()

    cfs, cf_dates, units = [], [], 0.0
    for dt, row in monthly_nav.iterrows():
        units += monthly / row["nav"]
        cfs.append(-monthly)
        cf_dates.append(dt)

    if len(monthly_nav) == 0:
        return {"xirr": np.nan, "final_value": 0, "invested": 0}

    final_val = units * df["nav"].iloc[-1]
    cfs.append(final_val)
    cf_dates.append(df.index[-1])

    return {
        "xirr": xirr(cf_dates, cfs),
        "final_value": final_val,
        "invested": monthly * len(monthly_nav),
    }


def run_sip_backtest(df=None):
    if df is None:
        df = pd.read_csv(DATA_DIR / "fund_dataset.csv",
                         parse_dates=["quarter_end"])

    nav_df = pd.read_csv(DATA_DIR / "raw_nav.csv", parse_dates=["date"])

    # Load DNN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        arch = joblib.load(MODEL_DIR / "dnn_arch.pkl")
        scaler = joblib.load(MODEL_DIR / "dnn_scaler.pkl")
        model = CommissionNet(
            arch["input_dim"], arch["hidden_layers"], arch["dropout"]
        ).to(device)
        model.load_state_dict(torch.load(MODEL_DIR / "dnn_model.pt",
                                         map_location=device, weights_only=True))
        model.eval()
    except FileNotFoundError:
        print("  ⚠ No trained model found. Run model_training.py first.")
        return None, None

    print("  ── SIP Back-Validation ──\n")

    # Get model predictions on test period
    sorted_quarters = sorted(df["quarter_end"].unique())
    test_start = sorted_quarters[-TEST_QUARTERS]
    test_data = df[df["quarter_end"] >= test_start].copy()

    available = [c for c in FEATURE_COLS if c in test_data.columns]
    X = test_data[available].fillna(0)
    X_scaled = scaler.transform(X)

    with torch.no_grad():
        _, logits = model(torch.FloatTensor(X_scaled).to(device))
        test_data["model_prob"] = torch.sigmoid(logits).cpu().numpy()

    # Select top-K funds per quarter
    selected_funds = set()
    for qtr, grp in test_data.groupby("quarter_end"):
        top = grp.nlargest(TOP_K_FUNDS, "model_prob")
        selected_funds.update(top["fund_name"].unique())

    all_funds = df["fund_name"].unique()
    print(f"    Model selected {len(selected_funds)} unique funds "
          f"from {len(all_funds)} total.\n")

    # Run SIPs
    results = {"strategy": [], "fund": [], "xirr": [],
               "invested": [], "final_value": []}

    for fund_name in all_funds:
        fnav = nav_df[nav_df["fund_name"] == fund_name]
        if len(fnav) < 12:
            continue

        # Naive Regular
        res = simulate_sip(fnav["nav_regular"], fnav["date"], SIP_MONTHLY_AMOUNT)
        results["strategy"].append("Naive Regular")
        results["fund"].append(fund_name)
        results["xirr"].append(res["xirr"])
        results["invested"].append(res["invested"])
        results["final_value"].append(res["final_value"])

        # Direct Plan
        res = simulate_sip(fnav["nav_direct"], fnav["date"], SIP_MONTHLY_AMOUNT)
        results["strategy"].append("Direct Plan")
        results["fund"].append(fund_name)
        results["xirr"].append(res["xirr"])
        results["invested"].append(res["invested"])
        results["final_value"].append(res["final_value"])

        # Model-Guided
        if fund_name in selected_funds:
            res = simulate_sip(fnav["nav_regular"], fnav["date"], SIP_MONTHLY_AMOUNT)
            results["strategy"].append("Model-Guided Regular")
            results["fund"].append(fund_name)
            results["xirr"].append(res["xirr"])
            results["invested"].append(res["invested"])
            results["final_value"].append(res["final_value"])

    results_df = pd.DataFrame(results).dropna(subset=["xirr"])

    summary = results_df.groupby("strategy").agg(
        mean_xirr=("xirr", "mean"),
        median_xirr=("xirr", "median"),
        mean_final=("final_value", "mean"),
        n_funds=("fund", "nunique"),
    ).round(4)

    print("  ── SIP XIRR Comparison ──")
    print(summary.to_string())

    # Chart
    palette = {"Naive Regular": "#E74C3C", "Direct Plan": "#2ECC71",
               "Model-Guided Regular": "#F1C40F"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.boxplot(data=results_df, x="strategy", y="xirr",
                palette=palette, ax=axes[0])
    axes[0].set_title("SIP XIRR by Strategy", fontsize=13)
    axes[0].set_ylabel("Annualised XIRR")
    axes[0].set_xlabel("")
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)

    mv = results_df.groupby("strategy")["final_value"].mean()
    bars = axes[1].bar(mv.index, mv.values,
                       color=[palette.get(s, "#999") for s in mv.index])
    axes[1].set_title("Mean Final Portfolio Value (₹)", fontsize=13)
    axes[1].set_ylabel("₹")
    axes[1].set_xlabel("")
    for bar, val in zip(bars, mv.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"₹{val:,.0f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "sip_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Chart → {REPORT_DIR / 'sip_comparison.png'}")

    results_df.to_csv(REPORT_DIR / "sip_results.csv", index=False)
    summary.to_csv(REPORT_DIR / "sip_summary.csv")

    return results_df, summary


if __name__ == "__main__":
    run_sip_backtest()
