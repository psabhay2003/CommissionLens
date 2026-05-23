"""
target_builder.py — Construct the prediction targets.

KEY CHANGE: Smoothed multi-quarter target.

Old target: "will net_alpha be > 0 NEXT quarter?"
  Problem: flips randomly quarter to quarter — unpredictable noise.

New target: "will the AVERAGE net_alpha over the next 2 quarters be > 0?"
  This is far more stable and learnable. A fund that consistently
  earns back its commission will have positive average even if one
  quarter dips. A fund that consistently fails will have negative
  average even if one quarter spikes.

  For regression, we also predict the 2-quarter average — smoother
  target = better R².

Run standalone:
    python -m src.target_builder
"""

import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, COMMISSION_JUSTIFIED_THRESHOLD


# How many forward quarters to average for the target
FORWARD_QUARTERS = 2


def build_targets(features_df=None):
    if features_df is None:
        features_df = pd.read_csv(
            DATA_DIR / "fund_features.csv", parse_dates=["quarter_end"]
        )

    df = features_df.sort_values(["fund_name", "quarter_end"]).copy()

    # ── Compute current-quarter net alpha ──
    df["gross_alpha"] = df["return_regular"] - df["bench_return"]
    df["net_alpha"] = df["gross_alpha"] - df["expense_gap_quarterly"]
    df["commission_justified"] = (
        df["net_alpha"] > COMMISSION_JUSTIFIED_THRESHOLD
    ).astype(int)

    # ── Smoothed forward-looking target ──
    # Average net_alpha over next FORWARD_QUARTERS quarters
    shifted = []
    for i in range(1, FORWARD_QUARTERS + 1):
        shifted.append(
            df.groupby("fund_name")["net_alpha"].shift(-i)
        )

    # Stack and average
    df["target_net_alpha"] = pd.concat(shifted, axis=1).mean(axis=1)

    # Binary: is the average forward net_alpha positive?
    df["target_justified"] = (
        df["target_net_alpha"] > COMMISSION_JUSTIFIED_THRESHOLD
    ).astype(float)  # float first for NaN handling

    # Drop rows without enough future data
    df = df.dropna(subset=["target_net_alpha", "target_justified"])
    df["target_justified"] = df["target_justified"].astype(int)

    # ── Clip extreme regression targets ──
    lo, hi = df["target_net_alpha"].quantile([0.02, 0.98])
    df["target_net_alpha"] = df["target_net_alpha"].clip(lo, hi)

    # Save
    out_path = DATA_DIR / "fund_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"  ✓ Dataset with targets saved → {out_path}")
    print(f"    Smoothing: average of next {FORWARD_QUARTERS} quarters")
    print(f"    Total samples: {len(df)}")
    print(f"    Justified: {df['target_justified'].sum()} "
          f"({df['target_justified'].mean():.1%})")
    print(f"    Unjustified: {(1 - df['target_justified']).sum()} "
          f"({1 - df['target_justified'].mean():.1%})")

    return df


if __name__ == "__main__":
    build_targets()
