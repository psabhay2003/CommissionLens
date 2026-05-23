"""
target_builder.py — Construct the prediction targets.

Two targets:
  1. net_alpha (regression) = fund's quarterly excess return over benchmark
     MINUS the commission (expense gap).  Positive means the fund earned
     enough alpha to justify the commission cost.

  2. commission_justified (binary classification) = 1 if net_alpha > 0.

Run standalone:
    python -m src.target_builder
"""

import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, COMMISSION_JUSTIFIED_THRESHOLD


def build_targets(features_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add target columns to the feature DataFrame.

    Parameters
    ----------
    features_df : DataFrame
        Output of feature_engineering.py. If None, reads from CSV.

    Returns
    -------
    DataFrame with two new columns:
        - net_alpha           : float (quarterly)
        - commission_justified: int {0, 1}

    The targets are for the NEXT quarter (shifted), so the model learns to
    predict future commission justification from current features.
    """
    if features_df is None:
        features_df = pd.read_csv(
            DATA_DIR / "fund_features.csv", parse_dates=["quarter_end"]
        )

    df = features_df.sort_values(["fund_name", "quarter_end"]).copy()

    # ── Compute net alpha for the CURRENT quarter ──
    # net_alpha = (fund return - benchmark return) - expense_gap
    # This is the "true" alpha after subtracting the commission
    df["gross_alpha"] = df["return_regular"] - df["bench_return"]
    df["net_alpha"] = df["gross_alpha"] - df["expense_gap_quarterly"]

    # ── Binary label ──
    df["commission_justified"] = (
        df["net_alpha"] > COMMISSION_JUSTIFIED_THRESHOLD
    ).astype(int)

    # ── Shift targets forward: we want to PREDICT next quarter's outcome ──
    # Features at time t predict targets at time t+1
    df["target_net_alpha"] = df.groupby("fund_name")["net_alpha"].shift(-1)
    df["target_justified"] = df.groupby("fund_name")["commission_justified"].shift(-1)

    # Drop last quarter per fund (no future target available)
    df = df.dropna(subset=["target_net_alpha", "target_justified"])
    df["target_justified"] = df["target_justified"].astype(int)

    # Save
    out_path = DATA_DIR / "fund_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"  ✓ Dataset with targets saved → {out_path}")
    print(f"    Total samples: {len(df)}")
    print(f"    Justified: {df['target_justified'].sum()} "
          f"({df['target_justified'].mean():.1%})")
    print(f"    Unjustified: {(1 - df['target_justified']).sum()} "
          f"({1 - df['target_justified'].mean():.1%})")

    return df


if __name__ == "__main__":
    build_targets()
