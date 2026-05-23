"""
feature_engineering.py — Transform raw NAV data into quarterly features.

KEY FIX in this version:
  Added aggressive data quality validation. The expense_gap between
  regular and direct plans should be 0.1-0.4% per quarter (0.5-1.5%
  annualised). Any fund showing gaps of 2%+ per quarter has mismatched
  scheme codes — the regular and direct aren't the same underlying fund.
  These are detected and removed before they corrupt everything downstream.

Run standalone:
    python -m src.feature_engineering
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (DATA_DIR, MACRO_CSV, ROLLING_WINDOW_QUARTERS,
                    RISK_FREE_RATE, QUARTER_FREQ)
from src.utils import (rolling_beta, rolling_alpha, rolling_sharpe,
                       rolling_information_ratio, rolling_sortino)


def resample_nav_to_quarterly(nav_df: pd.DataFrame) -> pd.DataFrame:
    """Convert daily NAV data to quarter-end NAV values."""
    nav_df = nav_df.copy()
    nav_df["date"] = pd.to_datetime(nav_df["date"])
    nav_df["quarter_end"] = nav_df["date"] + pd.offsets.QuarterEnd(0)

    quarterly = (
        nav_df
        .sort_values("date")
        .groupby(["fund_name", "quarter_end"])
        .last()
        .reset_index()
    )
    return quarterly[["fund_name", "quarter_end",
                       "nav_regular", "nav_direct",
                       "scheme_regular", "scheme_direct"]]


def compute_quarterly_returns(quarterly_df: pd.DataFrame) -> pd.DataFrame:
    """Add quarterly return columns for both regular and direct plans."""
    df = quarterly_df.sort_values(["fund_name", "quarter_end"]).copy()
    df["return_regular"] = df.groupby("fund_name")["nav_regular"].pct_change()
    df["return_direct"] = df.groupby("fund_name")["nav_direct"].pct_change()
    return df


def compute_expense_ratio_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate the implied commission (expense ratio gap) per quarter.
    expense_gap ≈ return_direct - return_regular  (per quarter)
    """
    df = df.copy()
    df["expense_gap_quarterly"] = df["return_direct"] - df["return_regular"]
    df["expense_gap_annualised"] = df["expense_gap_quarterly"] * 4
    return df


# ═══════════════════════════════════════════
#  DATA QUALITY VALIDATION — CRITICAL
# ═══════════════════════════════════════════
def validate_fund_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove funds with mismatched scheme codes.

    Real regular-vs-direct expense gap is 0.1–0.4% per quarter
    (both plans hold identical portfolios).  If a fund consistently
    shows gaps > 1.5% per quarter, the scheme codes are wrong —
    they're pulling NAVs from two different funds entirely.

    This is the #1 reason the model was performing at random.
    Garbage targets → garbage predictions.
    """
    MAX_QUARTERLY_GAP = 0.015   # 1.5% per quarter = 6% annual (generous)
    MIN_CLEAN_RATIO = 0.5       # at least 50% of quarters must be clean

    df = df.copy()
    before_count = df["fund_name"].nunique()

    fund_quality = []
    for fund_name, group in df.groupby("fund_name"):
        valid = group["expense_gap_quarterly"].dropna()
        if len(valid) == 0:
            fund_quality.append((fund_name, 0, 0))
            continue

        # Count quarters where gap is within reasonable range
        clean = (valid.abs() <= MAX_QUARTERLY_GAP).sum()
        ratio = clean / len(valid)
        median_gap = valid.median()
        fund_quality.append((fund_name, ratio, median_gap))

    quality_df = pd.DataFrame(fund_quality,
                              columns=["fund_name", "clean_ratio", "median_gap"])

    # Keep funds where majority of quarters have reasonable gaps
    good_funds = quality_df[quality_df["clean_ratio"] >= MIN_CLEAN_RATIO]["fund_name"]
    bad_funds = quality_df[quality_df["clean_ratio"] < MIN_CLEAN_RATIO]["fund_name"]

    if len(bad_funds) > 0:
        print(f"  ⚠ Removed {len(bad_funds)} funds with mismatched scheme codes:")
        for _, row in quality_df[quality_df["clean_ratio"] < MIN_CLEAN_RATIO].iterrows():
            print(f"      {row['fund_name']:>40s}  "
                  f"(clean: {row['clean_ratio']:.0%}, "
                  f"median gap: {row['median_gap']*400:.1f}% ann.)")

    df = df[df["fund_name"].isin(good_funds)].copy()

    # Even for good funds, clip individual outlier quarters
    gap_lo = df["expense_gap_quarterly"].quantile(0.02)
    gap_hi = df["expense_gap_quarterly"].quantile(0.98)
    df["expense_gap_quarterly"] = df["expense_gap_quarterly"].clip(gap_lo, gap_hi)
    df["expense_gap_annualised"] = df["expense_gap_quarterly"] * 4

    after_count = df["fund_name"].nunique()
    print(f"  ✓ Fund validation: {before_count} → {after_count} funds "
          f"({before_count - after_count} removed)")

    return df


def compute_benchmark_returns(bench_df: pd.DataFrame) -> pd.DataFrame:
    """Convert daily benchmark close to quarterly returns."""
    bench = bench_df.copy()
    bench["date"] = pd.to_datetime(bench["date"])
    bench = bench.set_index("date").resample(QUARTER_FREQ).last().reset_index()
    bench = bench.rename(columns={"date": "quarter_end"})
    bench["bench_return"] = bench["benchmark_close"].pct_change()
    return bench[["quarter_end", "bench_return", "benchmark_close"]]


def add_rolling_features(df: pd.DataFrame,
                         bench_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Merge benchmark returns and compute all rolling financial metrics.
    """
    df = df.merge(bench_returns, on="quarter_end", how="left")
    window = ROLLING_WINDOW_QUARTERS
    rf = RISK_FREE_RATE
    results = []

    for fund_name, group in df.groupby("fund_name"):
        g = group.sort_values("quarter_end").copy()

        g["rolling_beta"] = rolling_beta(
            g["return_regular"], g["bench_return"], window
        ).values

        g["rolling_alpha"] = rolling_alpha(
            g["return_regular"], g["bench_return"],
            g["rolling_beta"], rf
        ).values

        g["rolling_sharpe"] = rolling_sharpe(
            g["return_regular"], rf, window
        ).values

        g["rolling_sortino"] = rolling_sortino(
            g["return_regular"], rf, window
        ).values

        g["information_ratio"] = rolling_information_ratio(
            g["return_regular"], g["bench_return"], window
        ).values

        g["volatility"] = (
            g["return_regular"]
            .rolling(window)
            .std() * np.sqrt(4)
        )

        # Max drawdown (rolling on NAV)
        g["cum_max"] = g["nav_regular"].cummax()
        g["drawdown"] = (g["nav_regular"] - g["cum_max"]) / g["cum_max"]
        g["max_drawdown"] = g["drawdown"].rolling(window).min()
        g = g.drop(columns=["cum_max", "drawdown"])

        results.append(g)

    return pd.concat(results, ignore_index=True)


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge quarterly macro indicators onto the fund features."""
    if not MACRO_CSV.exists():
        print("  ⚠ Macro CSV not found, skipping macro features.")
        return df

    macro = pd.read_csv(MACRO_CSV, parse_dates=["quarter_end"])
    df = df.merge(macro, on="quarter_end", how="left")

    macro_cols = ["repo_rate", "cpi_inflation", "yield_curve_slope_bps",
                  "fii_net_flow_cr", "dii_net_flow_cr"]
    for col in macro_cols:
        if col in df.columns:
            df[col] = df[col].ffill()
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 1-quarter lagged features to prevent data leakage."""
    lag_cols = ["return_regular", "return_direct", "expense_gap_quarterly",
                "rolling_alpha", "rolling_beta", "rolling_sharpe",
                "information_ratio", "volatility"]

    df = df.sort_values(["fund_name", "quarter_end"])
    for col in lag_cols:
        if col in df.columns:
            df[f"{col}_lag1"] = df.groupby("fund_name")[col].shift(1)
    return df


def clip_feature_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Winsorise extreme values in rolling features.
    Financial data is fat-tailed; extreme values hurt gradient-based learning.
    """
    clip_cols = ["rolling_beta", "rolling_alpha", "rolling_sharpe",
                 "rolling_sortino", "information_ratio", "volatility",
                 "return_regular", "return_direct"]
    clip_cols = [c for c in clip_cols if c in df.columns]

    for col in clip_cols:
        lo, hi = df[col].quantile([0.02, 0.98])
        df[col] = df[col].clip(lo, hi)
    return df


# ═══════════════════════════════════════════
#  MAIN PIPELINE FUNCTION
# ═══════════════════════════════════════════
def engineer_features(nav_df=None, bench_df=None) -> pd.DataFrame:
    if nav_df is None:
        nav_df = pd.read_csv(DATA_DIR / "raw_nav.csv", parse_dates=["date"])
    if bench_df is None:
        bench_df = pd.read_csv(DATA_DIR / "benchmark_nifty50.csv",
                               parse_dates=["date"])

    print("  → Resampling to quarterly NAVs...")
    quarterly = resample_nav_to_quarterly(nav_df)

    print("  → Computing quarterly returns...")
    quarterly = compute_quarterly_returns(quarterly)

    print("  → Computing expense ratio gap...")
    quarterly = compute_expense_ratio_gap(quarterly)

    print("  → Validating fund pair integrity...")
    quarterly = validate_fund_pairs(quarterly)

    print("  → Computing benchmark quarterly returns...")
    bench_q = compute_benchmark_returns(bench_df)

    print("  → Adding rolling financial features...")
    quarterly = add_rolling_features(quarterly, bench_q)

    print("  → Clipping feature outliers...")
    quarterly = clip_feature_outliers(quarterly)

    print("  → Adding macro features...")
    quarterly = add_macro_features(quarterly)

    print("  → Adding lag features...")
    quarterly = add_lag_features(quarterly)

    # Drop rows where rolling features are NaN (initial window)
    initial_rows = len(quarterly)
    quarterly = quarterly.dropna(
        subset=["rolling_alpha", "rolling_beta", "rolling_sharpe"]
    ).reset_index(drop=True)
    print(f"  → Dropped {initial_rows - len(quarterly)} rows with "
          f"insufficient history. {len(quarterly)} rows remain.")

    out_path = DATA_DIR / "fund_features.csv"
    quarterly.to_csv(out_path, index=False)
    print(f"  ✓ Saved features → {out_path}")
    return quarterly


if __name__ == "__main__":
    engineer_features()
