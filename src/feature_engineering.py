"""
feature_engineering.py — Transform raw NAV data into quarterly features.

KEY CHANGES in this version:
  Added FUND-SPECIFIC features that capture individual fund behavior:
    - alpha_consistency    : % of past quarters with positive net alpha
    - expense_gap_stability: how stable is the commission gap (low std = predictable)
    - alpha_momentum       : is the fund's alpha trending up or down
    - relative_return      : this fund vs median of all funds this quarter
    - net_alpha_streak     : consecutive quarters of positive/negative net alpha

  These give the model fund-level signal instead of only macro data.

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


def resample_nav_to_quarterly(nav_df):
    nav_df = nav_df.copy()
    nav_df["date"] = pd.to_datetime(nav_df["date"])
    nav_df["quarter_end"] = nav_df["date"] + pd.offsets.QuarterEnd(0)
    quarterly = (
        nav_df.sort_values("date")
        .groupby(["fund_name", "quarter_end"]).last().reset_index()
    )
    return quarterly[["fund_name", "quarter_end", "nav_regular", "nav_direct",
                       "scheme_regular", "scheme_direct"]]


def compute_quarterly_returns(df):
    df = df.sort_values(["fund_name", "quarter_end"]).copy()
    df["return_regular"] = df.groupby("fund_name")["nav_regular"].pct_change()
    df["return_direct"] = df.groupby("fund_name")["nav_direct"].pct_change()
    return df


def compute_expense_ratio_gap(df):
    df = df.copy()
    df["expense_gap_quarterly"] = df["return_direct"] - df["return_regular"]
    df["expense_gap_annualised"] = df["expense_gap_quarterly"] * 4
    return df


def validate_fund_pairs(df):
    MAX_QUARTERLY_GAP = 0.015
    MIN_CLEAN_RATIO = 0.5
    df = df.copy()
    before_count = df["fund_name"].nunique()

    fund_quality = []
    for fund_name, group in df.groupby("fund_name"):
        valid = group["expense_gap_quarterly"].dropna()
        if len(valid) == 0:
            fund_quality.append((fund_name, 0, 0))
            continue
        clean = (valid.abs() <= MAX_QUARTERLY_GAP).sum()
        ratio = clean / len(valid)
        median_gap = valid.median()
        fund_quality.append((fund_name, ratio, median_gap))

    quality_df = pd.DataFrame(fund_quality,
                              columns=["fund_name", "clean_ratio", "median_gap"])
    good_funds = quality_df[quality_df["clean_ratio"] >= MIN_CLEAN_RATIO]["fund_name"]
    bad_funds = quality_df[quality_df["clean_ratio"] < MIN_CLEAN_RATIO]["fund_name"]

    if len(bad_funds) > 0:
        print(f"  ⚠ Removed {len(bad_funds)} funds with mismatched scheme codes:")
        for _, row in quality_df[quality_df["clean_ratio"] < MIN_CLEAN_RATIO].iterrows():
            print(f"      {row['fund_name']:>40s}  "
                  f"(clean: {row['clean_ratio']:.0%}, "
                  f"median gap: {row['median_gap']*400:.1f}% ann.)")

    df = df[df["fund_name"].isin(good_funds)].copy()
    gap_lo = df["expense_gap_quarterly"].quantile(0.02)
    gap_hi = df["expense_gap_quarterly"].quantile(0.98)
    df["expense_gap_quarterly"] = df["expense_gap_quarterly"].clip(gap_lo, gap_hi)
    df["expense_gap_annualised"] = df["expense_gap_quarterly"] * 4

    after_count = df["fund_name"].nunique()
    print(f"  ✓ Fund validation: {before_count} → {after_count} funds "
          f"({before_count - after_count} removed)")
    return df


def compute_benchmark_returns(bench_df):
    bench = bench_df.copy()
    bench["date"] = pd.to_datetime(bench["date"])
    bench = bench.set_index("date").resample(QUARTER_FREQ).last().reset_index()
    bench = bench.rename(columns={"date": "quarter_end"})
    bench["bench_return"] = bench["benchmark_close"].pct_change()
    return bench[["quarter_end", "bench_return", "benchmark_close"]]


def add_rolling_features(df, bench_returns):
    df = df.merge(bench_returns, on="quarter_end", how="left")
    window = ROLLING_WINDOW_QUARTERS
    rf = RISK_FREE_RATE
    results = []

    for fund_name, group in df.groupby("fund_name"):
        g = group.sort_values("quarter_end").copy()
        g["rolling_beta"] = rolling_beta(g["return_regular"], g["bench_return"], window).values
        g["rolling_alpha"] = rolling_alpha(g["return_regular"], g["bench_return"], g["rolling_beta"], rf).values
        g["rolling_sharpe"] = rolling_sharpe(g["return_regular"], rf, window).values
        g["rolling_sortino"] = rolling_sortino(g["return_regular"], rf, window).values
        g["information_ratio"] = rolling_information_ratio(g["return_regular"], g["bench_return"], window).values
        g["volatility"] = g["return_regular"].rolling(window).std() * np.sqrt(4)

        cum_max = g["nav_regular"].cummax()
        dd = (g["nav_regular"] - cum_max) / cum_max
        g["max_drawdown"] = dd.rolling(window).min()
        results.append(g)

    return pd.concat(results, ignore_index=True)


# ═══════════════════════════════════════════
#  FUND-SPECIFIC FEATURES — NEW
# ═══════════════════════════════════════════
def add_fund_specific_features(df):
    """
    Add features that capture individual fund behavior patterns.
    These are what differentiate fund A from fund B in the same quarter.
    Without these, the model can only learn macro patterns (same for all funds).
    """
    df = df.sort_values(["fund_name", "quarter_end"]).copy()
    window = ROLLING_WINDOW_QUARTERS

    # 1. Gross alpha (fund return - benchmark return) for intermediate calcs
    df["gross_alpha"] = df["return_regular"] - df["bench_return"]

    # 2. Alpha consistency: % of past quarters with positive gross alpha
    #    Funds that consistently beat the benchmark are more likely to continue
    def pct_positive(x):
        if len(x) == 0:
            return 0.5
        return (x > 0).mean()

    df["alpha_consistency"] = (
        df.groupby("fund_name")["gross_alpha"]
        .transform(lambda x: x.rolling(window, min_periods=2).apply(pct_positive, raw=False))
    )

    # 3. Expense gap stability: std of expense gap over rolling window
    #    Low std = predictable, stable commission structure
    df["expense_gap_stability"] = (
        df.groupby("fund_name")["expense_gap_quarterly"]
        .transform(lambda x: x.rolling(window, min_periods=2).std())
    )
    # Invert: lower std = higher stability score
    max_stab = df["expense_gap_stability"].quantile(0.95)
    df["expense_gap_stability"] = (max_stab - df["expense_gap_stability"]).clip(0)

    # 4. Alpha momentum: trend of gross_alpha over last 4 quarters
    #    Positive = improving, negative = deteriorating
    def slope(x):
        if len(x) < 3:
            return 0.0
        t = np.arange(len(x))
        try:
            return np.polyfit(t, x, 1)[0]
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    df["alpha_momentum"] = (
        df.groupby("fund_name")["gross_alpha"]
        .transform(lambda x: x.rolling(window, min_periods=3).apply(slope, raw=True))
    )

    # 5. Relative return: this fund vs median of all funds this quarter
    #    >0 means top-half performer, <0 means bottom-half
    quarter_median = df.groupby("quarter_end")["return_regular"].transform("median")
    df["relative_return"] = df["return_regular"] - quarter_median

    # 6. Relative expense gap: this fund's gap vs median gap across funds
    quarter_median_gap = df.groupby("quarter_end")["expense_gap_quarterly"].transform("median")
    df["relative_expense_gap"] = df["expense_gap_quarterly"] - quarter_median_gap

    # 7. Net alpha streak: consecutive quarters of same sign
    def streak_count(series):
        result = np.zeros(len(series))
        streak = 0
        for i in range(len(series)):
            if pd.isna(series.iloc[i]):
                streak = 0
            elif series.iloc[i] > 0:
                streak = streak + 1 if streak > 0 else 1
            else:
                streak = streak - 1 if streak < 0 else -1
            result[i] = streak
        return pd.Series(result, index=series.index)

    net_alpha_temp = df["gross_alpha"] - df["expense_gap_quarterly"]
    df["net_alpha_streak"] = df.groupby("fund_name").apply(
        lambda g: streak_count(net_alpha_temp.loc[g.index])
    ).reset_index(level=0, drop=True)

    # Drop the intermediate gross_alpha (will be recomputed in target_builder)
    df = df.drop(columns=["gross_alpha"])

    print(f"  ✓ Added 7 fund-specific features")
    return df


def add_macro_features(df):
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


def add_lag_features(df):
    lag_cols = ["return_regular", "return_direct", "expense_gap_quarterly",
                "rolling_alpha", "rolling_beta", "rolling_sharpe",
                "information_ratio", "volatility",
                # Also lag the new fund-specific features
                "alpha_consistency", "alpha_momentum", "relative_return",
                "net_alpha_streak"]

    df = df.sort_values(["fund_name", "quarter_end"])
    for col in lag_cols:
        if col in df.columns:
            df[f"{col}_lag1"] = df.groupby("fund_name")[col].shift(1)
    return df


def clip_feature_outliers(df):
    clip_cols = ["rolling_beta", "rolling_alpha", "rolling_sharpe",
                 "rolling_sortino", "information_ratio", "volatility",
                 "return_regular", "return_direct",
                 "alpha_momentum", "relative_return"]
    clip_cols = [c for c in clip_cols if c in df.columns]
    for col in clip_cols:
        lo, hi = df[col].quantile([0.02, 0.98])
        df[col] = df[col].clip(lo, hi)
    return df


# ═══════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════
def engineer_features(nav_df=None, bench_df=None):
    if nav_df is None:
        nav_df = pd.read_csv(DATA_DIR / "raw_nav.csv", parse_dates=["date"])
    if bench_df is None:
        bench_df = pd.read_csv(DATA_DIR / "benchmark_nifty50.csv", parse_dates=["date"])

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

    print("  → Adding fund-specific features...")
    quarterly = add_fund_specific_features(quarterly)

    print("  → Clipping feature outliers...")
    quarterly = clip_feature_outliers(quarterly)

    print("  → Adding macro features...")
    quarterly = add_macro_features(quarterly)

    print("  → Adding lag features...")
    quarterly = add_lag_features(quarterly)

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
