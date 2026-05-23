"""
feature_engineering.py — Transform raw NAV data into quarterly features.

This is the analytical core.  For each fund × quarter, we compute:

  Fund-level features:
    - quarterly_return_regular   : % return of regular plan
    - quarterly_return_direct    : % return of direct plan
    - expense_ratio_gap          : implied commission (direct NAV growth - regular)
    - rolling_beta               : CAPM beta vs Nifty 50 (4-quarter window)
    - rolling_alpha              : Jensen's alpha (4-quarter window)
    - rolling_sharpe             : Annualised Sharpe ratio
    - rolling_sortino            : Annualised Sortino ratio
    - information_ratio          : Active return / tracking error
    - volatility                 : Annualised std of quarterly returns
    - max_drawdown               : Worst peak-to-trough loss (rolling 1Y)

  Macro features (merged from macro CSV):
    - repo_rate, cpi_inflation, yield_curve_slope_bps
    - fii_net_flow_cr, dii_net_flow_cr

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
    """
    Convert daily NAV data to quarter-end NAV values.

    For each fund, take the last available NAV in each calendar quarter.
    """
    nav_df = nav_df.copy()
    nav_df["date"] = pd.to_datetime(nav_df["date"])
    nav_df["quarter_end"] = nav_df["date"] + pd.offsets.QuarterEnd(0)

    # Take the last NAV observation within each quarter for each fund
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

    Logic: If both plans hold identical portfolios, any difference in NAV
    growth is due to the expense difference (which includes the distributor
    commission in the regular plan).

    expense_gap ≈ return_direct - return_regular  (per quarter)
    Annualised: expense_gap * 4
    """
    df = df.copy()
    df["expense_gap_quarterly"] = df["return_direct"] - df["return_regular"]
    df["expense_gap_annualised"] = df["expense_gap_quarterly"] * 4
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
    Merge benchmark returns and compute all rolling financial metrics
    for each fund.
    """
    df = df.merge(bench_returns, on="quarter_end", how="left")
    window = ROLLING_WINDOW_QUARTERS
    rf = RISK_FREE_RATE
    results = []

    for fund_name, group in df.groupby("fund_name"):
        g = group.sort_values("quarter_end").copy()

        # Beta
        g["rolling_beta"] = rolling_beta(
            g["return_regular"], g["bench_return"], window
        ).values

        # Alpha (Jensen's)
        g["rolling_alpha"] = rolling_alpha(
            g["return_regular"], g["bench_return"],
            g["rolling_beta"], rf
        ).values

        # Sharpe
        g["rolling_sharpe"] = rolling_sharpe(
            g["return_regular"], rf, window
        ).values

        # Sortino
        g["rolling_sortino"] = rolling_sortino(
            g["return_regular"], rf, window
        ).values

        # Information ratio
        g["information_ratio"] = rolling_information_ratio(
            g["return_regular"], g["bench_return"], window
        ).values

        # Volatility (annualised)
        g["volatility"] = (
            g["return_regular"]
            .rolling(window)
            .std() * np.sqrt(4)
        )

        # Max drawdown (rolling 1-year on NAV)
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

    # Forward-fill any missing macro values
    macro_cols = ["repo_rate", "cpi_inflation", "yield_curve_slope_bps",
                  "fii_net_flow_cr", "dii_net_flow_cr"]
    for col in macro_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 1-quarter lagged versions of key features.
    These prevent data leakage — the model sees only past information.
    """
    lag_cols = ["return_regular", "return_direct", "expense_gap_quarterly",
                "rolling_alpha", "rolling_beta", "rolling_sharpe",
                "information_ratio", "volatility"]

    df = df.sort_values(["fund_name", "quarter_end"])
    for col in lag_cols:
        if col in df.columns:
            df[f"{col}_lag1"] = df.groupby("fund_name")[col].shift(1)

    return df


# ═══════════════════════════════════════════
#  MAIN PIPELINE FUNCTION
# ═══════════════════════════════════════════
def engineer_features(nav_df: pd.DataFrame = None,
                      bench_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    nav_df : DataFrame
        Raw NAV data (from data_collection). If None, reads from CSV.
    bench_df : DataFrame
        Benchmark close prices. If None, reads from CSV.

    Returns
    -------
    DataFrame — one row per (fund, quarter) with all features.
    """
    # Load if not provided
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

    print("  → Computing benchmark quarterly returns...")
    bench_q = compute_benchmark_returns(bench_df)

    print("  → Adding rolling financial features...")
    quarterly = add_rolling_features(quarterly, bench_q)

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

    # Save
    out_path = DATA_DIR / "fund_features.csv"
    quarterly.to_csv(out_path, index=False)
    print(f"  ✓ Saved features → {out_path}")

    return quarterly


if __name__ == "__main__":
    engineer_features()
