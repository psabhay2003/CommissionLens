"""
utils.py — Shared helper functions used across the pipeline.

Key functions:
  - xirr()          : compute XIRR from a cashflow series
  - rolling_beta()  : CAPM beta over a rolling window
  - rolling_sharpe(): annualised Sharpe ratio over a rolling window
  - quarter_label() : convert a date to "2021Q3" style label
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from typing import List, Tuple


# ─────────────────────────────────────────
#  XIRR CALCULATION
# ─────────────────────────────────────────
def xirr(dates: List, cashflows: List, guess: float = 0.1) -> float:
    """
    Compute the annualised internal rate of return for irregular cashflows.

    Parameters
    ----------
    dates : list of datetime-like
        Dates of each cashflow.
    cashflows : list of float
        Negative = investment (outflow), Positive = redemption (inflow).
    guess : float
        Starting estimate for the solver.

    Returns
    -------
    float  — annualised rate (e.g. 0.12 means 12%)
    """
    if len(dates) != len(cashflows) or len(dates) < 2:
        return np.nan

    dates = pd.to_datetime(dates)
    t0 = dates.min()

    # year fractions from first date
    years = np.array([(d - t0).days / 365.25 for d in dates])
    cfs = np.array(cashflows, dtype=float)

    def npv(rate):
        # Net present value at a given discount rate
        return np.sum(cfs / (1.0 + rate) ** years)

    try:
        return brentq(npv, -0.5, 10.0, maxiter=1000)
    except (ValueError, RuntimeError):
        return np.nan


# ─────────────────────────────────────────
#  ROLLING FINANCIAL METRICS
# ─────────────────────────────────────────
def rolling_beta(fund_returns: pd.Series, bench_returns: pd.Series,
                 window: int = 4) -> pd.Series:
    """
    Compute rolling CAPM beta (covariance / variance of benchmark).

    Both series should be quarterly returns aligned on the same index.
    """
    cov = fund_returns.rolling(window).cov(bench_returns)
    var = bench_returns.rolling(window).var()
    return (cov / var).rename("rolling_beta")


def rolling_alpha(fund_returns: pd.Series, bench_returns: pd.Series,
                  beta: pd.Series, rf: float = 0.065,
                  periods_per_year: int = 4) -> pd.Series:
    """
    Jensen's alpha: R_fund - [Rf + beta * (R_bench - Rf)]
    Inputs are quarterly returns; output is quarterly alpha.
    """
    rf_q = rf / periods_per_year
    expected = rf_q + beta * (bench_returns - rf_q)
    return (fund_returns - expected).rename("rolling_alpha")


def rolling_sharpe(returns: pd.Series, rf: float = 0.065,
                   window: int = 4,
                   periods_per_year: int = 4) -> pd.Series:
    """Annualised Sharpe ratio over a rolling window of quarters."""
    rf_q = rf / periods_per_year
    excess = returns - rf_q
    mean = excess.rolling(window).mean() * periods_per_year
    std = excess.rolling(window).std() * np.sqrt(periods_per_year)
    return (mean / std).rename("rolling_sharpe")


def rolling_information_ratio(fund_returns: pd.Series,
                              bench_returns: pd.Series,
                              window: int = 4,
                              periods_per_year: int = 4) -> pd.Series:
    """
    Information ratio = mean(active return) / std(active return), annualised.
    """
    active = fund_returns - bench_returns
    mean_active = active.rolling(window).mean() * periods_per_year
    te = active.rolling(window).std() * np.sqrt(periods_per_year)
    return (mean_active / te).rename("information_ratio")


def rolling_sortino(returns: pd.Series, rf: float = 0.065,
                    window: int = 4,
                    periods_per_year: int = 4) -> pd.Series:
    """Annualised Sortino ratio (downside deviation only)."""
    rf_q = rf / periods_per_year
    excess = returns - rf_q

    def downside_std(x):
        neg = x[x < 0]
        if len(neg) < 2:
            return np.nan
        return neg.std() * np.sqrt(periods_per_year)

    ds = excess.rolling(window).apply(downside_std, raw=False)
    mean_ann = excess.rolling(window).mean() * periods_per_year
    return (mean_ann / ds).rename("rolling_sortino")


# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────
def quarter_label(dt) -> str:
    """Convert a datetime to '2021Q3' format."""
    dt = pd.Timestamp(dt)
    return f"{dt.year}Q{dt.quarter}"


def safe_pct_change(series: pd.Series) -> pd.Series:
    """Percentage change that replaces inf/nan with 0."""
    pct = series.pct_change()
    return pct.replace([np.inf, -np.inf], np.nan).fillna(0)


def clip_outliers(df: pd.DataFrame, cols: List[str],
                  lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Winsorise columns to the [lower, upper] quantile range."""
    df = df.copy()
    for col in cols:
        lo, hi = df[col].quantile([lower, upper])
        df[col] = df[col].clip(lo, hi)
    return df
