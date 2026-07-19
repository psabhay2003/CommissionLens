"""Risk metrics and the quarterly fund-macro panel with a forward net-alpha target."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config
from data import SyntheticPanel

MONTHS_PER_YEAR = 12
QUARTERS_PER_YEAR = 4

ROLLING_FEATURES = [
    "alpha_annual", "beta", "sharpe", "info_ratio",
    "volatility", "tracking_error", "max_drawdown",
]
FUND_FEATURES = [
    "expense_ratio_regular", "expense_gap", "log_aum",
    "portfolio_turnover", "fund_manager_tenure", "fund_age",
]
MACRO_FEATURES = ["repo_rate", "cpi_inflation", "yield_slope", "fii_flow", "dii_flow"]
FEATURE_COLUMNS = ROLLING_FEATURES + FUND_FEATURES + MACRO_FEATURES

TARGET_REG = "net_alpha_next"
TARGET_CLS = "commission_justified_next"


def capm_alpha_beta(fund_excess: np.ndarray, market_excess: np.ndarray) -> tuple[float, float]:
    if len(fund_excess) < 3:
        return np.nan, np.nan
    design = np.column_stack([np.ones_like(market_excess), market_excess])
    coeffs, *_ = np.linalg.lstsq(design, fund_excess, rcond=None)
    return float(coeffs[0]), float(coeffs[1])


def annualized_alpha(fund: np.ndarray, market: np.ndarray, rf_annual: float) -> tuple[float, float]:
    rf = rf_annual / MONTHS_PER_YEAR
    alpha_month, beta = capm_alpha_beta(fund - rf, market - rf)
    return alpha_month * MONTHS_PER_YEAR, beta


def sharpe_ratio(fund: np.ndarray, rf_annual: float) -> float:
    excess = fund - rf_annual / MONTHS_PER_YEAR
    scale = excess.std(ddof=1)
    return float(excess.mean() / scale * np.sqrt(MONTHS_PER_YEAR)) if scale else np.nan


def information_ratio(fund: np.ndarray, market: np.ndarray) -> float:
    active = fund - market
    scale = active.std(ddof=1)
    return float(active.mean() / scale * np.sqrt(MONTHS_PER_YEAR)) if scale else np.nan


def tracking_error(fund: np.ndarray, market: np.ndarray) -> float:
    return float((fund - market).std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))


def annualized_volatility(fund: np.ndarray) -> float:
    return float(fund.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))


def max_drawdown(returns: np.ndarray) -> float:
    curve = np.cumprod(1.0 + returns)
    return float((curve / np.maximum.accumulate(curve) - 1.0).min())


def build_panel(panel: SyntheticPanel, config: Config) -> pd.DataFrame:
    window = config.features.rolling_window_months
    min_history = config.features.min_history_months
    rf_annual = config.data.risk_free_annual
    rf_quarter = rf_annual / QUARTERS_PER_YEAR

    monthly_fund = panel.nav_direct.resample("ME").last().pct_change()
    monthly_bench = panel.benchmark.resample("ME").last().pct_change()
    quarterly_fund = panel.nav_direct.resample("QE").last().pct_change()
    quarterly_bench = panel.benchmark.resample("QE").last().pct_change()
    macro_quarter = panel.macro.resample("QE").last().ffill()

    quarter_ends = sorted(pd.to_datetime(panel.fund_attributes["quarter_end"].unique()))
    next_quarter = {q: quarter_ends[i + 1] for i, q in enumerate(quarter_ends[:-1])}
    attrs = panel.fund_attributes.set_index(["fund_id", "quarter_end"])

    rows = []
    for fund_id, fund_series in monthly_fund.items():
        for quarter_end in quarter_ends:
            following = next_quarter.get(quarter_end)
            if following is None:
                continue

            start = quarter_end - pd.DateOffset(months=window)
            mask = (fund_series.index > start) & (fund_series.index <= quarter_end)
            fund_slice = fund_series[mask].dropna()
            bench_slice = monthly_bench[mask].reindex(fund_slice.index).dropna()
            fund_slice = fund_slice.reindex(bench_slice.index)
            if len(fund_slice) < min_history:
                continue

            fund_vals = fund_slice.to_numpy()
            bench_vals = bench_slice.to_numpy()
            alpha_annual, beta = annualized_alpha(fund_vals, bench_vals, rf_annual)
            if np.isnan(beta):
                continue

            fund_fwd = quarterly_fund.loc[following, fund_id]
            bench_fwd = quarterly_bench.loc[following]
            if pd.isna(fund_fwd) or pd.isna(bench_fwd):
                continue

            forward_alpha = ((fund_fwd - rf_quarter) - beta * (bench_fwd - rf_quarter)) * QUARTERS_PER_YEAR
            try:
                row_attrs = attrs.loc[(fund_id, quarter_end)]
            except KeyError:
                continue
            net_alpha = forward_alpha - float(row_attrs["expense_gap"])
            macro_row = macro_quarter.loc[quarter_end]

            rows.append(
                {
                    "fund_id": fund_id,
                    "fund_name": row_attrs["fund_name"],
                    "quarter_end": quarter_end,
                    "alpha_annual": alpha_annual,
                    "beta": beta,
                    "sharpe": sharpe_ratio(fund_vals, rf_annual),
                    "info_ratio": information_ratio(fund_vals, bench_vals),
                    "volatility": annualized_volatility(fund_vals),
                    "tracking_error": tracking_error(fund_vals, bench_vals),
                    "max_drawdown": max_drawdown(fund_vals),
                    "expense_ratio_regular": float(row_attrs["expense_ratio_regular"]),
                    "expense_gap": float(row_attrs["expense_gap"]),
                    "log_aum": float(row_attrs["log_aum"]),
                    "portfolio_turnover": float(row_attrs["portfolio_turnover"]),
                    "fund_manager_tenure": float(row_attrs["fund_manager_tenure"]),
                    "fund_age": float(row_attrs["fund_age"]),
                    **{c: float(macro_row[c]) for c in MACRO_FEATURES},
                    TARGET_REG: net_alpha,
                    TARGET_CLS: int(net_alpha > 0),
                }
            )

    frame = pd.DataFrame(rows).dropna(subset=FEATURE_COLUMNS + [TARGET_REG])
    return frame.sort_values(["quarter_end", "fund_id"]).reset_index(drop=True)
