"""Data acquisition: a synthetic fund panel plus live mfapi.in and macro loaders."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

TRADING_DAYS = 252
MACRO_COLUMNS = ["repo_rate", "cpi_inflation", "yield_slope", "fii_flow", "dii_flow"]
FUND_FAMILIES = [
    "Bluechip", "Flexi Cap", "Large Cap", "Mid Cap", "Focused",
    "Value", "Multi Cap", "Dividend Yield", "Contra", "Emerging",
]
AMC_NAMES = [
    "Aditya", "Axis", "HDFC", "ICICI", "Kotak", "Nippon",
    "SBI", "UTI", "Mirae", "Tata", "DSP", "Franklin",
]


@dataclass
class SyntheticPanel:
    nav_direct: pd.DataFrame
    nav_regular: pd.DataFrame
    benchmark: pd.Series
    macro: pd.DataFrame
    fund_attributes: pd.DataFrame


def _ar1_series(length: int, mean: float, persistence: float,
                shock_scale: float, rng: np.random.Generator) -> np.ndarray:
    values = np.empty(length)
    values[0] = mean
    for t in range(1, length):
        values[t] = mean + persistence * (values[t - 1] - mean) + rng.normal(0.0, shock_scale)
    return values


def _generate_macro(month_index: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    n = len(month_index)
    return pd.DataFrame(
        {
            "repo_rate": np.clip(_ar1_series(n, 0.062, 0.97, 0.0015, rng), 0.04, 0.085),
            "cpi_inflation": np.clip(_ar1_series(n, 0.052, 0.94, 0.004, rng), 0.02, 0.09),
            "yield_slope": _ar1_series(n, 0.009, 0.95, 0.0025, rng),
            "fii_flow": _ar1_series(n, 0.0, 0.55, 12000.0, rng),
            "dii_flow": _ar1_series(n, 3000.0, 0.6, 9000.0, rng),
        },
        index=month_index,
    )


def _standardize(values: np.ndarray) -> np.ndarray:
    scale = values.std()
    centered = values - values.mean()
    return centered / scale if scale > 0 else centered


def generate_synthetic_panel(
    n_funds: int,
    start_date: str,
    end_date: str,
    risk_free_annual: float,
    seed: int,
) -> SyntheticPanel:
    """Simulate direct and regular NAV paths, fund attributes, and a macro block.

    Each fund has a latent skill that reacts to the macro regime, a market beta,
    and idiosyncratic noise. The regular plan trails the direct plan by the
    annual expense gap, which is the cost the model must decide is worth paying.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, end=end_date)
    month_index = pd.date_range(start=start_date, end=end_date, freq="ME")
    quarter_index = pd.date_range(start=start_date, end=end_date, freq="QE")
    rf_daily = risk_free_annual / TRADING_DAYS

    macro = _generate_macro(month_index, rng)

    market_drift = (risk_free_annual + 0.055) / TRADING_DAYS
    market_vol = 0.16 / np.sqrt(TRADING_DAYS)
    benchmark_returns = pd.Series(rng.normal(market_drift, market_vol, len(dates)), index=dates)
    benchmark = (1000.0 * (1.0 + benchmark_returns).cumprod()).rename("benchmark")
    excess_market = benchmark_returns.to_numpy() - rf_daily

    regime = pd.Series(
        _standardize(macro["yield_slope"].to_numpy()) - _standardize(macro["repo_rate"].to_numpy()),
        index=month_index,
    )
    regime_daily = regime.reindex(dates, method="ffill").bfill().to_numpy()

    fund_ids = [f"MF{idx:04d}" for idx in range(n_funds)]
    beta = np.clip(rng.normal(1.0, 0.13, n_funds), 0.6, 1.4)
    skill_base = rng.normal(0.007, 0.040, n_funds)
    skill_sensitivity = rng.normal(0.0, 0.025, n_funds)
    idio_vol = rng.uniform(0.02, 0.05, n_funds) / np.sqrt(TRADING_DAYS)
    expense_direct = rng.uniform(0.004, 0.012, n_funds)
    expense_gap = rng.uniform(0.005, 0.015, n_funds)
    expense_regular = expense_direct + expense_gap
    inception_offset = rng.integers(0, 8 * 12, n_funds)
    aum_start = rng.lognormal(mean=7.0, sigma=0.9, size=n_funds)
    turnover_base = rng.uniform(0.25, 1.4, n_funds)
    tenure_start = rng.uniform(0.5, 9.0, n_funds)

    direct_nav, regular_nav, records = {}, {}, []
    for i, fund_id in enumerate(fund_ids):
        alpha_daily = (skill_base[i] + skill_sensitivity[i] * regime_daily) / TRADING_DAYS
        gross = rf_daily + beta[i] * excess_market + alpha_daily + rng.normal(0.0, idio_vol[i], len(dates))
        direct_nav[fund_id] = 10.0 * (1.0 + gross - expense_direct[i] / TRADING_DAYS).cumprod()
        regular_nav[fund_id] = 10.0 * (1.0 + gross - expense_regular[i] / TRADING_DAYS).cumprod()

        name = f"{AMC_NAMES[i % len(AMC_NAMES)]} {FUND_FAMILIES[i % len(FUND_FAMILIES)]} Fund"
        for q, quarter_end in enumerate(quarter_index):
            aum_cr = max(50.0, aum_start[i] * (1.0 + rng.normal(0.02, 0.05)) ** q)
            months_live = q * 3 + (96 - inception_offset[i])
            records.append(
                {
                    "quarter_end": quarter_end,
                    "fund_id": fund_id,
                    "fund_name": name,
                    "log_aum": float(np.log(aum_cr)),
                    "portfolio_turnover": float(np.clip(turnover_base[i] + rng.normal(0, 0.1), 0.05, 3.0)),
                    "fund_manager_tenure": float(tenure_start[i] + q * 0.25),
                    "fund_age": float(max(0.5, months_live / 12.0)),
                    "expense_ratio_direct": float(expense_direct[i]),
                    "expense_ratio_regular": float(expense_regular[i]),
                    "expense_gap": float(expense_gap[i]),
                }
            )

    return SyntheticPanel(
        nav_direct=pd.DataFrame(direct_nav, index=dates),
        nav_regular=pd.DataFrame(regular_nav, index=dates),
        benchmark=benchmark,
        macro=macro,
        fund_attributes=pd.DataFrame.from_records(records),
    )


# --- Live data connectors (used when config.data.source == "mfapi") ---------

MFAPI_BASE = "https://api.mfapi.in"


def mfapi_nav_history(scheme_code: int | str, timeout: int = 20) -> pd.Series:
    response = requests.get(f"{MFAPI_BASE}/mf/{scheme_code}", timeout=timeout)
    response.raise_for_status()
    frame = pd.DataFrame(response.json()["data"])
    if frame.empty:
        return pd.Series(dtype=float, name=str(scheme_code))
    frame["date"] = pd.to_datetime(frame["date"], format="%d-%m-%Y")
    frame["nav"] = pd.to_numeric(frame["nav"], errors="coerce")
    series = frame.set_index("date")["nav"].sort_index().dropna()
    series.name = str(scheme_code)
    return series


def mfapi_fetch_many(scheme_codes: Iterable[int | str], pause: float = 0.2) -> pd.DataFrame:
    columns = {}
    for code in scheme_codes:
        try:
            columns[str(code)] = mfapi_nav_history(code)
        except requests.RequestException:
            continue
        time.sleep(pause)
    return pd.concat(columns, axis=1).sort_index() if columns else pd.DataFrame()


def load_macro_csv(path: str | Path) -> pd.DataFrame:
    """Load an RBI DBIE / NSDL macro export. Rates as decimals, flows in crore."""
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError("macro csv must contain a 'date' column")
    frame["date"] = pd.to_datetime(frame["date"])
    missing = [c for c in MACRO_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"macro csv missing columns: {missing}")
    return frame.set_index("date")[MACRO_COLUMNS].sort_index()
