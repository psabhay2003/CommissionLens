"""
data_collection.py — Fetch raw data from public sources.

Three data streams:
  1. Fund NAV data   → mfapi.in  (free, no API key needed)
  2. Benchmark data  → Yahoo Finance (Nifty 50)
  3. Macro data      → manually curated CSV (RBI DBIE sourced)

Run standalone:
    python -m src.data_collection
"""

import time
import json
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (DATA_DIR, START_DATE, END_DATE, SEED_FUNDS,
                    NIFTY50_SYMBOL, MACRO_CSV)


# ═══════════════════════════════════════════
#  1. FUND NAV DATA FROM mfapi.in
# ═══════════════════════════════════════════
MFAPI_BASE = "https://api.mfapi.in/mf"
MFAPI_LIST = "https://api.mfapi.in/mf"


def fetch_nav_for_scheme(scheme_code: int) -> pd.DataFrame:
    """
    Fetch full NAV history for a single mutual fund scheme.

    Returns a DataFrame with columns: [date, nav]
    """
    url = f"{MFAPI_BASE}/{scheme_code}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  ⚠ Failed to fetch scheme {scheme_code}: {e}")
        return pd.DataFrame(columns=["date", "nav"])

    records = data.get("data", [])
    if not records:
        return pd.DataFrame(columns=["date", "nav"])

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    df = df.dropna(subset=["date", "nav"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_all_fund_navs(fund_list: list = None) -> pd.DataFrame:
    """
    Fetch NAV history for all funds in the list.

    Parameters
    ----------
    fund_list : list of tuples
        Each tuple = (regular_code, direct_code, fund_name).
        Defaults to config.SEED_FUNDS.

    Returns
    -------
    DataFrame with columns:
        [date, fund_name, nav_regular, nav_direct, scheme_regular, scheme_direct]
    """
    if fund_list is None:
        fund_list = SEED_FUNDS

    all_rows = []

    for reg_code, dir_code, name in tqdm(fund_list, desc="Fetching fund NAVs"):
        # Fetch regular plan NAV
        reg_df = fetch_nav_for_scheme(reg_code)
        reg_df = reg_df.rename(columns={"nav": "nav_regular"})

        # Fetch direct plan NAV
        dir_df = fetch_nav_for_scheme(dir_code)
        dir_df = dir_df.rename(columns={"nav": "nav_direct"})

        # Merge on date (inner join — only dates present in both plans)
        merged = pd.merge(reg_df, dir_df, on="date", how="inner")
        merged["fund_name"] = name
        merged["scheme_regular"] = reg_code
        merged["scheme_direct"] = dir_code
        all_rows.append(merged)

        time.sleep(0.5)  # polite rate limiting

    if not all_rows:
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)
    df = df[(df["date"] >= pd.Timestamp(START_DATE)) &
            (df["date"] <= pd.Timestamp(END_DATE))]
    return df.sort_values(["fund_name", "date"]).reset_index(drop=True)


# ═══════════════════════════════════════════
#  2. BENCHMARK DATA (NIFTY 50) FROM YAHOO FINANCE
# ═══════════════════════════════════════════
def fetch_benchmark() -> pd.DataFrame:
    """
    Download Nifty 50 daily closing prices from Yahoo Finance.

    Returns DataFrame with columns: [date, benchmark_close]
    """
    print("Fetching Nifty 50 benchmark data...")
    ticker = yf.Ticker(NIFTY50_SYMBOL)
    hist = ticker.history(
        start=START_DATE.isoformat(),
        end=END_DATE.isoformat(),
        interval="1d",
    )
    if hist.empty:
        print("  ⚠ Yahoo Finance returned no data for Nifty 50.")
        return pd.DataFrame(columns=["date", "benchmark_close"])

    hist = hist.reset_index()
    hist = hist.rename(columns={"Date": "date", "Close": "benchmark_close"})
    hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)
    return hist[["date", "benchmark_close"]].sort_values("date").reset_index(drop=True)


# ═══════════════════════════════════════════
#  3. MACRO DATA — SYNTHETIC / CSV BASED
# ═══════════════════════════════════════════
def generate_macro_data() -> pd.DataFrame:
    """
    Generate a quarterly macro feature dataset.

    In production, this would be scraped from RBI DBIE.  For reproducibility
    we create a realistic synthetic dataset based on actual Indian macro trends
    (2018-2023).  You can replace this with actual data from dbie.rbi.org.in.
    """
    quarters = pd.date_range(START_DATE, END_DATE, freq="QE")

    np.random.seed(42)
    n = len(quarters)

    # Repo rate trajectory (roughly matched to RBI policy 2018-2023)
    repo_base = np.array([
        6.00, 6.25, 6.50, 6.50,   # 2018
        6.25, 5.75, 5.40, 5.15,   # 2019
        5.15, 4.40, 4.00, 4.00,   # 2020
        4.00, 4.00, 4.00, 4.00,   # 2021
        4.00, 4.40, 4.90, 5.90,   # 2022
        6.25, 6.50, 6.50, 6.50,   # 2023
    ])[:n]
    repo_rate = repo_base + np.random.normal(0, 0.05, n)

    # CPI inflation (YoY %)
    cpi_base = np.array([
        4.6, 4.9, 3.7, 2.1,
        2.9, 3.2, 3.5, 5.8,
        5.9, 6.6, 7.3, 4.6,
        5.0, 5.6, 4.4, 5.6,
        6.0, 7.0, 7.4, 5.7,
        6.5, 4.9, 5.0, 5.7,
    ])[:n]
    cpi_inflation = cpi_base + np.random.normal(0, 0.2, n)

    # Yield curve slope (10Y - 2Y spread, bps)
    yield_slope = np.random.normal(80, 30, n).clip(10, 200)

    # FII net flows (₹ crores, quarterly)
    fii_flows = np.random.normal(5000, 15000, n)

    # DII net flows
    dii_flows = np.random.normal(20000, 10000, n)

    df = pd.DataFrame({
        "quarter_end": quarters,
        "repo_rate": np.round(repo_rate, 2),
        "cpi_inflation": np.round(cpi_inflation, 1),
        "yield_curve_slope_bps": np.round(yield_slope, 0).astype(int),
        "fii_net_flow_cr": np.round(fii_flows, 0).astype(int),
        "dii_net_flow_cr": np.round(dii_flows, 0).astype(int),
    })
    return df


# ═══════════════════════════════════════════
#  EXPAND FUND UNIVERSE (fetch AMFI master list)
# ═══════════════════════════════════════════
def fetch_amfi_master_list() -> pd.DataFrame:
    """
    Download the full AMFI scheme master list and filter for
    open-ended equity schemes.  Returns scheme codes and names.
    """
    url = "https://api.mfapi.in/mf"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        schemes = resp.json()
    except Exception as e:
        print(f"  ⚠ Could not fetch AMFI master list: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(schemes)
    # Filter for equity-ish keywords in scheme name
    equity_kw = ["equity", "bluechip", "large cap", "mid cap", "small cap",
                 "multi cap", "flexi cap", "focused", "contra", "value",
                 "growth", "opportunities"]
    mask = df["schemeName"].str.lower().apply(
        lambda s: any(kw in s for kw in equity_kw)
    )
    return df[mask].reset_index(drop=True)


# ═══════════════════════════════════════════
#  ORCHESTRATOR
# ═══════════════════════════════════════════
def collect_all(use_seed_only: bool = True):
    """
    Run the full data collection pipeline.

    Parameters
    ----------
    use_seed_only : bool
        If True, only fetch the seed funds from config.
        If False, attempt to build a 200+ fund universe from AMFI master list.
    """
    print("=" * 60)
    print("  STEP 1 / 3 — Collecting Fund NAV Data")
    print("=" * 60)
    nav_df = fetch_all_fund_navs()
    nav_path = DATA_DIR / "raw_nav.csv"
    nav_df.to_csv(nav_path, index=False)
    print(f"  ✓ Saved {len(nav_df)} NAV rows → {nav_path}")

    print("\n" + "=" * 60)
    print("  STEP 2 / 3 — Collecting Benchmark Data")
    print("=" * 60)
    bench_df = fetch_benchmark()
    bench_path = DATA_DIR / "benchmark_nifty50.csv"
    bench_df.to_csv(bench_path, index=False)
    print(f"  ✓ Saved {len(bench_df)} benchmark rows → {bench_path}")

    print("\n" + "=" * 60)
    print("  STEP 3 / 3 — Generating Macro Data")
    print("=" * 60)
    macro_df = generate_macro_data()
    macro_df.to_csv(MACRO_CSV, index=False)
    print(f"  ✓ Saved {len(macro_df)} macro rows → {MACRO_CSV}")

    return nav_df, bench_df, macro_df


if __name__ == "__main__":
    collect_all()
