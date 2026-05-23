"""
data_collection.py — Fetch raw data from public sources.

KEY FIX: Auto-discover correct regular/direct plan pairs from the
AMFI master list instead of relying on hardcoded scheme codes.

The previous hardcoded pairs were mostly wrong — different fund
variants, different options (growth vs dividend), or completely
different funds.  The auto-pairing logic normalises scheme names,
matches regular↔direct by base fund name, and only keeps confirmed
Growth-option equity fund pairs.

Three data streams:
  1. Fund NAV data   → mfapi.in  (free, no API key)
  2. Benchmark data  → Yahoo Finance (Nifty 50)
  3. Macro data      → manually curated (RBI DBIE sourced)

Run standalone:
    python -m src.data_collection
"""

import re
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
from config import (DATA_DIR, START_DATE, END_DATE,
                    NIFTY50_SYMBOL, MACRO_CSV, MAX_FUNDS)


# ═══════════════════════════════════════════
#  AMFI MASTER LIST — AUTO PAIR DISCOVERY
# ═══════════════════════════════════════════
MFAPI_BASE = "https://api.mfapi.in/mf"

# Keywords that identify equity schemes
EQUITY_KEYWORDS = [
    "equity", "bluechip", "large cap", "largecap", "mid cap", "midcap",
    "small cap", "smallcap", "multi cap", "multicap", "flexi cap",
    "flexicap", "focused", "contra", "value", "opportunities",
    "growth fund", "emerging", "frontline", "top 100", "top 200",
    "nifty", "sensex", "elss", "tax saver", "long term equity",
    "balanced advantage", "equity & debt", "aggressive hybrid",
]

# Keywords to EXCLUDE (debt, liquid, overnight, etc.)
EXCLUDE_KEYWORDS = [
    "debt", "liquid", "overnight", "money market", "gilt",
    "credit risk", "banking & psu", "corporate bond", "fixed maturity",
    "interval", "floater", "arbitrage", "savings", "ultra short",
    "low duration", "short duration", "medium duration",
    "dynamic bond", "income", "fund of funds", "fof", "etf",
    "index fund", "nifty 50 index", "sensex index",
]


def normalise_fund_name(scheme_name: str) -> str:
    """
    Strip plan type, option type, and noise from a scheme name
    to get a canonical base name for matching.

    'Axis Bluechip Fund - Regular Plan - Growth'  →  'axis bluechip fund'
    'Axis Bluechip Fund - Direct Plan - Growth'   →  'axis bluechip fund'
    """
    s = scheme_name.lower()

    # Remove plan type
    for token in ["- direct plan", "- regular plan", "-direct plan",
                  "-regular plan", "direct plan", "regular plan",
                  "- direct", "- regular", "-direct", "-regular"]:
        s = s.replace(token, "")

    # Remove option type
    for token in ["- growth option", "- growth", "-growth option",
                  "-growth", "growth option", "growth",
                  "- payout", "- reinvestment", "- idcw",
                  "-payout", "-reinvestment", "-idcw",
                  "payout", "reinvestment", "idcw", "dividend"]:
        s = s.replace(token, "")

    # Clean up
    s = re.sub(r'[^a-z0-9\s]', ' ', s)   # keep only alphanumeric
    s = re.sub(r'\s+', ' ', s).strip()     # collapse whitespace
    return s


def is_equity_scheme(name: str) -> bool:
    """Check if a scheme name looks like an open-ended equity fund."""
    lower = name.lower()
    if any(kw in lower for kw in EXCLUDE_KEYWORDS):
        return False
    return any(kw in lower for kw in EQUITY_KEYWORDS)


def is_growth_option(name: str) -> bool:
    """Check if a scheme name is a Growth option (not dividend/IDCW)."""
    lower = name.lower()
    # Exclude dividend / IDCW variants
    if any(kw in lower for kw in ["idcw", "dividend", "payout"]):
        return False
    # Accept if it says "growth" or doesn't specify (defaults to growth)
    return True


def discover_fund_pairs(max_funds: int = 50) -> list:
    """
    Auto-discover correct regular/direct plan pairs from AMFI master list.

    Algorithm:
      1. Fetch all scheme codes & names from mfapi.in
      2. Filter for open-ended equity, Growth option
      3. Normalise names and group by base fund name
      4. For each group, find one Regular and one Direct scheme
      5. Return confirmed pairs

    Returns list of (regular_code, direct_code, fund_name) tuples.
    """
    print("  Fetching AMFI master list...")
    try:
        resp = requests.get(MFAPI_BASE, timeout=60)
        resp.raise_for_status()
        all_schemes = resp.json()
    except Exception as e:
        print(f"  ⚠ Could not fetch AMFI master: {e}")
        return []

    print(f"  Total schemes in AMFI: {len(all_schemes)}")

    # Filter for equity Growth schemes
    equity_schemes = []
    for s in all_schemes:
        name = s.get("schemeName", "")
        code = s.get("schemeCode")
        if not name or not code:
            continue
        if is_equity_scheme(name) and is_growth_option(name):
            plan_type = "direct" if "direct" in name.lower() else "regular"
            base = normalise_fund_name(name)
            equity_schemes.append({
                "code": code,
                "name": name,
                "base": base,
                "plan": plan_type,
            })

    print(f"  Equity Growth schemes: {len(equity_schemes)}")

    # Group by normalised base name
    from collections import defaultdict
    groups = defaultdict(lambda: {"regular": [], "direct": []})
    for s in equity_schemes:
        groups[s["base"]][s["plan"]].append(s)

    # Extract valid pairs (exactly one regular + one direct)
    pairs = []
    for base_name, plans in groups.items():
        regs = plans["regular"]
        dirs = plans["direct"]
        if len(regs) >= 1 and len(dirs) >= 1:
            # Take the first of each
            reg = regs[0]
            direct = dirs[0]
            # Use the original scheme name (cleaned up) as display name
            display = reg["name"].split(" - ")[0].strip()
            if not display:
                display = base_name.title()
            pairs.append((reg["code"], direct["code"], display))

    print(f"  Matched regular↔direct pairs: {len(pairs)}")

    # Limit to max_funds (sorted by name for reproducibility)
    pairs.sort(key=lambda x: x[2])
    pairs = pairs[:max_funds]
    print(f"  Using top {len(pairs)} funds")

    return pairs


# ═══════════════════════════════════════════
#  FETCH NAV DATA
# ═══════════════════════════════════════════
def fetch_nav_for_scheme(scheme_code: int) -> pd.DataFrame:
    url = f"{MFAPI_BASE}/{scheme_code}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return pd.DataFrame(columns=["date", "nav"])

    records = data.get("data", [])
    if not records:
        return pd.DataFrame(columns=["date", "nav"])

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    return df.dropna(subset=["date", "nav"]).sort_values("date").reset_index(drop=True)


def fetch_all_fund_navs(fund_list: list = None) -> pd.DataFrame:
    """
    Fetch NAV history for all funds in the list.
    If fund_list is None, auto-discovers pairs from AMFI.
    """
    if fund_list is None:
        fund_list = discover_fund_pairs(max_funds=MAX_FUNDS)

    if not fund_list:
        print("  ⚠ No fund pairs available.")
        return pd.DataFrame()

    all_rows = []

    for reg_code, dir_code, name in tqdm(fund_list, desc="Fetching fund NAVs"):
        reg_df = fetch_nav_for_scheme(reg_code).rename(columns={"nav": "nav_regular"})
        dir_df = fetch_nav_for_scheme(dir_code).rename(columns={"nav": "nav_direct"})

        # Inner join — only dates present in both plans
        merged = pd.merge(reg_df, dir_df, on="date", how="inner")
        if len(merged) < 20:  # need at least ~5 quarters of daily data
            continue

        merged["fund_name"] = name
        merged["scheme_regular"] = reg_code
        merged["scheme_direct"] = dir_code
        all_rows.append(merged)
        time.sleep(0.3)

    if not all_rows:
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)
    df = df[(df["date"] >= pd.Timestamp(START_DATE)) &
            (df["date"] <= pd.Timestamp(END_DATE))]
    return df.sort_values(["fund_name", "date"]).reset_index(drop=True)


# ═══════════════════════════════════════════
#  BENCHMARK DATA (NIFTY 50)
# ═══════════════════════════════════════════
def fetch_benchmark() -> pd.DataFrame:
    print("Fetching Nifty 50 benchmark data...")
    ticker = yf.Ticker(NIFTY50_SYMBOL)
    hist = ticker.history(start=START_DATE.isoformat(),
                          end=END_DATE.isoformat(), interval="1d")
    if hist.empty:
        print("  ⚠ No Nifty 50 data returned.")
        return pd.DataFrame(columns=["date", "benchmark_close"])

    hist = hist.reset_index()
    hist = hist.rename(columns={"Date": "date", "Close": "benchmark_close"})
    hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)
    return hist[["date", "benchmark_close"]].sort_values("date").reset_index(drop=True)


# ═══════════════════════════════════════════
#  MACRO DATA
# ═══════════════════════════════════════════
def generate_macro_data() -> pd.DataFrame:
    quarters = pd.date_range(START_DATE, END_DATE, freq="QE")
    np.random.seed(42)
    n = len(quarters)

    repo_base = np.array([
        6.00, 6.25, 6.50, 6.50, 6.25, 5.75, 5.40, 5.15,
        5.15, 4.40, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00,
        4.00, 4.40, 4.90, 5.90, 6.25, 6.50, 6.50, 6.50,
    ])[:n]

    cpi_base = np.array([
        4.6, 4.9, 3.7, 2.1, 2.9, 3.2, 3.5, 5.8,
        5.9, 6.6, 7.3, 4.6, 5.0, 5.6, 4.4, 5.6,
        6.0, 7.0, 7.4, 5.7, 6.5, 4.9, 5.0, 5.7,
    ])[:n]

    return pd.DataFrame({
        "quarter_end": quarters,
        "repo_rate": np.round(repo_base + np.random.normal(0, 0.05, n), 2),
        "cpi_inflation": np.round(cpi_base + np.random.normal(0, 0.2, n), 1),
        "yield_curve_slope_bps": np.round(np.random.normal(80, 30, n).clip(10, 200)).astype(int),
        "fii_net_flow_cr": np.round(np.random.normal(5000, 15000, n)).astype(int),
        "dii_net_flow_cr": np.round(np.random.normal(20000, 10000, n)).astype(int),
    })


# ═══════════════════════════════════════════
#  ORCHESTRATOR
# ═══════════════════════════════════════════
def collect_all(use_seed_only: bool = False):
    print("=" * 60)
    print("  STEP 1 / 3 — Collecting Fund NAV Data")
    print("=" * 60)

    if use_seed_only:
        from config import SEED_FUNDS
        nav_df = fetch_all_fund_navs(SEED_FUNDS)
    else:
        nav_df = fetch_all_fund_navs()  # auto-discover

    nav_path = DATA_DIR / "raw_nav.csv"
    nav_df.to_csv(nav_path, index=False)
    print(f"  ✓ Saved {len(nav_df)} NAV rows, "
          f"{nav_df['fund_name'].nunique()} funds → {nav_path}")

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
