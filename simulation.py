"""Dated-cashflow XIRR and the SIP back-validation of model-guided fund selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd
from scipy.optimize import brentq

DAYS_PER_YEAR = 365.0


def _to_date(value) -> date:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def xirr(cashflows: list[tuple], lower: float = -0.9999, upper: float = 10.0) -> float:
    """Annualised IRR for dated cashflows (outflows negative, inflows positive)."""
    if len(cashflows) < 2:
        return float("nan")
    dates = [_to_date(d) for d, _ in cashflows]
    amounts = np.array([a for _, a in cashflows], dtype=float)
    if amounts.min() >= 0 or amounts.max() <= 0:
        return float("nan")
    origin = min(dates)
    years = np.array([(d - origin).days / DAYS_PER_YEAR for d in dates])

    def npv(rate: float) -> float:
        return float(np.sum(amounts / (1.0 + rate) ** years))

    if np.sign(npv(lower)) == np.sign(npv(upper)):
        return float("nan")
    return float(brentq(npv, lower, upper))


@dataclass
class SipResult:
    xirr: float
    invested: float
    final_value: float
    n_installments: int


def build_selections(predictions: pd.DataFrame, fraction: float, min_funds: int = 5) -> dict:
    """Top-fraction funds by score for each quarter."""
    selections = {}
    for quarter_end, group in predictions.groupby("quarter_end"):
        ranked = group.sort_values("probability", ascending=False)
        k = max(min_funds, int(round(len(ranked) * fraction)))
        selections[pd.Timestamp(quarter_end)] = ranked.head(k)["fund_id"].tolist()
    return selections


def build_all_fund_selections(predictions: pd.DataFrame) -> dict:
    return {
        pd.Timestamp(q): g["fund_id"].tolist()
        for q, g in predictions.groupby("quarter_end")
    }


def _selection_for(day: pd.Timestamp, selections: dict) -> list:
    eligible = [q for q in selections if q <= day]
    return selections[max(eligible)] if eligible else []


def run_sip(nav: pd.DataFrame, selections: dict, monthly_investment: float,
            start_date: str, end_date: str) -> SipResult:
    end_stamp = pd.Timestamp(end_date)
    units: dict[str, float] = {}
    cashflows: list[tuple] = []
    installments = 0

    for day in pd.date_range(start=start_date, end=end_date, freq="MS"):
        priced = []
        for fund_id in _selection_for(day, selections):
            if fund_id not in nav.columns:
                continue
            value = nav[fund_id].asof(day)
            if pd.notna(value) and value > 0:
                priced.append((fund_id, value))
        if not priced:
            continue
        per_fund = monthly_investment / len(priced)
        for fund_id, value in priced:
            units[fund_id] = units.get(fund_id, 0.0) + per_fund / value
        cashflows.append((day, -monthly_investment))
        installments += 1

    final_value = 0.0
    for fund_id, held in units.items():
        value = nav[fund_id].asof(end_stamp)
        if pd.notna(value):
            final_value += held * value
    cashflows.append((end_stamp, final_value))

    return SipResult(
        xirr=xirr(cashflows),
        invested=monthly_investment * installments,
        final_value=final_value,
        n_installments=installments,
    )
