"""Torch-free unit tests: metrics, XIRR, synthetic data, panel building, and SIP."""

import numpy as np
import pandas as pd

import features
from config import Config, DataConfig
from data import generate_synthetic_panel
from features import FEATURE_COLUMNS, TARGET_CLS, TARGET_REG, build_panel
from simulation import run_sip, xirr


def _panel(n_funds=20, end="2022-12-31"):
    config = Config(data=DataConfig(n_funds=n_funds, start_date="2016-01-01", end_date=end))
    source = generate_synthetic_panel(
        config.data.n_funds, config.data.start_date, config.data.end_date,
        config.data.risk_free_annual, config.seed,
    )
    return source, build_panel(source, config)


# --- metrics ---------------------------------------------------------------

def test_beta_and_alpha_recovery():
    rng = np.random.default_rng(0)
    market = rng.normal(0.01, 0.04, 240)
    rf = 0.06 / 12
    fund = rf + 0.002 + 1.3 * (market - rf)
    alpha_annual, beta = features.annualized_alpha(fund, market, 0.06)
    assert abs(beta - 1.3) < 1e-6
    assert abs(alpha_annual - 0.002 * 12) < 1e-6


def test_max_drawdown_known_path():
    assert abs(features.max_drawdown(np.array([0.10, -0.50, 0.05])) - (-0.50)) < 1e-9


def test_sharpe_positive_for_steady_gains():
    assert features.sharpe_ratio(np.full(36, 0.01), 0.0) > 0


# --- xirr ------------------------------------------------------------------

def test_xirr_single_year():
    flows = [(pd.Timestamp("2020-01-01"), -1000.0), (pd.Timestamp("2021-01-01"), 1100.0)]
    assert abs(xirr(flows) - 0.10) < 1e-3


def test_xirr_flat_is_zero():
    flows = [(pd.Timestamp("2020-01-01"), -1000.0), (pd.Timestamp("2022-01-01"), 1000.0)]
    assert abs(xirr(flows)) < 1e-4


def test_xirr_all_negative_is_nan():
    flows = [(pd.Timestamp("2020-01-01"), -100.0), (pd.Timestamp("2021-01-01"), -50.0)]
    assert pd.isna(xirr(flows))


# --- synthetic data --------------------------------------------------------

def test_regular_plan_trails_direct():
    source, _ = _panel()
    assert (source.nav_regular.iloc[-1] < source.nav_direct.iloc[-1]).all()
    assert not source.nav_direct.isna().any().any()


def test_expense_gap_positive():
    source, _ = _panel()
    assert (source.fund_attributes["expense_gap"] > 0).all()


# --- panel -----------------------------------------------------------------

def test_panel_has_features_and_targets():
    _, panel = _panel()
    assert not panel.empty
    for column in FEATURE_COLUMNS + [TARGET_REG, TARGET_CLS]:
        assert column in panel.columns
    assert panel[FEATURE_COLUMNS].notna().all().all()


def test_label_matches_regression_sign():
    _, panel = _panel()
    assert set(panel[TARGET_CLS].unique()).issubset({0, 1})
    assert (panel[panel[TARGET_REG] > 0][TARGET_CLS] == 1).all()


# --- sip -------------------------------------------------------------------

def test_sip_positive_xirr_for_rising_nav():
    dates = pd.bdate_range("2019-12-01", "2021-02-01")
    levels = 10.0 * (1.0 + 0.12 / 252) ** np.arange(len(dates))
    nav = pd.DataFrame({"MF0001": levels}, index=dates)
    result = run_sip(nav, {pd.Timestamp("2019-12-31"): ["MF0001"]}, 5000.0, "2020-01-01", "2020-12-31")
    assert result.n_installments == 12
    assert result.final_value > result.invested
    assert 0.05 < result.xirr < 0.20
