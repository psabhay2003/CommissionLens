"""CommissionLens command line: build the dataset, train, explain, and back-test.

Usage:
    py main.py build
    py main.py train
    py main.py explain
    py main.py simulate
    py main.py all
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config, load_config
from data import generate_synthetic_panel
from features import TARGET_CLS, build_panel
from model import (
    TrainedModel,
    build_sequences,
    evaluate_model,
    subset_bundle,
    temporal_split,
    train_model,
)
from simulation import build_all_fund_selections, build_selections, run_sip

GOLD, GREY = "#c9a227", "#6b7280"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def cmd_build(config: Config) -> pd.DataFrame:
    if config.data.source != "synthetic":
        raise NotImplementedError(
            "Only the synthetic source is wired end to end. Use data.mfapi_fetch_many "
            "and data.load_macro_csv to assemble a live panel."
        )
    source = generate_synthetic_panel(
        config.data.n_funds, config.data.start_date, config.data.end_date,
        config.data.risk_free_annual, config.seed,
    )
    panel = build_panel(source, config)
    out = _ensure(config.data_path)
    panel.to_parquet(out / "panel.parquet", index=False)
    source.nav_regular.to_parquet(out / "nav_regular.parquet")

    print(f"funds:              {panel['fund_id'].nunique()}")
    print(f"fund-quarter rows:  {len(panel)}")
    print(f"quarters:           {panel['quarter_end'].nunique()}")
    print(f"justified base rate:{panel[TARGET_CLS].mean(): .3f}")
    print(f"saved:              {out / 'panel.parquet'}")
    return panel


def _load_panel(config: Config) -> pd.DataFrame:
    path = config.data_path / "panel.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: py main.py build")
    return pd.read_parquet(path)


def cmd_train(config: Config) -> None:
    panel = _load_panel(config)
    bundle = build_sequences(panel, config.model.sequence_length)
    train, val, test = temporal_split(bundle, config.model.test_fraction, config.model.val_fraction)
    print(f"train/val/test samples: {len(train)}/{len(val)}/{len(test)}")

    model, history = train_model(train, val, config.model)
    metrics = {
        "validation": evaluate_model(model, val, config.simulation.top_decile),
        "test": evaluate_model(model, test, config.simulation.top_decile),
    }

    out = _ensure(config.artifacts_path)
    model.save(out / "commissionnet.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"epochs trained: {len(history)}")
    for key, value in metrics["test"].items():
        print(f"  test {key}: {value:.4f}" if isinstance(value, float) else f"  test {key}: {value}")
    print(f"saved: {out / 'commissionnet.pt'}")


def cmd_explain(config: Config, top: int = 5) -> None:
    from explain import compute_feature_importance

    panel = _load_panel(config)
    bundle = build_sequences(panel, config.model.sequence_length)
    train, _, test = temporal_split(bundle, config.model.test_fraction, config.model.val_fraction)
    model = TrainedModel.load(config.artifacts_path / "commissionnet.pt")

    importance = compute_feature_importance(model, train.features, test.features)
    out = _ensure(config.artifacts_path)
    importance.to_csv(out / "shap_importance.csv", index=False)

    ranked = importance.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ranked["feature"], ranked["mean_abs_shap"], color=GOLD)
    ax.set_xlabel("mean |SHAP| on commission-justification logit")
    ax.set_title("Most predictive features of commission justification")
    fig.tight_layout()
    fig.savefig(out / "shap_importance.png", dpi=150)

    print("top features:")
    for _, row in importance.head(top).iterrows():
        print(f"  {row['feature']:<24} {row['mean_abs_shap']:.4f}")
    print(f"saved: {out / 'shap_importance.png'}")


def cmd_simulate(config: Config) -> None:
    panel = _load_panel(config)
    nav_regular = pd.read_parquet(config.data_path / "nav_regular.parquet")
    bundle = build_sequences(panel, config.model.sequence_length)

    # Train only on quarters closing before the simulation starts (no lookahead).
    quarters = bundle.meta["quarter_end"].to_numpy()
    pool = quarters < np.datetime64(config.simulation.start_date)
    pool_quarters = np.sort(np.unique(quarters[pool]))
    n_val = max(1, int(round(len(pool_quarters) * config.model.val_fraction)))
    val_mask = np.isin(quarters, pool_quarters[-n_val:])
    print(f"pre-simulation train/val samples: {int((pool & ~val_mask).sum())}/{int(val_mask.sum())}")
    model, _ = train_model(subset_bundle(bundle, pool & ~val_mask), subset_bundle(bundle, val_mask), config.model)

    net_alpha, prob = model.predict(bundle.features)
    predictions = bundle.meta.copy()
    predictions["probability"] = prob
    predictions = predictions[pd.to_datetime(predictions["quarter_end"]) < config.simulation.end_date]

    guided = run_sip(nav_regular, build_selections(predictions, config.simulation.top_decile),
                     config.simulation.monthly_investment, config.simulation.start_date, config.simulation.end_date)
    naive = run_sip(nav_regular, build_all_fund_selections(predictions),
                    config.simulation.monthly_investment, config.simulation.start_date, config.simulation.end_date)

    summary = {
        "model_guided": {"xirr": guided.xirr, "final_value": guided.final_value, "invested": guided.invested},
        "naive_regular": {"xirr": naive.xirr, "final_value": naive.final_value, "invested": naive.invested},
        "xirr_edge_pct": (guided.xirr - naive.xirr) * 100,
    }
    out = _ensure(config.artifacts_path)
    (out / "simulation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 5))
    values = [guided.xirr * 100, naive.xirr * 100]
    ax.bar(["Model-guided", "Naive regular"], values, color=[GOLD, GREY])
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.2f}%", ha="center", va="bottom")
    ax.set_ylabel("SIP XIRR (%)")
    ax.set_title("Commission-aware selection vs naive regular investing")
    fig.tight_layout()
    fig.savefig(out / "sip_xirr.png", dpi=150)

    print(f"model-guided XIRR: {guided.xirr * 100:.2f}%  (final Rs {guided.final_value:,.0f})")
    print(f"naive regular XIRR: {naive.xirr * 100:.2f}%  (final Rs {naive.final_value:,.0f})")
    print(f"edge: {summary['xirr_edge_pct']:.2f} percentage points")


def main() -> None:
    parser = argparse.ArgumentParser(description="CommissionLens pipeline")
    parser.add_argument("command", choices=["build", "train", "explain", "simulate", "all"])
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    if args.command in ("build", "all"):
        cmd_build(config)
    if args.command in ("train", "all"):
        cmd_train(config)
    if args.command in ("explain", "all"):
        cmd_explain(config)
    if args.command in ("simulate", "all"):
        cmd_simulate(config)


if __name__ == "__main__":
    main()
