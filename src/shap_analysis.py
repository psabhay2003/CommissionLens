"""
shap_analysis.py — SHAP explainability using TreeExplainer.

TreeExplainer gives EXACT Shapley values for tree-based models
(XGBoost, Random Forest, GBM) — far more reliable than the
approximate GradientExplainer used for neural networks.

Run standalone:
    python -m src.shap_analysis
"""

import numpy as np
import pandas as pd
import shap
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, MODEL_DIR, REPORT_DIR, TEST_QUARTERS
from src.model_training import FEATURE_COLS, temporal_split, prepare_Xy


def run_shap_analysis(df=None):
    if df is None:
        df = pd.read_csv(DATA_DIR / "fund_dataset.csv",
                         parse_dates=["quarter_end"])

    train, test = temporal_split(df)

    available = [c for c in FEATURE_COLS if c in train.columns]
    X_train_raw = train[available].copy()
    X_test_raw = test[available].copy()
    for col in available:
        med = X_train_raw[col].median()
        X_train_raw[col] = X_train_raw[col].fillna(med)
        X_test_raw[col] = X_test_raw[col].fillna(med)

    # Load scaler & model
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    best_tree = joblib.load(MODEL_DIR / "best_tree_classifier.pkl")
    meta = joblib.load(MODEL_DIR / "model_meta.pkl")

    X_te_scaled = pd.DataFrame(
        scaler.transform(X_test_raw), columns=available
    )

    print(f"  → Using {meta['best_tree_name']} for SHAP (TreeExplainer)")
    print("  → Computing SHAP values (exact)...")

    explainer = shap.TreeExplainer(best_tree)
    shap_values = explainer.shap_values(X_te_scaled)

    # For binary classifiers, shap_values may be a list of [neg, pos]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class

    # ── Beeswarm ──
    print("  → Generating SHAP beeswarm plot...")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_te_scaled, show=False, max_display=15)
    plt.title("SHAP Feature Importance — Commission Justification", fontsize=13)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Bar ──
    print("  → Generating SHAP bar plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_te_scaled, plot_type="bar",
                      show=False, max_display=15)
    plt.title("Mean |SHAP| — Feature Importance Ranking", fontsize=13)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Top features ──
    mean_abs = np.abs(shap_values).mean(axis=0)
    fi = pd.DataFrame({"feature": available, "mean_abs_shap": mean_abs})
    fi = fi.sort_values("mean_abs_shap", ascending=False)

    print("\n  ── Top 5 Features ──")
    for _, row in fi.head(5).iterrows():
        print(f"    {row['feature']:>35s}  →  SHAP = {row['mean_abs_shap']:.4f}")

    # ── Written report ──
    lines = [
        "=" * 60,
        "  SHAP EXPLAINABILITY REPORT",
        f"  CommissionLens — {meta['best_tree_name'].upper()} + Ensemble",
        "=" * 60, "",
        "SHAP TreeExplainer (exact Shapley values) was used to decompose",
        "the tree model's predictions into per-feature contributions.", "",
        "TOP FEATURES (ranked by mean |SHAP value|):",
        "-" * 50,
    ]
    for rank, (_, row) in enumerate(fi.head(5).iterrows(), 1):
        lines.append(f"  {rank}. {row['feature']}  "
                     f"(mean |SHAP| = {row['mean_abs_shap']:.4f})")
    lines += ["", "See reports/shap_summary.png and shap_bar.png for visuals."]

    with open(REPORT_DIR / "shap_report.txt", "w") as f:
        f.write("\n".join(lines))

    fi.to_json(REPORT_DIR / "feature_importance.json", orient="records", indent=2)
    print(f"\n  ✓ SHAP report saved → {REPORT_DIR / 'shap_report.txt'}")

    return fi


if __name__ == "__main__":
    run_shap_analysis()
