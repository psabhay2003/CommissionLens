"""
shap_analysis.py — SHAP explainability for the DNN classifier.

Uses SHAP GradientExplainer (designed for neural networks) to decompose
each prediction into per-feature contributions.

Generates:
  1. SHAP beeswarm plot
  2. SHAP bar plot (mean |SHAP|)
  3. Written report of top 3–5 features

Run standalone:
    python -m src.shap_analysis
"""

import numpy as np
import pandas as pd
import shap
import joblib
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, MODEL_DIR, REPORT_DIR, TEST_QUARTERS, DNN_PARAMS
from src.model_training import (
    CommissionNet, FEATURE_COLS, prepare_Xy, temporal_split
)


class _ClsWrapper(nn.Module):
    """Wraps the full model to output only classification logits."""
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, x):
        _, cls_logit = self.base(x)
        return cls_logit


def run_shap_analysis(df=None):
    if df is None:
        df = pd.read_csv(DATA_DIR / "fund_dataset.csv",
                         parse_dates=["quarter_end"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, test = temporal_split(df)

    available = [c for c in FEATURE_COLS if c in train.columns]
    X_train_raw = train[available].copy()
    X_test_raw = test[available].copy()
    for col in available:
        med = X_train_raw[col].median()
        X_train_raw[col] = X_train_raw[col].fillna(med)
        X_test_raw[col] = X_test_raw[col].fillna(med)

    # Load scaler & model
    scaler = joblib.load(MODEL_DIR / "dnn_scaler.pkl")
    arch = joblib.load(MODEL_DIR / "dnn_arch.pkl")

    model = CommissionNet(
        arch["input_dim"], arch["hidden_layers"], arch["dropout"]
    ).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / "dnn_model.pt",
                                     map_location=device, weights_only=True))
    model.eval()

    cls_model = _ClsWrapper(model).to(device)
    cls_model.eval()

    X_tr_scaled = scaler.transform(X_train_raw)
    X_te_scaled = scaler.transform(X_test_raw)

    X_tr_t = torch.FloatTensor(X_tr_scaled).to(device)
    X_te_t = torch.FloatTensor(X_te_scaled).to(device)

    # Background sample
    bg_size = min(100, len(X_tr_t))
    bg_idx = np.random.choice(len(X_tr_t), bg_size, replace=False)
    background = X_tr_t[bg_idx]

    print("  → Computing SHAP values (GradientExplainer)...")
    explainer = shap.GradientExplainer(cls_model, background)
    shap_values = explainer.shap_values(X_te_t)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if torch.is_tensor(shap_values):
        shap_values = shap_values.cpu().numpy()

    X_display = pd.DataFrame(X_te_scaled, columns=available)

    # ── Beeswarm ──
    print("  → Generating SHAP beeswarm plot...")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_display, show=False, max_display=15)
    plt.title("SHAP Feature Importance — Commission Justification", fontsize=13)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Bar ──
    print("  → Generating SHAP bar plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_display, plot_type="bar",
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
        "  CommissionLens — Deep Neural Network",
        "=" * 60, "",
        "SHAP GradientExplainer was used to decompose the DNN's",
        "classification predictions into per-feature contributions.", "",
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
