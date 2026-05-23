"""
run_pipeline.py — Execute the full CommissionLens pipeline end-to-end.

Usage:
    python run_pipeline.py

This runs all 6 stages in sequence:
  1. Data Collection   (fetch NAV, benchmark, macro data)
  2. Feature Engineering (quarterly features for each fund)
  3. Target Building   (net alpha & binary labels)
  4. Model Training    (Dual-Head Deep Neural Network)
  5. SHAP Analysis     (explainability report)
  6. SIP Simulation    (back-validation)
"""

import time
import warnings
warnings.filterwarnings("ignore")

from src.data_collection import collect_all
from src.feature_engineering import engineer_features
from src.target_builder import build_targets
from src.model_training import train_all_models
from src.shap_analysis import run_shap_analysis
from src.sip_simulation import run_sip_backtest


def main():
    start = time.time()

    print("\n" + "═" * 60)
    print("  CommissionLens — Full Pipeline")
    print("═" * 60)

    # ── Stage 1: Data Collection ──
    print("\n\n🔹 STAGE 1 / 6 — DATA COLLECTION\n")
    nav_df, bench_df, macro_df = collect_all()

    # ── Stage 2: Feature Engineering ──
    print("\n\n🔹 STAGE 2 / 6 — FEATURE ENGINEERING\n")
    features_df = engineer_features(nav_df, bench_df)

    # ── Stage 3: Target Building ──
    print("\n\n🔹 STAGE 3 / 6 — TARGET BUILDING\n")
    dataset_df = build_targets(features_df)

    # ── Stage 4: Model Training ──
    print("\n\n🔹 STAGE 4 / 6 — MODEL TRAINING\n")
    metrics, models = train_all_models(dataset_df)

    # ── Stage 5: SHAP Explainability ──
    print("\n\n🔹 STAGE 5 / 6 — SHAP ANALYSIS\n")
    feature_importance = run_shap_analysis(dataset_df)

    # ── Stage 6: SIP Back-Validation ──
    print("\n\n🔹 STAGE 6 / 6 — SIP BACK-VALIDATION\n")
    sip_results, sip_summary = run_sip_backtest(dataset_df)

    elapsed = time.time() - start
    print("\n\n" + "═" * 60)
    print(f"  ✅ Pipeline complete in {elapsed:.1f}s")
    print("═" * 60)
    print("\n  Outputs:")
    print("    📁 data/fund_dataset.csv      — Engineered dataset")
    print("    📁 models/                     — Trained model files")
    print("    📁 reports/metrics.json        — Model performance metrics")
    print("    📁 reports/shap_summary.png    — SHAP beeswarm plot")
    print("    📁 reports/shap_bar.png        — Feature importance bar chart")
    print("    📁 reports/shap_report.txt     — Written explainability report")
    print("    📁 reports/sip_comparison.png  — SIP strategy comparison")
    print()


if __name__ == "__main__":
    main()
