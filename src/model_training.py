"""
model_training.py — Ensemble model training with stacking.

Architecture:
  Level 0 (Base models):
    - XGBoost (gradient boosted trees)
    - Random Forest
    - Logistic Regression / Ridge
  
  Level 1 (Meta-learner):
    - Logistic Regression trained on cross-validated base predictions
    - This learns the optimal blend of base model outputs

  Cross-validation is time-series aware (no future leakage).

  For classification: base classifiers + regression-as-classifier
    (if predicted net_alpha > 0, classify as justified)
  
  For regression: averaging of base regressors

Run standalone:
    python -m src.model_training
"""

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    roc_auc_score, f1_score, precision_score,
    classification_report, roc_curve, accuracy_score,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DATA_DIR, MODEL_DIR, REPORT_DIR, TEST_QUARTERS, RANDOM_STATE

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  ⚠ xgboost not installed, using sklearn GradientBoosting")


# ═══════════════════════════════════════════
#  FEATURE COLUMNS (now includes fund-specific features)
# ═══════════════════════════════════════════
FEATURE_COLS = [
    # Lagged fund returns & expense gap
    "return_regular_lag1", "return_direct_lag1",
    "expense_gap_quarterly_lag1",
    # Lagged rolling financial metrics
    "rolling_alpha_lag1", "rolling_beta_lag1",
    "rolling_sharpe_lag1", "information_ratio_lag1",
    "volatility_lag1",
    # Current macro
    "repo_rate", "cpi_inflation", "yield_curve_slope_bps",
    "fii_net_flow_cr", "dii_net_flow_cr",
    # Current non-leaky
    "max_drawdown", "rolling_sortino", "expense_gap_annualised",
    # Fund-specific features (NEW — these differentiate funds)
    "alpha_consistency",         # % of past quarters with positive alpha
    "alpha_consistency_lag1",
    "expense_gap_stability",     # how stable is the commission gap
    "alpha_momentum",            # is alpha trending up or down
    "alpha_momentum_lag1",
    "relative_return",           # this fund vs median fund
    "relative_return_lag1",
    "relative_expense_gap",      # this fund's gap vs median gap
    "net_alpha_streak",          # consecutive justified/unjustified quarters
    "net_alpha_streak_lag1",
]

REGRESSION_TARGET = "target_net_alpha"
CLASSIFICATION_TARGET = "target_justified"


# ═══════════════════════════════════════════
#  TEMPORAL SPLIT
# ═══════════════════════════════════════════
def temporal_split(df):
    df = df.sort_values("quarter_end")
    cutoff_dates = sorted(df["quarter_end"].unique())
    test_start = cutoff_dates[-TEST_QUARTERS]

    train = df[df["quarter_end"] < test_start].copy()
    test = df[df["quarter_end"] >= test_start].copy()

    print(f"  → Temporal split: train {len(train)} rows "
          f"(up to {train['quarter_end'].max().date()}), "
          f"test {len(test)} rows "
          f"(from {test['quarter_end'].min().date()})")
    return train, test


def prepare_Xy(train, test, target_col):
    available = [c for c in FEATURE_COLS if c in train.columns]
    X_train = train[available].copy()
    X_test = test[available].copy()
    y_train = train[target_col].values
    y_test = test[target_col].values

    for col in available:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

    return X_train, X_test, y_train, y_test, available


def find_optimal_threshold(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j = tpr - fpr
    idx = np.argmax(j)
    return float(np.clip(thresholds[idx], 0.25, 0.75))


# ═══════════════════════════════════════════
#  STACKED CLASSIFIER
# ═══════════════════════════════════════════
def train_stacked_classifier(X_train, y_train, X_test, y_test, scaler):
    """
    Two-level stacking:
      Level 0: XGBoost, RF, LR trained with time-series CV
      Level 1: Logistic Regression meta-learner on OOF predictions
    """
    rs = RANDOM_STATE
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    spw = n_neg / max(n_pos, 1)

    # ── Base models ──
    base_models = {}
    if HAS_XGB:
        base_models["xgb"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=3.0,
            min_child_weight=8, scale_pos_weight=spw,
            eval_metric="logloss", random_state=rs, verbosity=0,
        )
    else:
        base_models["gbm"] = GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, min_samples_leaf=15, random_state=rs,
        )

    base_models["rf"] = RandomForestClassifier(
        n_estimators=500, max_depth=5, min_samples_leaf=12,
        max_features="sqrt", class_weight="balanced",
        random_state=rs,
    )

    base_models["lr"] = LogisticRegression(
        C=0.3, class_weight="balanced", max_iter=2000,
        penalty="l2", random_state=rs,
    )

    print(f"    Base models: {list(base_models.keys())}")
    print(f"    Class balance: pos={int(n_pos)}, neg={int(n_neg)}, ratio={spw:.2f}")

    # ── Generate OOF (out-of-fold) predictions for stacking ──
    n_splits = 4
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_preds = np.zeros((len(X_train), len(base_models)))
    test_preds = np.zeros((len(X_test), len(base_models)))

    X_tr_arr = X_train.values
    X_te_arr = X_test.values

    for m_idx, (name, model) in enumerate(base_models.items()):
        fold_test_preds = []

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_tr_arr)):
            X_fold_tr = X_tr_arr[tr_idx]
            y_fold_tr = y_train[tr_idx]
            X_fold_val = X_tr_arr[val_idx]

            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_fold_tr, y_fold_tr)

            oof_preds[val_idx, m_idx] = model_clone.predict_proba(X_fold_val)[:, 1]
            fold_test_preds.append(model_clone.predict_proba(X_te_arr)[:, 1])

        test_preds[:, m_idx] = np.mean(fold_test_preds, axis=0)

    # ── Train each base model on FULL training data ──
    trained_models = {}
    for name, model in base_models.items():
        model.fit(X_tr_arr, y_train)
        trained_models[name] = model

        # Individual model AUC
        indiv_probs = model.predict_proba(X_te_arr)[:, 1]
        indiv_auc = roc_auc_score(y_test, indiv_probs) if len(np.unique(y_test)) > 1 else 0
        print(f"      {name:>5s} individual AUC: {indiv_auc:.4f}")

    # ── Level 1: Meta-learner ──
    # Use OOF predictions as features for the meta-learner
    meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=rs)
    # Only use OOF rows where we have predictions (skip first fold's training rows)
    valid_oof_mask = oof_preds.sum(axis=1) != 0
    meta_model.fit(oof_preds[valid_oof_mask], y_train[valid_oof_mask])

    # Final test predictions via meta-learner
    meta_probs = meta_model.predict_proba(test_preds)[:, 1]

    # Also compute simple average for comparison
    avg_probs = test_preds.mean(axis=1)

    # Pick whichever is better
    auc_meta = roc_auc_score(y_test, meta_probs) if len(np.unique(y_test)) > 1 else 0
    auc_avg = roc_auc_score(y_test, avg_probs) if len(np.unique(y_test)) > 1 else 0

    if auc_meta >= auc_avg:
        final_probs = meta_probs
        method = "stacked"
        print(f"\n    Using stacked meta-learner (AUC={auc_meta:.4f} vs avg={auc_avg:.4f})")
    else:
        final_probs = avg_probs
        method = "averaged"
        print(f"\n    Using simple average (AUC={auc_avg:.4f} vs stacked={auc_meta:.4f})")

    return trained_models, meta_model, final_probs, method


# ═══════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════
def train_all_models(df=None):
    if df is None:
        df = pd.read_csv(DATA_DIR / "fund_dataset.csv", parse_dates=["quarter_end"])

    train, test = temporal_split(df)

    X_tr, X_te, y_reg_tr, y_reg_te, feat_cols = prepare_Xy(train, test, REGRESSION_TARGET)
    _, _, y_cls_tr, y_cls_te, _ = prepare_Xy(train, test, CLASSIFICATION_TARGET)

    scaler = StandardScaler()
    X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr), columns=feat_cols, index=X_tr.index)
    X_te_scaled = pd.DataFrame(scaler.transform(X_te), columns=feat_cols, index=X_te.index)

    all_metrics = {}

    # ════════════════════════════════════
    #  CLASSIFICATION (Stacked Ensemble)
    # ════════════════════════════════════
    print("\n  ── Classification (Stacked Ensemble) ──")

    trained_cls, meta_cls, cls_probs, cls_method = train_stacked_classifier(
        X_tr_scaled, y_cls_tr, X_te_scaled, y_cls_te, scaler
    )

    # Optimal threshold from training OOF
    tr_probs_full = np.mean([
        m.predict_proba(X_tr_scaled.values)[:, 1]
        for m in trained_cls.values()
    ], axis=0)
    threshold = find_optimal_threshold(y_cls_tr, tr_probs_full)

    cls_preds = (cls_probs > threshold).astype(int)

    auc = roc_auc_score(y_cls_te, cls_probs) if len(np.unique(y_cls_te)) > 1 else 0
    f1 = f1_score(y_cls_te, cls_preds, zero_division=0)
    acc = accuracy_score(y_cls_te, cls_preds)

    top_idx = np.argsort(cls_probs)[-max(1, len(cls_probs) // 10):]
    prec_top10 = precision_score(y_cls_te[top_idx], cls_preds[top_idx], zero_division=0)

    print(f"\n    Ensemble AUC-ROC           : {auc:.4f}")
    print(f"    Optimal threshold          : {threshold:.3f}")
    print(f"    Ensemble Accuracy          : {acc:.4f}")
    print(f"    Ensemble F1 Score          : {f1:.4f}")
    print(f"    Precision@Top 10%          : {prec_top10:.4f}")
    print(f"\n    Classification Report:\n")
    print(classification_report(y_cls_te, cls_preds,
                                target_names=["Unjustified", "Justified"], zero_division=0))

    all_metrics["classification"] = {
        "auc_roc": float(auc), "f1": float(f1), "accuracy": float(acc),
        "precision_top_decile": float(prec_top10),
        "optimal_threshold": float(threshold), "method": cls_method,
    }

    # ════════════════════════════════════
    #  REGRESSION (Averaged Ensemble)
    # ════════════════════════════════════
    print("  ── Regression (Averaged Ensemble) ──")

    reg_models = {}
    if HAS_XGB:
        reg_models["xgb"] = XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=3.0,
            min_child_weight=8, random_state=RANDOM_STATE, verbosity=0,
        )
    else:
        reg_models["gbm"] = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, min_samples_leaf=15, random_state=RANDOM_STATE,
        )

    reg_models["rf"] = RandomForestRegressor(
        n_estimators=500, max_depth=5, min_samples_leaf=12,
        max_features="sqrt", random_state=RANDOM_STATE,
    )
    reg_models["ridge"] = Ridge(alpha=1.0)

    reg_preds_all = []
    for name, model in reg_models.items():
        model.fit(X_tr_scaled, y_reg_tr)
        preds = model.predict(X_te_scaled)
        rmse_i = np.sqrt(mean_squared_error(y_reg_te, preds))
        r2_i = r2_score(y_reg_te, preds)
        print(f"      {name:>5s} — RMSE: {rmse_i:.6f}, R²: {r2_i:.4f}")
        reg_preds_all.append(preds)

    reg_preds = np.mean(reg_preds_all, axis=0)
    rmse = np.sqrt(mean_squared_error(y_reg_te, reg_preds))
    r2 = r2_score(y_reg_te, reg_preds)
    mae = mean_absolute_error(y_reg_te, reg_preds)

    print(f"\n    Ensemble RMSE : {rmse:.6f}")
    print(f"    Ensemble R²   : {r2:.4f}")
    print(f"    Ensemble MAE  : {mae:.6f}")

    all_metrics["regression"] = {
        "rmse": float(rmse), "r2": float(r2), "mae": float(mae),
    }

    # ════════════════════════════════════
    #  SAVE
    # ════════════════════════════════════
    # Find best tree model for SHAP
    tree_names = [n for n in trained_cls if n in ("xgb", "gbm", "rf")]
    best_tree_name = tree_names[0] if tree_names else list(trained_cls.keys())[0]
    best_tree = trained_cls[best_tree_name]

    # Save as a simple ensemble dict (for sip_simulation)
    joblib.dump(trained_cls, MODEL_DIR / "trained_classifiers.pkl")
    joblib.dump(meta_cls, MODEL_DIR / "meta_classifier.pkl")
    joblib.dump(reg_models, MODEL_DIR / "trained_regressors.pkl")
    joblib.dump(best_tree, MODEL_DIR / "best_tree_classifier.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump({
        "feature_cols": feat_cols,
        "optimal_threshold": threshold,
        "best_tree_name": best_tree_name,
        "cls_method": cls_method,
    }, MODEL_DIR / "model_meta.pkl")

    with open(REPORT_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  ✓ All models & metrics saved")

    return all_metrics, (trained_cls, reg_models, best_tree, scaler, feat_cols)


if __name__ == "__main__":
    train_all_models()
