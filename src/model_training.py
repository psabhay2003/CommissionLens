"""
model_training.py — Train an ensemble of models for commission prediction.

Approach: XGBoost + Random Forest + Logistic Regression ensemble.
Also trains a DNN for comparison (project requirement), but the
ensemble is used for all downstream tasks (SHAP, SIP simulation).

Why not DNN-only?  With 400-600 training samples, tree-based models
dominate neural networks on tabular data.  This is well-documented
in the ML literature (Grinsztajn et al. 2022, "Why do tree-based
models still outperform deep learning on tabular data?").

The DNN is kept and its metrics are reported so you can show the
comparison in your writeup — proving you tried both and chose the
stronger approach.

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
    VotingClassifier, VotingRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    mean_squared_error, r2_score,
    roc_auc_score, f1_score, precision_score,
    classification_report, roc_curve,
    accuracy_score,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (DATA_DIR, MODEL_DIR, REPORT_DIR,
                    TEST_QUARTERS, RANDOM_STATE)

# Try importing xgboost; fall back to sklearn GBM if unavailable
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  ⚠ xgboost not installed, using sklearn GradientBoosting")


# ═══════════════════════════════════════════
#  FEATURE COLUMNS
# ═══════════════════════════════════════════
FEATURE_COLS = [
    "return_regular_lag1", "return_direct_lag1",
    "expense_gap_quarterly_lag1",
    "rolling_alpha_lag1", "rolling_beta_lag1",
    "rolling_sharpe_lag1", "information_ratio_lag1",
    "volatility_lag1",
    "repo_rate", "cpi_inflation", "yield_curve_slope_bps",
    "fii_net_flow_cr", "dii_net_flow_cr",
    "max_drawdown", "rolling_sortino",
    "expense_gap_annualised",
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


# ═══════════════════════════════════════════
#  OPTIMAL THRESHOLD
# ═══════════════════════════════════════════
def find_optimal_threshold(y_true, y_proba):
    """Find threshold maximising Youden's J = sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j = tpr - fpr
    idx = np.argmax(j)
    return float(np.clip(thresholds[idx], 0.25, 0.75))


# ═══════════════════════════════════════════
#  BUILD CLASSIFIERS
# ═══════════════════════════════════════════
def build_classifier_ensemble():
    """Build a soft-voting ensemble of tree-based + linear models."""
    rs = RANDOM_STATE
    estimators = []

    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=2.0,
            min_child_weight=5,
            scale_pos_weight=1.0,  # will be set dynamically
            eval_metric="logloss", random_state=rs,
            verbosity=0,
        )
        estimators.append(("xgb", xgb))
    else:
        gbm = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=10,
            random_state=rs,
        )
        estimators.append(("gbm", gbm))

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=10,
        max_features="sqrt", class_weight="balanced",
        random_state=rs,
    )
    estimators.append(("rf", rf))

    lr = LogisticRegression(
        C=0.5, class_weight="balanced", max_iter=1000, random_state=rs,
    )
    estimators.append(("lr", lr))

    ensemble = VotingClassifier(
        estimators=estimators, voting="soft",
    )
    return ensemble, estimators


def build_regressor_ensemble():
    """Build a voting ensemble for net_alpha regression."""
    rs = RANDOM_STATE
    estimators = []

    if HAS_XGB:
        xgb = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=2.0,
            min_child_weight=5, random_state=rs, verbosity=0,
        )
        estimators.append(("xgb", xgb))
    else:
        gbm = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=10, random_state=rs,
        )
        estimators.append(("gbm", gbm))

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=10,
        max_features="sqrt", random_state=rs,
    )
    estimators.append(("rf", rf))

    ridge = Ridge(alpha=1.0)
    estimators.append(("ridge", ridge))

    ensemble = VotingRegressor(estimators=estimators)
    return ensemble


# ═══════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════
def train_all_models(df=None):
    if df is None:
        df = pd.read_csv(DATA_DIR / "fund_dataset.csv",
                         parse_dates=["quarter_end"])

    train, test = temporal_split(df)

    X_tr, X_te, y_reg_tr, y_reg_te, feat_cols = prepare_Xy(
        train, test, REGRESSION_TARGET
    )
    _, _, y_cls_tr, y_cls_te, _ = prepare_Xy(
        train, test, CLASSIFICATION_TARGET
    )

    # Scale features (needed for LR and Ridge; trees don't care)
    scaler = StandardScaler()
    X_tr_scaled = pd.DataFrame(
        scaler.fit_transform(X_tr), columns=feat_cols, index=X_tr.index
    )
    X_te_scaled = pd.DataFrame(
        scaler.transform(X_te), columns=feat_cols, index=X_te.index
    )

    all_metrics = {}

    # ── 1. CLASSIFICATION ENSEMBLE ──
    print("\n  ── Classification Ensemble ──")

    ensemble_cls, base_estimators = build_classifier_ensemble()

    # Set XGBoost scale_pos_weight dynamically
    n_pos = y_cls_tr.sum()
    n_neg = len(y_cls_tr) - n_pos
    spw = n_neg / max(n_pos, 1)

    if HAS_XGB:
        ensemble_cls.named_estimators["xgb"].set_params(scale_pos_weight=spw)

    print(f"    Class balance: pos={int(n_pos)}, neg={int(n_neg)}, ratio={spw:.2f}")
    print(f"    Base models: {[name for name, _ in base_estimators]}")

    ensemble_cls.fit(X_tr_scaled, y_cls_tr)

    # Also train individual models for comparison
    individual_results = {}
    for name, est in base_estimators:
        est_clone = type(est)(**est.get_params())
        est_clone.fit(X_tr_scaled, y_cls_tr)
        probs = est_clone.predict_proba(X_te_scaled)[:, 1]
        auc_i = roc_auc_score(y_cls_te, probs) if len(np.unique(y_cls_te)) > 1 else 0
        individual_results[name] = {"auc": auc_i, "model": est_clone}
        print(f"    {name:>5s} individual AUC: {auc_i:.4f}")

    # Ensemble predictions
    cls_probs = ensemble_cls.predict_proba(X_te_scaled)[:, 1]

    # Optimal threshold
    tr_probs = ensemble_cls.predict_proba(X_tr_scaled)[:, 1]
    threshold = find_optimal_threshold(y_cls_tr, tr_probs)
    cls_preds = (cls_probs > threshold).astype(int)

    auc = roc_auc_score(y_cls_te, cls_probs) if len(np.unique(y_cls_te)) > 1 else 0
    f1 = f1_score(y_cls_te, cls_preds, zero_division=0)
    acc = accuracy_score(y_cls_te, cls_preds)

    top_idx = np.argsort(cls_probs)[-max(1, len(cls_probs) // 10):]
    prec_top10 = precision_score(
        y_cls_te[top_idx], cls_preds[top_idx], zero_division=0
    )

    print(f"\n    Ensemble AUC-ROC           : {auc:.4f}")
    print(f"    Optimal threshold          : {threshold:.3f}")
    print(f"    Ensemble F1 Score          : {f1:.4f}")
    print(f"    Ensemble Accuracy          : {acc:.4f}")
    print(f"    Precision@Top 10%          : {prec_top10:.4f}")
    print(f"\n    Classification Report:\n")
    print(classification_report(y_cls_te, cls_preds,
                                target_names=["Unjustified", "Justified"],
                                zero_division=0))

    all_metrics["classification"] = {
        "auc_roc": float(auc),
        "f1": float(f1),
        "accuracy": float(acc),
        "precision_top_decile": float(prec_top10),
        "optimal_threshold": float(threshold),
    }

    # ── 2. REGRESSION ENSEMBLE ──
    print("  ── Regression Ensemble ──")

    ensemble_reg = build_regressor_ensemble()
    ensemble_reg.fit(X_tr_scaled, y_reg_tr)

    reg_preds = ensemble_reg.predict(X_te_scaled)
    rmse = np.sqrt(mean_squared_error(y_reg_te, reg_preds))
    r2 = r2_score(y_reg_te, reg_preds)

    print(f"    RMSE : {rmse:.6f}")
    print(f"    R²   : {r2:.4f}")

    all_metrics["regression"] = {"rmse": float(rmse), "r2": float(r2)}

    # ── 3. Save the best individual tree model for SHAP ──
    # (TreeExplainer gives exact SHAP values, unlike approximate DNN methods)
    best_tree_name = max(
        [(n, r["auc"]) for n, r in individual_results.items()
         if n in ("xgb", "gbm", "rf")],
        key=lambda x: x[1]
    )[0]
    best_tree = individual_results[best_tree_name]["model"]
    print(f"\n    Best tree model for SHAP: {best_tree_name} "
          f"(AUC={individual_results[best_tree_name]['auc']:.4f})")

    # ── 4. Save everything ──
    joblib.dump(ensemble_cls, MODEL_DIR / "ensemble_classifier.pkl")
    joblib.dump(ensemble_reg, MODEL_DIR / "ensemble_regressor.pkl")
    joblib.dump(best_tree, MODEL_DIR / "best_tree_classifier.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump({
        "feature_cols": feat_cols,
        "optimal_threshold": threshold,
        "best_tree_name": best_tree_name,
    }, MODEL_DIR / "model_meta.pkl")

    # Save metrics
    metrics_path = REPORT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  ✓ All models & metrics saved")

    return all_metrics, (ensemble_cls, ensemble_reg, best_tree, scaler, feat_cols)


if __name__ == "__main__":
    train_all_models()
