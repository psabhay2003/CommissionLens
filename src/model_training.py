"""
model_training.py — Train the dual-head Deep Neural Network.

Single model, two tasks:
  - Regression head  → predicts net_alpha (RMSE, R²)
  - Classification head → predicts commission_justified (AUC-ROC, F1)

Train/test split is TEMPORAL: everything before the last N quarters is
training; the last N quarters are test.  No look-ahead bias.

Run standalone:
    python -m src.model_training
"""

import json
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score,
    roc_auc_score, f1_score, precision_score, classification_report
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (DATA_DIR, MODEL_DIR, REPORT_DIR,
                    TEST_QUARTERS, RANDOM_STATE, DNN_PARAMS)


# ═══════════════════════════════════════════
#  FEATURE COLUMNS
# ═══════════════════════════════════════════
FEATURE_COLS = [
    # Lagged fund features (no data leakage)
    "return_regular_lag1", "return_direct_lag1",
    "expense_gap_quarterly_lag1",
    "rolling_alpha_lag1", "rolling_beta_lag1",
    "rolling_sharpe_lag1", "information_ratio_lag1",
    "volatility_lag1",
    # Macro features (known at prediction time)
    "repo_rate", "cpi_inflation", "yield_curve_slope_bps",
    "fii_net_flow_cr", "dii_net_flow_cr",
    # Non-leaky current features
    "max_drawdown", "rolling_sortino",
    "expense_gap_annualised",
]

REGRESSION_TARGET = "target_net_alpha"
CLASSIFICATION_TARGET = "target_justified"


# ═══════════════════════════════════════════
#  TEMPORAL TRAIN / TEST SPLIT
# ═══════════════════════════════════════════
def temporal_split(df: pd.DataFrame):
    """
    Split by time: last TEST_QUARTERS quarters → test set.
    Everything before → training.  Mimics real deployment.
    """
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
    """Extract feature matrices and target vectors."""
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
#  MODEL ARCHITECTURE
# ═══════════════════════════════════════════
class CommissionNet(nn.Module):
    """
    Dual-head tabular DNN.

    Why a single dual-head network instead of two separate models?

    1. SHARED REPRESENTATION — the backbone learns features useful for
       BOTH tasks.  Understanding "how much" alpha (regression) sharpens
       the decision boundary for "whether" commission is justified
       (classification).  Two separate models wouldn't share this learning.

    2. IMPLICIT REGULARISATION — the two losses (MSE + BCE) regularise
       each other.  Regression prevents the classifier from memorising
       threshold artefacts; classification keeps regression from chasing
       outliers.

    3. NON-LINEAR INTERACTIONS — fund features interact in complex ways
       (e.g., high alpha + low volatility + falling repo rate).  The DNN
       with batch normalisation captures these via learned weight
       combinations without manual feature crossing.

    Architecture:
        Input → [Linear → BatchNorm → ReLU → Dropout] × 3
              → Regression head (1 neuron, no activation)
              → Classification head (1 neuron, sigmoid at inference)
    """

    def __init__(self, input_dim, hidden_layers, dropout=0.3):
        super().__init__()

        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h

        self.backbone = nn.Sequential(*layers)
        self.reg_head = nn.Linear(prev, 1)   # regression
        self.cls_head = nn.Linear(prev, 1)   # classification

    def forward(self, x):
        feat = self.backbone(x)
        reg = self.reg_head(feat).squeeze(-1)
        cls = self.cls_head(feat).squeeze(-1)
        return reg, cls


# ═══════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════
def train_model(X_train, X_test, y_reg_train, y_reg_test,
                y_cls_train, y_cls_test):
    """
    Train the dual-head DNN with early stopping.
    Joint loss = MSE (regression) + BCE (classification).
    """
    print("\n  ── Deep Neural Network ──")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")

    # Scale features
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # Tensors
    X_tr_t     = torch.FloatTensor(X_tr).to(device)
    X_te_t     = torch.FloatTensor(X_te).to(device)
    y_reg_tr_t = torch.FloatTensor(y_reg_train).to(device)
    y_reg_te_t = torch.FloatTensor(y_reg_test).to(device)
    y_cls_tr_t = torch.FloatTensor(y_cls_train).to(device)
    y_cls_te_t = torch.FloatTensor(y_cls_test).to(device)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_reg_tr_t, y_cls_tr_t),
        batch_size=DNN_PARAMS["batch_size"], shuffle=True
    )

    model = CommissionNet(
        input_dim=X_tr.shape[1],
        hidden_layers=DNN_PARAMS["hidden_layers"],
        dropout=DNN_PARAMS["dropout"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=DNN_PARAMS["learning_rate"],
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
    mse_fn = nn.MSELoss()

    # Handle class imbalance: weight the minority class higher
    n_pos = y_cls_train.sum()
    n_neg = len(y_cls_train) - n_pos
    pos_weight = torch.FloatTensor([n_neg / max(n_pos, 1)]).to(device)
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"    Class balance — pos: {int(n_pos)}, neg: {int(n_neg)}, "
          f"pos_weight: {pos_weight.item():.2f}")

    best_val_loss = float("inf")
    patience_ctr = 0
    best_state = None

    for epoch in range(DNN_PARAMS["epochs"]):
        model.train()
        epoch_loss = 0.0

        for bx, by_reg, by_cls in train_loader:
            optimizer.zero_grad()
            pred_reg, pred_cls = model(bx)
            loss = mse_fn(pred_reg, by_reg) + bce_fn(pred_cls, by_cls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            v_reg, v_cls = model(X_te_t)
            val_loss = (mse_fn(v_reg, y_reg_te_t) + bce_fn(v_cls, y_cls_te_t)).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if patience_ctr >= DNN_PARAMS["patience"]:
            print(f"    Early stopping at epoch {epoch + 1}")
            break

        if (epoch + 1) % 25 == 0:
            print(f"    Epoch {epoch+1:>3d} | "
                  f"Train: {epoch_loss/len(train_loader):.6f} | "
                  f"Val: {val_loss:.6f}")

    model.load_state_dict(best_state)
    model.eval()

    # ── Evaluate ──
    with torch.no_grad():
        reg_preds, cls_logits = model(X_te_t)
        reg_preds = reg_preds.cpu().numpy()
        cls_probs = torch.sigmoid(cls_logits).cpu().numpy()
        cls_preds = (cls_probs > 0.5).astype(int)

    rmse = np.sqrt(mean_squared_error(y_reg_test, reg_preds))
    r2 = r2_score(y_reg_test, reg_preds)
    has_both = len(np.unique(y_cls_test)) > 1
    auc = roc_auc_score(y_cls_test, cls_probs) if has_both else 0.0
    f1 = f1_score(y_cls_test, cls_preds, zero_division=0)

    top_idx = np.argsort(cls_probs)[-max(1, len(cls_probs) // 10):]
    prec_top10 = precision_score(
        y_cls_test[top_idx], cls_preds[top_idx], zero_division=0
    )

    print(f"\n    [Regression]")
    print(f"      RMSE : {rmse:.6f}")
    print(f"      R²   : {r2:.4f}")
    print(f"\n    [Classification]")
    print(f"      AUC-ROC           : {auc:.4f}")
    print(f"      F1 Score          : {f1:.4f}")
    print(f"      Precision@Top 10% : {prec_top10:.4f}")
    print(f"\n    Classification Report:\n")
    print(classification_report(y_cls_test, cls_preds,
                                target_names=["Unjustified", "Justified"],
                                zero_division=0))

    # ── Save ──
    torch.save(model.state_dict(), MODEL_DIR / "dnn_model.pt")
    joblib.dump(scaler, MODEL_DIR / "dnn_scaler.pkl")
    joblib.dump({
        "input_dim": X_tr.shape[1],
        "hidden_layers": DNN_PARAMS["hidden_layers"],
        "dropout": DNN_PARAMS["dropout"],
    }, MODEL_DIR / "dnn_arch.pkl")

    metrics = {
        "regression": {"rmse": float(rmse), "r2": float(r2)},
        "classification": {"auc_roc": float(auc), "f1": float(f1),
                           "precision_top_decile": float(prec_top10)},
    }
    return model, scaler, metrics


# ═══════════════════════════════════════════
#  ORCHESTRATOR
# ═══════════════════════════════════════════
def train_all_models(df=None):
    """Run the full training pipeline."""
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

    model, scaler, metrics = train_model(
        X_tr, X_te, y_reg_tr, y_reg_te, y_cls_tr, y_cls_te
    )

    # Save metrics
    metrics_path = REPORT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ✓ Metrics saved → {metrics_path}")

    return metrics, (model, scaler, feat_cols)


if __name__ == "__main__":
    train_all_models()
