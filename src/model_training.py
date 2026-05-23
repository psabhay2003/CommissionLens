"""
model_training.py — Train the dual-head Deep Neural Network.

Deep learning techniques applied for small, noisy tabular data:

  1. RESIDUAL CONNECTIONS — skip connections in the backbone help gradient
     flow and prevent degradation in deeper layers.

  2. MIXUP AUGMENTATION — blends pairs of training samples to create
     synthetic examples (Zhang et al. 2018).  On small datasets this is
     the single most impactful technique — it smooths decision boundaries
     and acts as a powerful regulariser without needing more real data.

  3. FOCAL LOSS — down-weights easy-to-classify examples and focuses the
     gradient on hard boundary cases (Lin et al. 2017).  Better than
     pos_weight for imbalanced financial data where the boundary between
     "justified" and "unjustified" commissions is inherently fuzzy.

  4. LABEL SMOOTHING — softens binary targets (0 → 0.05, 1 → 0.95) to
     prevent the model from becoming overconfident on noisy labels.
     Quarterly commission justification is inherently uncertain.

  5. LEARNED MULTI-TASK LOSS WEIGHTING — instead of loss = MSE + BCE,
     learns the optimal balance: loss = (1/2σ₁²)·MSE + (1/2σ₂²)·Focal
     + log(σ₁) + log(σ₂).  From Kendall et al. "Multi-Task Learning
     Using Uncertainty to Weigh Losses" (CVPR 2018).

  6. COSINE ANNEALING WITH WARM RESTARTS — periodically increases the
     learning rate to escape sharp local minima.  Better than
     ReduceLROnPlateau for small datasets where the loss landscape
     has many narrow valleys.

  7. STOCHASTIC WEIGHT AVERAGING (SWA) — averages model weights from
     the final epochs to find wider, flatter minima that generalise
     better (Izmailov et al. 2018).

Run standalone:
    python -m src.model_training
"""

import json
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.swa_utils import AveragedModel, SWALR
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
#  FOCAL LOSS  (Lin et al. 2017)
# ═══════════════════════════════════════════
class FocalLoss(nn.Module):
    """
    Focal loss focuses learning on hard-to-classify examples by
    down-weighting the loss contribution from easy examples.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    gamma=0 reduces to standard BCE.  gamma=2 (default) strongly
    suppresses easy examples.
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        pt = torch.exp(-bce)  # p_t = probability of correct class
        focal_weight = self.alpha * (1.0 - pt) ** self.gamma
        return (focal_weight * bce).mean()


# ═══════════════════════════════════════════
#  MIXUP AUGMENTATION  (Zhang et al. 2018)
# ═══════════════════════════════════════════
def mixup_batch(x, y_reg, y_cls, alpha=0.4):
    """
    Blend pairs of samples: x_mix = λ·x_i + (1-λ)·x_j

    Creates synthetic training examples that lie between real ones.
    This smooths the decision boundary and is the single most effective
    regulariser for small tabular datasets.

    alpha controls the Beta distribution — higher = more aggressive mixing.
    """
    if alpha <= 0:
        return x, y_reg, y_cls

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # ensure lam >= 0.5 (closer to real sample)

    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_reg_mix = lam * y_reg + (1 - lam) * y_reg[idx]
    y_cls_mix = lam * y_cls + (1 - lam) * y_cls[idx]

    return x_mix, y_reg_mix, y_cls_mix


# ═══════════════════════════════════════════
#  RESIDUAL BLOCK
# ═══════════════════════════════════════════
class ResidualBlock(nn.Module):
    """
    Linear → LayerNorm → ReLU → Dropout, with a skip connection.

    If input and output dimensions differ, a linear projection
    is used for the skip path.  LayerNorm (not BatchNorm) is more
    stable for small batch sizes common in financial data.
    """
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Skip connection (project if dimensions differ)
        self.skip = (nn.Linear(in_dim, out_dim)
                     if in_dim != out_dim else nn.Identity())

    def forward(self, x):
        return self.block(x) + self.skip(x)


# ═══════════════════════════════════════════
#  MODEL ARCHITECTURE
# ═══════════════════════════════════════════
class CommissionNet(nn.Module):
    """
    Dual-head tabular DNN with residual connections.

    Architecture:
        Input → Feature Embedding (Linear + LayerNorm)
              → ResidualBlock × N  (shared backbone)
              → Regression head  (1 neuron)
              → Classification head (1 neuron)
    """

    def __init__(self, input_dim, hidden_layers, dropout=0.2):
        super().__init__()

        # Feature embedding: project raw features into a richer space
        first_hidden = hidden_layers[0] if hidden_layers else input_dim
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, first_hidden),
            nn.LayerNorm(first_hidden),
            nn.ReLU(),
        )

        # Residual backbone
        blocks = []
        prev = first_hidden
        for h_dim in hidden_layers:
            blocks.append(ResidualBlock(prev, h_dim, dropout))
            prev = h_dim
        self.backbone = nn.Sequential(*blocks)

        # Task-specific heads (small hidden layer + output)
        self.reg_head = nn.Sequential(
            nn.Linear(prev, prev // 2),
            nn.ReLU(),
            nn.Linear(prev // 2, 1),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(prev, prev // 2),
            nn.ReLU(),
            nn.Linear(prev // 2, 1),
        )

    def forward(self, x):
        emb = self.embedding(x)
        feat = self.backbone(emb)
        reg = self.reg_head(feat).squeeze(-1)
        cls = self.cls_head(feat).squeeze(-1)
        return reg, cls


# ═══════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════
def train_model(X_train, X_test, y_reg_train, y_reg_test,
                y_cls_train, y_cls_test):
    print("\n  ── Deep Neural Network (with advanced techniques) ──")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")

    # ── Scale features ──
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # ── Label smoothing ──
    smoothing = DNN_PARAMS.get("label_smoothing", 0.1)
    y_cls_train_smooth = y_cls_train * (1 - smoothing) + 0.5 * smoothing
    print(f"    Label smoothing: {smoothing} "
          f"(0 → {0.5*smoothing:.2f}, 1 → {1-0.5*smoothing:.2f})")

    # ── Tensors ──
    X_tr_t     = torch.FloatTensor(X_tr).to(device)
    X_te_t     = torch.FloatTensor(X_te).to(device)
    y_reg_tr_t = torch.FloatTensor(y_reg_train).to(device)
    y_reg_te_t = torch.FloatTensor(y_reg_test).to(device)
    y_cls_tr_t = torch.FloatTensor(y_cls_train_smooth).to(device)
    y_cls_te_t = torch.FloatTensor(y_cls_test).to(device)  # no smoothing for eval

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_reg_tr_t, y_cls_tr_t),
        batch_size=DNN_PARAMS["batch_size"], shuffle=True,
        drop_last=False,
    )

    # ── Model ──
    model = CommissionNet(
        input_dim=X_tr.shape[1],
        hidden_layers=DNN_PARAMS["hidden_layers"],
        dropout=DNN_PARAMS["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Parameters: {total_params:,}")

    # ── Losses ──
    mse_fn = nn.MSELoss()
    focal_fn = FocalLoss(
        gamma=DNN_PARAMS.get("focal_gamma", 2.0),
        alpha=DNN_PARAMS.get("focal_alpha", 0.25),
    )
    print(f"    Focal loss: γ={focal_fn.gamma}, α={focal_fn.alpha}")

    # ── Learned multi-task loss weighting (Kendall et al. 2018) ──
    # log_sigma parameters learn the optimal balance between tasks
    log_sigma_reg = torch.nn.Parameter(torch.zeros(1, device=device))
    log_sigma_cls = torch.nn.Parameter(torch.zeros(1, device=device))

    # ── Optimizer with weight decay ──
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + [log_sigma_reg, log_sigma_cls],
        lr=DNN_PARAMS["learning_rate"],
        weight_decay=DNN_PARAMS.get("weight_decay", 1e-3),
    )

    # ── Cosine annealing with warm restarts ──
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=DNN_PARAMS.get("cosine_T0", 30), T_mult=2
    )

    # ── SWA setup ──
    swa_start = DNN_PARAMS.get("swa_start", 100)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=DNN_PARAMS["learning_rate"] * 0.2)
    swa_active = False

    # ── Mixup config ──
    mixup_alpha = DNN_PARAMS.get("mixup_alpha", 0.4)
    print(f"    Mixup: α={mixup_alpha}")
    print(f"    SWA starts at epoch {swa_start}")
    print(f"    Cosine annealing: T_0={DNN_PARAMS.get('cosine_T0', 30)}, T_mult=2")

    # ── Training loop ──
    best_val_loss = float("inf")
    patience_ctr = 0
    best_state = None

    for epoch in range(DNN_PARAMS["epochs"]):
        model.train()
        epoch_loss = 0.0

        for bx, by_reg, by_cls in train_loader:
            # ── Apply mixup ──
            bx_mix, by_reg_mix, by_cls_mix = mixup_batch(
                bx, by_reg, by_cls, alpha=mixup_alpha
            )

            optimizer.zero_grad()
            pred_reg, pred_cls = model(bx_mix)

            # ── Learned multi-task weighting ──
            loss_reg = mse_fn(pred_reg, by_reg_mix)
            loss_cls = focal_fn(pred_cls, by_cls_mix)

            # loss = (1/2σ²)·L + log(σ)  per task
            loss = (torch.exp(-log_sigma_reg) * loss_reg + log_sigma_reg +
                    torch.exp(-log_sigma_cls) * loss_cls + log_sigma_cls)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # ── Scheduler / SWA switch ──
        if epoch >= swa_start and not swa_active:
            swa_active = True
            print(f"    ↳ SWA activated at epoch {epoch + 1}")

        if swa_active:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # ── Validation ──
        model.eval()
        with torch.no_grad():
            v_reg, v_cls = model(X_te_t)
            val_loss_reg = mse_fn(v_reg, y_reg_te_t)
            val_loss_cls = focal_fn(v_cls, y_cls_te_t)
            val_loss = (val_loss_reg + val_loss_cls).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if not swa_active and patience_ctr >= DNN_PARAMS["patience"]:
            # If haven't reached SWA yet, jump to it instead of stopping
            if epoch < swa_start:
                swa_start = epoch + 1
                patience_ctr = 0
                print(f"    ↳ Patience hit — fast-forwarding to SWA at epoch {swa_start}")
                continue
            else:
                print(f"    Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 50 == 0:
            w_reg = torch.exp(-log_sigma_reg).item()
            w_cls = torch.exp(-log_sigma_cls).item()
            print(f"    Epoch {epoch+1:>3d} | "
                  f"Train: {epoch_loss/len(train_loader):.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"Weights: reg={w_reg:.2f} cls={w_cls:.2f}")

    # ── Use SWA model if available, otherwise best snapshot ──
    if swa_active:
        # Update batch norm statistics for SWA model
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        eval_model = swa_model
        print(f"    ✓ Using SWA-averaged model")
        # Save the inner model's state dict
        final_state = swa_model.module.state_dict()
    else:
        model.load_state_dict(best_state)
        eval_model = model
        final_state = best_state

    # ── Evaluate ──
    eval_model.eval()
    with torch.no_grad():
        reg_preds, cls_logits = eval_model(X_te_t)
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
    # Save the base model (not the SWA wrapper)
    base_model = CommissionNet(
        input_dim=X_tr.shape[1],
        hidden_layers=DNN_PARAMS["hidden_layers"],
        dropout=DNN_PARAMS["dropout"],
    )
    base_model.load_state_dict(final_state)
    torch.save(base_model.state_dict(), MODEL_DIR / "dnn_model.pt")
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
    return base_model, scaler, metrics


# ═══════════════════════════════════════════
#  ORCHESTRATOR
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

    model, scaler, metrics = train_model(
        X_tr, X_te, y_reg_tr, y_reg_te, y_cls_tr, y_cls_te
    )

    metrics_path = REPORT_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ✓ Metrics saved → {metrics_path}")

    return metrics, (model, scaler, feat_cols)


if __name__ == "__main__":
    train_all_models()
