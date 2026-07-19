"""CommissionNet: a multi-task temporal network, plus sequence prep and evaluation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import ModelConfig
from features import FEATURE_COLUMNS, TARGET_CLS, TARGET_REG


@dataclass
class SequenceBundle:
    features: np.ndarray          # (n_samples, seq_len, n_features)
    target_reg: np.ndarray
    target_cls: np.ndarray
    meta: pd.DataFrame            # fund_id, fund_name, quarter_end per sample

    def __len__(self) -> int:
        return self.features.shape[0]


@dataclass
class StandardScaler:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray) -> "StandardScaler":
        flat = features.reshape(-1, features.shape[-1])
        std = flat.std(axis=0)
        std[std == 0] = 1.0
        return cls(mean=flat.mean(axis=0), std=std)

    def transform(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean) / self.std


def build_sequences(panel: pd.DataFrame, sequence_length: int) -> SequenceBundle:
    windows, reg, cls, meta = [], [], [], []
    for fund_id, group in panel.groupby("fund_id", sort=False):
        ordered = group.sort_values("quarter_end")
        matrix = ordered[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        reg_col = ordered[TARGET_REG].to_numpy(dtype=np.float32)
        cls_col = ordered[TARGET_CLS].to_numpy(dtype=np.float32)
        names = ordered["fund_name"].to_numpy()
        quarters = ordered["quarter_end"].to_numpy()
        for end in range(sequence_length - 1, len(ordered)):
            windows.append(matrix[end - sequence_length + 1 : end + 1])
            reg.append(reg_col[end])
            cls.append(cls_col[end])
            meta.append((fund_id, names[end], quarters[end]))
    features = np.stack(windows) if windows else np.empty((0, sequence_length, len(FEATURE_COLUMNS)))
    return SequenceBundle(
        features=features,
        target_reg=np.asarray(reg, dtype=np.float32),
        target_cls=np.asarray(cls, dtype=np.float32),
        meta=pd.DataFrame(meta, columns=["fund_id", "fund_name", "quarter_end"]),
    )


def subset_bundle(bundle: SequenceBundle, mask: np.ndarray) -> SequenceBundle:
    return SequenceBundle(
        features=bundle.features[mask],
        target_reg=bundle.target_reg[mask],
        target_cls=bundle.target_cls[mask],
        meta=bundle.meta[mask].reset_index(drop=True),
    )


def temporal_split(bundle: SequenceBundle, test_fraction: float, val_fraction: float):
    quarters = np.sort(bundle.meta["quarter_end"].unique())
    n_test = max(1, int(round(len(quarters) * test_fraction)))
    remaining = quarters[:-n_test]
    n_val = max(1, int(round(len(remaining) * val_fraction)))
    values = bundle.meta["quarter_end"].to_numpy()
    test_mask = np.isin(values, quarters[-n_test:])
    val_mask = np.isin(values, remaining[-n_val:])
    train_mask = ~test_mask & ~val_mask
    return subset_bundle(bundle, train_mask), subset_bundle(bundle, val_mask), subset_bundle(bundle, test_mask)


class AttentionPool(nn.Module):
    """Learned soft attention over the temporal dimension."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        return torch.einsum("bt,bth->bh", weights, sequence)


class CommissionNet(nn.Module):
    """A bidirectional GRU reads a window of quarterly features, attention pools
    the timesteps, and two heads jointly predict next-quarter net alpha and the
    commission-justified label."""

    def __init__(self, n_features: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.encoder = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.pool = AttentionPool(hidden_size * 2)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.regression_head = nn.Linear(hidden_size, 1)
        self.classification_head = nn.Linear(hidden_size, 1)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, _ = self.encoder(sequence)
        shared = self.trunk(self.pool(encoded))
        return self.regression_head(shared).squeeze(-1), self.classification_head(shared).squeeze(-1)


@dataclass
class TrainedModel:
    model: CommissionNet
    scaler: StandardScaler
    target_mean: float
    target_std: float
    threshold: float = 0.5
    feature_names: tuple = tuple(FEATURE_COLUMNS)

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        scaled = self.scaler.transform(features.astype(np.float32))
        with torch.no_grad():
            net_alpha, logit = self.model(torch.from_numpy(scaled).float())
        return net_alpha.numpy() * self.target_std + self.target_mean, torch.sigmoid(logit).numpy()

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "scaler_mean": self.scaler.mean,
                "scaler_std": self.scaler.std,
                "target_mean": self.target_mean,
                "target_std": self.target_std,
                "threshold": self.threshold,
                "hidden_size": self.model.encoder.hidden_size,
                "num_layers": self.model.encoder.num_layers,
                "n_features": self.model.encoder.input_size,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TrainedModel":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model = CommissionNet(blob["n_features"], blob["hidden_size"], blob["num_layers"])
        model.load_state_dict(blob["state_dict"])
        scaler = StandardScaler(mean=blob["scaler_mean"], std=blob["scaler_std"])
        return cls(model, scaler, blob["target_mean"], blob["target_std"], blob["threshold"])


def train_model(train: SequenceBundle, val: SequenceBundle, config: ModelConfig) -> tuple[TrainedModel, list[dict]]:
    if len(train) == 0 or len(val) == 0:
        raise ValueError(
            f"empty split (train={len(train)}, val={len(val)}). The date range is too short "
            "for the rolling and sequence windows; widen data.start_date or shorten the windows."
        )
    scaler = StandardScaler.fit(train.features)
    target_mean = float(train.target_reg.mean())
    target_std = float(train.target_reg.std()) or 1.0

    def prepare(bundle: SequenceBundle):
        return (
            scaler.transform(bundle.features),
            (bundle.target_reg - target_mean) / target_std,
            bundle.target_cls,
        )

    train_x, train_reg, train_cls = prepare(train)
    val_x, val_reg, val_cls = prepare(val)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_x).float(),
            torch.from_numpy(train_reg).float(),
            torch.from_numpy(train_cls).float(),
        ),
        batch_size=config.batch_size,
        shuffle=True,
    )

    model = CommissionNet(train.features.shape[-1], config.hidden_size, config.num_layers, config.dropout)
    positives = train.target_cls.sum()
    negatives = len(train.target_cls) - positives
    pos_weight = torch.tensor([negatives / positives if positives else 1.0])
    reg_loss = nn.SmoothL1Loss()
    cls_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    val_x_t = torch.from_numpy(val_x).float()
    val_reg_t = torch.from_numpy(val_reg).float()
    val_cls_t = torch.from_numpy(val_cls).float()

    history, best_val, best_state, stale = [], float("inf"), copy.deepcopy(model.state_dict()), 0
    for epoch in range(config.max_epochs):
        model.train()
        for batch_x, batch_reg, batch_cls in loader:
            optimizer.zero_grad()
            pred_reg, pred_logit = model(batch_x)
            loss = reg_loss(pred_reg, batch_reg) + config.classification_weight * cls_loss(pred_logit, batch_cls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_reg, pred_logit = model(val_x_t)
            val_loss = reg_loss(pred_reg, val_reg_t) + config.classification_weight * cls_loss(pred_logit, val_cls_t)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "val_loss": float(val_loss)})

        if val_loss.item() < best_val - 1e-5:
            best_val, best_state, stale = val_loss.item(), copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    model.load_state_dict(best_state)
    return TrainedModel(model, scaler, target_mean, target_std), history


def precision_at_top_decile(prob: np.ndarray, labels: np.ndarray, fraction: float = 0.10) -> float:
    if len(labels) == 0:
        return float("nan")
    k = max(1, int(round(len(labels) * fraction)))
    return float(labels[np.argsort(prob)[::-1][:k]].mean())


def evaluate_model(model: TrainedModel, bundle: SequenceBundle, top_decile: float = 0.10) -> dict:
    net_alpha, prob = model.predict(bundle.features)
    labels = bundle.target_cls.astype(int)
    predicted = (prob >= model.threshold).astype(int)
    metrics = {
        "n_samples": int(len(bundle)),
        "rmse": float(np.sqrt(mean_squared_error(bundle.target_reg, net_alpha))),
        "mae": float(mean_absolute_error(bundle.target_reg, net_alpha)),
        "r2": float(r2_score(bundle.target_reg, net_alpha)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "precision_top_decile": precision_at_top_decile(prob, labels, top_decile),
        "base_rate": float(labels.mean()),
        "auc_roc": float(roc_auc_score(labels, prob)) if len(np.unique(labels)) > 1 else float("nan"),
    }
    return metrics
