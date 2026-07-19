"""SHAP feature importance for CommissionNet's commission-justification head."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import torch
from torch import nn

from features import FEATURE_COLUMNS
from model import TrainedModel


class _JustificationWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        _, logit = self.model(sequence)
        return logit.unsqueeze(-1)


def compute_feature_importance(
    model: TrainedModel,
    background: np.ndarray,
    samples: np.ndarray,
    background_size: int = 128,
) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, averaged over the temporal window."""
    model.model.eval()
    bg = model.scaler.transform(background.astype(np.float32))
    xs = model.scaler.transform(samples.astype(np.float32))
    if len(bg) > background_size:
        idx = np.random.default_rng(0).choice(len(bg), background_size, replace=False)
        bg = bg[idx]

    explainer = shap.GradientExplainer(_JustificationWrapper(model.model), torch.from_numpy(bg).float())
    shap_values = explainer.shap_values(torch.from_numpy(xs).float())
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 4:
        shap_values = shap_values[..., 0]

    per_feature = np.abs(shap_values).mean(axis=(0, 1))
    return (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "mean_abs_shap": per_feature})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
