from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class DataConfig:
    source: str = "synthetic"
    n_funds: int = 220
    start_date: str = "2013-01-01"
    end_date: str = "2023-12-31"
    risk_free_annual: float = 0.06
    benchmark_name: str = "NIFTY 50 TRI"


@dataclass
class FeatureConfig:
    rolling_window_months: int = 36
    min_history_months: int = 24


@dataclass
class ModelConfig:
    sequence_length: int = 8
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.25
    classification_weight: float = 0.5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 120
    patience: int = 15
    val_fraction: float = 0.15
    test_fraction: float = 0.20


@dataclass
class SimulationConfig:
    monthly_investment: float = 5000.0
    start_date: str = "2018-07-01"
    end_date: str = "2023-06-30"
    top_decile: float = 0.10


@dataclass
class Config:
    seed: int = 7
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    def resolve(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def data_path(self) -> Path:
        return self.resolve(self.data_dir)

    @property
    def artifacts_path(self) -> Path:
        return self.resolve(self.artifacts_dir)


def _build(cls: type, values: dict[str, Any]) -> Any:
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in values:
            continue
        raw = values[f.name]
        field_type = hints.get(f.name)
        if is_dataclass(field_type) and isinstance(raw, dict):
            kwargs[f.name] = _build(field_type, raw)
        else:
            kwargs[f.name] = raw
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> Config:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return Config()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _build(Config, raw)
