from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    model_name: str
    max_length: int
    batch_size: int
    lr: float
    epochs: int
    weight_decay: float
    warmup_ratio: float
    eval_every_steps: int
    synthesis_multiplier: int
    alpha: float
    beta: float
    gamma: float
    dropout_passes: int
    output_root: str


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return Config(**data)

