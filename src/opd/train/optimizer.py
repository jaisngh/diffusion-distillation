from __future__ import annotations

import torch

from opd.utils.io import TrainConfig


def build_optimizer(model: torch.nn.Module, train_config: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
