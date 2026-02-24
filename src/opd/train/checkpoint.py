from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    student_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config_hash: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "student_model": student_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config_hash": config_hash,
        "extra": extra or {},
    }
    torch.save(payload, target)
    return target


def load_checkpoint(
    path: str | Path,
    student_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location)
    student_model.load_state_dict(payload["student_model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
