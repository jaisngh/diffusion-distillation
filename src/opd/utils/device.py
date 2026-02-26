from __future__ import annotations

import torch


def _mps_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is None:
        return False
    is_available = getattr(mps_backend, "is_available", None)
    if not callable(is_available):
        return False
    is_built = getattr(mps_backend, "is_built", None)
    if callable(is_built) and not is_built():
        return False
    return bool(is_available())


def select_device(prefer_cuda: bool = True, prefer_mps: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and _mps_available():
        return torch.device("mps")
    return torch.device("cpu")
