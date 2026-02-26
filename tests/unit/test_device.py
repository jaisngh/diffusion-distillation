from __future__ import annotations

from opd.utils import device as device_utils


def test_select_device_prefers_cuda_over_mps(monkeypatch) -> None:
    monkeypatch.setattr(device_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device_utils, "_mps_available", lambda: True)
    assert device_utils.select_device().type == "cuda"


def test_select_device_uses_mps_when_cuda_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(device_utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device_utils, "_mps_available", lambda: True)
    assert device_utils.select_device().type == "mps"


def test_select_device_respects_preference_flags(monkeypatch) -> None:
    monkeypatch.setattr(device_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device_utils, "_mps_available", lambda: True)

    assert device_utils.select_device(prefer_cuda=False, prefer_mps=True).type == "mps"
    assert device_utils.select_device(prefer_cuda=False, prefer_mps=False).type == "cpu"

