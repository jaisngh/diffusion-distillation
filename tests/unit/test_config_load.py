from __future__ import annotations

from pathlib import Path

from opd.utils.io import load_app_config


def test_load_app_config_defaults() -> None:
    cfg = load_app_config(str(Path("configs/pilot_run.yaml")))
    assert cfg.stage == "pilot"
    assert cfg.model.backend == "mock"
    assert cfg.config_hash
    assert cfg.git_commit


def test_cuda_config_uses_diffusers_backend() -> None:
    cfg = load_app_config(str(Path("configs/sd35m_sd3m_run.yaml")))
    assert cfg.stage == "final"
    assert cfg.model.backend == "diffusers"
