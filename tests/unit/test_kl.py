from __future__ import annotations

import torch

from opd.reward.kl import gaussian_kl


def test_gaussian_kl_zero_for_identical_distributions() -> None:
    mean = torch.zeros(3, 8)
    logvar = torch.zeros(3, 8)
    result = gaussian_kl(mean, logvar, mean, logvar)
    assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)


def test_gaussian_kl_non_negative_and_finite() -> None:
    mean_q = torch.randn(2, 4)
    logvar_q = torch.randn(2, 4) * 0.1
    mean_p = torch.randn(2, 4)
    logvar_p = torch.randn(2, 4) * 0.1
    result = gaussian_kl(mean_q, logvar_q, mean_p, logvar_p)
    assert torch.isfinite(result).all()
    assert (result >= 0).all()
