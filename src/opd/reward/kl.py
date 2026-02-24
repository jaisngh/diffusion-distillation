from __future__ import annotations

import torch


def gaussian_kl(
    mean_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mean_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    var_ratio = torch.exp(logvar_q - logvar_p)
    mean_diff = (mean_p - mean_q).pow(2) * torch.exp(-logvar_p)
    kl = 0.5 * (logvar_p - logvar_q + var_ratio + mean_diff - 1.0)
    return kl
