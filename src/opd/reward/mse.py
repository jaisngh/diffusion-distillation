from __future__ import annotations

import torch


def epsilon_mse(student_epsilon: torch.Tensor, teacher_epsilon: torch.Tensor) -> torch.Tensor:
    return torch.mean((student_epsilon - teacher_epsilon).pow(2))


def latent_mse(student_latent: torch.Tensor, teacher_latent: torch.Tensor) -> torch.Tensor:
    return torch.mean((student_latent - teacher_latent).pow(2))
