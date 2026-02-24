from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TrajectoryStep:
    timestep: int
    latent: torch.Tensor
    student_mean: torch.Tensor
    student_logvar: torch.Tensor
    student_epsilon: torch.Tensor
    teacher_mean: torch.Tensor
    teacher_logvar: torch.Tensor
    teacher_epsilon: torch.Tensor


@dataclass
class TrajectoryBatch:
    prompts: list[str]
    steps: list[TrajectoryStep]
    final_latent: torch.Tensor
