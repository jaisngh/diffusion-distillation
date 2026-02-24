from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LinearTimeScheduler:
    num_inference_steps: int
    step_size: float

    @property
    def timesteps(self) -> list[int]:
        return list(range(self.num_inference_steps - 1, -1, -1))

    def step(self, latent: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        return latent - self.step_size * epsilon
