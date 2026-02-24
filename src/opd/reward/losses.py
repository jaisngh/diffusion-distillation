from __future__ import annotations

from dataclasses import dataclass

import torch

from opd.reward.kl import gaussian_kl
from opd.reward.mse import epsilon_mse, latent_mse
from opd.rollout.trajectory import TrajectoryBatch
from opd.utils.io import DistillConfig


@dataclass
class LossBreakdown:
    total: torch.Tensor
    kl_loss: torch.Tensor
    epsilon_mse_loss: torch.Tensor
    latent_mse_loss: torch.Tensor

    def as_metrics(self) -> dict[str, float]:
        return {
            "total_loss": float(self.total.detach().cpu().item()),
            "kl_loss": float(self.kl_loss.detach().cpu().item()),
            "epsilon_mse_loss": float(self.epsilon_mse_loss.detach().cpu().item()),
            "latent_mse_loss": float(self.latent_mse_loss.detach().cpu().item()),
        }


class RewardComputer:
    def __init__(self, config: DistillConfig) -> None:
        self.config = config

    def compute(self, trajectory: TrajectoryBatch, step: int) -> LossBreakdown:
        kl_terms = []
        eps_terms = []
        for item in trajectory.steps:
            per_dim_kl = gaussian_kl(
                mean_q=item.student_mean,
                logvar_q=item.student_logvar,
                mean_p=item.teacher_mean,
                logvar_p=item.teacher_logvar,
            )
            kl_terms.append(per_dim_kl.mean())
            eps_terms.append(epsilon_mse(item.student_epsilon, item.teacher_epsilon))

        kl_loss = torch.stack(kl_terms).mean()
        eps_loss = torch.stack(eps_terms).mean()
        last_step = trajectory.steps[-1]
        final_latent_loss = latent_mse(last_step.student_mean, last_step.teacher_mean)

        if self.config.warmup_steps > 0 and step < self.config.warmup_steps:
            kl_weight = self.config.kl_weight * float(step + 1) / float(self.config.warmup_steps)
        else:
            kl_weight = self.config.kl_weight

        total = (
            kl_weight * kl_loss
            + self.config.epsilon_mse_weight * eps_loss
            + self.config.latent_mse_weight * final_latent_loss
        )
        return LossBreakdown(
            total=total,
            kl_loss=kl_loss,
            epsilon_mse_loss=eps_loss,
            latent_mse_loss=final_latent_loss,
        )
