from __future__ import annotations

from dataclasses import dataclass

import torch

from opd.models.pipeline_factory import DiffusionModelWrapper
from opd.models.schedulers import LinearTimeScheduler
from opd.rollout.trajectory import TrajectoryBatch, TrajectoryStep


@dataclass
class RolloutEngine:
    student: DiffusionModelWrapper
    teacher: DiffusionModelWrapper
    scheduler: LinearTimeScheduler
    initial_noise_scale: float = 1.0

    def rollout(
        self,
        prompt_batch: list[str],
        device: torch.device,
        seed: int | None = None,
    ) -> TrajectoryBatch:
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            latent = torch.randn(
                len(prompt_batch),
                self.student.latent_dim,
                device=device,
                generator=generator,
            )
        else:
            latent = torch.randn(len(prompt_batch), self.student.latent_dim, device=device)
        latent = latent * self.initial_noise_scale

        embeddings = self.student.encode_prompts(prompt_batch, device=device)
        steps: list[TrajectoryStep] = []
        for timestep in self.scheduler.timesteps:
            with torch.no_grad():
                teacher_pred = self.teacher.predict(latent, timestep, embeddings)
            student_pred = self.student.predict(latent, timestep, embeddings)

            steps.append(
                TrajectoryStep(
                    timestep=timestep,
                    latent=latent,
                    student_mean=student_pred.mean,
                    student_logvar=student_pred.logvar,
                    student_epsilon=student_pred.epsilon,
                    teacher_mean=teacher_pred.mean,
                    teacher_logvar=teacher_pred.logvar,
                    teacher_epsilon=teacher_pred.epsilon,
                )
            )
            latent = self.scheduler.step(latent, student_pred.epsilon)

        return TrajectoryBatch(prompts=prompt_batch, steps=steps, final_latent=latent)
