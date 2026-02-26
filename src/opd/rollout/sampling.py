from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
        latent = self.student.prepare_initial_latents(
            batch_size=len(prompt_batch),
            device=device,
            seed=seed,
            scale=self.initial_noise_scale,
        )

        embeddings = self.student.encode_prompts(prompt_batch, device=device)
        timesteps: list[Any]
        if self.student.config.backend == "diffusers":
            timesteps = self.student.get_rollout_timesteps(self.scheduler.num_inference_steps, device=device)
        else:
            timesteps = self.scheduler.timesteps

        steps: list[TrajectoryStep] = []
        for timestep in timesteps:
            with torch.no_grad():
                teacher_pred = self.teacher.predict(latent, timestep, embeddings)
            student_pred = self.student.predict(latent, timestep, embeddings)
            if isinstance(timestep, torch.Tensor):
                step_value = float(timestep.detach().cpu().item())
            else:
                step_value = float(timestep)

            steps.append(
                TrajectoryStep(
                    timestep=step_value,
                    latent=latent,
                    student_mean=student_pred.mean,
                    student_logvar=student_pred.logvar,
                    student_epsilon=student_pred.epsilon,
                    teacher_mean=teacher_pred.mean,
                    teacher_logvar=teacher_pred.logvar,
                    teacher_epsilon=teacher_pred.epsilon,
                )
            )
            latent = self.student.step_latents(
                latents=latent,
                epsilon=student_pred.epsilon,
                timestep=timestep,
                fallback_step_size=self.scheduler.step_size,
            )

        return TrajectoryBatch(prompts=prompt_batch, steps=steps, final_latent=latent)
