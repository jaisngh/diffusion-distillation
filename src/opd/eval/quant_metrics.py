from __future__ import annotations

import json
from pathlib import Path

import torch

from opd.models.pipeline_factory import DiffusionModelWrapper
from opd.models.schedulers import LinearTimeScheduler
from opd.rollout.sampling import RolloutEngine
from opd.reward.mse import epsilon_mse, latent_mse


def evaluate_alignment(
    student: DiffusionModelWrapper,
    teacher: DiffusionModelWrapper,
    scheduler: LinearTimeScheduler,
    prompts: list[str],
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    engine = RolloutEngine(
        student=student,
        teacher=teacher,
        scheduler=scheduler,
        initial_noise_scale=1.0,
    )
    trajectory = engine.rollout(prompt_batch=prompts, device=device, seed=seed)
    eps_losses = [
        epsilon_mse(step.student_epsilon, step.teacher_epsilon)
        for step in trajectory.steps
    ]
    latent_loss = latent_mse(trajectory.steps[-1].student_mean, trajectory.steps[-1].teacher_mean)

    payload = {
        "num_prompts": float(len(prompts)),
        "epsilon_mse": float(torch.stack(eps_losses).mean().detach().cpu().item()),
        "final_latent_mse": float(latent_loss.detach().cpu().item()),
    }
    payload["clip_similarity"] = -1.0
    payload["diversity"] = -1.0
    return payload


def save_eval_report(path: str | Path, report: dict[str, float]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return target
