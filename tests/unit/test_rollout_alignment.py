from __future__ import annotations

import torch

from opd.models.schedulers import LinearTimeScheduler
from opd.models.student import build_student_model
from opd.models.teacher import build_teacher_model
from opd.rollout.sampling import RolloutEngine
from opd.utils.io import ModelConfig


def test_rollout_shapes_and_timestep_alignment() -> None:
    model_cfg = ModelConfig(
        backend="mock",
        latent_dim=16,
        text_embed_dim=16,
        num_inference_steps=8,
    )
    student = build_student_model(model_cfg)
    teacher = build_teacher_model(model_cfg)
    scheduler = LinearTimeScheduler(num_inference_steps=8, step_size=0.1)
    engine = RolloutEngine(student=student, teacher=teacher, scheduler=scheduler, initial_noise_scale=1.0)

    trajectory = engine.rollout(["prompt one", "prompt two"], device=torch.device("cpu"), seed=7)
    assert len(trajectory.steps) == 8
    assert [step.timestep for step in trajectory.steps] == [float(i) for i in range(7, -1, -1)]
    assert trajectory.final_latent.shape == (2, 16)
    for step in trajectory.steps:
        assert step.student_mean.shape == step.teacher_mean.shape == (2, 16)
        assert step.student_logvar.shape == step.teacher_logvar.shape == (2, 16)
