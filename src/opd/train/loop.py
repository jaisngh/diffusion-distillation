from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast

from opd.data.dataset import PromptDataset, iter_prompt_batches
from opd.reward.losses import RewardComputer
from opd.rollout.sampling import RolloutEngine
from opd.train.checkpoint import save_checkpoint
from opd.train.optimizer import build_optimizer
from opd.utils.io import AppConfig
from opd.utils.logging import CsvMetricWriter, JsonlMetricWriter, init_logger


@dataclass
class DistillTrainer:
    config: AppConfig
    rollout_engine: RolloutEngine
    reward_computer: RewardComputer
    student: torch.nn.Module
    teacher: torch.nn.Module
    device: torch.device
    run_paths: dict[str, Path]

    def __post_init__(self) -> None:
        self.logger = init_logger("opd.trainer")
        self.optimizer = build_optimizer(self.student, self.config.train)
        self.scaler = GradScaler(enabled=self.config.train.mixed_precision and self.device.type == "cuda")
        self.global_step = 0
        self.student.to(self.device)
        self.teacher.to(self.device)
        self.teacher.eval()
        self.metrics_jsonl = JsonlMetricWriter(self.run_paths["metrics"] / "metrics.jsonl")
        self.metrics_csv = CsvMetricWriter(
            self.run_paths["metrics"] / "metrics.csv",
            fieldnames=[
                "step",
                "total_loss",
                "kl_loss",
                "epsilon_mse_loss",
                "latent_mse_loss",
                "lr",
                "grad_norm",
            ],
        )

    def step(self, prompt_batch: list[str]) -> dict[str, float]:
        self.student.train()
        with autocast(enabled=self.config.train.mixed_precision and self.device.type == "cuda"):
            trajectory = self.rollout_engine.rollout(
                prompt_batch=prompt_batch,
                device=self.device,
                seed=self.config.seed + self.global_step,
            )
            breakdown = self.reward_computer.compute(trajectory=trajectory, step=self.global_step)
            loss = breakdown.total / self.config.train.grad_accum_steps

        self.scaler.scale(loss).backward()

        grad_norm = 0.0
        will_step = (self.global_step + 1) % self.config.train.grad_accum_steps == 0
        if will_step:
            self.scaler.unscale_(self.optimizer)
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                self.student.parameters(),
                self.config.train.max_grad_norm,
            )
            grad_norm = float(grad_norm_tensor.detach().cpu().item())
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        metrics = {
            "step": float(self.global_step),
            **breakdown.as_metrics(),
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "grad_norm": grad_norm,
        }
        if self.global_step % self.config.train.log_every == 0:
            self.metrics_jsonl.write(metrics)
            self.metrics_csv.write(metrics)

        if self.global_step > 0 and self.global_step % self.config.train.save_every == 0:
            ckpt_path = self.run_paths["checkpoints"] / f"step_{self.global_step:07d}.pt"
            save_checkpoint(
                path=ckpt_path,
                student_model=self.student,
                optimizer=self.optimizer,
                step=self.global_step,
                config_hash=self.config.config_hash,
                extra={"stage": self.config.stage},
            )
            self.logger.info("Saved checkpoint: %s", ckpt_path)

        self.global_step += 1
        return metrics

    def train(self, dataset: PromptDataset) -> dict[str, float]:
        if len(dataset) == 0:
            raise ValueError("Dataset has no prompts.")
        batches = iter_prompt_batches(
            dataset=dataset,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            seed=self.config.seed,
        )
        cycle = itertools.cycle(batches)
        last_metrics: dict[str, float] = {}
        for _ in range(self.config.train.max_train_steps):
            batch_records = next(cycle)
            prompts = [record.prompt for record in batch_records]
            last_metrics = self.step(prompts)
            if not torch.isfinite(torch.tensor(last_metrics["total_loss"])):
                raise RuntimeError(f"Non-finite loss detected at step {self.global_step}.")
        final_ckpt = self.run_paths["checkpoints"] / "final.pt"
        save_checkpoint(
            path=final_ckpt,
            student_model=self.student,
            optimizer=self.optimizer,
            step=self.global_step,
            config_hash=self.config.config_hash,
            extra={"stage": self.config.stage},
        )
        self.logger.info("Training complete. Final checkpoint: %s", final_ckpt)
        return last_metrics
