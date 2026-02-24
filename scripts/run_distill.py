#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opd.utils.io import ensure_output_layout, load_app_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run on-policy distillation training.")
    parser.add_argument("--config", required=True, help="Path to top-level run config YAML.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path.")
    parser.add_argument("--stage", default=None, help="Optional stage override: pilot|final.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and print resolved config without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config, stage_override=args.stage)
    paths = ensure_output_layout(config)
    resolved = config.to_dict()
    print(json.dumps(resolved, indent=2, sort_keys=True))
    if args.dry_run:
        return

    from opd.data.dataset import PromptDataset
    from opd.models.schedulers import LinearTimeScheduler
    from opd.models.student import build_student_model
    from opd.models.teacher import build_teacher_model
    from opd.reward.losses import RewardComputer
    from opd.rollout.sampling import RolloutEngine
    from opd.train.checkpoint import load_checkpoint
    from opd.train.loop import DistillTrainer
    from opd.utils.device import select_device
    from opd.utils.seed import set_seed

    set_seed(config.seed)
    device = select_device(prefer_cuda=True)
    dataset = PromptDataset.from_jsonl(config.prompt_file)
    if len(dataset) == 0:
        raise RuntimeError(
            f"No prompts found in {config.prompt_file}. Run scripts/generate_prompts.py first."
        )

    student = build_student_model(config.model)
    teacher = build_teacher_model(config.model)
    scheduler = LinearTimeScheduler(
        num_inference_steps=config.distill.num_inference_steps,
        step_size=config.model.scheduler_step_size,
    )
    rollout = RolloutEngine(
        student=student,
        teacher=teacher,
        scheduler=scheduler,
        initial_noise_scale=config.distill.initial_noise_scale,
    )
    reward = RewardComputer(config.distill)
    trainer = DistillTrainer(
        config=config,
        rollout_engine=rollout,
        reward_computer=reward,
        student=student,
        teacher=teacher,
        device=device,
        run_paths=paths,
    )

    if args.resume:
        payload = load_checkpoint(args.resume, student_model=student, optimizer=trainer.optimizer)
        trainer.global_step = int(payload.get("step", 0))

    final_metrics = trainer.train(dataset)
    summary = {
        "config_hash": config.config_hash,
        "stage": config.stage,
        "run_name": config.run_name,
        "global_step": trainer.global_step,
        "final_metrics": final_metrics,
    }
    save_json(paths["run_dir"] / "train_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
