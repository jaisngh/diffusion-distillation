#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opd.data.dataset import PromptDataset
from opd.eval.quant_metrics import evaluate_alignment, save_eval_report
from opd.models.schedulers import LinearTimeScheduler
from opd.models.student import build_student_model
from opd.models.teacher import build_teacher_model
from opd.train.checkpoint import load_checkpoint
from opd.utils.device import select_device
from opd.utils.io import ensure_output_layout, load_app_config
from opd.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quantitative evaluation for OPD.")
    parser.add_argument("--config", required=True, help="Path to top-level run config YAML.")
    parser.add_argument("--ckpt", default=None, help="Optional student checkpoint path.")
    parser.add_argument("--stage", default=None, help="Optional stage override: pilot|final.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config, stage_override=args.stage)
    paths = ensure_output_layout(config)
    set_seed(config.seed)

    dataset = PromptDataset.from_jsonl(config.prompt_file)
    heldout = [row for row in dataset.records if row.category in set(config.eval.heldout_categories)]
    if not heldout:
        heldout = dataset.records
    selected = heldout[: config.eval.num_eval_prompts]
    prompts = [row.prompt for row in selected]

    device = select_device(prefer_cuda=False)
    student = build_student_model(config.model).to(device)
    teacher = build_teacher_model(config.model).to(device)
    if args.ckpt:
        load_checkpoint(args.ckpt, student_model=student, optimizer=None)

    scheduler = LinearTimeScheduler(
        num_inference_steps=config.distill.num_inference_steps,
        step_size=config.model.scheduler_step_size,
    )
    report = evaluate_alignment(
        student=student,
        teacher=teacher,
        scheduler=scheduler,
        prompts=prompts,
        device=device,
        seed=config.seed,
    )
    report.update({"stage": config.stage, "run_name": config.run_name})
    report_path = save_eval_report(paths["metrics"] / "eval_report.json", report)
    print(json.dumps({"report_path": str(report_path), **report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
