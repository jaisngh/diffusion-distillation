#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from opd.data.dataset import PromptDataset
from opd.models.schedulers import LinearTimeScheduler
from opd.models.teacher import build_teacher_model
from opd.rollout.sampling import RolloutEngine
from opd.utils.device import select_device
from opd.utils.io import ensure_output_layout, load_app_config, save_json
from opd.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic teacher dataset manifest.")
    parser.add_argument("--config", required=True, help="Path to top-level run config YAML.")
    parser.add_argument("--stage", default=None, help="Optional stage override: pilot|final.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config, stage_override=args.stage)
    paths = ensure_output_layout(config)
    set_seed(config.seed)

    prompt_dataset = PromptDataset.from_jsonl(config.prompt_file)
    device = select_device(prefer_cuda=False)
    print(f"Using device: {device}")
    teacher = build_teacher_model(config.model).to(device)
    scheduler = LinearTimeScheduler(
        num_inference_steps=config.distill.num_inference_steps,
        step_size=config.model.scheduler_step_size,
    )
    engine = RolloutEngine(
        student=teacher,
        teacher=teacher,
        scheduler=scheduler,
        initial_noise_scale=config.distill.initial_noise_scale,
    )

    latent_dir = paths["datasets"] / "teacher_latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []

    for record in prompt_dataset.records:
        print(f"starting rollout for prompt: ", record.prompt)
        trajectory = engine.rollout([record.prompt], device=device, seed=record.seed)
        latent_path = latent_dir / f"{record.prompt_id}.pt"
        torch.save({"final_latent": trajectory.final_latent.detach().cpu()}, latent_path)
        manifest_rows.append(
            {
                "prompt_id": record.prompt_id,
                "category": record.category,
                "seed": record.seed,
                "latent_path": str(latent_path),
                "image_path": str(paths["images"] / "teacher" / f"{record.prompt_id}.png"),
            }
        )
        print(f"finished rollout for prompt: ", record.prompt)
    manifest_payload = {
        "teacher_model_id": config.model.teacher_model_id,
        "config_hash": config.config_hash,
        "num_samples": len(manifest_rows),
        "samples": manifest_rows,
    }
    save_json(config.teacher_dataset_manifest, manifest_payload)
    print(json.dumps({"manifest": config.teacher_dataset_manifest, "num_samples": len(manifest_rows)}, indent=2))
    


if __name__ == "__main__":
    main()
