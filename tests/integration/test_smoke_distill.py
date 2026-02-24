from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import os
import yaml


def _write_smoke_config(tmp_path: Path, repo_root: Path) -> Path:
    train_cfg = {
        "output_root": str(tmp_path / "outputs"),
        "run_name": "smoke",
        "seed": 123,
        "batch_size": 2,
        "grad_accum_steps": 1,
        "max_train_steps": 3,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "mixed_precision": False,
        "max_grad_norm": 1.0,
        "log_every": 1,
        "save_every": 2,
        "num_workers": 0,
    }
    train_cfg_path = tmp_path / "train.yaml"
    train_cfg_path.write_text(yaml.safe_dump(train_cfg), encoding="utf-8")

    root_cfg = {
        "experiment_name": "opd-test",
        "stage": "pilot",
        "seed": 123,
        "output_root": str(tmp_path / "outputs"),
        "model_config": str(repo_root / "configs/model/pilot.yaml"),
        "train_config": str(train_cfg_path),
        "distill_config": str(repo_root / "configs/distill/kl_per_timestep.yaml"),
        "eval_config": str(repo_root / "configs/eval/quant.yaml"),
        "prompt_file": str(tmp_path / "outputs/prompts/prompts.jsonl"),
        "teacher_dataset_manifest": str(tmp_path / "outputs/datasets/teacher_manifest.json"),
    }
    root_cfg_path = tmp_path / "run.yaml"
    root_cfg_path.write_text(yaml.safe_dump(root_cfg), encoding="utf-8")
    return root_cfg_path


def test_smoke_distill_end_to_end(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = _write_smoke_config(tmp_path, repo_root)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/generate_prompts.py"),
            "--config",
            str(config_path),
            "--num-per-category",
            "1",
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/run_distill.py"),
            "--config",
            str(config_path),
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )
    checkpoints = sorted((tmp_path / "outputs/checkpoints").rglob("final.pt"))
    assert checkpoints, "Expected final checkpoint to be produced."
