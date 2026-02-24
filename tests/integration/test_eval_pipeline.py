from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import yaml


def _write_eval_config(tmp_path: Path, repo_root: Path) -> Path:
    train_cfg = {
        "output_root": str(tmp_path / "outputs"),
        "run_name": "eval-smoke",
        "seed": 777,
        "batch_size": 2,
        "grad_accum_steps": 1,
        "max_train_steps": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "mixed_precision": False,
        "max_grad_norm": 1.0,
        "log_every": 1,
        "save_every": 1,
        "num_workers": 0,
    }
    train_cfg_path = tmp_path / "train.yaml"
    train_cfg_path.write_text(yaml.safe_dump(train_cfg), encoding="utf-8")

    root_cfg = {
        "experiment_name": "opd-eval-test",
        "stage": "pilot",
        "seed": 777,
        "output_root": str(tmp_path / "outputs"),
        "model_config": str(repo_root / "configs/model/pilot.yaml"),
        "train_config": str(train_cfg_path),
        "distill_config": str(repo_root / "configs/distill/kl_per_timestep.yaml"),
        "eval_config": str(repo_root / "configs/eval/qualtrics.yaml"),
        "prompt_file": str(tmp_path / "outputs/prompts/prompts.jsonl"),
        "teacher_dataset_manifest": str(tmp_path / "outputs/datasets/teacher_manifest.json"),
    }
    root_cfg_path = tmp_path / "run.yaml"
    root_cfg_path.write_text(yaml.safe_dump(root_cfg), encoding="utf-8")
    return root_cfg_path


def test_eval_and_qualtrics_export(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = _write_eval_config(tmp_path, repo_root)
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
            str(repo_root / "scripts/run_eval.py"),
            "--config",
            str(config_path),
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/export_qualtrics.py"),
            "--config",
            str(config_path),
        ],
        cwd=repo_root,
        env=env,
        check=True,
    )

    eval_report = list((tmp_path / "outputs/metrics").rglob("eval_report.json"))
    assert eval_report, "Expected eval report."

    surveys = list((tmp_path / "outputs/surveys").rglob("survey.csv"))
    assert surveys, "Expected survey CSV."
    with surveys[0].open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert rows
    assert set(rows[0]).issuperset(
        {"question_id", "prompt_id", "prompt_text", "image_a_path", "image_b_path", "randomized_label"}
    )
