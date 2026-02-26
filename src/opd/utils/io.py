from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    backend: str = "mock"
    teacher_model_id: str = "opd/mock-teacher"
    student_model_id: str = "opd/mock-student"
    latent_dim: int = 32
    text_embed_dim: int = 32
    num_inference_steps: int = 16
    scheduler_step_size: float = 0.1
    guidance_scale: float = 1.0
    dtype: str = "float32"
    revision: str | None = None
    variant: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = False
    use_safetensors: bool = True
    low_cpu_mem_usage: bool = True
    image_height: int = 512
    image_width: int = 512
    max_sequence_length: int = 256
    freeze_text_encoders: bool = True
    freeze_vae: bool = True


@dataclass
class TrainConfig:
    output_root: str = "outputs"
    run_name: str = "opd-run"
    seed: int = 42
    batch_size: int = 2
    grad_accum_steps: int = 1
    max_train_steps: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    mixed_precision: bool = False
    max_grad_norm: float = 1.0
    log_every: int = 1
    save_every: int = 20
    num_workers: int = 0


@dataclass
class DistillConfig:
    num_inference_steps: int = 16
    initial_noise_scale: float = 1.0
    kl_weight: float = 1.0
    epsilon_mse_weight: float = 0.1
    latent_mse_weight: float = 0.1
    warmup_steps: int = 0


@dataclass
class EvalConfig:
    heldout_categories: list[str] = field(default_factory=list)
    num_eval_prompts: int = 16
    compute_clip_metrics: bool = False
    export_num_pairs: int = 24


@dataclass
class AppConfig:
    experiment_name: str
    stage: str
    model: ModelConfig
    train: TrainConfig
    distill: DistillConfig
    eval: EvalConfig
    prompt_file: str = "outputs/prompts/prompts.jsonl"
    teacher_dataset_manifest: str = "outputs/datasets/teacher_manifest.json"
    output_root: str = "outputs"
    seed: int = 42
    config_hash: str = ""
    git_commit: str = "unknown"

    @property
    def run_id(self) -> str:
        now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        return f"{self.experiment_name}-{self.stage}-{now}-{self.config_hash[:8]}"

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / "runs" / self.run_name

    @property
    def run_name(self) -> str:
        return f"{self.train.run_name}-{self.stage}-{self.config_hash[:8]}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["run_id"] = self.run_id
        payload["run_name"] = self.run_name
        payload["run_dir"] = str(self.run_dir)
        return payload


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(raw).__name__}.")
    return raw


def _resolve_path(base: Path, child: str) -> Path:
    candidate = Path(child)
    if candidate.is_absolute():
        return candidate
    return (base / candidate).resolve()


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _git_commit() -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return "unknown"
    return output.strip() or "unknown"


def _hash_payload(payload: dict[str, Any]) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_app_config(config_path: str, stage_override: str | None = None) -> AppConfig:
    config_file = Path(config_path).resolve()
    root = _read_yaml(config_file)
    base_dir = config_file.parent

    stage = stage_override or root.get("stage") or "pilot"
    stage_overrides = root.get("stages", {}).get(stage, {})
    if stage_overrides and not isinstance(stage_overrides, dict):
        raise ValueError(f"Expected mapping for stage override '{stage}'.")
    merged_root = _deep_merge(root, stage_overrides if isinstance(stage_overrides, dict) else {})

    model_cfg = _read_yaml(_resolve_path(base_dir, merged_root["model_config"]))
    train_cfg = _read_yaml(_resolve_path(base_dir, merged_root["train_config"]))
    distill_cfg = _read_yaml(_resolve_path(base_dir, merged_root["distill_config"]))
    eval_cfg = _read_yaml(_resolve_path(base_dir, merged_root["eval_config"]))

    payload_for_hash = {
        "root": merged_root,
        "model": model_cfg,
        "train": train_cfg,
        "distill": distill_cfg,
        "eval": eval_cfg,
    }
    cfg_hash = _hash_payload(payload_for_hash)

    app = AppConfig(
        experiment_name=merged_root.get("experiment_name", "opd"),
        stage=stage,
        model=ModelConfig(**model_cfg),
        train=TrainConfig(**train_cfg),
        distill=DistillConfig(**distill_cfg),
        eval=EvalConfig(**eval_cfg),
        prompt_file=merged_root.get("prompt_file", "outputs/prompts/prompts.jsonl"),
        teacher_dataset_manifest=merged_root.get(
            "teacher_dataset_manifest",
            "outputs/datasets/teacher_manifest.json",
        ),
        output_root=merged_root.get("output_root", train_cfg.get("output_root", "outputs")),
        seed=int(merged_root.get("seed", train_cfg.get("seed", 42))),
        config_hash=cfg_hash,
        git_commit=_git_commit(),
    )
    return app


def ensure_output_layout(config: AppConfig) -> dict[str, Path]:
    root = Path(config.output_root)
    run_dir = root / "runs" / config.run_name
    layout = {
        "root": root,
        "run_dir": run_dir,
        "checkpoints": root / "checkpoints" / config.run_name,
        "metrics": root / "metrics" / config.run_name,
        "images": root / "images" / config.run_name,
        "surveys": root / "surveys" / config.run_name,
        "prompts": root / "prompts",
        "datasets": root / "datasets",
    }
    for path in layout.values():
        path.mkdir(parents=True, exist_ok=True)
    return layout


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object payload at {target}.")
    return data
