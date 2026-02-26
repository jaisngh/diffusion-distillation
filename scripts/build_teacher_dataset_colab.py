#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Hard-coded config (keep shared values aligned with run_distill_colab.py)
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "opd-cs234"
STAGE = "final"
SEED = 42
OUTPUT_ROOT = Path("outputs_colab")
RUN_NAME = "opd-distill-sd35m-sd3m-colab"

TEACHER_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"
STUDENT_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
DTYPE_NAME = "float16"
NUM_INFERENCE_STEPS = 12
SCHEDULER_STEP_SIZE = 0.1
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
MAX_SEQUENCE_LENGTH = 256
FREEZE_TEXT_ENCODERS = True
FREEZE_VAE = True
LOCAL_FILES_ONLY = False
LOW_CPU_MEM_USAGE = True
USE_SAFETENSORS = True
REVISION: str | None = None
VARIANT: str | None = None
CACHE_DIR: str | None = None

# Prompt + rollout paths.
# Set these paths in Colab before running.
PROMPTS_DIR = "/content/path/to/prompts"
PROMPTS_FILENAME = "prompts.jsonl"
PROMPTS_JSONL = str(Path(PROMPTS_DIR) / PROMPTS_FILENAME)
TEACHER_ROLLOUT_DIR = "/content/path/to/teacher_rollouts"
TEACHER_MANIFEST_PATH = "/content/path/to/teacher_manifest.json"

# Optional local cache for explicit CLI prefetch downloads.
COLAB_MODEL_CACHE = Path("/content/hf_models")

# Optional one-click bootstrap toggles (kept hard-coded instead of CLI args).
PRINT_COLAB_SETUP = False
INSTALL_DEPS = False
PREFETCH_MODELS = False


def _colab_setup_commands() -> list[str]:
    return [
        "python -m pip install -U diffusers transformers accelerate safetensors sentencepiece huggingface_hub",
        "huggingface-cli login",
        f"huggingface-cli download {TEACHER_MODEL_ID} --local-dir /content/hf_models/teacher",
    ]


def print_colab_setup_commands() -> None:
    print("# Run these in Colab before building teacher rollouts:")
    for command in _colab_setup_commands():
        print(command)


def _run_shell(command: str) -> None:
    print(f"[bootstrap] {command}")
    subprocess.run(command, shell=True, check=True)


def maybe_bootstrap_colab(install_deps: bool, prefetch_models: bool) -> None:
    if install_deps:
        _run_shell(
            f"{sys.executable} -m pip install -U diffusers transformers accelerate "
            "safetensors sentencepiece huggingface_hub"
        )

    if prefetch_models:
        COLAB_MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        teacher_dir = COLAB_MODEL_CACHE / "teacher"
        _run_shell(f"huggingface-cli download {TEACHER_MODEL_ID} --local-dir \"{teacher_dir}\"")


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    key = dtype_name.lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_name}'.")
    return mapping[key]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class PromptRecord:
    prompt_id: str
    prompt: str
    category: str
    seed: int


class PromptDataset:
    def __init__(self, records: list[PromptRecord]) -> None:
        self.records = records

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "PromptDataset":
        target = Path(path)
        rows: list[PromptRecord] = []
        with target.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows.append(
                    PromptRecord(
                        prompt_id=row["prompt_id"],
                        prompt=row["prompt"],
                        category=row.get("category", "unknown"),
                        seed=int(row.get("seed", SEED)),
                    )
                )
        return cls(rows)

    def __len__(self) -> int:
        return len(self.records)


class TeacherDiffusionModel:
    def __init__(self) -> None:
        try:
            from diffusers import StableDiffusion3Pipeline
        except Exception as exc:
            raise RuntimeError("Please install diffusers/transformers/accelerate in Colab.") from exc

        torch_dtype = _resolve_dtype(DTYPE_NAME)
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            TEACHER_MODEL_ID,
            torch_dtype=torch_dtype,
            revision=REVISION,
            variant=VARIANT,
            cache_dir=CACHE_DIR,
            local_files_only=LOCAL_FILES_ONLY,
            use_safetensors=USE_SAFETENSORS,
            low_cpu_mem_usage=LOW_CPU_MEM_USAGE,
        )
        self.pipeline.scheduler.config.shift = SCHEDULER_STEP_SIZE

    def to(self, device: torch.device) -> "TeacherDiffusionModel":
        self.pipeline = self.pipeline.to(device)
        return self

    def encode_prompts(self, prompts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
                prompt=prompts,
                prompt_2=prompts,
                prompt_3=prompts,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
            )
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }

    def prepare_initial_latents(self, batch_size: int, seed: int, device: torch.device) -> torch.Tensor:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=self.pipeline.transformer.config.in_channels,
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
            dtype=_resolve_dtype(DTYPE_NAME),
            device=device,
            generator=generator,
            latents=None,
        )

    def timesteps(self, device: torch.device) -> list[Any]:
        self.pipeline.scheduler.set_timesteps(NUM_INFERENCE_STEPS, device=device)
        return list(self.pipeline.scheduler.timesteps)

    def predict_epsilon(
        self,
        latents: torch.Tensor,
        timestep: Any,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        model_dtype = next(self.pipeline.transformer.parameters()).dtype
        latents_input = latents.to(dtype=model_dtype)
        prompt_embeds = prompt_embeds.to(device=latents.device, dtype=model_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=latents.device, dtype=model_dtype)
        if isinstance(timestep, torch.Tensor):
            timestep_tensor = timestep.to(device=latents.device, dtype=model_dtype)
        else:
            timestep_tensor = torch.tensor(timestep, device=latents.device, dtype=model_dtype)
        timestep_tensor = timestep_tensor.expand(latents_input.shape[0])
        epsilon = self.pipeline.transformer(
            hidden_states=latents_input,
            timestep=timestep_tensor,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
        return epsilon.to(dtype=latents.dtype)

    def step_latents(self, latents: torch.Tensor, epsilon: torch.Tensor, timestep: Any) -> torch.Tensor:
        return self.pipeline.scheduler.step(
            model_output=epsilon.to(latents.dtype),
            timestep=timestep,
            sample=latents,
            return_dict=False,
        )[0]


def main() -> None:
    if PRINT_COLAB_SETUP:
        print_colab_setup_commands()
        return

    maybe_bootstrap_colab(
        install_deps=INSTALL_DEPS,
        prefetch_models=PREFETCH_MODELS,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for build_teacher_dataset_colab.py. Please use a GPU Colab runtime."
        )

    prompts_path = Path(PROMPTS_JSONL)
    rollout_dir = Path(TEACHER_ROLLOUT_DIR)
    manifest_path = Path(TEACHER_MANIFEST_PATH)
    if not prompts_path.exists():
        raise FileNotFoundError(
            f"PROMPTS_JSONL does not exist: {prompts_path}. Please edit the placeholder path."
        )
    rollout_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"Teacher model: {TEACHER_MODEL_ID}")

    dataset = PromptDataset.from_jsonl(prompts_path)
    if len(dataset) == 0:
        raise RuntimeError(f"No prompts found in: {prompts_path}")

    teacher = TeacherDiffusionModel().to(device)
    manifest_rows: list[dict[str, object]] = []

    for idx, record in enumerate(dataset.records, start=1):
        print(f"[{idx}/{len(dataset)}] starting rollout for prompt_id={record.prompt_id}")
        with torch.no_grad():
            latents = teacher.prepare_initial_latents(batch_size=1, seed=record.seed, device=device)
            embeddings = teacher.encode_prompts([record.prompt], device=device)
            for timestep in teacher.timesteps(device=device):
                epsilon = teacher.predict_epsilon(
                    latents=latents,
                    timestep=timestep,
                    prompt_embeds=embeddings["prompt_embeds"],
                    pooled_prompt_embeds=embeddings["pooled_prompt_embeds"],
                )
                latents = teacher.step_latents(latents=latents, epsilon=epsilon, timestep=timestep)

        latent_path = rollout_dir / f"{record.prompt_id}.pt"
        torch.save({"final_latent": latents.detach().cpu()}, latent_path)
        manifest_rows.append(
            {
                "prompt_id": record.prompt_id,
                "category": record.category,
                "seed": record.seed,
                "latent_path": str(latent_path),
                "image_path": str((OUTPUT_ROOT / "images" / "teacher" / f"{record.prompt_id}.png").resolve()),
            }
        )
        print(f"[{idx}/{len(dataset)}] finished rollout for prompt_id={record.prompt_id}")

    manifest_payload = {
        "teacher_model_id": TEACHER_MODEL_ID,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "scheduler_step_size": SCHEDULER_STEP_SIZE,
        "num_samples": len(manifest_rows),
        "prompt_file": str(prompts_path),
        "samples": manifest_rows,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2, sort_keys=True)

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "stage": STAGE,
        "run_name": RUN_NAME,
        "teacher_model_id": TEACHER_MODEL_ID,
        "prompt_file": str(prompts_path),
        "rollout_dir": str(rollout_dir),
        "manifest": str(manifest_path),
        "num_samples": len(manifest_rows),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
