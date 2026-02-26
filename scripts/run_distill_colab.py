#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast


# ---------------------------------------------------------------------------
# Hard-coded config (mirrors configs/sd35m_sd3m_run.yaml + referenced configs)
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

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
MAX_TRAIN_STEPS = 40
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MIXED_PRECISION = True
MAX_GRAD_NORM = 1.0
LOG_EVERY = 1
SAVE_EVERY = 10

# Placeholder data paths (replace these in Colab before running).
PROMPTS_JSONL = "/content/path/to/prompts.jsonl"
TEACHER_ROLLOUT_DIR = "/content/path/to/teacher_rollouts"
# Expected file format for each prompt_id:
#   f"{TEACHER_ROLLOUT_DIR}/{prompt_id}.pt"
# containing: {"final_latent": <torch.Tensor>}

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
        f"huggingface-cli download {STUDENT_MODEL_ID} --local-dir /content/hf_models/student",
    ]


def print_colab_setup_commands() -> None:
    print("# Run these in Colab before training:")
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
        student_dir = COLAB_MODEL_CACHE / "student"
        _run_shell(f"huggingface-cli download {TEACHER_MODEL_ID} --local-dir \"{teacher_dir}\"")
        _run_shell(f"huggingface-cli download {STUDENT_MODEL_ID} --local-dir \"{student_dir}\"")


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


def iter_prompt_batches(dataset: PromptDataset, batch_size: int, seed: int) -> list[list[PromptRecord]]:
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    batches: list[list[PromptRecord]] = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batches.append([dataset.records[idx] for idx in batch_indices])
    return batches


class CsvMetricWriter:
    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        if not self.path.exists():
            with self.path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()

    def write(self, payload: dict[str, float]) -> None:
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow({name: payload.get(name) for name in self.fieldnames})


class JsonlMetricWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict[str, float]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


class StudentDiffusionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        try:
            from diffusers import StableDiffusion3Pipeline
        except Exception as exc:
            raise RuntimeError("Please install diffusers/transformers/accelerate in Colab.") from exc

        torch_dtype = _resolve_dtype(DTYPE_NAME)
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            STUDENT_MODEL_ID,
            torch_dtype=torch_dtype,
            revision=REVISION,
            variant=VARIANT,
            cache_dir=CACHE_DIR,
            local_files_only=LOCAL_FILES_ONLY,
            use_safetensors=USE_SAFETENSORS,
            low_cpu_mem_usage=LOW_CPU_MEM_USAGE,
        )
        if FREEZE_TEXT_ENCODERS:
            for encoder_name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
                encoder = getattr(self.pipeline, encoder_name, None)
                if encoder is None:
                    continue
                for parameter in encoder.parameters():
                    parameter.requires_grad = False
        if FREEZE_VAE and getattr(self.pipeline, "vae", None) is not None:
            for parameter in self.pipeline.vae.parameters():
                parameter.requires_grad = False
        for parameter in self.pipeline.transformer.parameters():
            parameter.requires_grad = True

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
        latents = self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=self.pipeline.transformer.config.in_channels,
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
            dtype=_resolve_dtype(DTYPE_NAME),
            device=device,
            generator=generator,
            latents=None,
        )
        return latents

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


def load_teacher_final_latent(prompt_id: str, rollout_dir: Path, device: torch.device) -> torch.Tensor:
    latent_path = rollout_dir / f"{prompt_id}.pt"
    if not latent_path.exists():
        raise FileNotFoundError(
            f"Missing teacher rollout file for prompt_id={prompt_id}: {latent_path}"
        )
    payload = torch.load(latent_path, map_location="cpu")
    if "final_latent" not in payload:
        raise KeyError(f"Expected key 'final_latent' in {latent_path}")
    return payload["final_latent"].to(device=device)


def save_checkpoint(path: Path, model: StudentDiffusionModel, optimizer: torch.optim.Optimizer, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "student_transformer": model.pipeline.transformer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        path,
    )


def main() -> None:
    if PRINT_COLAB_SETUP:
        print_colab_setup_commands()
        return

    maybe_bootstrap_colab(
        install_deps=INSTALL_DEPS,
        prefetch_models=PREFETCH_MODELS,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for run_distill_colab.py. Please use a GPU Colab runtime.")

    prompts_path = Path(PROMPTS_JSONL)
    rollout_dir = Path(TEACHER_ROLLOUT_DIR)
    if not prompts_path.exists():
        raise FileNotFoundError(
            f"PROMPTS_JSONL does not exist: {prompts_path}. Please edit the placeholder path."
        )
    if not rollout_dir.exists():
        raise FileNotFoundError(
            f"TEACHER_ROLLOUT_DIR does not exist: {rollout_dir}. Please edit the placeholder path."
        )

    set_seed(SEED)
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"Teacher model (reference): {TEACHER_MODEL_ID}")
    print(f"Student model: {STUDENT_MODEL_ID}")

    run_dir = OUTPUT_ROOT / "runs" / RUN_NAME
    ckpt_dir = OUTPUT_ROOT / "checkpoints" / RUN_NAME
    metrics_dir = OUTPUT_ROOT / "metrics" / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    dataset = PromptDataset.from_jsonl(prompts_path)
    if len(dataset) == 0:
        raise RuntimeError(f"No prompts found in: {prompts_path}")

    model = StudentDiffusionModel().to(device)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = GradScaler("cuda", enabled=MIXED_PRECISION)
    metrics_jsonl = JsonlMetricWriter(metrics_dir / "metrics.jsonl")
    metrics_csv = CsvMetricWriter(
        metrics_dir / "metrics.csv",
        fieldnames=["step", "total_loss", "latent_mse_loss", "lr", "grad_norm"],
    )

    batches = iter_prompt_batches(dataset=dataset, batch_size=BATCH_SIZE, seed=SEED)
    if not batches:
        raise RuntimeError("No batches were created from prompts.")

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    while global_step < MAX_TRAIN_STEPS:
        batch = batches[global_step % len(batches)]
        prompts = [row.prompt for row in batch]
        if len(batch) != 1:
            raise RuntimeError("This Colab script currently assumes BATCH_SIZE=1 for seeded rollout alignment.")
        batch_record = batch[0]

        teacher_final = load_teacher_final_latent(
            prompt_id=batch_record.prompt_id,
            rollout_dir=rollout_dir,
            device=device,
        )

        with autocast("cuda", enabled=MIXED_PRECISION):
            latents = model.prepare_initial_latents(
                batch_size=1,
                seed=batch_record.seed,
                device=device,
            )
            embeddings = model.encode_prompts(prompts, device=device)
            for timestep in model.timesteps(device=device):
                epsilon = model.predict_epsilon(
                    latents=latents,
                    timestep=timestep,
                    prompt_embeds=embeddings["prompt_embeds"],
                    pooled_prompt_embeds=embeddings["pooled_prompt_embeds"],
                )
                latents = model.step_latents(latents=latents, epsilon=epsilon, timestep=timestep)

            latent_mse = F.mse_loss(latents.float(), teacher_final.float())
            loss = latent_mse / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        grad_norm = 0.0
        will_step = (global_step + 1) % GRAD_ACCUM_STEPS == 0
        if will_step:
            scaler.unscale_(optimizer)
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad],
                MAX_GRAD_NORM,
            )
            grad_norm = float(grad_norm_tensor.detach().cpu().item())
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        metrics = {
            "step": float(global_step),
            "total_loss": float((latent_mse).detach().cpu().item()),
            "latent_mse_loss": float((latent_mse).detach().cpu().item()),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_norm": grad_norm,
        }

        if global_step % LOG_EVERY == 0:
            print(json.dumps(metrics, sort_keys=True))
            metrics_jsonl.write(metrics)
            metrics_csv.write(metrics)

        if global_step > 0 and global_step % SAVE_EVERY == 0:
            save_checkpoint(ckpt_dir / f"step_{global_step:07d}.pt", model, optimizer, global_step)

        global_step += 1

    final_ckpt = ckpt_dir / "final.pt"
    save_checkpoint(final_ckpt, model, optimizer, global_step)
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "stage": STAGE,
        "run_name": RUN_NAME,
        "num_prompts": len(dataset),
        "max_train_steps": MAX_TRAIN_STEPS,
        "checkpoint": str(final_ckpt),
        "metrics_jsonl": str(metrics_dir / "metrics.jsonl"),
    }
    with (run_dir / "train_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
