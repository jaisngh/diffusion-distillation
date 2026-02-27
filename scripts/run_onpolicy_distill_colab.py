#!/usr/bin/env python3
from __future__ import annotations

"""Self-contained Colab script for on-policy SD3 distillation.

High-level flow:
1) Optional Colab bootstrap (install deps, optional HF prefetch).
2) Prompt generation (or load an existing prompt JSONL).
3) On-policy teacher-student rollout training with KL/MSE losses.
"""

import csv
import hashlib
import json
import random
import re
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
# Hard-coded config (equivalent to configs/sd35m_sd3m_run.yaml on CUDA)
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "opd-cs234"
STAGE = "final"
SEED = 42
OUTPUT_ROOT = Path("outputs_colab")
RUN_NAME = "opd-distill-sd35m-sd3m-onpolicy-colab"

# Model setup (configs/model/sd35m_sd3m.yaml)
TEACHER_MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"  # Teacher checkpoint from HF Hub.
STUDENT_MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"  # Student checkpoint from HF Hub.
DTYPE_NAME = "bfloat16"  # Runtime dtype for model weights.
TRAIN_NUM_INFERENCE_STEPS = 12  # Number of denoising steps during training rollouts.
EVAL_NUM_INFERENCE_STEPS = 24  # Number of denoising steps when generating eval images.
SCHEDULER_STEP_SIZE = 0.1  # Kept for config parity; scheduler uses its own internal stepping.
IMAGE_HEIGHT = 512  # Latent/image height used when sampling.
IMAGE_WIDTH = 512  # Latent/image width used when sampling.
MAX_SEQUENCE_LENGTH = 256  # Max token length for SD3 prompt encoding.
FREEZE_TEXT_ENCODERS = True  # Freeze text encoders during distillation.
FREEZE_VAE = True  # Freeze VAE during distillation.
LOCAL_FILES_ONLY = False  # Set True only when models are already cached locally.
LOW_CPU_MEM_USAGE = True  # Use diffusers low-memory loading path.
USE_SAFETENSORS = True  # Prefer safetensors over pickle weights.
REVISION: str | None = None  # Optional HF revision/tag override.
VARIANT: str | None = None  # Optional variant (e.g., fp16).
CACHE_DIR: str | None = None  # Optional custom cache directory.

# Train setup (configs/train/sd35m_sd3m_single_gpu.yaml)
BATCH_SIZE = 1  # Prompts per optimization micro-step.
GRAD_ACCUM_STEPS = 8  # Micro-steps to accumulate before optimizer.step().
MAX_TRAIN_STEPS = 40  # Total micro-steps to run.
LEARNING_RATE = 1e-5  # AdamW learning rate (lowered for SD3 stability).
WEIGHT_DECAY = 1e-4  # AdamW weight decay.
MIXED_PRECISION = True  # Use autocast for bf16/fp16 kernels; often more stable than plain low-precision.
LOG_EVERY = 1  # Metric logging interval (steps).
SAVE_EVERY = 10  # Checkpoint save interval (steps).
EVAL_OUTPUT_DIR = OUTPUT_ROOT / "eval_images" / RUN_NAME  # Directory for post-train qualitative samples.
EVAL_PROMPTS = [
    "A cozy coffee shop interior at golden hour, cinematic photo style",
    "A futuristic electric sports car driving through a neon city at night",
    "A vibrant fruit tart on a marble table, studio food photography",
    "A hiker standing on a snowy mountain ridge under dramatic clouds",
    "A minimal product ad for wireless earbuds with clean typography backdrop",
]  # Held-out prompts for qualitative teacher/student comparisons.

# Distill setup (configs/distill/sd35_kl.yaml)
INITIAL_NOISE_SCALE = 1.0  # Scale applied to initial sampled latents.
KL_WEIGHT = 1.0  # Base weight for Gaussian KL term.
EPSILON_MSE_WEIGHT = 0.1  # Weight for epsilon prediction MSE.
LATENT_MSE_WEIGHT = 0.05  # Weight for final-step latent mean MSE.
WARMUP_STEPS = 20  # Linear warmup steps for KL_WEIGHT.

# Prompt generation setup (self-contained replacement for generate_prompts.py)
PROMPTS_JSONL = "/content/path/to/prompts.jsonl"  # Prompt file path (generated or pre-existing).
GENERATE_PROMPTS = True  # True: generate prompts into PROMPTS_JSONL before training.
NUM_PROMPTS_PER_CATEGORY = 8  # Prompts sampled for each category.
PROMPT_CATEGORIES = ["fashion", "food", "travel", "tech", "fitness"]  # Categories to generate.

CATEGORY_TEMPLATES: dict[str, list[str]] = {
    "fashion": [
        "A premium {adjective} fashion ad for {audience}, featuring {style} styling and studio lighting",
        "A social media banner advertising {adjective} apparel for {audience} with {style} aesthetics",
    ],
    "food": [
        "A mouthwatering ad for {adjective} {product} aimed at {audience}, cinematic close-up",
        "A digital campaign visual promoting {product} with {style} composition and bold typography",
    ],
    "travel": [
        "An aspirational ad for {adjective} travel packages to {destination}, {style} visual style",
        "A conversion-focused ad creative for {destination} vacations targeting {audience}",
    ],
    "tech": [
        "A modern ad for a {adjective} {product} targeting {audience}, rendered in {style} style",
        "A launch campaign visual for {product} with {style} gradients and product hero shot",
    ],
    "fitness": [
        "A high-energy ad promoting {adjective} fitness programs for {audience}, {style} visual language",
        "A performance marketing image for {product} with dynamic motion and {style} palette",
    ],
}
ADJECTIVES = ["minimal", "bold", "luxury", "vibrant", "futuristic", "clean", "playful"]
AUDIENCES = ["young professionals", "college students", "new parents", "small business owners"]
PRODUCTS = ["smartwatch", "meal kit", "running shoes", "wireless earbuds", "protein shake"]
STYLES = ["high-contrast", "editorial", "retro-modern", "photoreal", "flat-design"]
DESTINATIONS = ["Kyoto", "Barcelona", "Iceland", "Bali", "New York City"]

# ---------------------------------------------------------------------------
# Colab setup options (optional one-time bootstrap)
# ---------------------------------------------------------------------------
PRINT_COLAB_SETUP = False  # True: print setup shell commands.
INSTALL_DEPS = False  # True: pip install required Python packages.
PREFETCH_MODELS = False  # True: download teacher/student snapshots to local cache.
COLAB_MODEL_CACHE = Path("/content/hf_models")  # Cache dir used by PREFETCH_MODELS.


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    """Map a string dtype name to a torch dtype object."""
    mapping: dict[str, torch.dtype] = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_name}'.")
    return mapping[key]


def set_seed(seed: int) -> None:
    """Seed Python and Torch RNGs for reproducible runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _colab_setup_commands() -> list[str]:
    """Return setup commands to run in a fresh Colab runtime."""
    return [
        "python -m pip install -U diffusers transformers accelerate safetensors sentencepiece huggingface_hub",
        "huggingface-cli login",
        f"huggingface-cli download {TEACHER_MODEL_ID} --local-dir /content/hf_models/teacher",
        f"huggingface-cli download {STUDENT_MODEL_ID} --local-dir /content/hf_models/student",
    ]


def print_colab_setup_commands() -> None:
    """Print setup commands without executing them."""
    print("# Run these in Colab before training:")
    for command in _colab_setup_commands():
        print(command)


def _run_shell(command: str) -> None:
    """Execute a shell command and fail on non-zero exit."""
    print(f"[bootstrap] {command}")
    subprocess.run(command, shell=True, check=True)


def maybe_bootstrap_colab(install_deps: bool, prefetch_models: bool) -> None:
    """Optionally install dependencies and prefetch model weights."""
    if install_deps:
        _run_shell(
            f"{sys.executable} -m pip install -U diffusers transformers accelerate "
            "safetensors sentencepiece huggingface_hub"
        )
    if prefetch_models:
        COLAB_MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        _run_shell(
            f"huggingface-cli download {TEACHER_MODEL_ID} "
            f"--local-dir \"{COLAB_MODEL_CACHE / 'teacher'}\""
        )
        _run_shell(
            f"huggingface-cli download {STUDENT_MODEL_ID} "
            f"--local-dir \"{COLAB_MODEL_CACHE / 'student'}\""
        )


def dep_install() -> None:
    """Optional Colab bootstrap helper.

    Call this once in a fresh Colab runtime, then comment it out to avoid
    re-installing packages or re-downloading model snapshots.
    """
    if PRINT_COLAB_SETUP:
        print_colab_setup_commands()
    maybe_bootstrap_colab(install_deps=INSTALL_DEPS, prefetch_models=PREFETCH_MODELS)


@dataclass
class PromptRecord:
    """One prompt row used by training."""
    prompt_id: str
    prompt: str
    category: str
    seed: int


class PromptDataset:
    """Container for prompt records loaded from JSONL."""

    def __init__(self, records: list[PromptRecord]) -> None:
        """Initialize dataset with already-parsed records."""
        self.records = records

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "PromptDataset":
        """Load prompt records from a JSONL file."""
        target = Path(path)
        records: list[PromptRecord] = []
        # Each line is a JSON object with prompt_id/prompt/category/seed.
        with target.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                records.append(
                    PromptRecord(
                        prompt_id=row["prompt_id"],
                        prompt=row["prompt"],
                        category=row.get("category", "unknown"),
                        seed=int(row.get("seed", SEED)),
                    )
                )
        return cls(records)

    def __len__(self) -> int:
        """Return number of prompt records."""
        return len(self.records)


def _prompt_id(category: str, prompt: str, seed: int) -> str:
    """Create a stable short id for a prompt/category/seed triple."""
    digest = hashlib.sha1(f"{category}|{prompt}|{seed}".encode("utf-8")).hexdigest()
    return digest[:16]


def generate_prompt_records(
    num_per_category: int,
    seed: int,
    categories: list[str],
) -> list[dict[str, object]]:
    """Generate synthetic prompt rows from category templates."""
    rng = random.Random(seed)
    records: list[dict[str, object]] = []
    for category in categories:
        templates = CATEGORY_TEMPLATES.get(category)
        if not templates:
            raise ValueError(f"Unknown category '{category}'.")
        for index in range(num_per_category):
            template = rng.choice(templates)
            prompt = template.format(
                adjective=rng.choice(ADJECTIVES),
                audience=rng.choice(AUDIENCES),
                product=rng.choice(PRODUCTS),
                style=rng.choice(STYLES),
                destination=rng.choice(DESTINATIONS),
            )
            prompt_seed = seed + index
            records.append(
                {
                    "prompt_id": _prompt_id(category, prompt, prompt_seed),
                    "prompt": prompt,
                    "category": category,
                    "seed": prompt_seed,
                }
            )
    return records


def save_prompts_jsonl(path: str | Path, records: list[dict[str, object]]) -> None:
    """Write generated prompt rows to JSONL."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def maybe_generate_prompts(path: Path) -> None:
    """Generate prompts if enabled; otherwise keep existing prompt file unchanged."""
    if not GENERATE_PROMPTS:
        return
    records = generate_prompt_records(
        num_per_category=NUM_PROMPTS_PER_CATEGORY,
        seed=SEED,
        categories=PROMPT_CATEGORIES,
    )
    save_prompts_jsonl(path, records)
    print(
        json.dumps(
            {
                "generated_prompt_file": str(path),
                "num_records": len(records),
                "categories": PROMPT_CATEGORIES,
            },
            indent=2,
            sort_keys=True,
        )
    )


def iter_prompt_batches(dataset: PromptDataset, batch_size: int, seed: int) -> list[list[PromptRecord]]:
    """Shuffle records deterministically and split into fixed-size batches."""
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    batches: list[list[PromptRecord]] = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batches.append([dataset.records[idx] for idx in batch_indices])
    return batches


class CsvMetricWriter:
    """Simple append-only CSV metric logger."""

    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        """Create metric file and header if missing."""
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        # Create the file with header exactly once.
        if not self.path.exists():
            with self.path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
                writer.writeheader()

    def write(self, payload: dict[str, float]) -> None:
        """Append one metrics row to CSV."""
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow({name: payload.get(name) for name in self.fieldnames})


class JsonlMetricWriter:
    """Simple append-only JSONL metric logger."""

    def __init__(self, path: Path) -> None:
        """Ensure output directory exists for JSONL metrics."""
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict[str, float]) -> None:
        """Append one JSON metrics object per line."""
        # JSONL makes it easy to stream metrics during long runs.
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


class DiffusionModelWrapper(nn.Module):
    """Thin wrapper around SD3 pipeline used for teacher/student roles."""

    def __init__(self, model_id: str, role: str) -> None:
        """Load pipeline and set trainable parameters by role."""
        super().__init__()
        self.role = role
        try:
            from diffusers import StableDiffusion3Pipeline
        except Exception as exc:
            raise RuntimeError("Please install diffusers/transformers/accelerate in Colab.") from exc

        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=_resolve_dtype(DTYPE_NAME),
            revision=REVISION,
            variant=VARIANT,
            cache_dir=CACHE_DIR,
            local_files_only=LOCAL_FILES_ONLY,
            use_safetensors=USE_SAFETENSORS,
            low_cpu_mem_usage=LOW_CPU_MEM_USAGE,
        )
        # Register the trainable UNet/transformer on this module so
        # .parameters(), .train(), and .eval() recurse as expected.
        self.transformer = self.pipeline.transformer

        if FREEZE_TEXT_ENCODERS:
            # Text encoders are used for conditioning but not updated during distillation.
            for encoder_name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
                encoder = getattr(self.pipeline, encoder_name, None)
                if encoder is None:
                    continue
                for param in encoder.parameters():
                    param.requires_grad = False
        if FREEZE_VAE and getattr(self.pipeline, "vae", None) is not None:
            # VAE is not part of this optimization objective.
            for param in self.pipeline.vae.parameters():
                param.requires_grad = False

        # Only student transformer should be trainable.
        train_transformer = role == "student"
        for param in self.transformer.parameters():
            param.requires_grad = train_transformer

    def encode_prompts(self, prompts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        """Encode string prompts into SD3 conditioning embeddings."""
        for encoder_name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
            encoder = getattr(self.pipeline, encoder_name, None)
            if encoder is not None:
                encoder.to(device=device, dtype=torch.float32)
        # Prompt encoding is inference-only for both teacher and student.
        with torch.no_grad(), autocast("cuda", enabled=False):
            prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
                prompt=prompts,
                prompt_2=prompts,
                prompt_3=prompts,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
            )
        prompt_embeds = prompt_embeds.to(dtype=torch.float32)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=torch.float32)
        return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}

    def prepare_initial_latents(
        self,
        batch_size: int,
        seed: int | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample initial latent noise for rollout starts."""
        generator = None
        if seed is not None:
            # Deterministic seed keeps training reproducible between runs.
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
        return self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=self.transformer.config.in_channels,
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
            # Keep scheduler state in fp32 to reduce NaNs during iterative updates.
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=None,
        ) * INITIAL_NOISE_SCALE

    def get_timesteps(self, num_inference_steps: int, device: torch.device) -> list[Any]:
        """Prepare scheduler timesteps for the current rollout horizon."""
        self.pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        return list(self.pipeline.scheduler.timesteps)

    def predict(
        self,
        latents: torch.Tensor,
        timestep: Any,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict mean/logvar/epsilon at one denoising step."""
        # Cast inputs to transformer dtype for stable mixed-precision execution.
        model_dtype = next(self.transformer.parameters()).dtype
        latent_input = self.pipeline.scheduler.scale_model_input(latents, timestep).to(dtype=model_dtype)
        prompt_embeds = prompt_embeds.to(device=latents.device, dtype=model_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=latents.device, dtype=model_dtype)
        if isinstance(timestep, torch.Tensor):
            # Timestep conditioning is numerically safer in fp32.
            timestep_tensor = timestep.to(device=latents.device, dtype=torch.float32)
        else:
            timestep_tensor = torch.tensor(timestep, device=latents.device, dtype=torch.float32)
        timestep_tensor = timestep_tensor.expand(latent_input.shape[0])

        epsilon = self.transformer(
            hidden_states=latent_input,
            timestep=timestep_tensor,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
        epsilon = epsilon.to(dtype=latents.dtype)
        # Keep variance fixed to zero in this simplified Gaussian parameterization.
        logvar = torch.zeros_like(epsilon)
        mean = latents - epsilon
        return mean, logvar, epsilon

    def step_latents(self, latents: torch.Tensor, epsilon: torch.Tensor, timestep: Any) -> torch.Tensor:
        """Advance latents by one scheduler step using epsilon prediction."""
        return self.pipeline.scheduler.step(
            model_output=epsilon.to(latents.dtype),
            timestep=timestep,
            sample=latents,
            return_dict=False,
        )[0]


def gaussian_kl(mean_q: torch.Tensor, logvar_q: torch.Tensor, mean_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    """Compute per-dimension KL divergence between two diagonal Gaussians."""
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    return 0.5 * (logvar_p - logvar_q + (var_q + (mean_q - mean_p).pow(2)) / var_p - 1.0)


def save_checkpoint(path: Path, student: DiffusionModelWrapper, optimizer: torch.optim.Optimizer, step: int) -> None:
    """Persist student transformer weights plus optimizer state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "student_transformer": student.transformer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "run_name": RUN_NAME,
        },
        path,
    )


def _config_hash() -> str:
    """Create a short reproducibility hash from key config values."""
    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "stage": STAGE,
        "teacher_model_id": TEACHER_MODEL_ID,
        "student_model_id": STUDENT_MODEL_ID,
        "dtype": DTYPE_NAME,
        "train_num_inference_steps": TRAIN_NUM_INFERENCE_STEPS,
        "eval_num_inference_steps": EVAL_NUM_INFERENCE_STEPS,
        "train_steps": MAX_TRAIN_STEPS,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _slugify(text: str) -> str:
    """Convert prompt text into a short filesystem-safe slug."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    if not cleaned:
        return "prompt"
    return cleaned[:60]


def _save_eval_images(
    *,
    model: DiffusionModelWrapper,
    model_name: str,
    prompts: list[str],
    output_dir: Path,
    seed_base: int,
    num_inference_steps: int,
    device: torch.device,
) -> None:
    """Generate and save one image per prompt for a specific model."""
    model.eval()
    model.pipeline.to(device)
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for prompt_idx, prompt in enumerate(prompts):
            prompt_seed = seed_base + prompt_idx
            generator = torch.Generator(device=device).manual_seed(prompt_seed)
            result = model.pipeline(
                prompt=prompt,
                prompt_2=prompt,
                prompt_3=prompt,
                num_inference_steps=num_inference_steps,
                height=IMAGE_HEIGHT,
                width=IMAGE_WIDTH,
                generator=generator,
            )
            image = result.images[0]
            image_name = f"{prompt_idx:02d}_{_slugify(prompt)}.png"
            image.save(model_dir / image_name)

    manifest = {
        "model": model_name,
        "seed_base": seed_base,
        "num_inference_steps": num_inference_steps,
        "height": IMAGE_HEIGHT,
        "width": IMAGE_WIDTH,
        "prompts": prompts,
    }
    with (model_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def main() -> None:
    """Run full pipeline: optional setup, prompts, then on-policy distillation."""
    # Optional one-time setup for fresh Colab runtimes.
    # Uncomment, run once, then comment back out to avoid re-installs.
    # dep_install()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Use a GPU Colab runtime.")

    # -----------------------------------------------------------------------
    # Phase 1: Prompt generation/loading
    # - Optionally generate a prompt file.
    # - Verify the prompt file exists.
    # - Load prompt records for training.
    # -----------------------------------------------------------------------
    prompts_path = Path(PROMPTS_JSONL)
    maybe_generate_prompts(prompts_path)
    if not prompts_path.exists():
        raise FileNotFoundError(
            f"PROMPTS_JSONL does not exist: {prompts_path}. Update PROMPTS_JSONL at top of this file."
        )

    set_seed(SEED)
    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"Teacher model: {TEACHER_MODEL_ID}")
    print(f"Student model: {STUDENT_MODEL_ID}")

    run_dir = OUTPUT_ROOT / "runs" / RUN_NAME
    ckpt_dir = OUTPUT_ROOT / "checkpoints" / RUN_NAME
    metrics_dir = OUTPUT_ROOT / "metrics" / RUN_NAME
    for path in (run_dir, ckpt_dir, metrics_dir):
        path.mkdir(parents=True, exist_ok=True)

    dataset = PromptDataset.from_jsonl(prompts_path)
    if len(dataset) == 0:
        raise RuntimeError(f"No prompts found in {prompts_path}")

    # -----------------------------------------------------------------------
    # Phase 2: On-policy teacher-student distillation
    # - Initialize teacher (frozen) and student (trainable transformer).
    # - For each step: rollout on student states, supervise with teacher.
    # - Optimize weighted KL + epsilon MSE + latent MSE objective.
    # -----------------------------------------------------------------------
    teacher = DiffusionModelWrapper(TEACHER_MODEL_ID, role="teacher").to(device)
    student = DiffusionModelWrapper(STUDENT_MODEL_ID, role="student").to(device)
    # Move full pipelines so text encoders/scheduler buffers match CUDA tensors.
    teacher.pipeline.to(device)
    student.pipeline.to(device)
    teacher.eval()
    student.train()

    optimizer = torch.optim.AdamW(
        (param for param in student.parameters() if param.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    model_dtype = _resolve_dtype(DTYPE_NAME)
    use_autocast = MIXED_PRECISION and model_dtype in (torch.float16, torch.bfloat16)
    # GradScaler is only supported/needed for fp16 (not bf16).
    use_grad_scaler = MIXED_PRECISION and model_dtype == torch.float16
    scaler = GradScaler("cuda", enabled=use_grad_scaler)

    metrics_jsonl = JsonlMetricWriter(metrics_dir / "metrics.jsonl")
    metrics_csv = CsvMetricWriter(
        metrics_dir / "metrics.csv",
        fieldnames=[
            "step",
            "total_loss",
            "epsilon_mse_loss",
            "latent_mse_loss",
            "lr",
        ],
    )

    batches = iter_prompt_batches(dataset=dataset, batch_size=BATCH_SIZE, seed=SEED)
    if not batches:
        raise RuntimeError("No batches created from prompts.")

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    while global_step < MAX_TRAIN_STEPS:
        # Select batch deterministically and update the per-step noise seed.
        batch = batches[global_step % len(batches)]
        prompts = [row.prompt for row in batch]
        step_seed = SEED + global_step

        with autocast("cuda", dtype=model_dtype, enabled=use_autocast):
            # Student starts from noise and defines the rollout states we supervise on.
            latents = student.prepare_initial_latents(
                batch_size=len(prompts),
                seed=step_seed,
                device=device,
            )
            embeddings = student.encode_prompts(prompts, device=device)
            timesteps = student.get_timesteps(num_inference_steps=TRAIN_NUM_INFERENCE_STEPS, device=device)

            eps_terms: list[torch.Tensor] = []
            final_student_mean: torch.Tensor | None = None
            final_teacher_mean: torch.Tensor | None = None

            for timestep in timesteps:
                # Teacher labels the student's current latent state (on-policy supervision).
                with torch.no_grad():
                    teacher_mean, _, teacher_eps = teacher.predict(
                        latents=latents,
                        timestep=timestep,
                        prompt_embeds=embeddings["prompt_embeds"],
                        pooled_prompt_embeds=embeddings["pooled_prompt_embeds"],
                    )
                student_mean, _, student_eps = student.predict(
                    latents=latents,
                    timestep=timestep,
                    prompt_embeds=embeddings["prompt_embeds"],
                    pooled_prompt_embeds=embeddings["pooled_prompt_embeds"],
                )
                if not (
                    torch.isfinite(teacher_eps).all()
                    and torch.isfinite(student_eps).all()
                    and torch.isfinite(latents).all()
                ):
                    raise RuntimeError(
                        f"Non-finite tensor detected at global_step={global_step}, "
                        f"timestep={timestep}. Try lower LR or disable mixed precision."
                    )

                eps_terms.append(F.mse_loss(student_eps, teacher_eps))
                final_student_mean = student_mean
                final_teacher_mean = teacher_mean
                # Next state comes from the student's own action.
                latents = student.step_latents(latents=latents, epsilon=student_eps, timestep=timestep)

            if final_student_mean is None or final_teacher_mean is None:
                raise RuntimeError("No timesteps were produced by scheduler.")

            # Loss terms are averaged over rollout steps, then combined by weights.
            epsilon_mse_loss = torch.stack(eps_terms).mean()
            latent_mse_loss = F.mse_loss(final_student_mean, final_teacher_mean)

            total = (
                EPSILON_MSE_WEIGHT * epsilon_mse_loss
                + LATENT_MSE_WEIGHT * latent_mse_loss
            )
            if not torch.isfinite(total):
                raise RuntimeError(
                    f"Non-finite total loss at global_step={global_step}. "
                    "Try lower LR or disable mixed precision."
                )
            # Gradient accumulation keeps effective batch larger on limited VRAM.
            loss = total / GRAD_ACCUM_STEPS

        if use_grad_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (global_step + 1) % GRAD_ACCUM_STEPS == 0:
            if use_grad_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        metrics = {
            "step": float(global_step),
            "total_loss": float(total.detach().cpu().item()),
            "epsilon_mse_loss": float(epsilon_mse_loss.detach().cpu().item()),
            "latent_mse_loss": float(latent_mse_loss.detach().cpu().item()),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        if global_step % LOG_EVERY == 0:
            print(json.dumps(metrics, sort_keys=True))
            metrics_jsonl.write(metrics)
            metrics_csv.write(metrics)

        if global_step > 0 and global_step % SAVE_EVERY == 0:
            save_checkpoint(ckpt_dir / f"step_{global_step:07d}.pt", student, optimizer, global_step)

        global_step += 1

    final_ckpt = ckpt_dir / "final.pt"
    save_checkpoint(final_ckpt, student, optimizer, global_step)

    # -----------------------------------------------------------------------
    # Phase 3: Qualitative evaluation images
    # - Save teacher/original-student/distilled-student outputs on held-out prompts.
    # - Use a fixed seed schedule for comparable visual outputs per prompt.
    # -----------------------------------------------------------------------
    eval_dir = Path(EVAL_OUTPUT_DIR)
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_prompts = list(EVAL_PROMPTS)
    if len(eval_prompts) == 0:
        raise RuntimeError("EVAL_PROMPTS is empty. Add at least one prompt for evaluation.")

    _save_eval_images(
        model=teacher,
        model_name="teacher",
        prompts=eval_prompts,
        output_dir=eval_dir,
        seed_base=SEED + 10_000,
        num_inference_steps=EVAL_NUM_INFERENCE_STEPS,
        device=device,
    )
    del teacher
    torch.cuda.empty_cache()

    del student
    torch.cuda.empty_cache()

    original_student = DiffusionModelWrapper(STUDENT_MODEL_ID, role="student").to(device)
    _save_eval_images(
        model=original_student,
        model_name="original_student",
        prompts=eval_prompts,
        output_dir=eval_dir,
        seed_base=SEED + 10_000,
        num_inference_steps=EVAL_NUM_INFERENCE_STEPS,
        device=device,
    )
    del original_student
    torch.cuda.empty_cache()

    distilled_student = DiffusionModelWrapper(STUDENT_MODEL_ID, role="student").to(device)
    distilled_state = torch.load(final_ckpt, map_location="cpu")
    transformer_state = (
        distilled_state["student_transformer"]
        if isinstance(distilled_state, dict) and "student_transformer" in distilled_state
        else distilled_state
    )
    distilled_student.transformer.load_state_dict(transformer_state)
    _save_eval_images(
        model=distilled_student,
        model_name="distilled_student",
        prompts=eval_prompts,
        output_dir=eval_dir,
        seed_base=SEED + 10_000,
        num_inference_steps=EVAL_NUM_INFERENCE_STEPS,
        device=device,
    )
    del distilled_student
    torch.cuda.empty_cache()

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "stage": STAGE,
        "run_name": RUN_NAME,
        "config_hash": _config_hash(),
        "num_prompts": len(dataset),
        "max_train_steps": MAX_TRAIN_STEPS,
        "train_num_inference_steps": TRAIN_NUM_INFERENCE_STEPS,
        "eval_num_inference_steps": EVAL_NUM_INFERENCE_STEPS,
        "checkpoint": str(final_ckpt),
        "metrics_jsonl": str(metrics_dir / "metrics.jsonl"),
        "eval_output_dir": str(eval_dir),
        "eval_prompts": eval_prompts,
    }
    with (run_dir / "train_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
