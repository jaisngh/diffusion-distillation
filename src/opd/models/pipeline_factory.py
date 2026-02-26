from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from opd.utils.io import ModelConfig


@dataclass
class ModelPrediction:
    mean: torch.Tensor
    logvar: torch.Tensor
    epsilon: torch.Tensor


class MockDiffusionBackbone(nn.Module):
    def __init__(self, latent_dim: int, text_embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + text_embed_dim + 1, 128),
            nn.SiLU(),
            nn.Linear(128, latent_dim),
        )
        self.logvar_bias = nn.Parameter(torch.zeros(latent_dim))

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timestep = timestep.view(-1, 1)
        stacked = torch.cat([latent, prompt_embed, timestep], dim=-1)
        epsilon = self.net(stacked)
        logvar = self.logvar_bias.unsqueeze(0).expand_as(epsilon)
        return epsilon, logvar


class DiffusionModelWrapper(nn.Module):
    def __init__(self, config: ModelConfig, role: str) -> None:
        super().__init__()
        self.config = config
        self.role = role
        self.latent_dim = config.latent_dim
        self.text_embed_dim = config.text_embed_dim
        self.pipeline: Any | None = None
        if config.backend == "mock":
            self.backbone = MockDiffusionBackbone(config.latent_dim, config.text_embed_dim)
        elif config.backend == "diffusers":
            self.backbone = self._build_diffusers_backbone()
        else:
            raise ValueError(f"Unsupported backend '{config.backend}'.")

    def _resolve_model_id(self) -> str:
        if self.role == "teacher":
            return self.config.teacher_model_id
        return self.config.student_model_id

    @staticmethod
    def _resolve_dtype(dtype_name: str) -> torch.dtype:
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

    def _build_diffusers_backbone(self) -> nn.Module:
        try:
            from diffusers import StableDiffusion3Pipeline
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Diffusers backend requested but import failed. "
                "Install required packages from requirements.txt."
            ) from exc

        model_id = self._resolve_model_id()
        torch_dtype = self._resolve_dtype(self.config.dtype)
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            revision=self.config.revision,
            variant=self.config.variant,
            cache_dir=self.config.cache_dir,
            local_files_only=self.config.local_files_only,
            use_safetensors=self.config.use_safetensors,
            low_cpu_mem_usage=self.config.low_cpu_mem_usage,
        )
        self.pipeline = pipeline
        if self.config.freeze_text_encoders:
            for encoder_name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
                encoder = getattr(pipeline, encoder_name, None)
                if encoder is None:
                    continue
                for parameter in encoder.parameters():
                    parameter.requires_grad = False
        if self.config.freeze_vae and getattr(pipeline, "vae", None) is not None:
            for parameter in pipeline.vae.parameters():
                parameter.requires_grad = False

        # Always train/freeze only the diffusion transformer for this setup.
        for parameter in pipeline.transformer.parameters():
            parameter.requires_grad = self.role != "teacher"

        self.latent_dim = int(getattr(pipeline.transformer.config, "in_channels", self.latent_dim))
        return pipeline.transformer

    def encode_prompts(self, prompts: list[str], device: torch.device) -> Any:
        if self.config.backend == "diffusers":
            if self.pipeline is None:
                raise RuntimeError("Diffusers pipeline not initialized.")
            with torch.no_grad():
                prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
                    prompt=prompts,
                    prompt_2=prompts,
                    prompt_3=prompts,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    max_sequence_length=self.config.max_sequence_length,
                )
            return {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
            }

        embeds = []
        for prompt in prompts:
            digest = hashlib.sha1(prompt.encode("utf-8")).digest()
            raw = torch.tensor(list(digest), dtype=torch.float32, device=device)
            repeats = (self.text_embed_dim + raw.numel() - 1) // raw.numel()
            tiled = raw.repeat(repeats)[: self.text_embed_dim]
            embeds.append((tiled / 255.0) * 2.0 - 1.0)
        return torch.stack(embeds, dim=0)

    def prepare_initial_latents(
        self,
        batch_size: int,
        device: torch.device,
        seed: int | None,
        scale: float,
    ) -> torch.Tensor:
        generator = None
        if seed is not None:
            # MPS does not reliably support `torch.Generator(device="mps")` across PyTorch versions.
            # Use a CPU generator and CPU sampling fallback in that case.
            generator_device = device if device.type != "mps" else torch.device("cpu")
            generator = torch.Generator(device=generator_device)
            generator.manual_seed(seed)

        if self.config.backend == "diffusers":
            if self.pipeline is None:
                raise RuntimeError("Diffusers pipeline not initialized.")
            latents = self.pipeline.prepare_latents(
                batch_size=batch_size,
                num_channels_latents=self.pipeline.transformer.config.in_channels,
                height=self.config.image_height,
                width=self.config.image_width,
                dtype=self._resolve_dtype(self.config.dtype),
                device=device,
                generator=generator,
                latents=None,
            )
            return latents * scale

        if generator is None:
            latents = torch.randn(batch_size, self.latent_dim, device=device)
        elif device.type == "mps":
            latents = torch.randn(batch_size, self.latent_dim, device="cpu", generator=generator).to(device)
        else:
            latents = torch.randn(batch_size, self.latent_dim, device=device, generator=generator)
        return latents * scale

    def get_rollout_timesteps(self, num_steps: int, device: torch.device) -> list[Any]:
        if self.config.backend == "diffusers":
            if self.pipeline is None:
                raise RuntimeError("Diffusers pipeline not initialized.")
            self.pipeline.scheduler.set_timesteps(num_steps, device=device)
            return list(self.pipeline.scheduler.timesteps)
        return list(range(num_steps - 1, -1, -1))

    def step_latents(
        self,
        latents: torch.Tensor,
        epsilon: torch.Tensor,
        timestep: Any,
        fallback_step_size: float,
    ) -> torch.Tensor:
        if self.config.backend == "diffusers":
            if self.pipeline is None:
                raise RuntimeError("Diffusers pipeline not initialized.")
            return self.pipeline.scheduler.step(
                model_output=epsilon.to(latents.dtype),
                timestep=timestep,
                sample=latents,
                return_dict=False,
            )[0]
        return latents - fallback_step_size * epsilon

    def predict(
        self,
        latent: torch.Tensor,
        timestep: Any,
        prompt_embed: Any,
    ) -> ModelPrediction:
        if self.config.backend == "diffusers":
            prompt_embeds = prompt_embed["prompt_embeds"].to(device=latent.device)
            pooled_prompt_embeds = prompt_embed["pooled_prompt_embeds"].to(device=latent.device)
            model_dtype = next(self.backbone.parameters()).dtype
            latent_input = latent.to(dtype=model_dtype)
            prompt_embeds = prompt_embeds.to(dtype=model_dtype)
            pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=model_dtype)

            if isinstance(timestep, torch.Tensor):
                timestep_tensor = timestep.to(device=latent.device, dtype=model_dtype)
            else:
                timestep_tensor = torch.tensor(timestep, device=latent.device, dtype=model_dtype)
            timestep_tensor = timestep_tensor.expand(latent_input.shape[0])

            epsilon = self.backbone(
                hidden_states=latent_input,
                timestep=timestep_tensor,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]
            epsilon = epsilon.to(dtype=latent.dtype)
            logvar = torch.zeros_like(epsilon)
            mean = latent - epsilon
            return ModelPrediction(mean=mean, logvar=logvar, epsilon=epsilon)

        if isinstance(timestep, torch.Tensor):
            timestep_scalar = float(timestep.detach().cpu().item())
        else:
            timestep_scalar = float(timestep)
        ts = torch.full(
            (latent.shape[0],),
            fill_value=timestep_scalar / max(1, self.config.num_inference_steps - 1),
            device=latent.device,
            dtype=latent.dtype,
        )
        epsilon, logvar = self.backbone(latent, ts, prompt_embed)
        mean = latent - epsilon
        return ModelPrediction(mean=mean, logvar=logvar, epsilon=epsilon)

    def to(self, *args: Any, **kwargs: Any) -> "DiffusionModelWrapper":
        super().to(*args, **kwargs)
        if self.config.backend == "diffusers" and self.pipeline is not None:
            self.pipeline.to(*args, **kwargs)
        return self


def build_model(config: ModelConfig, role: str) -> DiffusionModelWrapper:
    model = DiffusionModelWrapper(config=config, role=role)
    if role == "teacher":
        for parameter in model.parameters():
            parameter.requires_grad = False
    return model
