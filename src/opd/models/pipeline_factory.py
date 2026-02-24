from __future__ import annotations

import hashlib
from dataclasses import dataclass

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
        if config.backend == "mock":
            self.backbone = MockDiffusionBackbone(config.latent_dim, config.text_embed_dim)
        elif config.backend == "diffusers":
            raise NotImplementedError(
                "Diffusers backend is declared but not yet wired in this scaffold. "
                "Use backend='mock' for local smoke runs."
            )
        else:
            raise ValueError(f"Unsupported backend '{config.backend}'.")

    def encode_prompts(self, prompts: list[str], device: torch.device) -> torch.Tensor:
        embeds = []
        for prompt in prompts:
            digest = hashlib.sha1(prompt.encode("utf-8")).digest()
            raw = torch.tensor(list(digest), dtype=torch.float32, device=device)
            repeats = (self.text_embed_dim + raw.numel() - 1) // raw.numel()
            tiled = raw.repeat(repeats)[: self.text_embed_dim]
            embeds.append((tiled / 255.0) * 2.0 - 1.0)
        return torch.stack(embeds, dim=0)

    def predict(
        self,
        latent: torch.Tensor,
        timestep: int,
        prompt_embed: torch.Tensor,
    ) -> ModelPrediction:
        ts = torch.full(
            (latent.shape[0],),
            fill_value=float(timestep) / max(1, self.config.num_inference_steps - 1),
            device=latent.device,
            dtype=latent.dtype,
        )
        epsilon, logvar = self.backbone(latent, ts, prompt_embed)
        mean = latent - epsilon
        return ModelPrediction(mean=mean, logvar=logvar, epsilon=epsilon)


def build_model(config: ModelConfig, role: str) -> DiffusionModelWrapper:
    model = DiffusionModelWrapper(config=config, role=role)
    if role == "teacher":
        for parameter in model.parameters():
            parameter.requires_grad = False
    return model
