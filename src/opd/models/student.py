from __future__ import annotations

from opd.models.pipeline_factory import DiffusionModelWrapper, build_model
from opd.utils.io import ModelConfig


def build_student_model(model_config: ModelConfig) -> DiffusionModelWrapper:
    return build_model(model_config, role="student")
