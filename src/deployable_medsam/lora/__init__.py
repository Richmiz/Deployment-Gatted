"""LoRA teacher-adaptation utilities for Deployable MedSAM (Stage 3)."""

from .adapters import (
    LoRAConfig,
    LoRALinear,
    inject_lora,
    load_lora_state_dict,
    lora_state_dict,
    lora_trainable_parameters,
    mark_only_lora_as_trainable,
)
from .sam_lora import (
    FALLBACK_TEACHER_MODEL_ID,
    PRIMARY_TEACHER_MODEL_ID,
    SamLoraModel,
    load_lora_config_from_checkpoint,
)

__all__ = [
    "LoRAConfig",
    "LoRALinear",
    "inject_lora",
    "load_lora_state_dict",
    "lora_state_dict",
    "lora_trainable_parameters",
    "mark_only_lora_as_trainable",
    "FALLBACK_TEACHER_MODEL_ID",
    "PRIMARY_TEACHER_MODEL_ID",
    "SamLoraModel",
    "load_lora_config_from_checkpoint",
]
