from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class LoRAConfig:
    """Configuration for low-rank adaptation of selected ``nn.Linear`` layers.

    ``target_modules`` are matched as substrings of the fully qualified module
    name (e.g. ``vision_encoder.layers.0.attn.qkv``). The defaults cover the
    attention projection naming used by Hugging Face ``SamModel`` for both the
    ViT image encoder (``qkv``) and the two-way mask decoder (``q_proj`` /
    ``v_proj``).
    """

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("qkv", "q_proj", "v_proj", "query", "value")

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("LoRA rank must be a positive integer.")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1).")
        if not self.target_modules:
            raise ValueError("At least one target module substring is required.")


class LoRALinear(nn.Module):
    """Wrap a frozen ``nn.Linear`` with a trainable low-rank update.

    Computes ``y = W0 x + (alpha / rank) * (x A^T) B^T``. ``lora_B`` is
    initialized to zero so the adapter is an identity at initialization
    (``y == W0 x``); training then learns the low-rank delta while ``W0`` stays
    frozen.
    """

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear can only wrap an nn.Linear layer.")
        if rank <= 0:
            raise ValueError("LoRA rank must be a positive integer.")

        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Standard LoRA init: A ~ Kaiming, B = 0 so the initial delta is exactly 0.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scaling * lora_out


def inject_lora(model: nn.Module, config: LoRAConfig) -> list[str]:
    """Replace matching ``nn.Linear`` submodules with :class:`LoRALinear` in place.

    Returns the list of fully qualified module names that were adapted. Modules
    already wrapped are skipped so the call is idempotent.
    """

    replaced: list[str] = []
    for module_name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            continue
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            qualified_name = f"{module_name}.{child_name}" if module_name else child_name
            if not _matches_target(qualified_name, config.target_modules):
                continue
            wrapped = LoRALinear(child, rank=config.rank, alpha=config.alpha, dropout=config.dropout)
            setattr(module, child_name, wrapped)
            replaced.append(qualified_name)

    if not replaced:
        raise ValueError(
            "inject_lora replaced no layers. Check that target_modules "
            f"{config.target_modules} match the model's nn.Linear names."
        )
    return replaced


def mark_only_lora_as_trainable(model: nn.Module, *, also_train: Sequence[str] = ()) -> None:
    """Freeze every parameter except LoRA adapters (and any ``also_train`` matches).

    ``also_train`` substrings are matched against parameter names so callers can,
    for example, additionally fine-tune the mask decoder via ``("mask_decoder",)``.
    """

    for name, parameter in model.named_parameters():
        is_lora = "lora_A" in name or "lora_B" in name
        is_extra = any(token in name for token in also_train)
        parameter.requires_grad_(is_lora or is_extra)


def lora_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def lora_state_dict(model: nn.Module, *, also_train: Sequence[str] = ()) -> dict[str, torch.Tensor]:
    """Return only the trainable adapter (and ``also_train``) tensors for checkpointing."""

    state: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        is_lora = "lora_A" in name or "lora_B" in name
        is_extra = any(token in name for token in also_train)
        if is_lora or is_extra:
            state[name] = parameter.detach().cpu().clone()
    return state


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor], *, strict: bool = True) -> None:
    """Load adapter tensors produced by :func:`lora_state_dict` back into ``model``."""

    own = dict(model.named_parameters())
    missing = [name for name in state if name not in own]
    if strict and missing:
        raise KeyError(f"LoRA state contains parameters absent from the model: {missing[:5]}")
    with torch.no_grad():
        for name, tensor in state.items():
            if name in own:
                own[name].copy_(tensor.to(own[name].device, own[name].dtype))


def _matches_target(qualified_name: str, target_modules: Sequence[str]) -> bool:
    return any(token in qualified_name for token in target_modules)
