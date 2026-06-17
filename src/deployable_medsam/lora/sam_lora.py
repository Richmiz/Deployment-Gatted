from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from deployable_medsam.data.prompts import BoxPrompt

from .adapters import (
    LoRAConfig,
    inject_lora,
    load_lora_state_dict,
    lora_state_dict,
    lora_trainable_parameters,
    mark_only_lora_as_trainable,
)


PRIMARY_TEACHER_MODEL_ID = "wanglab/medsam-vit-base"
FALLBACK_TEACHER_MODEL_ID = "facebook/sam-vit-base"


class SamLoraModel:
    """LoRA-adapted SAM/MedSAM teacher with a differentiable box-prompted forward.

    The base SAM weights are frozen and only the injected LoRA adapters (plus any
    ``also_train`` parameters such as the mask decoder) are updated. This is the
    Stage 3 teacher adaptation used by the study workflow.
    """

    def __init__(
        self,
        model_id: str = PRIMARY_TEACHER_MODEL_ID,
        *,
        lora_config: LoRAConfig | None = None,
        also_train: Sequence[str] = (),
        device: str = "auto",
        allow_fallback: bool = False,
        fallback_model_id: str = FALLBACK_TEACHER_MODEL_ID,
    ) -> None:
        self.device = _resolve_device(device)
        self.lora_config = lora_config or LoRAConfig()
        self.also_train = tuple(also_train)
        self.model_id = model_id
        try:
            self.processor, self.model = self._load_model(model_id)
        except Exception:
            if not allow_fallback or model_id == fallback_model_id:
                raise
            self.model_id = fallback_model_id
            self.processor, self.model = self._load_model(fallback_model_id)

        self.adapted_modules = inject_lora(self.model, self.lora_config)
        mark_only_lora_as_trainable(self.model, also_train=self.also_train)
        self.model.to(self.device)

    def _load_model(self, model_id: str):
        try:
            from transformers import SamModel, SamProcessor
        except ImportError as exc:
            raise ImportError(
                "LoRA teacher adaptation requires torch and transformers. "
                "Install them with: python -m pip install -r requirements-experiments.txt"
            ) from exc
        processor = SamProcessor.from_pretrained(model_id)
        model = SamModel.from_pretrained(model_id)
        return processor, model

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def trainable_parameters(self) -> int:
        return lora_trainable_parameters(self.model)

    def forward_logits(self, image, box: BoxPrompt, *, target_size: tuple[int, int]):
        """Return box-prompted mask logits of shape ``[1, 1, H, W]`` with gradient."""
        import torch

        input_boxes = [[[box.x_min, box.y_min, box.x_max, box.y_max]]]
        inputs = self.processor(image, input_boxes=input_boxes, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, multimask_output=False)
        logits = outputs.pred_masks[:, 0]  # [batch, num_masks=1, H_low, W_low]
        height, width = target_size
        if tuple(logits.shape[-2:]) != (height, width):
            logits = torch.nn.functional.interpolate(
                logits, size=(height, width), mode="bilinear", align_corners=False
            )
        return logits

    def save_adapters(self, path: str | Path) -> Path:
        import torch

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "lora_state_dict": lora_state_dict(self.model, also_train=self.also_train),
            "lora_config": asdict(self.lora_config),
            "also_train": list(self.also_train),
            "model_id": self.model_id,
            "adapted_modules": list(self.adapted_modules),
            "trainable_parameters": self.trainable_parameters(),
        }
        torch.save(payload, output)
        return output

    def load_adapters(self, path: str | Path, *, strict: bool = True) -> None:
        import torch

        payload = torch.load(path, map_location=self.device)
        load_lora_state_dict(self.model, payload["lora_state_dict"], strict=strict)


def load_lora_config_from_checkpoint(path: str | Path) -> LoRAConfig:
    import torch

    payload = torch.load(path, map_location="cpu")
    config_dict = payload.get("lora_config", {})
    return LoRAConfig(**config_dict) if config_dict else LoRAConfig()


def _resolve_device(device: str) -> str:
    if device != "auto":
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda.")
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
