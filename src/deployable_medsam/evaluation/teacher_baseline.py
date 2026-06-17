from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

from deployable_medsam.data import SegmentationSample
from deployable_medsam.data.prompts import BoxPrompt
from research_runtime.metrics import (
    binary_brier_score,
    binary_ece,
    boundary_mask_from_binary_masks,
    dice_score,
    foreground_mask,
    iou_score,
    precision_score,
    recall_score,
    roi_mask_from_binary_masks,
)


PRIMARY_TEACHER_MODEL_ID = "wanglab/medsam-vit-base"
FALLBACK_TEACHER_MODEL_ID = "facebook/sam-vit-base"


@dataclass(frozen=True)
class PromptSpec:
    prompt_type: str
    jitter_pixels: int


@dataclass(frozen=True)
class MetricConfig:
    roi_padding_pixels: int = 10
    boundary_radius_pixels: int = 2
    threshold: float = 0.5
    ece_bins: int = 10


@dataclass(frozen=True)
class TeacherPrediction:
    probabilities: list[list[float]]
    latency_ms: float
    model_id: str
    device: str
    precision: str


def expand_prompt_specs(prompt_mode: str, jitter_levels: Sequence[int]) -> list[PromptSpec]:
    normalized_mode = prompt_mode.lower()
    if normalized_mode not in {"clean", "noisy", "both"}:
        raise ValueError("prompt_mode must be one of: clean, noisy, both.")

    unique_jitters = sorted({int(level) for level in jitter_levels})
    if any(level < 0 for level in unique_jitters):
        raise ValueError("jitter levels must be non-negative.")

    specs: list[PromptSpec] = []
    if normalized_mode in {"clean", "both"}:
        specs.append(PromptSpec(prompt_type="clean", jitter_pixels=0))
    if normalized_mode in {"noisy", "both"}:
        noisy_levels = unique_jitters if normalized_mode == "noisy" else [level for level in unique_jitters if level > 0]
        specs.extend(PromptSpec(prompt_type="noisy", jitter_pixels=level) for level in noisy_levels)
    return specs


def compute_metric_row(
    *,
    sample: SegmentationSample,
    split: str,
    model_id: str,
    prompt_type: str,
    jitter_pixels: int,
    box: BoxPrompt,
    probabilities: Sequence[Sequence[float]],
    target_mask: Sequence[Sequence[int]],
    latency_ms: float,
    device: str,
    precision: str,
    metric_config: MetricConfig | None = None,
) -> dict[str, object]:
    config = metric_config or MetricConfig()
    foreground = foreground_mask(target_mask, threshold=config.threshold)
    roi = roi_mask_from_binary_masks(
        probabilities,
        target_mask,
        threshold=config.threshold,
        padding=config.roi_padding_pixels,
    )
    boundary = boundary_mask_from_binary_masks(
        probabilities,
        target_mask,
        threshold=config.threshold,
        radius=config.boundary_radius_pixels,
    )

    return {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "split": split,
        "model_id": model_id,
        "prompt_type": prompt_type,
        "jitter_pixels": jitter_pixels,
        "box_xyxy": str(box.as_xyxy()),
        "dice": _rounded(dice_score(probabilities, target_mask, threshold=config.threshold)),
        "iou": _rounded(iou_score(probabilities, target_mask, threshold=config.threshold)),
        "precision": _rounded(precision_score(probabilities, target_mask, threshold=config.threshold)),
        "recall": _rounded(recall_score(probabilities, target_mask, threshold=config.threshold)),
        "brier_full": _rounded(binary_brier_score(probabilities, target_mask)),
        "brier_roi": _rounded(binary_brier_score(probabilities, target_mask, sample_mask=roi)),
        "binary_ece_full": _rounded(binary_ece(probabilities, target_mask, n_bins=config.ece_bins)),
        "binary_ece_foreground": _rounded(
            binary_ece(probabilities, target_mask, n_bins=config.ece_bins, sample_mask=foreground)
        ),
        "binary_ece_roi": _rounded(binary_ece(probabilities, target_mask, n_bins=config.ece_bins, sample_mask=roi)),
        "binary_ece_boundary": _rounded(
            binary_ece(probabilities, target_mask, n_bins=config.ece_bins, sample_mask=boundary)
        ),
        "latency_ms": _rounded(latency_ms, digits=3),
        "device": device,
        "inference_precision": precision,
    }


def summarize_metric_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        raise ValueError("At least one metric row is required.")

    group_fields = ["dataset", "split", "model_id", "prompt_type", "jitter_pixels", "device", "inference_precision"]
    metric_fields = [
        "dice",
        "iou",
        "precision",
        "recall",
        "brier_full",
        "brier_roi",
        "binary_ece_full",
        "binary_ece_foreground",
        "binary_ece_roi",
        "binary_ece_boundary",
        "latency_ms",
    ]

    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped.setdefault(key, []).append(row)

    summaries = []
    for key, group_rows in sorted(grouped.items()):
        summary = {field: value for field, value in zip(group_fields, key)}
        summary["sample_count"] = len(group_rows)
        for field in metric_fields:
            values = [float(row[field]) for row in group_rows]
            summary[f"mean_{field}"] = _rounded(sum(values) / len(values), digits=6)
        summaries.append(summary)
    return summaries


class SamTeacherPredictor:
    def __init__(
        self,
        model_id: str = PRIMARY_TEACHER_MODEL_ID,
        *,
        fallback_model_id: str = FALLBACK_TEACHER_MODEL_ID,
        allow_fallback: bool = False,
        device: str = "auto",
        precision: str = "fp32",
        lora_checkpoint: str | None = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.precision = _resolve_precision(precision, self.device)
        self.model_id = model_id
        try:
            self.processor, self.model = self._load_model(model_id)
        except Exception:
            if not allow_fallback or model_id == fallback_model_id:
                raise
            self.model_id = fallback_model_id
            self.processor, self.model = self._load_model(fallback_model_id)

        self.lora_checkpoint = lora_checkpoint
        if lora_checkpoint is not None:
            self._apply_lora_adapters(lora_checkpoint)
            self.model_id = f"{self.model_id}-lora"

    def _load_model(self, model_id: str):
        try:
            import torch
            from transformers import SamModel, SamProcessor
        except ImportError as exc:
            raise ImportError(
                "Teacher baseline inference requires torch and transformers. "
                "Install them with: python -m pip install -r requirements-experiments.txt"
            ) from exc

        processor = SamProcessor.from_pretrained(model_id)
        model = SamModel.from_pretrained(model_id)
        model.to(self.device)
        if self.precision == "fp16":
            model.half()
        model.eval()
        return processor, model

    def _apply_lora_adapters(self, lora_checkpoint: str) -> None:
        """Inject and load Stage 3 LoRA adapters so this teacher matches the trained run."""
        import torch

        from deployable_medsam.lora import LoRAConfig, inject_lora, load_lora_state_dict

        payload = torch.load(lora_checkpoint, map_location=self.device)
        config = LoRAConfig(**payload["lora_config"]) if payload.get("lora_config") else LoRAConfig()
        inject_lora(self.model, config)
        load_lora_state_dict(self.model, payload["lora_state_dict"], strict=False)
        self.model.to(self.device)
        if self.precision == "fp16":
            self.model.half()
        self.model.eval()

    def predict(self, image, box: BoxPrompt) -> TeacherPrediction:
        import torch

        width, height = image.size
        input_boxes = [[[box.x_min, box.y_min, box.x_max, box.y_max]]]
        inputs = self.processor(image, input_boxes=input_boxes, return_tensors="pt")
        inputs = inputs.to(self.device)
        if self.precision == "fp16" and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].half()

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)
        if self.device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000.0

        try:
            masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.detach().cpu(),
                inputs["original_sizes"].detach().cpu(),
                inputs["reshaped_input_sizes"].detach().cpu(),
                binarize=False,
            )
        except TypeError:
            masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.detach().cpu(),
                inputs["original_sizes"].detach().cpu(),
                inputs["reshaped_input_sizes"].detach().cpu(),
            )

        mask_tensor = masks[0]
        while mask_tensor.ndim > 2:
            mask_tensor = mask_tensor[0]
        if tuple(mask_tensor.shape[-2:]) != (height, width):
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor.float().view(1, 1, *mask_tensor.shape[-2:]),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        probabilities = mask_tensor.float() if mask_tensor.dtype == torch.bool else torch.sigmoid(mask_tensor.float())
        return TeacherPrediction(
            probabilities=probabilities.detach().cpu().tolist(),
            latency_ms=latency_ms,
            model_id=self.model_id,
            device=self.device,
            precision=self.precision,
        )


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


def _resolve_precision(precision: str, device: str) -> str:
    normalized = precision.lower()
    if normalized not in {"fp32", "fp16"}:
        raise ValueError("precision must be one of: fp32, fp16.")
    if normalized == "fp16" and device != "cuda":
        raise ValueError("fp16 teacher inference is only supported on CUDA.")
    return normalized


def _rounded(value: float, digits: int = 6) -> float:
    return round(float(value), digits)
