from __future__ import annotations

import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

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
from .losses import distillation_loss, segmentation_loss


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return torch.device(device)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    dice_weight: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        images = batch["image"].to(device)
        targets = batch["ground_truth_mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = segmentation_loss(logits, targets, dice_weight=dice_weight)
        loss.backward()
        optimizer.step()
        batch_size = images.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


def train_one_epoch_distillation(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    dice_weight: float = 1.0,
    teacher_loss_weight: float = 0.5,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch in dataloader:
        images = batch["image"].to(device)
        targets = batch["ground_truth_mask"].to(device)
        teacher_probabilities = batch["teacher_probabilities"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = distillation_loss(
            logits,
            targets,
            teacher_probabilities,
            dice_weight=dice_weight,
            teacher_loss_weight=teacher_loss_weight,
        )
        loss.backward()
        optimizer.step()
        batch_size = images.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_student_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    split: str,
    model_id: str = "student_baseline",
    threshold: float = 0.5,
    roi_padding_pixels: int = 10,
    boundary_radius_pixels: int = 2,
    ece_bins: int = 10,
    trainable_parameters: int | None = None,
    prediction_output_dir: str | Path | None = None,
    project_root: str | Path | None = None,
) -> list[dict[str, object]]:
    model.eval()
    rows: list[dict[str, object]] = []
    prediction_dir = Path(prediction_output_dir) if prediction_output_dir is not None else None
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    root = Path(project_root).resolve() if project_root is not None else None

    for batch in dataloader:
        images = batch["image"].to(device)
        targets = batch["ground_truth_mask"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000.0 / max(images.shape[0], 1)
        probabilities = torch.sigmoid(logits).detach().cpu()
        target_cpu = targets.detach().cpu()

        for index in range(images.shape[0]):
            probability_array = probabilities[index, 0].numpy().astype(np.float32)
            binary_array = (probability_array >= threshold).astype(np.uint8)
            target_array = target_cpu[index, 0].numpy().astype(np.float32)
            probability = probability_array.tolist()
            target = target_array.tolist()
            foreground = foreground_mask(target, threshold=threshold)
            roi = roi_mask_from_binary_masks(probability, target, threshold=threshold, padding=roi_padding_pixels)
            boundary = boundary_mask_from_binary_masks(
                probability,
                target,
                threshold=threshold,
                radius=boundary_radius_pixels,
            )
            sample_id = str(batch["sample_id"][index])
            row = {
                "sample_id": sample_id,
                "split": split,
                "model_id": model_id,
                "dice": _rounded(dice_score(probability, target, threshold=threshold)),
                "iou": _rounded(iou_score(probability, target, threshold=threshold)),
                "precision": _rounded(precision_score(probability, target, threshold=threshold)),
                "recall": _rounded(recall_score(probability, target, threshold=threshold)),
                "brier_full": _rounded(binary_brier_score(probability, target)),
                "brier_roi": _rounded(binary_brier_score(probability, target, sample_mask=roi)),
                "binary_ece_full": _rounded(binary_ece(probability, target, n_bins=ece_bins)),
                "binary_ece_foreground": _rounded(
                    binary_ece(probability, target, n_bins=ece_bins, sample_mask=foreground)
                ),
                "binary_ece_roi": _rounded(binary_ece(probability, target, n_bins=ece_bins, sample_mask=roi)),
                "binary_ece_boundary": _rounded(
                    binary_ece(probability, target, n_bins=ece_bins, sample_mask=boundary)
                ),
                "latency_ms": _rounded(latency_ms, digits=3),
                "device": device.type,
            }
            if trainable_parameters is not None:
                row["trainable_parameters"] = int(trainable_parameters)
            if prediction_dir is not None:
                prediction_path = prediction_dir / f"{_safe_filename(sample_id)}.npz"
                np.savez_compressed(
                    prediction_path,
                    probabilities=probability_array,
                    binary_mask=binary_array,
                    sample_id=sample_id,
                    split=split,
                    model_name=model_id,
                    threshold=np.float32(threshold),
                )
                row["prediction_path"] = _relative_or_posix(prediction_path, root) if root is not None else prediction_path.as_posix()
            rows.append(row)
    return rows


def summarize_student_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        raise ValueError("At least one student metric row is required.")
    group_fields = ["split", "model_id", "device"]
    if "epoch" in rows[0]:
        group_fields.append("epoch")
    if "trainable_parameters" in rows[0]:
        group_fields.append("trainable_parameters")
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
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)

    summaries = []
    for key, group_rows in sorted(grouped.items()):
        summary = {field: value for field, value in zip(group_fields, key)}
        summary["sample_count"] = len(group_rows)
        for field in metric_fields:
            values = [float(row[field]) for row in group_rows]
            summary[f"mean_{field}"] = _rounded(sum(values) / len(values), digits=6)
        summaries.append(summary)
    return summaries


def _rounded(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._") or "sample"


def _relative_or_posix(path: Path, root: Path | None) -> str:
    if root is None:
        return path.as_posix()
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
