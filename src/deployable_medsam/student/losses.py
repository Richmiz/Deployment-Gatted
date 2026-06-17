from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    probabilities = probabilities.flatten(start_dim=1)
    targets = targets.float().flatten(start_dim=1)
    numerator = 2.0 * (probabilities * targets).sum(dim=1) + eps
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1) + eps
    return 1.0 - (numerator / denominator).mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor, dice_weight: float = 1.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets.float())
    return bce + dice_weight * dice_loss_from_logits(logits, targets)


def boundary_weighted_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    boundary_weight: float = 2.0,
    radius: int = 2,
) -> torch.Tensor:
    if radius < 0:
        raise ValueError("radius must be non-negative.")
    target = targets.float()
    kernel_size = 2 * radius + 1
    pooled_max = F.max_pool2d(target, kernel_size=kernel_size, stride=1, padding=radius)
    pooled_min = -F.max_pool2d(-target, kernel_size=kernel_size, stride=1, padding=radius)
    boundary = (pooled_max - pooled_min > 0).float()
    weights = 1.0 + float(boundary_weight) * boundary
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def teacher_soft_mask_loss(logits: torch.Tensor, teacher_probabilities: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, teacher_probabilities.float())


def distillation_loss(
    logits: torch.Tensor,
    ground_truth: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    *,
    dice_weight: float = 1.0,
    teacher_loss_weight: float = 0.5,
) -> torch.Tensor:
    return (
        segmentation_loss(logits, ground_truth, dice_weight=dice_weight)
        + teacher_loss_weight * teacher_soft_mask_loss(logits, teacher_probabilities)
    )
