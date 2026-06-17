"""Student model utilities for Deployable MedSAM."""

from .dataset import (
    PlainSegmentationDataset,
    StudentSegmentationDataset,
    build_plain_student_dataloader,
    build_student_dataloader,
)
from .losses import boundary_weighted_bce_loss, dice_loss_from_logits, distillation_loss, segmentation_loss, teacher_soft_mask_loss
from .model import LightweightUNet, PromptableLightweightUNet, box_prompt_to_mask, count_trainable_parameters
from .qat import FakeQuantizedConv2d, export_clean_unet_from_fake_quant, fake_quantize_tensor, prepare_fake_quant_unet
from .training import (
    evaluate_student_model,
    resolve_device,
    summarize_student_rows,
    train_one_epoch,
    train_one_epoch_distillation,
)

__all__ = [
    "StudentSegmentationDataset",
    "PlainSegmentationDataset",
    "build_plain_student_dataloader",
    "build_student_dataloader",
    "LightweightUNet",
    "PromptableLightweightUNet",
    "box_prompt_to_mask",
    "count_trainable_parameters",
    "boundary_weighted_bce_loss",
    "dice_loss_from_logits",
    "distillation_loss",
    "evaluate_student_model",
    "FakeQuantizedConv2d",
    "export_clean_unet_from_fake_quant",
    "fake_quantize_tensor",
    "prepare_fake_quant_unet",
    "resolve_device",
    "segmentation_loss",
    "summarize_student_rows",
    "teacher_soft_mask_loss",
    "train_one_epoch",
    "train_one_epoch_distillation",
]
