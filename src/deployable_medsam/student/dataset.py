from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from deployable_medsam.data import load_binary_mask, load_rgb_image, read_split_manifest
from deployable_medsam.distillation import TeacherDistillationRecord, read_jsonl_manifest
from deployable_medsam.distillation.artifacts import resolve_project_path, validate_prediction_artifact


class StudentSegmentationDataset(Dataset):
    def __init__(self, manifest_path: str | Path, *, project_root: str | Path | None = None) -> None:
        self.manifest_path = Path(manifest_path)
        self.project_root = Path(project_root) if project_root is not None else self.manifest_path.resolve().parents[2]
        self.records = read_jsonl_manifest(self.manifest_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        image_path = resolve_project_path(record.image_path, self.project_root)
        mask_path = resolve_project_path(record.mask_path, self.project_root)
        prediction_path = resolve_project_path(record.teacher_prediction_path, self.project_root)
        validate_prediction_artifact(prediction_path, expected_size=(record.input_size, record.input_size))

        image = load_rgb_image(image_path, size=(record.input_size, record.input_size))
        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)

        ground_truth = np.asarray(load_binary_mask(mask_path, size=(record.input_size, record.input_size)), dtype=np.float32)
        ground_truth_tensor = torch.from_numpy(ground_truth[None, :, :])

        with np.load(prediction_path) as artifact:
            teacher_probabilities = torch.from_numpy(artifact["probabilities"].astype(np.float32)[None, :, :].copy())
            teacher_binary_mask = torch.from_numpy(artifact["binary_mask"].astype(np.float32)[None, :, :].copy())

        return {
            "image": image_tensor,
            "ground_truth_mask": ground_truth_tensor,
            "teacher_probabilities": teacher_probabilities,
            "teacher_binary_mask": teacher_binary_mask,
            "sample_id": record.sample_id,
            "split": record.split,
            "image_path": record.image_path,
            "teacher_prediction_path": record.teacher_prediction_path,
        }


class PlainSegmentationDataset(Dataset):
    """Image/mask dataset for student evaluation without teacher artifacts."""

    def __init__(
        self,
        split_manifest_path: str | Path,
        *,
        split: str,
        project_root: str | Path | None = None,
        input_size: int = 256,
    ) -> None:
        self.split_manifest_path = Path(split_manifest_path)
        self.project_root = Path(project_root) if project_root is not None else self.split_manifest_path.resolve().parents[2]
        self.input_size = int(input_size)
        if self.input_size <= 0:
            raise ValueError("input_size must be positive.")
        manifest = read_split_manifest(self.split_manifest_path)
        if split not in manifest.samples:
            valid = ", ".join(sorted(manifest.samples))
            raise KeyError(f"Split {split!r} is not available. Valid splits: {valid}")
        self.split = split
        self.records = manifest.samples[split]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.records[index]
        image_path = resolve_project_path(sample.image_path, self.project_root)
        mask_path = resolve_project_path(sample.mask_path, self.project_root)
        image = load_rgb_image(image_path, size=(self.input_size, self.input_size))
        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        ground_truth = np.asarray(load_binary_mask(mask_path, size=(self.input_size, self.input_size)), dtype=np.float32)
        ground_truth_tensor = torch.from_numpy(ground_truth[None, :, :])
        return {
            "image": image_tensor,
            "ground_truth_mask": ground_truth_tensor,
            "sample_id": sample.sample_id,
            "split": self.split,
            "image_path": sample.image_path,
            "mask_path": sample.mask_path,
            "dataset": sample.dataset,
        }


def build_student_dataloader(
    manifest_path: str | Path,
    *,
    project_root: str | Path | None = None,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = StudentSegmentationDataset(manifest_path, project_root=project_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def build_plain_student_dataloader(
    split_manifest_path: str | Path,
    *,
    split: str,
    project_root: str | Path | None = None,
    input_size: int = 256,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = PlainSegmentationDataset(
        split_manifest_path,
        split=split,
        project_root=project_root,
        input_size=input_size,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
