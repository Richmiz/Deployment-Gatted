from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from deployable_medsam.data import SegmentationSample, SplitManifest, load_binary_mask, load_rgb_image


@dataclass(frozen=True)
class TeacherDistillationRecord:
    sample_id: str
    dataset: str
    split: str
    image_path: str
    mask_path: str
    teacher_prediction_path: str
    teacher_model_id: str
    prompt_type: str
    jitter_pixels: int
    box_xyxy: str
    input_size: int
    source_teacher_csv: str


def read_teacher_baseline_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Teacher baseline CSV does not exist: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Teacher baseline CSV has no rows: {csv_path}")
    return rows


def build_teacher_distillation_records(
    *,
    teacher_csv_path: str | Path,
    split_manifest: SplitManifest,
    split: str,
    project_root: str | Path,
    prompt_type: str = "clean",
    jitter_pixels: int = 0,
    input_size: int = 256,
) -> list[TeacherDistillationRecord]:
    project_path = Path(project_root)
    csv_path = Path(teacher_csv_path)
    rows = read_teacher_baseline_rows(csv_path)
    samples = _samples_by_id(split_manifest, split)
    records: list[TeacherDistillationRecord] = []

    for row in rows:
        row_split = row.get("split")
        row_prompt_type = row.get("prompt_type")
        row_jitter = int(row.get("jitter_pixels", -1))
        if row_split != split or row_prompt_type != prompt_type or row_jitter != jitter_pixels:
            continue

        sample_id = row.get("sample_id", "")
        if sample_id not in samples:
            raise ValueError(f"Sample {sample_id!r} from {csv_path} is not present in split {split!r}.")
        prediction_path = row.get("prediction_path", "")
        if not prediction_path:
            raise ValueError(f"Teacher row for sample {sample_id!r} is missing prediction_path.")
        resolved_prediction_path = resolve_project_path(prediction_path, project_path)
        validate_prediction_artifact(resolved_prediction_path, expected_size=(input_size, input_size))

        sample = samples[sample_id]
        records.append(
            TeacherDistillationRecord(
                sample_id=sample.sample_id,
                dataset=sample.dataset,
                split=split,
                image_path=_relative_or_posix(resolve_project_path(sample.image_path, project_path), project_path),
                mask_path=_relative_or_posix(resolve_project_path(sample.mask_path, project_path), project_path),
                teacher_prediction_path=_relative_or_posix(resolved_prediction_path, project_path),
                teacher_model_id=row.get("model_id", ""),
                prompt_type=row_prompt_type,
                jitter_pixels=row_jitter,
                box_xyxy=row.get("box_xyxy", ""),
                input_size=input_size,
                source_teacher_csv=_relative_or_posix(csv_path, project_path),
            )
        )

    if not records:
        raise ValueError(
            f"No teacher rows matched split={split!r}, prompt_type={prompt_type!r}, jitter_pixels={jitter_pixels} in {csv_path}."
        )
    return records


def validate_prediction_artifact(path: str | Path, expected_size: tuple[int, int] = (256, 256)) -> None:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Teacher prediction artifact does not exist: {artifact_path}")
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("Validating teacher prediction artifacts requires numpy.") from exc

    with np.load(artifact_path) as artifact:
        required = {"probabilities", "binary_mask"}
        missing = required - set(artifact.files)
        if missing:
            raise ValueError(f"Teacher prediction artifact {artifact_path} is missing arrays: {sorted(missing)}")
        probabilities = artifact["probabilities"]
        binary_mask = artifact["binary_mask"]
        if tuple(probabilities.shape) != expected_size:
            raise ValueError(f"Artifact {artifact_path} probabilities shape {probabilities.shape} != {expected_size}.")
        if tuple(binary_mask.shape) != expected_size:
            raise ValueError(f"Artifact {artifact_path} binary_mask shape {binary_mask.shape} != {expected_size}.")
        if probabilities.dtype != np.float32:
            raise ValueError(f"Artifact {artifact_path} probabilities dtype {probabilities.dtype} != float32.")
        if binary_mask.dtype != np.uint8:
            raise ValueError(f"Artifact {artifact_path} binary_mask dtype {binary_mask.dtype} != uint8.")


def write_jsonl_manifest(path: str | Path, records: Sequence[TeacherDistillationRecord]) -> None:
    if not records:
        raise ValueError("At least one distillation record is required.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def read_jsonl_manifest(path: str | Path) -> list[TeacherDistillationRecord]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Distillation manifest does not exist: {manifest_path}")
    records = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(TeacherDistillationRecord(**json.loads(line)))
            except TypeError as exc:
                raise ValueError(f"Invalid distillation record at {manifest_path}:{line_number}") from exc
    if not records:
        raise ValueError(f"Distillation manifest has no records: {manifest_path}")
    return records


def load_distillation_preview(record: TeacherDistillationRecord, project_root: str | Path):
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("Loading distillation previews requires numpy.") from exc

    project_path = Path(project_root)
    image = load_rgb_image(resolve_project_path(record.image_path, project_path), size=(record.input_size, record.input_size))
    mask = load_binary_mask(resolve_project_path(record.mask_path, project_path), size=(record.input_size, record.input_size))
    artifact_path = resolve_project_path(record.teacher_prediction_path, project_path)
    validate_prediction_artifact(artifact_path, expected_size=(record.input_size, record.input_size))
    with np.load(artifact_path) as artifact:
        probabilities = artifact["probabilities"].copy()
        binary_mask = artifact["binary_mask"].copy()
    return {
        "image": image,
        "ground_truth_mask": mask,
        "teacher_probabilities": probabilities,
        "teacher_binary_mask": binary_mask,
    }


def resolve_project_path(path: str | Path, project_root: str | Path) -> Path:
    raw_path = str(path)
    if _is_wsl_mount_path(raw_path) and _running_on_windows():
        return _wsl_mount_to_windows_path(raw_path)
    if _is_windows_drive_path(raw_path) and not _running_on_windows():
        return _windows_drive_to_wsl_mount(raw_path)

    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return Path(project_root) / candidate


def _running_on_windows() -> bool:
    return Path.cwd().drive != ""


def _is_wsl_mount_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    parts = normalized.split("/")
    return len(parts) >= 4 and parts[0] == "" and parts[1] == "mnt" and len(parts[2]) == 1


def _wsl_mount_to_windows_path(path: str) -> Path:
    normalized = path.replace("\\", "/")
    _, _, drive, remainder = normalized.split("/", 3)
    return Path(f"{drive.upper()}:/{remainder}")


def _is_windows_drive_path(path: str) -> bool:
    return len(path) >= 3 and path[1] == ":" and path[2] in {"\\", "/"}


def _windows_drive_to_wsl_mount(path: str) -> Path:
    drive = path[0].lower()
    remainder = path[3:].replace("\\", "/")
    return Path(f"/mnt/{drive}/{remainder}")


def _samples_by_id(split_manifest: SplitManifest, split: str) -> dict[str, SegmentationSample]:
    if split not in split_manifest.samples:
        raise ValueError(f"Split {split!r} is not available in the split manifest.")
    return {sample.sample_id: sample for sample in split_manifest.samples[split]}


def _relative_or_posix(path: str | Path, project_root: Path) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return candidate.as_posix()
