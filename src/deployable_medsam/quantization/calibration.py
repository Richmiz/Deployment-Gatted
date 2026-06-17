from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class CalibrationSample:
    sample_id: str
    image_path: str
    mask_path: str | None
    dataset: str
    split: str


@dataclass(frozen=True)
class CalibrationManifest:
    track: str
    purpose: str
    quantization_method: str
    dataset: str
    split: str
    preprocessing_contract: str
    samples: list[CalibrationSample]


def discover_image_mask_pairs(
    image_dir: str | Path,
    mask_dir: str | Path | None = None,
    dataset: str = "unknown",
    split: str = "validation",
) -> list[CalibrationSample]:
    image_root = Path(image_dir)
    if not image_root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_root}")

    mask_root = Path(mask_dir) if mask_dir else None
    if mask_root is not None and not mask_root.exists():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_root}")

    samples: list[CalibrationSample] = []
    for image_path in sorted(image_root.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = _find_matching_mask(image_path, mask_root) if mask_root else None
        samples.append(
            CalibrationSample(
                sample_id=image_path.stem,
                image_path=str(image_path),
                mask_path=str(mask_path) if mask_path else None,
                dataset=dataset,
                split=split,
            )
        )
    if not samples:
        raise ValueError(f"No calibration images found in {image_root}")
    return samples


def select_representative_samples(samples: Sequence[CalibrationSample], target_samples: int) -> list[CalibrationSample]:
    if target_samples <= 0:
        raise ValueError("target_samples must be positive.")
    if len(samples) <= target_samples:
        return list(samples)

    if target_samples == 1:
        return [samples[len(samples) // 2]]

    step = (len(samples) - 1) / (target_samples - 1)
    selected_indices = [round(index * step) for index in range(target_samples)]

    deduped_indices = []
    seen = set()
    for index in selected_indices:
        if index not in seen:
            deduped_indices.append(index)
            seen.add(index)

    cursor = 0
    while len(deduped_indices) < target_samples:
        if cursor not in seen:
            deduped_indices.append(cursor)
            seen.add(cursor)
        cursor += 1

    return [samples[index] for index in sorted(deduped_indices[:target_samples])]


def build_calibration_manifest(
    samples: Iterable[CalibrationSample],
    dataset: str,
    split: str,
    preprocessing_contract: str,
) -> CalibrationManifest:
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("Calibration manifest requires at least one sample.")
    return CalibrationManifest(
        track="deployable_medsam",
        purpose="representative_int8_calibration",
        quantization_method="tensorrt_or_torch_tensorrt_modelopt_calibrated_int8",
        dataset=dataset,
        split=split,
        preprocessing_contract=preprocessing_contract,
        samples=sample_list,
    )


def write_manifest(manifest: CalibrationManifest, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")


def _find_matching_mask(image_path: Path, mask_root: Path | None) -> Path | None:
    if mask_root is None:
        return None
    for suffix in IMAGE_EXTENSIONS:
        candidate = mask_root / f"{image_path.stem}{suffix}"
        if candidate.exists():
            return candidate
    return None
