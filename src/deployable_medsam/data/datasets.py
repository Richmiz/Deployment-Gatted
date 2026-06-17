from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SegmentationSample:
    sample_id: str
    image_path: str
    mask_path: str
    dataset: str


@dataclass(frozen=True)
class SplitManifest:
    dataset: str
    seed: int
    split_ratios: dict[str, float]
    counts: dict[str, int]
    samples: dict[str, list[SegmentationSample]]


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    display_name: str
    default_root: str
    task: str
    image_subdir: str = "images"
    mask_subdir: str = "masks"
    notes: str = ""

    def image_dir(self, root: str | Path | None = None) -> Path:
        return Path(root or self.default_root) / self.image_subdir

    def mask_dir(self, root: str | Path | None = None) -> Path:
        return Path(root or self.default_root) / self.mask_subdir


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "kvasir_seg": DatasetSpec(
        dataset="kvasir_seg",
        display_name="Kvasir-SEG",
        default_root="data/kvasir_seg",
        task="polyp segmentation",
        notes="Expected local folders: images/ and masks/ with matching filename stems.",
    ),
    "isic_2018_task1": DatasetSpec(
        dataset="isic_2018_task1",
        display_name="ISIC 2018 Task 1",
        default_root="data/isic_2018_task1",
        task="skin lesion boundary segmentation",
        notes="Place challenge images under images/ and binary lesion masks under masks/ after accepting dataset terms.",
    ),
    "busi": DatasetSpec(
        dataset="busi",
        display_name="BUSI",
        default_root="data/busi",
        task="breast ultrasound lesion segmentation",
        notes="Flatten or symlink benign/malignant image and mask files into images/ and masks/ with matching stems.",
    ),
}


def list_dataset_specs() -> list[DatasetSpec]:
    return [DATASET_REGISTRY[key] for key in sorted(DATASET_REGISTRY)]


def get_dataset_spec(dataset: str) -> DatasetSpec:
    try:
        return DATASET_REGISTRY[dataset]
    except KeyError as exc:
        valid = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset {dataset!r}. Valid datasets: {valid}") from exc


def discover_registered_dataset(dataset: str, root: str | Path | None = None) -> list[SegmentationSample]:
    spec = get_dataset_spec(dataset)
    return discover_segmentation_samples(
        image_dir=spec.image_dir(root),
        mask_dir=spec.mask_dir(root),
        dataset=spec.dataset,
    )


def discover_segmentation_samples(
    image_dir: str | Path,
    mask_dir: str | Path,
    dataset: str = "kvasir_seg",
) -> list[SegmentationSample]:
    image_root = Path(image_dir)
    mask_root = Path(mask_dir)
    if not image_root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_root}")
    if not mask_root.exists():
        raise FileNotFoundError(f"Mask directory does not exist: {mask_root}")

    images = _files_by_stem(image_root, IMAGE_EXTENSIONS)
    masks = _files_by_stem(mask_root, MASK_EXTENSIONS)

    missing_masks = sorted(set(images) - set(masks))
    orphan_masks = sorted(set(masks) - set(images))
    if missing_masks:
        example_stems = ", ".join(missing_masks[:5])
        raise ValueError(f"Found {len(missing_masks)} image(s) without matching masks: {example_stems}")
    if orphan_masks:
        example_stems = ", ".join(orphan_masks[:5])
        raise ValueError(f"Found {len(orphan_masks)} mask(s) without matching images: {example_stems}")

    samples = [
        SegmentationSample(
            sample_id=stem,
            image_path=str(images[stem]),
            mask_path=str(masks[stem]),
            dataset=dataset,
        )
        for stem in sorted(images)
    ]
    if not samples:
        raise ValueError(f"No image/mask pairs found in {image_root} and {mask_root}")
    return samples


def create_split_manifest(
    samples: Sequence[SegmentationSample],
    dataset: str = "kvasir_seg",
    seed: int = 1,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> SplitManifest:
    if not samples:
        raise ValueError("At least one sample is required to create splits.")
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")

    ordered = list(samples)
    rng = random.Random(seed)
    rng.shuffle(ordered)

    train_count, val_count, test_count = _split_counts(len(ordered), train_ratio, val_ratio)
    split_samples = {
        "train": ordered[:train_count],
        "validation": ordered[train_count : train_count + val_count],
        "test": ordered[train_count + val_count : train_count + val_count + test_count],
    }
    counts = {name: len(items) for name, items in split_samples.items()}
    return SplitManifest(
        dataset=dataset,
        seed=seed,
        split_ratios={"train": train_ratio, "validation": val_ratio, "test": test_ratio},
        counts=counts,
        samples=split_samples,
    )


def write_split_manifest(manifest: SplitManifest, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")


def read_split_manifest(path: str | Path) -> SplitManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = {
        split_name: [SegmentationSample(**sample) for sample in split_samples]
        for split_name, split_samples in payload["samples"].items()
    }
    return SplitManifest(
        dataset=payload["dataset"],
        seed=int(payload["seed"]),
        split_ratios={name: float(value) for name, value in payload["split_ratios"].items()},
        counts={name: int(value) for name, value in payload["counts"].items()},
        samples=samples,
    )

def load_rgb_image(path: str | Path, size: tuple[int, int] | None = None):
    Image = _import_pillow_image()
    image = Image.open(path).convert("RGB")
    if size:
        image = image.resize(size, Image.BILINEAR)
    return image


def load_binary_mask(path: str | Path, size: tuple[int, int] | None = None, threshold: int = 128) -> list[list[int]]:
    Image = _import_pillow_image()
    mask = Image.open(path).convert("L")
    if size:
        mask = mask.resize(size, Image.NEAREST)
    width, height = mask.size
    pixels = list(mask.getdata())
    return [
        [1 if pixels[y * width + x] >= threshold else 0 for x in range(width)]
        for y in range(height)
    ]


def mask_to_uint8_image(mask: Sequence[Sequence[int]]):
    Image = _import_pillow_image()
    rows = [list(row) for row in mask]
    if not rows or not rows[0]:
        raise ValueError("Mask must be non-empty.")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("All mask rows must have the same width.")
    data = [255 if value else 0 for row in rows for value in row]
    image = Image.new("L", (width, len(rows)))
    image.putdata(data)
    return image


def _files_by_stem(root: Path, extensions: set[str]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    duplicates = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if path.stem in files:
            duplicates.append(path.stem)
        files[path.stem] = path
    if duplicates:
        example_stems = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(f"Duplicate file stems found in {root}: {example_stems}")
    return files


def _split_counts(sample_count: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if sample_count == 1:
        return 1, 0, 0
    if sample_count == 2:
        return 1, 0, 1

    train_count = max(1, int(sample_count * train_ratio))
    val_count = max(1, int(sample_count * val_ratio))
    if train_count + val_count >= sample_count:
        val_count = 1
        train_count = sample_count - 2
    test_count = sample_count - train_count - val_count
    if test_count == 0:
        test_count = 1
        train_count -= 1
    return train_count, val_count, test_count


def _import_pillow_image():
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for image preprocessing. Install it with: pip install pillow") from exc
    return Image
