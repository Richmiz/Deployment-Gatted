from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

try:
    from onnxruntime.quantization import CalibrationDataReader as _CalibrationDataReader
except ImportError:  # ONNX Runtime is optional until Stage 7 is executed.
    class _CalibrationDataReader:  # type: ignore[no-redef]
        pass

from deployable_medsam.student import LightweightUNet, StudentSegmentationDataset
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


@dataclass(frozen=True)
class OnnxExportMetadata:
    checkpoint_path: str
    onnx_path: str
    input_size: int
    base_channels: int
    opset_version: int
    max_abs_diff: float | None
    mean_abs_diff: float | None


@dataclass(frozen=True)
class QuantizationMetadata:
    fp32_onnx_path: str
    int8_onnx_path: str
    calibration_manifest: str
    calibration_samples: int
    requested_calibration_method: str
    calibration_method: str
    quant_format: str
    activation_type: str
    weight_type: str
    op_types_to_quantize: list[str]
    fp32_model_size_mb: float
    int8_model_size_mb: float
    compression_ratio_vs_fp32: float
    quantization_input_path: str
    preprocessing_applied: bool
    preprocessing_status: str
    nodes_to_exclude: list[str]


class StudentCalibrationDataReader(_CalibrationDataReader):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        project_root: str | Path,
        input_name: str,
        batch_size: int = 1,
        sample_limit: int = 128,
        num_workers: int = 0,
    ) -> None:
        dataset = StudentSegmentationDataset(manifest_path, project_root=project_root)
        if sample_limit > 0:
            dataset = Subset(dataset, list(range(min(sample_limit, len(dataset)))))
        self.input_name = input_name
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.iterator = iter(self.dataloader)

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            batch = next(self.iterator)
        except StopIteration:
            return None
        images = batch["image"].numpy().astype(np.float32)
        return {self.input_name: images}

    def rewind(self) -> None:
        self.iterator = iter(self.dataloader)


def load_lightweight_unet_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    base_channels: int = 16,
    device: str | torch.device = "cpu",
) -> LightweightUNet:
    target_device = torch.device(device)
    model = LightweightUNet(base_channels=base_channels).to(target_device)
    checkpoint = torch.load(checkpoint_path, map_location=target_device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_student_to_onnx(
    *,
    checkpoint_path: str | Path,
    output_path: str | Path,
    input_size: int = 256,
    base_channels: int = 16,
    opset_version: int = 17,
    validate_parity: bool = True,
) -> OnnxExportMetadata:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = load_lightweight_unet_from_checkpoint(checkpoint_path, base_channels=base_channels, device="cpu")
    dummy_input = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)
    _export_model_to_onnx(
        model=model,
        dummy_input=dummy_input,
        output=output,
        opset_version=opset_version,
    )

    max_abs_diff = None
    mean_abs_diff = None
    if validate_parity:
        max_abs_diff, mean_abs_diff = validate_onnx_parity(model, output, dummy_input)

    metadata = OnnxExportMetadata(
        checkpoint_path=_as_posix(checkpoint_path),
        onnx_path=_as_posix(output),
        input_size=input_size,
        base_channels=base_channels,
        opset_version=opset_version,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
    )
    write_json(output.with_suffix(".export.json"), asdict(metadata))
    return metadata


def _export_model_to_onnx(
    *,
    model: LightweightUNet,
    dummy_input: torch.Tensor,
    output: Path,
    opset_version: int,
) -> None:
    export_kwargs = {
        "export_params": True,
        "opset_version": opset_version,
        "do_constant_folding": True,
        "input_names": ["image"],
        "output_names": ["logits"],
        "dynamic_axes": {"image": {0: "batch"}, "logits": {0: "batch"}},
    }
    try:
        # Keep Stage 7 stable across PyTorch 2.9+ environments where the default
        # dynamo exporter requires onnxscript. This U-Net exports cleanly through
        # the legacy-compatible path and is validated immediately with ONNX Runtime.
        torch.onnx.export(model, dummy_input, output, dynamo=False, **export_kwargs)
    except TypeError as exc:
        if "dynamo" not in str(exc):
            raise
        torch.onnx.export(model, dummy_input, output, **export_kwargs)


def validate_onnx_parity(
    model: LightweightUNet,
    onnx_path: str | Path,
    input_tensor: torch.Tensor,
) -> tuple[float, float]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("ONNX Runtime is required to validate ONNX parity.") from exc

    with torch.no_grad():
        torch_logits = model(input_tensor).detach().cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_logits = session.run(None, {input_name: input_tensor.numpy().astype(np.float32)})[0]
    diff = np.abs(torch_logits - onnx_logits)
    return float(diff.max()), float(diff.mean())


def quantize_student_onnx_static(
    *,
    fp32_onnx_path: str | Path,
    int8_onnx_path: str | Path,
    calibration_manifest: str | Path,
    project_root: str | Path,
    calibration_samples: int = 128,
    calibration_batch_size: int = 1,
    requested_calibration_method: str = "percentile",
    op_types_to_quantize: Sequence[str] = ("Conv",),
    nodes_to_exclude: Sequence[str] = (),
    preprocess: bool = False,
    preprocessed_onnx_path: str | Path | None = None,
) -> QuantizationMetadata:
    try:
        from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    except ImportError as exc:
        raise ImportError("ONNX Runtime quantization is required. Install requirements-experiments.txt in WSL.") from exc

    fp32_path = Path(fp32_onnx_path)
    int8_path = Path(int8_onnx_path)
    int8_path.parent.mkdir(parents=True, exist_ok=True)
    quantization_input_path = fp32_path
    preprocessing_applied = False
    preprocessing_status = "not_requested"
    if preprocess:
        preprocessed_path = Path(preprocessed_onnx_path) if preprocessed_onnx_path is not None else int8_path.with_name(f"{int8_path.stem}.preprocessed.onnx")
        preprocess_onnx_for_quantization(fp32_path, preprocessed_path)
        quantization_input_path = preprocessed_path
        preprocessing_applied = True
        preprocessing_status = "applied"

    method, method_name, extra_options = _choose_calibration_method(CalibrationMethod, requested_calibration_method)
    reader = StudentCalibrationDataReader(
        calibration_manifest,
        project_root=project_root,
        input_name="image",
        batch_size=calibration_batch_size,
        sample_limit=calibration_samples,
    )

    excluded_nodes = list(nodes_to_exclude)
    try:
        quantize_static(
            model_input=str(quantization_input_path),
            model_output=str(int8_path),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=method,
            op_types_to_quantize=list(op_types_to_quantize),
            nodes_to_exclude=excluded_nodes,
            extra_options=extra_options,
        )
    except Exception:
        if method_name != "MinMax":
            reader.rewind()
            method = CalibrationMethod.MinMax
            method_name = "MinMax"
            quantize_static(
                model_input=str(quantization_input_path),
                model_output=str(int8_path),
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                calibrate_method=method,
                op_types_to_quantize=list(op_types_to_quantize),
                nodes_to_exclude=excluded_nodes,
            )
        else:
            raise

    fp32_size = file_size_mb(fp32_path)
    int8_size = file_size_mb(int8_path)
    metadata = QuantizationMetadata(
        fp32_onnx_path=_as_posix(fp32_path),
        int8_onnx_path=_as_posix(int8_path),
        calibration_manifest=_as_posix(calibration_manifest),
        calibration_samples=calibration_samples,
        requested_calibration_method=requested_calibration_method,
        calibration_method=method_name,
        quant_format="QDQ",
        activation_type="QUInt8",
        weight_type="QInt8",
        op_types_to_quantize=list(op_types_to_quantize),
        fp32_model_size_mb=fp32_size,
        int8_model_size_mb=int8_size,
        compression_ratio_vs_fp32=round(fp32_size / int8_size, 6) if int8_size > 0 else 0.0,
        quantization_input_path=_as_posix(quantization_input_path),
        preprocessing_applied=preprocessing_applied,
        preprocessing_status=preprocessing_status,
        nodes_to_exclude=excluded_nodes,
    )
    write_json(int8_path.with_suffix(".quantization.json"), asdict(metadata))
    return metadata


def preprocess_onnx_for_quantization(
    input_path: str | Path,
    output_path: str | Path,
) -> Path:
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError:
        try:
            from onnxruntime.quantization import quant_pre_process
        except ImportError as exc:
            raise ImportError("ONNX Runtime quantization preprocessing is not available in this onnxruntime version.") from exc

    source = Path(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    quant_pre_process(str(source), str(output))
    return output


def evaluate_onnx_student_model(
    *,
    onnx_path: str | Path,
    manifest_path: str | Path,
    project_root: str | Path,
    split: str,
    model_id: str,
    batch_size: int = 8,
    threshold: float = 0.5,
    prediction_output_dir: str | Path | None = None,
    execution_provider: str = "CPUExecutionProvider",
    model_metadata: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("ONNX Runtime is required to evaluate ONNX student models.") from exc

    available = ort.get_available_providers()
    provider = execution_provider if execution_provider in available else "CPUExecutionProvider"
    session = ort.InferenceSession(str(onnx_path), providers=[provider])
    input_name = session.get_inputs()[0].name
    dataset = StudentSegmentationDataset(manifest_path, project_root=project_root)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    prediction_dir = Path(prediction_output_dir) if prediction_output_dir is not None else None
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    root = Path(project_root).resolve()
    metadata = model_metadata or {}
    rows: list[dict[str, object]] = []

    for batch in dataloader:
        images = batch["image"].numpy().astype(np.float32)
        start = time.perf_counter()
        logits = session.run(None, {input_name: images})[0]
        latency_ms = (time.perf_counter() - start) * 1000.0 / max(images.shape[0], 1)
        probabilities_batch = sigmoid(logits).astype(np.float32)
        targets = batch["ground_truth_mask"].numpy().astype(np.float32)

        for index in range(images.shape[0]):
            probability_array = probabilities_batch[index, 0]
            binary_array = (probability_array >= threshold).astype(np.uint8)
            target_array = targets[index, 0]
            probability = probability_array.tolist()
            target = target_array.tolist()
            foreground = foreground_mask(target, threshold=threshold)
            roi = roi_mask_from_binary_masks(probability, target, threshold=threshold, padding=10)
            boundary = boundary_mask_from_binary_masks(probability, target, threshold=threshold, radius=2)
            sample_id = str(batch["sample_id"][index])
            row: dict[str, object] = {
                "sample_id": sample_id,
                "split": split,
                "model_id": model_id,
                "dice": rounded(dice_score(probability, target, threshold=threshold)),
                "iou": rounded(iou_score(probability, target, threshold=threshold)),
                "precision": rounded(precision_score(probability, target, threshold=threshold)),
                "recall": rounded(recall_score(probability, target, threshold=threshold)),
                "brier_full": rounded(binary_brier_score(probability, target)),
                "brier_roi": rounded(binary_brier_score(probability, target, sample_mask=roi)),
                "binary_ece_full": rounded(binary_ece(probability, target, n_bins=10)),
                "binary_ece_foreground": rounded(binary_ece(probability, target, n_bins=10, sample_mask=foreground)),
                "binary_ece_roi": rounded(binary_ece(probability, target, n_bins=10, sample_mask=roi)),
                "binary_ece_boundary": rounded(binary_ece(probability, target, n_bins=10, sample_mask=boundary)),
                "latency_ms": rounded(latency_ms, digits=3),
                "device": f"onnxruntime:{provider}",
            }
            row.update(metadata)
            if prediction_dir is not None:
                prediction_path = prediction_dir / f"{safe_filename(sample_id)}.npz"
                np.savez_compressed(
                    prediction_path,
                    probabilities=probability_array,
                    binary_mask=binary_array,
                    sample_id=sample_id,
                    split=split,
                    model_name=model_id,
                    threshold=np.float32(threshold),
                )
                row["prediction_path"] = relative_or_posix(prediction_path, root)
            rows.append(row)
    return rows


def summarize_onnx_rows(rows: Sequence[dict[str, object]], metadata_fields: Sequence[str] = ()) -> list[dict[str, object]]:
    if not rows:
        raise ValueError("At least one ONNX metric row is required.")
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
    summary: dict[str, object] = {
        "split": rows[0]["split"],
        "model_id": rows[0]["model_id"],
        "device": rows[0]["device"],
        "sample_count": len(rows),
    }
    for field in metric_fields:
        values = [float(row[field]) for row in rows]
        summary[f"mean_{field}"] = rounded(sum(values) / len(values), digits=6)
    for field in metadata_fields:
        if field in rows[0]:
            summary[field] = rows[0][field]
    return [summary]


def _choose_calibration_method(calibration_method_cls, requested: str):
    requested_normalized = requested.strip().lower()
    if requested_normalized == "percentile" and hasattr(calibration_method_cls, "Percentile"):
        return calibration_method_cls.Percentile, "Percentile", {"CalibPercentile": 99.99}
    return calibration_method_cls.MinMax, "MinMax", {}


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def rounded(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def safe_filename(value: str) -> str:
    import re

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._") or "sample"


def file_size_mb(path: str | Path) -> float:
    return rounded(Path(path).stat().st_size / (1024 * 1024), digits=6)


def relative_or_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _as_posix(path: str | Path) -> str:
    return Path(path).as_posix()
