from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from deployable_medsam.data import load_binary_mask
from deployable_medsam.distillation import read_jsonl_manifest
from deployable_medsam.distillation.artifacts import resolve_project_path
from research_runtime.io import write_csv_rows


def threshold_sweep_from_prediction_csv(
    *,
    prediction_csv: str | Path,
    manifest_path: str | Path,
    project_root: str | Path,
    thresholds: Sequence[float],
    output_path: str | Path | None = None,
    label: str | None = None,
) -> list[dict[str, object]]:
    root = Path(project_root).resolve()
    rows = pd.read_csv(prediction_csv)
    if "prediction_path" not in rows.columns:
        raise ValueError(f"Prediction CSV does not include prediction_path: {prediction_csv}")
    records = {record.sample_id: record for record in read_jsonl_manifest(manifest_path)}
    probabilities = []
    targets = []
    for _, row in rows.iterrows():
        sample_id = str(row["sample_id"])
        if sample_id not in records:
            raise KeyError(f"Sample ID from prediction CSV is missing from manifest: {sample_id}")
        record = records[sample_id]
        prediction_path = resolve_project_path(row["prediction_path"], root)
        with np.load(prediction_path) as artifact:
            probabilities.append(artifact["probabilities"].astype(np.float32))
        targets.append(
            np.asarray(
                load_binary_mask(
                    resolve_project_path(record.mask_path, root),
                    size=(record.input_size, record.input_size),
                ),
                dtype=np.bool_,
            )
        )
    probabilities_array = np.stack(probabilities)
    targets_array = np.stack(targets)
    sweep_rows = threshold_sweep_from_arrays(
        probabilities_array,
        targets_array,
        thresholds=thresholds,
        label=label,
    )
    if output_path is not None:
        write_csv_rows(output_path, sweep_rows)
    return sweep_rows


def threshold_sweep_from_arrays(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    thresholds: Sequence[float],
    label: str | None = None,
) -> list[dict[str, object]]:
    if probabilities.shape != targets.shape:
        raise ValueError("probabilities and targets must have the same shape.")
    if probabilities.ndim != 3:
        raise ValueError("Expected arrays with shape [N, H, W].")
    target_bool = targets.astype(bool)
    sweep_rows: list[dict[str, object]] = []
    for threshold in thresholds:
        pred_bool = probabilities >= float(threshold)
        tp = np.logical_and(pred_bool, target_bool).sum(axis=(1, 2)).astype(np.float64)
        fp = np.logical_and(pred_bool, ~target_bool).sum(axis=(1, 2)).astype(np.float64)
        fn = np.logical_and(~pred_bool, target_bool).sum(axis=(1, 2)).astype(np.float64)
        dice_den = 2 * tp + fp + fn
        iou_den = tp + fp + fn
        precision_den = tp + fp
        recall_den = tp + fn
        dice = np.ones_like(tp, dtype=np.float64)
        np.divide(2 * tp, dice_den, out=dice, where=dice_den != 0)
        iou = np.ones_like(tp, dtype=np.float64)
        np.divide(tp, iou_den, out=iou, where=iou_den != 0)
        precision = np.where(fn == 0, 1.0, 0.0).astype(np.float64)
        np.divide(tp, precision_den, out=precision, where=precision_den != 0)
        recall = np.ones_like(tp, dtype=np.float64)
        np.divide(tp, recall_den, out=recall, where=recall_den != 0)
        row: dict[str, object] = {
            "threshold": round(float(threshold), 6),
            "sample_count": int(probabilities.shape[0]),
            "mean_dice": _rounded(np.mean(dice)),
            "median_dice": _rounded(np.median(dice)),
            "mean_iou": _rounded(np.mean(iou)),
            "mean_precision": _rounded(np.mean(precision)),
            "mean_recall": _rounded(np.mean(recall)),
            "samples_dice_below_0_5": int((dice < 0.5).sum()),
            "samples_dice_below_0_3": int((dice < 0.3).sum()),
        }
        if label is not None:
            row = {"label": label, **row}
        sweep_rows.append(row)
    return sorted(sweep_rows, key=lambda row: (float(row["mean_dice"]), float(row["mean_iou"])), reverse=True)


def parse_thresholds(values: Iterable[str | float]) -> list[float]:
    parsed = []
    for value in values:
        if isinstance(value, str) and ":" in value:
            start_text, stop_text, step_text = value.split(":")
            start = float(start_text)
            stop = float(stop_text)
            step = float(step_text)
            current = start
            while current <= stop + 1e-12:
                parsed.append(round(current, 6))
                current += step
        else:
            parsed.append(round(float(value), 6))
    return parsed


def _rounded(value: float, digits: int = 6) -> float:
    return round(float(value), digits)
