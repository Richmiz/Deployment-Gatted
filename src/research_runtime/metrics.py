from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence


EPS = 1e-12


def _flatten(values: Iterable[object]) -> list[float]:
    flattened: list[float] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            flattened.extend(_flatten(value))
        else:
            flattened.append(float(value))
    return flattened


def _flatten_selected(values: Iterable[object], sample_mask: Iterable[object] | None = None) -> list[float]:
    flattened = _flatten(values)
    if sample_mask is None:
        return flattened

    flattened_mask = [bool(value) for value in _flatten(sample_mask)]
    if len(flattened) != len(flattened_mask):
        raise ValueError("Values and sample mask must have the same number of elements.")
    return [value for value, keep in zip(flattened, flattened_mask) if keep]


def _as_2d(values: Iterable[Iterable[object]]) -> list[list[float]]:
    rows = [list(row) for row in values]
    if not rows or not rows[0]:
        raise ValueError("Expected a non-empty 2D array-like object.")

    width = len(rows[0])
    converted: list[list[float]] = []
    for row in rows:
        if len(row) != width:
            raise ValueError("All rows must have the same width.")
        converted.append([float(value) for value in row])
    return converted


def _binary_labels(values: Iterable[object], threshold: float = 0.5) -> list[int]:
    return [1 if value >= threshold else 0 for value in _flatten(values)]


def binary_confusion_counts(
    probabilities: Iterable[object],
    targets: Iterable[object],
    threshold: float = 0.5,
) -> dict[str, int]:
    preds = _binary_labels(probabilities, threshold)
    labels = _binary_labels(targets, threshold)
    if len(preds) != len(labels):
        raise ValueError("Predictions and targets must have the same number of elements.")

    tp = sum(1 for pred, label in zip(preds, labels) if pred == 1 and label == 1)
    fp = sum(1 for pred, label in zip(preds, labels) if pred == 1 and label == 0)
    tn = sum(1 for pred, label in zip(preds, labels) if pred == 0 and label == 0)
    fn = sum(1 for pred, label in zip(preds, labels) if pred == 0 and label == 1)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def dice_score(probabilities: Iterable[object], targets: Iterable[object], threshold: float = 0.5) -> float:
    counts = binary_confusion_counts(probabilities, targets, threshold)
    numerator = 2 * counts["tp"]
    denominator = 2 * counts["tp"] + counts["fp"] + counts["fn"]
    return 1.0 if denominator == 0 else numerator / denominator


def iou_score(probabilities: Iterable[object], targets: Iterable[object], threshold: float = 0.5) -> float:
    counts = binary_confusion_counts(probabilities, targets, threshold)
    denominator = counts["tp"] + counts["fp"] + counts["fn"]
    return 1.0 if denominator == 0 else counts["tp"] / denominator


def precision_score(probabilities: Iterable[object], targets: Iterable[object], threshold: float = 0.5) -> float:
    counts = binary_confusion_counts(probabilities, targets, threshold)
    denominator = counts["tp"] + counts["fp"]
    if denominator == 0:
        return 1.0 if counts["fn"] == 0 else 0.0
    return counts["tp"] / denominator


def recall_score(probabilities: Iterable[object], targets: Iterable[object], threshold: float = 0.5) -> float:
    counts = binary_confusion_counts(probabilities, targets, threshold)
    denominator = counts["tp"] + counts["fn"]
    return 1.0 if denominator == 0 else counts["tp"] / denominator


def binary_brier_score(
    probabilities: Iterable[object],
    targets: Iterable[object],
    sample_mask: Iterable[object] | None = None,
) -> float:
    probs = _flatten_selected(probabilities, sample_mask)
    labels = [1 if value >= 0.5 else 0 for value in _flatten_selected(targets, sample_mask)]
    if len(probs) != len(labels):
        raise ValueError("Predictions and targets must have the same number of elements.")
    return sum((prob - label) ** 2 for prob, label in zip(probs, labels)) / max(len(labels), 1)


def binary_ece(
    probabilities: Iterable[object],
    targets: Iterable[object],
    n_bins: int = 10,
    sample_mask: Iterable[object] | None = None,
) -> float:
    probs = _flatten_selected(probabilities, sample_mask)
    labels = [1 if value >= 0.5 else 0 for value in _flatten_selected(targets, sample_mask)]
    if len(probs) != len(labels):
        raise ValueError("Predictions and targets must have the same number of elements.")

    predicted_labels = [1 if prob >= 0.5 else 0 for prob in probs]
    confidences = [prob if pred == 1 else 1.0 - prob for prob, pred in zip(probs, predicted_labels)]
    correctness = [1.0 if pred == label else 0.0 for pred, label in zip(predicted_labels, labels)]
    return _ece_from_confidence(confidences, correctness, n_bins)


def foreground_mask(targets: Iterable[Iterable[object]], threshold: float = 0.5) -> list[list[bool]]:
    target_rows = _as_2d(targets)
    return [[value >= threshold for value in row] for row in target_rows]


def roi_mask_from_binary_masks(
    probabilities: Iterable[Iterable[object]],
    targets: Iterable[Iterable[object]],
    threshold: float = 0.5,
    padding: int = 0,
) -> list[list[bool]]:
    if padding < 0:
        raise ValueError("padding must be non-negative.")

    prob_rows = _as_2d(probabilities)
    target_rows = _as_2d(targets)
    _validate_same_2d_shape(prob_rows, target_rows)

    height = len(target_rows)
    width = len(target_rows[0])
    foreground_points = []
    for y in range(height):
        for x in range(width):
            if target_rows[y][x] >= threshold or prob_rows[y][x] >= threshold:
                foreground_points.append((x, y))

    if not foreground_points:
        return [[True for _ in range(width)] for _ in range(height)]

    x_values = [point[0] for point in foreground_points]
    y_values = [point[1] for point in foreground_points]
    x_min = max(0, min(x_values) - padding)
    y_min = max(0, min(y_values) - padding)
    x_max = min(width - 1, max(x_values) + padding)
    y_max = min(height - 1, max(y_values) + padding)

    return [
        [x_min <= x <= x_max and y_min <= y <= y_max for x in range(width)]
        for y in range(height)
    ]


def boundary_mask_from_binary_masks(
    probabilities: Iterable[Iterable[object]],
    targets: Iterable[Iterable[object]],
    threshold: float = 0.5,
    radius: int = 1,
) -> list[list[bool]]:
    if radius < 0:
        raise ValueError("radius must be non-negative.")

    prob_rows = _as_2d(probabilities)
    target_rows = _as_2d(targets)
    _validate_same_2d_shape(prob_rows, target_rows)

    height = len(target_rows)
    width = len(target_rows[0])
    combined = [
        [target_rows[y][x] >= threshold or prob_rows[y][x] >= threshold for x in range(width)]
        for y in range(height)
    ]

    edge_points = []
    for y in range(height):
        for x in range(width):
            current = combined[y][x]
            for nx, ny in _neighbors(x, y, width, height):
                if combined[ny][nx] != current:
                    edge_points.append((x, y))
                    break

    if not edge_points:
        return roi_mask_from_binary_masks(prob_rows, target_rows, threshold=threshold, padding=radius)

    boundary = [[False for _ in range(width)] for _ in range(height)]
    for x, y in edge_points:
        for ny in range(max(0, y - radius), min(height - 1, y + radius) + 1):
            for nx in range(max(0, x - radius), min(width - 1, x + radius) + 1):
                boundary[ny][nx] = True
    return boundary


def softmax(logits: Sequence[float]) -> list[float]:
    if not logits:
        raise ValueError("Logits cannot be empty.")
    max_logit = max(logits)
    exp_values = [math.exp(logit - max_logit) for logit in logits]
    denominator = sum(exp_values)
    return [value / denominator for value in exp_values]


def classification_metrics(logits: Sequence[Sequence[float]], labels: Sequence[int], n_bins: int = 10) -> dict[str, float]:
    if len(logits) != len(labels):
        raise ValueError("Logits and labels must have the same number of samples.")
    if not logits:
        raise ValueError("At least one sample is required.")

    probabilities = [softmax(row) for row in logits]
    predictions = [max(range(len(row)), key=row.__getitem__) for row in probabilities]
    num_classes = max(max(labels), max(predictions)) + 1

    accuracy = sum(1 for pred, label in zip(predictions, labels) if pred == label) / len(labels)
    nll = -sum(math.log(max(probabilities[i][labels[i]], EPS)) for i in range(len(labels))) / len(labels)
    brier = _multiclass_brier_score(probabilities, labels, num_classes)
    macro_f1 = _macro_f1(predictions, labels, num_classes)
    confidences = [max(row) for row in probabilities]
    correctness = [1.0 if pred == label else 0.0 for pred, label in zip(predictions, labels)]
    ece = _ece_from_confidence(confidences, correctness, n_bins)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "ece": ece,
        "nll": nll,
        "brier": brier,
    }


def topk_label_purity(retrieved_labels: Sequence[Sequence[int]], true_labels: Sequence[int]) -> float:
    if len(retrieved_labels) != len(true_labels):
        raise ValueError("Retrieved labels and true labels must have the same number of samples.")
    if not retrieved_labels:
        raise ValueError("At least one retrieval row is required.")

    purities = []
    for neighbors, true_label in zip(retrieved_labels, true_labels):
        if not neighbors:
            raise ValueError("Each retrieval row must contain at least one neighbor.")
        purities.append(sum(1 for label in neighbors if label == true_label) / len(neighbors))
    return sum(purities) / len(purities)


def _macro_f1(predictions: Sequence[int], labels: Sequence[int], num_classes: int) -> float:
    scores = []
    for class_id in range(num_classes):
        tp = sum(1 for pred, label in zip(predictions, labels) if pred == class_id and label == class_id)
        fp = sum(1 for pred, label in zip(predictions, labels) if pred == class_id and label != class_id)
        fn = sum(1 for pred, label in zip(predictions, labels) if pred != class_id and label == class_id)
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return sum(scores) / num_classes


def _multiclass_brier_score(probabilities: Sequence[Sequence[float]], labels: Sequence[int], num_classes: int) -> float:
    total = 0.0
    for row, label in zip(probabilities, labels):
        for class_id in range(num_classes):
            expected = 1.0 if class_id == label else 0.0
            total += (row[class_id] - expected) ** 2
    return total / len(labels)


def _validate_same_2d_shape(left: Sequence[Sequence[object]], right: Sequence[Sequence[object]]) -> None:
    if len(left) != len(right):
        raise ValueError("2D arrays must have the same height.")
    for left_row, right_row in zip(left, right):
        if len(left_row) != len(right_row):
            raise ValueError("2D arrays must have the same width.")


def _neighbors(x: int, y: int, width: int, height: int) -> list[tuple[int, int]]:
    points = []
    for offset_y in (-1, 0, 1):
        for offset_x in (-1, 0, 1):
            if offset_x == 0 and offset_y == 0:
                continue
            nx = x + offset_x
            ny = y + offset_y
            if 0 <= nx < width and 0 <= ny < height:
                points.append((nx, ny))
    return points


def _ece_from_confidence(confidences: Sequence[float], correctness: Sequence[float], n_bins: int) -> float:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    if len(confidences) != len(correctness):
        raise ValueError("Confidences and correctness must have the same number of samples.")

    total = len(confidences)
    if total == 0:
        return 0.0

    ece = 0.0
    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        in_bin = [
            idx
            for idx, confidence in enumerate(confidences)
            if confidence >= lower and (confidence < upper or (bin_index == n_bins - 1 and confidence <= upper))
        ]
        if not in_bin:
            continue
        bin_confidence = sum(confidences[idx] for idx in in_bin) / len(in_bin)
        bin_accuracy = sum(correctness[idx] for idx in in_bin) / len(in_bin)
        ece += (len(in_bin) / total) * abs(bin_confidence - bin_accuracy)
    return ece


def label_distribution(labels: Sequence[int]) -> dict[int, int]:
    return dict(Counter(labels))
