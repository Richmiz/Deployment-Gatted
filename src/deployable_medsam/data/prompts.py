from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class BoxPrompt:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)


def mask_to_box_prompt(mask: Iterable[Iterable[float]], threshold: float = 0.5, padding: int = 0) -> BoxPrompt:
    rows = [list(row) for row in mask]
    if not rows or not rows[0]:
        raise ValueError("Mask must be a non-empty 2D array-like object.")

    height = len(rows)
    width = len(rows[0])
    foreground = []
    for y, row in enumerate(rows):
        if len(row) != width:
            raise ValueError("Mask rows must all have the same width.")
        for x, value in enumerate(row):
            if float(value) >= threshold:
                foreground.append((x, y))

    if not foreground:
        raise ValueError("Cannot create a box prompt from an empty foreground mask.")

    x_values = [point[0] for point in foreground]
    y_values = [point[1] for point in foreground]
    return _clamped_box(
        x_min=min(x_values) - padding,
        y_min=min(y_values) - padding,
        x_max=max(x_values) + padding,
        y_max=max(y_values) + padding,
        width=width,
        height=height,
    )


def generate_noisy_box_prompt(
    mask: Iterable[Iterable[float]],
    jitter_pixels: int = 10,
    seed: int | None = None,
    threshold: float = 0.5,
    padding: int = 0,
) -> BoxPrompt:
    if jitter_pixels < 0:
        raise ValueError("jitter_pixels must be non-negative.")

    rows = [list(row) for row in mask]
    base_box = mask_to_box_prompt(rows, threshold=threshold, padding=padding)
    height = len(rows)
    width = len(rows[0])
    if jitter_pixels == 0:
        return base_box

    rng = random.Random(seed)
    x_min = base_box.x_min + rng.randint(-jitter_pixels, jitter_pixels)
    y_min = base_box.y_min + rng.randint(-jitter_pixels, jitter_pixels)
    x_max = base_box.x_max + rng.randint(-jitter_pixels, jitter_pixels)
    y_max = base_box.y_max + rng.randint(-jitter_pixels, jitter_pixels)

    if x_min > x_max:
        x_min, x_max = x_max, x_min
    if y_min > y_max:
        y_min, y_max = y_max, y_min

    return _clamped_box(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, width=width, height=height)


def _clamped_box(x_min: int, y_min: int, x_max: int, y_max: int, width: int, height: int) -> BoxPrompt:
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")

    clamped_x_min = max(0, min(width - 1, x_min))
    clamped_y_min = max(0, min(height - 1, y_min))
    clamped_x_max = max(0, min(width - 1, x_max))
    clamped_y_max = max(0, min(height - 1, y_max))

    if clamped_x_min > clamped_x_max:
        clamped_x_min, clamped_x_max = clamped_x_max, clamped_x_min
    if clamped_y_min > clamped_y_max:
        clamped_y_min, clamped_y_max = clamped_y_max, clamped_y_min

    return BoxPrompt(
        x_min=clamped_x_min,
        y_min=clamped_y_min,
        x_max=clamped_x_max,
        y_max=clamped_y_max,
    )
