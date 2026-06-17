from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


EXCLUSION_RECIPES = (
    "none",
    "out_head_fp32",
    "last_decoder_fp32",
    "decoder_fp32",
    "first_last_fp32",
)


def inspect_onnx_nodes(onnx_path: str | Path) -> list[dict[str, object]]:
    """Return a lightweight ONNX node inventory with U-Net block hints."""

    try:
        import onnx
    except ImportError as exc:
        raise ImportError("ONNX is required to inspect quantization nodes.") from exc

    model = onnx.load(str(onnx_path))
    conv_total = sum(1 for node in model.graph.node if node.op_type == "Conv")
    conv_index = 0
    rows: list[dict[str, object]] = []
    for index, node in enumerate(model.graph.node):
        this_conv_index: int | None = None
        if node.op_type == "Conv":
            this_conv_index = conv_index
            conv_index += 1
        block_hint = infer_unet_block_hint(
            node_name=node.name,
            op_type=node.op_type,
            conv_index=this_conv_index,
            conv_total=conv_total,
        )
        rows.append(
            {
                "node_index": index,
                "node_name": node.name,
                "node_display_name": node.name or f"node_{index}_{node.op_type}",
                "op_type": node.op_type,
                "conv_index": "" if this_conv_index is None else this_conv_index,
                "block_hint": block_hint,
                "input_names": "|".join(node.input),
                "output_names": "|".join(node.output),
            }
        )
    memberships = _recipe_memberships(rows)
    for row in rows:
        row["exclusion_recipes"] = ",".join(memberships.get(str(row["node_name"]), []))
    return rows


def resolve_exclusion_recipe(onnx_path: str | Path, recipes: Sequence[str] | str | None) -> list[str]:
    rows = inspect_onnx_nodes(onnx_path)
    return resolve_exclusion_recipe_from_rows(rows, recipes)


def resolve_exclusion_recipe_from_rows(
    rows: Sequence[dict[str, object]],
    recipes: Sequence[str] | str | None,
) -> list[str]:
    normalized = normalize_recipe_names(recipes)
    selected: list[str] = []
    for recipe in normalized:
        if recipe == "none":
            continue
        selected.extend(_nodes_for_recipe(rows, recipe))
    return _dedupe_keep_order(name for name in selected if name)


def normalize_recipe_names(recipes: Sequence[str] | str | None) -> list[str]:
    if recipes is None:
        return ["none"]
    if isinstance(recipes, str):
        values = [recipes]
    else:
        values = list(recipes)
    normalized = [value.strip().lower() for value in values if value and value.strip()]
    if not normalized:
        return ["none"]
    unknown = sorted(set(normalized) - set(EXCLUSION_RECIPES))
    if unknown:
        valid = ", ".join(EXCLUSION_RECIPES)
        raise ValueError(f"Unknown exclusion recipe(s): {', '.join(unknown)}. Valid recipes: {valid}")
    if "none" in normalized and len(normalized) > 1:
        normalized = [value for value in normalized if value != "none"]
    return _dedupe_keep_order(normalized)


def infer_unet_block_hint(
    *,
    node_name: str,
    op_type: str,
    conv_index: int | None,
    conv_total: int,
) -> str:
    normalized = node_name.replace("\\", "/").lower()
    for hint in ("out_conv", "up3", "up2", "up1", "down3", "down2", "down1", "in_conv"):
        if hint in normalized:
            return hint
    if op_type != "Conv" or conv_index is None:
        return "unknown"
    if conv_index == conv_total - 1:
        return "out_conv"
    if conv_index in {0, 1}:
        return "in_conv"
    if conv_index in {2, 3}:
        return "down1"
    if conv_index in {4, 5}:
        return "down2"
    if conv_index in {6, 7}:
        return "down3"
    if conv_index in {8, 9}:
        return "up1"
    if conv_index in {10, 11}:
        return "up2"
    if conv_index in {12, 13}:
        return "up3"
    return "unknown"


def _nodes_for_recipe(rows: Sequence[dict[str, object]], recipe: str) -> list[str]:
    conv_rows = [row for row in rows if row.get("op_type") == "Conv"]
    if recipe == "out_head_fp32":
        return _conv_names_with_hints(rows, {"out_conv"}) or _last_conv_names(conv_rows, 1)
    if recipe == "last_decoder_fp32":
        return _conv_names_with_hints(rows, {"up3", "out_conv"}) or _last_conv_names(conv_rows, 3)
    if recipe == "decoder_fp32":
        return _conv_names_with_hints(rows, {"up1", "up2", "up3", "out_conv"}) or _last_conv_names(conv_rows, 7)
    if recipe == "first_last_fp32":
        return (
            _conv_names_with_hints(rows, {"in_conv", "out_conv"})
            or _first_conv_names(conv_rows, 2) + _last_conv_names(conv_rows, 1)
        )
    if recipe == "none":
        return []
    raise ValueError(f"Unsupported exclusion recipe: {recipe}")


def _conv_names_with_hints(rows: Sequence[dict[str, object]], hints: set[str]) -> list[str]:
    return [
        str(row["node_name"])
        for row in rows
        if row.get("op_type") == "Conv" and row.get("block_hint") in hints and str(row.get("node_name", ""))
    ]


def _first_conv_names(conv_rows: Sequence[dict[str, object]], count: int) -> list[str]:
    return [str(row["node_name"]) for row in conv_rows[:count] if str(row.get("node_name", ""))]


def _last_conv_names(conv_rows: Sequence[dict[str, object]], count: int) -> list[str]:
    return [str(row["node_name"]) for row in conv_rows[-count:] if str(row.get("node_name", ""))]


def _recipe_memberships(rows: Sequence[dict[str, object]]) -> dict[str, list[str]]:
    memberships: dict[str, list[str]] = {}
    for recipe in EXCLUSION_RECIPES:
        if recipe == "none":
            continue
        for node_name in _nodes_for_recipe(rows, recipe):
            memberships.setdefault(node_name, []).append(recipe)
    return memberships


def _dedupe_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
