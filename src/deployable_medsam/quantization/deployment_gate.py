from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DeploymentThresholds:
    """Maximum tolerated degradation from the FP32 reference to the INT8 model.

    Mirrors ``quantization.success_thresholds`` in ``configs/deployable_medsam.json``
    and the Stage 5 failure criterion in the study configuration: an INT8 model
    that loses more than ~1-2 Dice points (or sharply worsens calibration) must
    not be reported as a successful deployment.
    """

    max_dice_drop: float = 0.02
    max_iou_drop: float = 0.02
    max_ece_roi_increase: float = 0.02


@dataclass(frozen=True)
class GateCheck:
    metric: str
    fp32_value: float
    int8_value: float
    delta: float  # degradation magnitude (drop for quality, increase for error)
    limit: float
    passed: bool


@dataclass(frozen=True)
class DeploymentGateResult:
    passed: bool
    checks: list[GateCheck] = field(default_factory=list)

    def to_rows(self) -> list[dict[str, object]]:
        return [
            {
                "metric": check.metric,
                "fp32_value": round(check.fp32_value, 6),
                "int8_value": round(check.int8_value, 6),
                "degradation": round(check.delta, 6),
                "limit": round(check.limit, 6),
                "verdict": "pass" if check.passed else "FAIL",
            }
            for check in self.checks
        ]

    def summary_line(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        failed = [c.metric for c in self.checks if not c.passed]
        detail = "" if self.passed else f" (breached: {', '.join(failed)})"
        return f"INT8 deployment gate: {verdict}{detail}"


def evaluate_deployment_gate(
    fp32_summary: dict[str, object],
    int8_summary: dict[str, object],
    thresholds: DeploymentThresholds | None = None,
) -> DeploymentGateResult:
    """Compare FP32 vs INT8 summary rows against the configured deployment limits.

    ``*_summary`` are summary dicts as written by the evaluation scripts (keys
    ``mean_dice``, ``mean_iou``, ``mean_binary_ece_roi``). Drops are measured as
    ``fp32 - int8`` for quality metrics and ``int8 - fp32`` for the ROI ECE error.
    """

    limits = thresholds or DeploymentThresholds()
    checks: list[GateCheck] = [
        _quality_check("mean_dice", fp32_summary, int8_summary, limits.max_dice_drop),
        _quality_check("mean_iou", fp32_summary, int8_summary, limits.max_iou_drop),
        _error_check("mean_binary_ece_roi", fp32_summary, int8_summary, limits.max_ece_roi_increase),
    ]
    return DeploymentGateResult(passed=all(check.passed for check in checks), checks=checks)


def _quality_check(metric: str, fp32: dict, int8: dict, limit: float) -> GateCheck:
    fp32_value = _require_float(fp32, metric)
    int8_value = _require_float(int8, metric)
    drop = fp32_value - int8_value
    return GateCheck(metric=metric, fp32_value=fp32_value, int8_value=int8_value, delta=drop, limit=limit, passed=drop <= limit)


def _error_check(metric: str, fp32: dict, int8: dict, limit: float) -> GateCheck:
    fp32_value = _require_float(fp32, metric)
    int8_value = _require_float(int8, metric)
    increase = int8_value - fp32_value
    return GateCheck(
        metric=metric, fp32_value=fp32_value, int8_value=int8_value, delta=increase, limit=limit, passed=increase <= limit
    )


def _require_float(summary: dict[str, object], key: str) -> float:
    if key not in summary:
        raise KeyError(f"Summary row is missing required metric {key!r}.")
    return float(summary[key])
