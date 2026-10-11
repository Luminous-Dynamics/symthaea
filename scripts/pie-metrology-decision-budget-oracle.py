#!/usr/bin/env python3
"""Independent PIE-009R conservative metrology decision-budget oracle.

This reference intentionally uses deterministic worst-case error bounds rather than
inventing probability distributions or claiming a full GUM uncertainty model.
"""

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math


class PrecisionStatus(str, Enum):
    QUALIFIED = "Qualified"
    INSUFFICIENT = "InsufficientPrecision"
    EXPIRED = "Expired"


class ClosureClass(str, Enum):
    LOCAL = "LocallyRenewable"
    IMPORT = "ImportDependent"


@dataclass(frozen=True)
class ErrorComponent:
    component_id: str
    bound_at_calibration: float
    drift_bound_per_step: float = 0.0
    sensitivity: float = 1.0


@dataclass(frozen=True)
class CalibrationRoute:
    route_id: str
    calibrated_step: int
    valid_through_step: int
    closure_class: ClosureClass
    components: tuple[ErrorComponent, ...]


@dataclass(frozen=True)
class DecisionRequirement:
    requirement_id: str
    max_total_error_bound: float


@dataclass(frozen=True)
class BudgetReport:
    route_id: str
    requirement_id: str
    age_steps: int
    total_error_bound: float
    allowed_error_bound: float
    status: PrecisionStatus
    closure_class: ClosureClass
    metrologically_autonomous: bool
    digest: str


def _finite_nonnegative(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value >= 0


def _validate(route: CalibrationRoute, requirement: DecisionRequirement, step: int) -> None:
    if not route.route_id or not requirement.requirement_id:
        raise ValueError("empty identifier")
    if not all(isinstance(v, int) for v in (route.calibrated_step, route.valid_through_step, step)):
        raise ValueError("steps must be integers")
    if route.calibrated_step < 0 or route.valid_through_step < route.calibrated_step:
        raise ValueError("invalid calibration interval")
    if step < route.calibrated_step:
        raise ValueError("cannot evaluate before calibration")
    if not _finite_nonnegative(requirement.max_total_error_bound):
        raise ValueError("invalid decision allowance")
    if not route.components:
        raise ValueError("calibration route must contain at least one error component")

    seen: set[str] = set()
    for component in route.components:
        if not component.component_id or component.component_id in seen:
            raise ValueError("empty or duplicate error component")
        seen.add(component.component_id)
        if not (
            _finite_nonnegative(component.bound_at_calibration)
            and _finite_nonnegative(component.drift_bound_per_step)
            and _finite_nonnegative(component.sensitivity)
        ):
            raise ValueError("invalid error component")


def _digest(route: CalibrationRoute, requirement: DecisionRequirement, step: int) -> str:
    canonical = {
        "route_id": route.route_id,
        "calibrated_step": route.calibrated_step,
        "valid_through_step": route.valid_through_step,
        "closure_class": route.closure_class.value,
        "components": [
            {
                "component_id": component.component_id,
                "bound_at_calibration": component.bound_at_calibration,
                "drift_bound_per_step": component.drift_bound_per_step,
                "sensitivity": component.sensitivity,
            }
            for component in sorted(route.components, key=lambda value: value.component_id)
        ],
        "requirement_id": requirement.requirement_id,
        "max_total_error_bound": requirement.max_total_error_bound,
        "evaluation_step": step,
    }
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def evaluate_budget(
    route: CalibrationRoute,
    requirement: DecisionRequirement,
    step: int,
) -> BudgetReport:
    _validate(route, requirement, step)
    age = step - route.calibrated_step

    # Conservative interval-style composition: no independence assumption and no
    # probabilistic cancellation. Every declared bounded contribution is summed.
    total = sum(
        component.sensitivity
        * (component.bound_at_calibration + component.drift_bound_per_step * age)
        for component in route.components
    )

    if step > route.valid_through_step:
        status = PrecisionStatus.EXPIRED
    elif total <= requirement.max_total_error_bound + 1e-15:
        status = PrecisionStatus.QUALIFIED
    else:
        status = PrecisionStatus.INSUFFICIENT

    return BudgetReport(
        route_id=route.route_id,
        requirement_id=requirement.requirement_id,
        age_steps=age,
        total_error_bound=total,
        allowed_error_bound=requirement.max_total_error_bound,
        status=status,
        closure_class=route.closure_class,
        metrologically_autonomous=(
            status == PrecisionStatus.QUALIFIED
            and route.closure_class == ClosureClass.LOCAL
        ),
        digest=_digest(route, requirement, step),
    )


def run_self_test() -> str:
    requirement = DecisionRequirement("restart-pressure", 0.5)

    exact_boundary = CalibrationRoute(
        "exact-boundary",
        0,
        10,
        ClosureClass.LOCAL,
        (
            ErrorComponent("reference", 0.2),
            ErrorComponent("transfer", 0.3),
        ),
    )
    assert evaluate_budget(exact_boundary, requirement, 0).status == PrecisionStatus.QUALIFIED

    aging = CalibrationRoute(
        "aging",
        0,
        10,
        ClosureClass.LOCAL,
        (
            ErrorComponent("reference", 0.2),
            ErrorComponent("transfer", 0.2, drift_bound_per_step=0.11),
        ),
    )
    assert evaluate_budget(aging, requirement, 0).status == PrecisionStatus.QUALIFIED
    assert evaluate_budget(aging, requirement, 1).status == PrecisionStatus.INSUFFICIENT
    aging_totals = [evaluate_budget(aging, requirement, step).total_error_bound for step in range(6)]
    assert aging_totals == sorted(aging_totals)

    local_coarse = CalibrationRoute(
        "local-coarse",
        0,
        20,
        ClosureClass.LOCAL,
        (
            ErrorComponent("local-standard", 0.35),
            ErrorComponent("transfer", 0.25),
        ),
    )
    imported_precise = CalibrationRoute(
        "import-precise",
        0,
        20,
        ClosureClass.IMPORT,
        (
            ErrorComponent("import-standard", 0.1),
            ErrorComponent("transfer", 0.1),
        ),
    )
    local_report = evaluate_budget(local_coarse, requirement, 0)
    import_report = evaluate_budget(imported_precise, requirement, 0)
    assert local_report.status == PrecisionStatus.INSUFFICIENT
    assert not local_report.metrologically_autonomous
    assert import_report.status == PrecisionStatus.QUALIFIED
    assert not import_report.metrologically_autonomous

    improved_local = CalibrationRoute(
        "local-improved",
        0,
        20,
        ClosureClass.LOCAL,
        (
            ErrorComponent("local-standard-v2", 0.15),
            ErrorComponent("transfer", 0.15),
        ),
    )
    improved_report = evaluate_budget(improved_local, requirement, 0)
    assert improved_report.status == PrecisionStatus.QUALIFIED
    assert improved_report.metrologically_autonomous

    expired = CalibrationRoute(
        "expired",
        0,
        2,
        ClosureClass.LOCAL,
        (ErrorComponent("reference", 0.01),),
    )
    assert evaluate_budget(expired, requirement, 3).status == PrecisionStatus.EXPIRED

    better_reference = CalibrationRoute(
        "better-reference",
        0,
        10,
        ClosureClass.LOCAL,
        (
            ErrorComponent("reference", 0.1),
            ErrorComponent("transfer", 0.2),
        ),
    )
    worse_reference = CalibrationRoute(
        "worse-reference",
        0,
        10,
        ClosureClass.LOCAL,
        (
            ErrorComponent("reference", 0.2),
            ErrorComponent("transfer", 0.2),
        ),
    )
    assert (
        evaluate_budget(better_reference, requirement, 0).total_error_bound
        <= evaluate_budget(worse_reference, requirement, 0).total_error_bound
    )

    digest = evaluate_budget(exact_boundary, requirement, 0).digest
    assert digest == evaluate_budget(exact_boundary, requirement, 0).digest
    metadata_changed = CalibrationRoute(
        exact_boundary.route_id,
        exact_boundary.calibrated_step,
        11,
        exact_boundary.closure_class,
        exact_boundary.components,
    )
    assert evaluate_budget(metadata_changed, requirement, 0).digest != digest

    malformed = [
        CalibrationRoute(
            "nan",
            0,
            1,
            ClosureClass.LOCAL,
            (ErrorComponent("x", float("nan")),),
        ),
        CalibrationRoute(
            "duplicate",
            0,
            1,
            ClosureClass.LOCAL,
            (ErrorComponent("x", 0.1), ErrorComponent("x", 0.2)),
        ),
    ]
    for route in malformed:
        try:
            evaluate_budget(route, requirement, 0)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed route must fail closed")

    return "ok"


if __name__ == "__main__":
    print(run_self_test())
