#!/usr/bin/env python3
"""Independent PIE-007/008 structural dependency-closure oracle.

This models capability reachability only. It does not model quantities,
throughput, lifetime, economics, inventory depletion, or manufacturing authority.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable


class TargetStatus(str, Enum):
    """Structural closure state of one target capability."""

    LOCALLY_CLOSED = "LocallyClosed"
    IMPORT_DEPENDENT = "ImportDependent"
    UNAVAILABLE = "Unavailable"


@dataclass(frozen=True)
class Recipe:
    """One AND-set route for producing an output capability."""

    output: str
    requires: tuple[str, ...]
    recipe_id: str


@dataclass(frozen=True)
class TargetReport:
    target: str
    status: str


@dataclass(frozen=True)
class ImportLeverage:
    import_id: str
    targets_lost_if_removed: tuple[str, ...]
    capability_count_lost: int


@dataclass(frozen=True)
class ClosureReport:
    locally_reproducible: tuple[str, ...]
    operationally_reachable: tuple[str, ...]
    targets: tuple[TargetReport, ...]
    import_leverage: tuple[ImportLeverage, ...]


def validate_ids(items: Iterable[str], label: str) -> set[str]:
    result: set[str] = set()
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{label} ids must be non-empty strings")
        if item in result:
            raise ValueError(f"duplicate {label} id: {item}")
        result.add(item)
    return result


def validate_recipes(recipes: list[Recipe]) -> None:
    seen: set[str] = set()
    for recipe in recipes:
        if not recipe.recipe_id or not recipe.output:
            raise ValueError("recipe_id/output must be non-empty")
        if recipe.recipe_id in seen:
            raise ValueError(f"duplicate recipe id: {recipe.recipe_id}")
        seen.add(recipe.recipe_id)
        if not recipe.requires:
            raise ValueError("recipe must declare at least one dependency")
        if any(not dep for dep in recipe.requires):
            raise ValueError("recipe dependency ids must be non-empty")
        if len(set(recipe.requires)) != len(recipe.requires):
            raise ValueError("recipe dependencies must be unique")


def fixed_point(seed: set[str], recipes: list[Recipe]) -> set[str]:
    """AND/OR closure: one complete recipe is sufficient for an output."""
    reachable = set(seed)
    changed = True
    while changed:
        changed = False
        for recipe in recipes:
            if recipe.output in reachable:
                continue
            if all(dep in reachable for dep in recipe.requires):
                reachable.add(recipe.output)
                changed = True
    return reachable


def evaluate(
    local_primitives: list[str],
    imports: list[str],
    targets: list[str],
    recipes: list[Recipe],
) -> ClosureReport:
    """Compare local reproductive closure with operation enabled by imports."""
    local = validate_ids(local_primitives, "local primitive")
    imported = validate_ids(imports, "import")
    target_set = validate_ids(targets, "target")
    validate_recipes(recipes)

    locally_closed = fixed_point(local, recipes)
    operational = fixed_point(local | imported, recipes)

    target_reports = []
    for target in sorted(target_set):
        if target in locally_closed:
            status = TargetStatus.LOCALLY_CLOSED
        elif target in operational:
            status = TargetStatus.IMPORT_DEPENDENT
        else:
            status = TargetStatus.UNAVAILABLE
        target_reports.append(TargetReport(target, status.value))

    leverage = []
    for imported_item in sorted(imported):
        without = fixed_point(local | (imported - {imported_item}), recipes)
        lost_targets = tuple(sorted(t for t in target_set if t in operational and t not in without))
        leverage.append(ImportLeverage(
            import_id=imported_item,
            targets_lost_if_removed=lost_targets,
            capability_count_lost=len(operational - without),
        ))

    return ClosureReport(
        locally_reproducible=tuple(sorted(locally_closed)),
        operationally_reachable=tuple(sorted(operational)),
        targets=tuple(target_reports),
        import_leverage=tuple(leverage),
    )


def self_test() -> None:
    recipes = [
        Recipe("frame", ("local-metal",), "frame-route"),
        Recipe("motor", ("local-copper", "bearing"), "motor-route"),
        Recipe("machine", ("frame", "motor"), "machine-route"),
    ]
    report = evaluate(
        ["local-metal", "local-copper"], ["bearing"],
        ["frame", "motor", "machine"], recipes,
    )
    statuses = {item.target: item.status for item in report.targets}
    assert statuses["frame"] == TargetStatus.LOCALLY_CLOSED.value
    assert statuses["motor"] == TargetStatus.IMPORT_DEPENDENT.value
    assert statuses["machine"] == TargetStatus.IMPORT_DEPENDENT.value

    no_import = evaluate(["local-metal", "local-copper"], [], ["machine"], recipes)
    assert no_import.targets[0].status == TargetStatus.UNAVAILABLE.value

    locally_closed = evaluate(
        ["local-metal", "local-copper", "ceramic-feed"], ["bearing"],
        ["motor", "machine"],
        recipes + [Recipe("bearing", ("ceramic-feed",), "local-bearing-route")],
    )
    assert all(item.status == TargetStatus.LOCALLY_CLOSED.value for item in locally_closed.targets)

    cyclic = evaluate(
        [], [], ["a", "b"],
        [Recipe("a", ("b",), "a-from-b"), Recipe("b", ("a",), "b-from-a")],
    )
    assert all(item.status == TargetStatus.UNAVAILABLE.value for item in cyclic.targets)

    alternatives = evaluate(
        ["local-metal", "local-magnet"], ["electronics"], ["actuator"],
        [
            Recipe("actuator", ("local-metal", "electronics"), "electronic-route"),
            Recipe("actuator", ("local-metal", "local-magnet"), "local-route"),
        ],
    )
    assert alternatives.targets[0].status == TargetStatus.LOCALLY_CLOSED.value

    imported_machine = evaluate([], ["machine"], ["machine"], [])
    assert imported_machine.targets[0].status == TargetStatus.IMPORT_DEPENDENT.value

    bearing = next(item for item in report.import_leverage if item.import_id == "bearing")
    assert bearing.targets_lost_if_removed == ("machine", "motor")
    assert bearing.capability_count_lost >= 2

    missing = evaluate(
        ["local-metal"], [], ["widget"],
        [Recipe("widget", ("local-metal", "missing-seal"), "widget-route")],
    )
    assert missing.targets[0].status == TargetStatus.UNAVAILABLE.value

    malformed = [
        lambda: evaluate([""], [], [], []),
        lambda: evaluate(["x", "x"], [], [], []),
        lambda: evaluate([], [], [], [Recipe("", ("x",), "r")]),
        lambda: evaluate([], [], [], [Recipe("x", (), "r")]),
    ]
    for case in malformed:
        try:
            case()
        except ValueError:
            pass
        else:
            raise AssertionError("malformed dependency input must fail closed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--json")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("ok")
        return
    if not args.json:
        parser.error("--self-test or --json is required")
    payload = json.loads(args.json)
    recipes = [
        Recipe(item["output"], tuple(item["requires"]), item["recipe_id"])
        for item in payload.get("recipes", [])
    ]
    report = evaluate(
        payload.get("local_primitives", []), payload.get("imports", []),
        payload.get("targets", []), recipes,
    )
    print(json.dumps(asdict(report), sort_keys=True))


if __name__ == "__main__":
    main()
