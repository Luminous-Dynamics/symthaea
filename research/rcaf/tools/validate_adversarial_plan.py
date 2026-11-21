#!/usr/bin/env python3
"""Validate an RCAF adversarial calibration plan and its scientific invariants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


class PlanValidationError(ValueError):
    """Raised when a calibration plan violates the plan/evidence contract."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise PlanValidationError(message)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanValidationError(f"cannot read valid JSON from {path}: {exc}") from exc


def validate_semantics(plan: dict[str, Any]) -> None:
    """Check invariants that are awkward or impossible to express in JSON Schema."""
    _need(isinstance(plan, dict), "plan root must be an object")
    seed_policy = plan["seed_policy"]
    _need(
        seed_policy["development_seed"] != seed_policy["confirmatory_seed"],
        "development and confirmatory seeds must differ",
    )
    _need(
        seed_policy["critical_null_replicates"] >= seed_policy["qualification_replicates"],
        "critical_null_replicates must be >= qualification_replicates",
    )

    latent = plan["latent_scenarios"]
    ids = [item["scenario_id"] for item in latent]
    _need(len(ids) == len(set(ids)), "latent scenario_id values must be unique")
    nulls = [item for item in latent if item["truth_class"] == "null"]
    effects = [item for item in latent if item["truth_class"] == "effect"]
    for item in nulls:
        _need(
            item["mechanism"] == "none"
            and item["injection_stage"] == "none"
            and all(dose == 0 for dose in item["dose_points"]),
            f"null scenario {item['scenario_id']} must not inject a latent effect",
        )
    _need(
        any(
            item["mechanism"] == "none"
            and item["injection_stage"] == "none"
            and all(dose == 0 for dose in item["dose_points"])
            for item in nulls
        ),
        "at least one exact same-state null with zero dose and no injection is required",
    )
    for item in effects:
        _need(
            item["mechanism"] != "none" and item["injection_stage"] != "none",
            f"effect scenario {item['scenario_id']} must declare a mechanism and injection stage",
        )
        _need(
            any(dose > 0 for dose in item["dose_points"]),
            f"effect scenario {item['scenario_id']} must include a positive dose",
        )
    _need(
        any(
            item["injection_stage"] in {"parcel_time_series", "sensor_time_series", "source_activity"}
            for item in effects
        ),
        "at least one pre-estimation signal-level injection is required",
    )
    _need(
        any(item["injection_stage"] == "connectivity_matrix" for item in effects),
        "at least one matrix-level control is required to isolate comparator semantics",
    )

    nuisance = plan["nuisance_scenarios"]
    nuisance_ids = [item["scenario_id"] for item in nuisance]
    _need(len(nuisance_ids) == len(set(nuisance_ids)), "nuisance scenario_id values must be unique")
    _need(
        any(item["preserves_latent_target"] is True for item in nuisance),
        "at least one nuisance-only null preserving the latent target is required",
    )

    outputs = set(plan["required_outputs"])
    for required in ("type_i_error", "power", "directional_power", "wrong_direction_rejection_rate", "nuisance_false_positives", "execution_receipt"):
        _need(required in outputs, f"required_outputs must include {required}")
    _need(bool(plan["connectivity_constructors"]), "at least one connectivity constructor is required")
    _need(bool(plan["matrix_comparators"]), "at least one matrix comparator is required")


def validate_plan(plan: dict[str, Any], schema: dict[str, Any]) -> None:
    """Run Draft 2020-12 schema validation and semantic validation."""
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(plan), key=lambda error: list(error.absolute_path))
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "<root>"
        raise PlanValidationError(f"schema violation at {location}: {first.message}")
    validate_semantics(plan)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="JSON calibration plan")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "adversarial-calibration-plan.schema.json",
    )
    args = parser.parse_args(argv)
    try:
        plan = load_json(args.plan)
        schema = load_json(args.schema)
        validate_plan(plan, schema)
    except PlanValidationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print("PASS: RCAF adversarial calibration plan schema and semantic invariants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
