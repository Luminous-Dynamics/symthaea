#!/usr/bin/env python3
"""Validate the proposed RCAF spatial-nuisance plan and its identifiability invariants."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


class SpatialPlanValidationError(ValueError):
    """Raised when a spatial-nuisance plan violates its schema or scientific contract."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise SpatialPlanValidationError(message)


def _reject_non_json_constant(value: str) -> None:
    raise SpatialPlanValidationError(f"non-standard JSON numeric constant is forbidden: {value}")


def load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_non_json_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise SpatialPlanValidationError(f"cannot read valid JSON from {path}: {exc}") from exc


def _strictly_increasing(values: list[float], label: str) -> None:
    _need(
        all(math.isfinite(value) for value in values),
        f"{label} must contain only finite numbers",
    )
    _need(
        all(left < right for left, right in zip(values, values[1:])),
        f"{label} must be strictly increasing",
    )


def validate_semantics(plan: dict[str, Any]) -> None:
    """Check cross-field invariants that JSON Schema cannot express cleanly."""
    _need(isinstance(plan, dict), "plan root must be an object")

    geometry = plan["geometry"]
    target = plan["target"]
    roi_labels = set(geometry["roi_labels"])
    _need(
        set(target["roi_labels"]).issubset(roi_labels),
        "target ROI labels must be a subset of the declared geometry ROI labels",
    )

    if geometry["source"] == "synthetic_fixture_graph":
        _need(
            geometry["interpretation_scope"] == "synthetic_topology_only",
            "a synthetic fixture graph must be labelled synthetic_topology_only",
        )
        _need(
            any("not anatomical" in item.lower() for item in plan["claim_limits"]),
            "synthetic graph claim limits must state that the topology is not anatomical",
        )

    cells = plan["counterfactual_cells"]
    _need(
        set(cells) == {
            "clean_null",
            "signal_only",
            "nuisance_only",
            "signal_plus_nuisance",
        },
        "all four counterfactual cells must be present exactly once",
    )

    seeds = plan["seed_policy"]
    _need(
        seeds["development_seed"] != seeds["confirmatory_seed"],
        "development and confirmatory seeds must differ",
    )
    _need(
        seeds["critical_null_replicates"] >= seeds["screening_replicates"],
        "critical null replicates must be >= screening replicates",
    )

    dose_grid = plan["dose_grid"]
    for key in (
        "nuisance_variance_fractions",
        "spatial_alignment",
        "temporal_coherence",
        "temporal_ar1_coefficients",
        "target_effect_sizes",
        "participant_response_fractions",
    ):
        _strictly_increasing(dose_grid[key], key)

    for key in ("nuisance_variance_fractions", "target_effect_sizes"):
        _need(dose_grid[key][0] == 0.0, f"{key} must include an exact zero-dose/control point")
        _need(any(value > 0.0 for value in dose_grid[key]), f"{key} must include a positive point")

    _need(
        0.0 in dose_grid["spatial_alignment"] and 1.0 in dose_grid["spatial_alignment"],
        "spatial alignment grid must include orthogonal and fully aligned endpoints",
    )
    _need(
        0.0 in dose_grid["temporal_coherence"] and 1.0 in dose_grid["temporal_coherence"],
        "temporal coherence grid must include independent and fully coherent endpoints",
    )
    _need(
        0.0 in dose_grid["participant_response_fractions"]
        and 1.0 in dose_grid["participant_response_fractions"],
        "participant response grid must include zero and full response controls",
    )

    families = plan["nuisance_families"]
    family_ids = [family["family_id"] for family in families]
    family_kinds = {family["kind"] for family in families}
    _need(len(family_ids) == len(set(family_ids)), "nuisance family identifiers must be unique")
    _need(
        family_kinds == {"graph_smooth_low_rank", "heat_kernel_localized"},
        "spatial v1 must include graph-smooth and heat-kernel nuisance families",
    )
    for family in families:
        _strictly_increasing(family["rank_points"], f"{family['family_id']}.rank_points")
        _strictly_increasing(
            family["spatial_scale_points"],
            f"{family['family_id']}.spatial_scale_points",
        )

    boundary = plan["identifiability_boundary"]
    _need(
        boundary["spatial_alignment"] in dose_grid["spatial_alignment"],
        "the non-identifiable spatial-alignment boundary must occur in the declared grid",
    )
    _need(
        boundary["temporal_coherence"] in dose_grid["temporal_coherence"],
        "the non-identifiable temporal-coherence boundary must occur in the declared grid",
    )
    _need(
        boundary["expected_disposition"] == "not_identifiable_abstain",
        "the observational-equivalence boundary must require abstention",
    )

    outputs = set(plan["required_outputs"])
    required = {
        "classical_null_calibration",
        "nuisance_only_operational_false_alarm",
        "clean_directional_recovery",
        "signal_plus_nuisance_directional_recovery",
        "wrong_direction_rejection",
        "effect_drift",
        "correct_abstention",
        "false_abstention",
        "execution_receipt",
    }
    _need(required.issubset(outputs), f"required_outputs is missing: {sorted(required - outputs)}")


def validate_plan(plan: dict[str, Any], schema: dict[str, Any]) -> None:
    """Run Draft 2020-12 validation followed by cross-field semantic checks."""
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(plan), key=lambda error: list(error.absolute_path))
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "<root>"
        raise SpatialPlanValidationError(f"schema violation at {location}: {first.message}")
    validate_semantics(plan)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="JSON spatial-nuisance plan")
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "spatial-nuisance-plan.schema.json",
    )
    args = parser.parse_args(argv)

    try:
        plan = load_json(args.plan)
        schema = load_json(args.schema)
        validate_plan(plan, schema)
    except SpatialPlanValidationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("PASS: RCAF spatial-nuisance plan schema and identifiability invariants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
