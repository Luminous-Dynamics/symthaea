#!/usr/bin/env python3
"""Validate the synthetic socioeconomic run receipt and its negative controls.

This is test/CI tooling only. It is not part of Symthaea's production runtime.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "research" / "finance"
SCHEMA_PATH = DOCS / "SYM-FIN-SYS-016_run_receipt_v1.schema.json"
FIXTURE_PATH = DOCS / "SYM-FIN-SYS-016_run_receipt_example_v1.json"
NEGATIVE_PATH = DOCS / "SYM-FIN-SYS-016_run_receipt_negative_controls_v1.json"
REGISTRY_PATH = DOCS / "SYM-FIN-SYS-016_scenario_registry_v1.json"

EXPECTED_NEGATIVE_CORPUS_SCHEMA = (
    "civos.socioeconomic_interoperability.run_receipt_negative_controls.v1"
)
EXPECTED_NEGATIVE_CONTROL_LAYERS = {
    **{f"SRC-NEG-{number:03d}": "json_schema" for number in range(1, 11)},
    **{f"SRC-SEM-{number:03d}": "semantic_check" for number in range(1, 13)},
}
EXPECTED_NEGATIVE_CONTROL_FIELDS = {
    "id",
    "description",
    "pointer",
    "replacement_value",
    "validation_layer",
}


class ValidationFailure(Exception):
    pass


def reject_duplicate_object_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous JSON objects instead of accepting the last duplicate key."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationFailure(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def reject_non_finite_constant(value: str) -> None:
    """Python's decoder accepts NaN/Infinity extensions that are not JSON values."""
    raise ValidationFailure(f"non-standard numeric constant is not permitted in JSON: {value}")


def parse_finite_float(value: str) -> float:
    """Reject numbers that overflow or underflow Python's floating-point data model."""
    try:
        parsed = float(value)
        exact = Decimal(value)
    except (ValueError, InvalidOperation) as exc:
        raise ValidationFailure(f"invalid JSON numeric value: {value}") from exc
    if not math.isfinite(parsed):
        raise ValidationFailure(f"JSON numeric value overflows finite float range: {value}")
    if parsed == 0.0 and exact != 0:
        raise ValidationFailure(f"JSON numeric value underflows float range: {value}")
    return parsed


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=reject_duplicate_object_keys,
                parse_constant=reject_non_finite_constant,
                parse_float=parse_finite_float,
            )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationFailure) as exc:
        raise ValidationFailure(f"{path}: cannot load strict JSON: {exc}") from exc


def schema_errors(validator: Draft202012Validator, value: Any) -> list[str]:
    errors = []
    for error in sorted(validator.iter_errors(value), key=lambda item: list(map(str, item.absolute_path))):
        location = "/" + "/".join(str(part) for part in error.absolute_path)
        errors.append(f"{location}: {error.message}")
    return errors


def semantic_errors(
    record: dict[str, Any], expected_registry_digest: str, known_scenario_ids: set[str]
) -> list[str]:
    errors: list[str] = []
    if record["run"]["scenario_registry_digest"] != expected_registry_digest:
        errors.append("scenario_registry_digest does not match the exact scenario registry bytes")
    if record["run"]["scenario_id"] not in known_scenario_ids:
        errors.append("scenario_id is not declared by the scenario registry")

    profiles = record["profiles"]
    assignments = profiles["regime_assignments"]
    community_ids = [entry["community_id"] for entry in assignments]
    known_communities = set(community_ids)

    if len(community_ids) != len(known_communities):
        errors.append("regime assignment community_id values must be unique")
    if len(known_communities) != record["world"]["communities"]:
        errors.append("regime assignment count must match world.communities")

    for bridge in profiles["bridge_profiles"]:
        source = bridge["source_community_id"]
        target = bridge["target_community_id"]
        if source not in known_communities or target not in known_communities:
            errors.append(f"bridge {bridge['bridge_id']} references an undeclared community")
        if source == target:
            errors.append(f"bridge {bridge['bridge_id']} must connect distinct communities")

    events = record["events"]
    event_ids = [event["event_id"] for event in events]
    if len(event_ids) != len(set(event_ids)):
        errors.append("event_id values must be unique")

    previous_sequence: int | None = None
    previous_tick: int | None = None
    for event in events:
        sequence = event["sequence"]
        tick = event["simulation_tick"]
        if previous_sequence is not None and sequence <= previous_sequence:
            errors.append("event sequence values must be strictly increasing")
        if previous_tick is not None and tick < previous_tick:
            errors.append("simulation_tick values must be non-decreasing")
        previous_sequence = sequence
        previous_tick = tick

    for metric in record["metrics"]:
        value = metric["value"]
        status = metric["measurement_status"]
        if status == "unmeasured" and value is not None:
            errors.append(f"unmeasured metric {metric['metric_id']} must have null value")
        if status in ("model_observed", "estimated") and value is None:
            errors.append(f"{status} metric {metric['metric_id']} must have a numeric value")

    evaluation = record["evaluation"]
    controls = set(evaluation["controls_applied"])
    if evaluation["accounting_oracle_applied"] != ("independent_accounting_oracle" in controls):
        errors.append("accounting_oracle_applied must agree with controls_applied")
    if evaluation["power_oracle_applied"] != ("independent_power_oracle" in controls):
        errors.append("power_oracle_applied must agree with controls_applied")

    return errors


def set_json_pointer(document: Any, pointer: str, replacement: Any) -> None:
    if not pointer.startswith("/"):
        raise ValidationFailure(f"negative control has invalid JSON pointer: {pointer}")
    pieces = [piece.replace("~1", "/").replace("~0", "~") for piece in pointer[1:].split("/")]
    cursor = document
    for piece in pieces[:-1]:
        if isinstance(cursor, list):
            if not piece.isdecimal():
                raise ValidationFailure(f"negative control array index is not a non-negative integer: {pointer}")
            index = int(piece)
            if index >= len(cursor):
                raise ValidationFailure(f"negative control array index is out of range: {pointer}")
            cursor = cursor[index]
        elif isinstance(cursor, dict) and piece in cursor:
            cursor = cursor[piece]
        else:
            raise ValidationFailure(f"negative control pointer parent does not resolve: {pointer}")

    leaf = pieces[-1]
    if isinstance(cursor, list):
        if not leaf.isdecimal():
            raise ValidationFailure(f"negative control array leaf is not a non-negative integer: {pointer}")
        index = int(leaf)
        if index >= len(cursor):
            raise ValidationFailure(f"negative control array leaf is out of range: {pointer}")
        cursor[index] = replacement
    elif isinstance(cursor, dict):
        cursor[leaf] = replacement
    else:
        raise ValidationFailure(f"negative control parent is not a container: {pointer}")


def main() -> int:
    schema = load_json(SCHEMA_PATH)
    fixture = load_json(FIXTURE_PATH)
    negative_corpus = load_json(NEGATIVE_PATH)
    scenario_registry = load_json(REGISTRY_PATH)
    if not isinstance(scenario_registry, dict) or (
        scenario_registry.get("schema") != "civos.socioeconomic_interoperability.scenario.v1"
        or scenario_registry.get("status") != "research"
        or scenario_registry.get("claim_ceiling") != "synthetic_deterministic_only"
    ):
        print("FAIL: scenario registry metadata is missing or outside its research-only contract")
        return 1
    scenario_ids = scenario_registry.get("scenario_ids")
    if (
        not isinstance(scenario_ids, list)
        or not scenario_ids
        or any(not isinstance(value, str) or not value.strip() for value in scenario_ids)
        or len(scenario_ids) != len(set(scenario_ids))
    ):
        print("FAIL: scenario registry must declare a non-empty unique list of scenario_ids")
        return 1
    known_scenario_ids = set(scenario_ids)
    expected_registry_digest = "sha256:" + hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()

    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    positive_errors = schema_errors(validator, fixture)
    if positive_errors:
        print("FAIL: positive example does not satisfy JSON Schema:")
        for error in positive_errors:
            print(f"  - {error}")
        return 1

    positive_semantics = semantic_errors(fixture, expected_registry_digest, known_scenario_ids)
    if positive_semantics:
        print("FAIL: positive example violates semantic invariants:")
        for error in positive_semantics:
            print(f"  - {error}")
        return 1

    if not isinstance(negative_corpus, dict):
        print("FAIL: negative control corpus root must be an object")
        return 1

    expected_metadata = {
        "schema": EXPECTED_NEGATIVE_CORPUS_SCHEMA,
        "status": "research_fixture",
        "base_fixture": FIXTURE_PATH.name,
    }
    metadata_failures = [
        f"{field}: expected {expected!r}, got {negative_corpus.get(field)!r}"
        for field, expected in expected_metadata.items()
        if negative_corpus.get(field) != expected
    ]
    if metadata_failures:
        print("FAIL: negative control corpus metadata:")
        for failure in metadata_failures:
            print(f"  - {failure}")
        return 1

    cases = negative_corpus.get("cases")
    if not isinstance(cases, list):
        print("FAIL: negative control corpus cases must be an array")
        return 1

    failures: list[str] = []
    if len(cases) != len(EXPECTED_NEGATIVE_CONTROL_LAYERS):
        failures.append(
            "negative control corpus size changed: "
            f"expected {len(EXPECTED_NEGATIVE_CONTROL_LAYERS)}, got {len(cases)}"
        )

    seen_case_ids: set[str] = set()
    checked_schema = 0
    checked_semantic = 0

    for case in cases:
        if not isinstance(case, dict):
            failures.append("negative control entry must be an object")
            continue

        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            failures.append("negative control is missing a non-empty string ID")
            continue
        if case_id in seen_case_ids:
            failures.append(f"{case_id}: duplicate case ID")
            continue
        seen_case_ids.add(case_id)

        expected_layer = EXPECTED_NEGATIVE_CONTROL_LAYERS.get(case_id)
        if expected_layer is None:
            failures.append(f"{case_id}: unexpected control ID; update the versioned corpus intentionally")
            continue

        missing_fields = EXPECTED_NEGATIVE_CONTROL_FIELDS - set(case)
        extra_fields = set(case) - EXPECTED_NEGATIVE_CONTROL_FIELDS
        if missing_fields or extra_fields:
            if missing_fields:
                failures.append(f"{case_id}: missing fields {sorted(missing_fields)}")
            if extra_fields:
                failures.append(f"{case_id}: unexpected fields {sorted(extra_fields)}")
            continue

        if not isinstance(case["description"], str) or not case["description"].strip():
            failures.append(f"{case_id}: description must be a non-empty string")
            continue
        if not isinstance(case["pointer"], str) or not case["pointer"].startswith("/"):
            failures.append(f"{case_id}: pointer must be an absolute JSON Pointer")
            continue
        layer = case["validation_layer"]
        if layer != expected_layer:
            failures.append(
                f"{case_id}: expected validation layer {expected_layer!r}, got {layer!r}"
            )
            continue

        mutated = copy.deepcopy(fixture)
        try:
            set_json_pointer(mutated, case["pointer"], case["replacement_value"])
        except (KeyError, ValidationFailure) as exc:
            failures.append(f"{case_id}: {exc}")
            continue

        errors = schema_errors(validator, mutated)
        if layer == "json_schema":
            checked_schema += 1
            if not errors:
                failures.append(f"{case_id}: invalid mutation unexpectedly passed JSON Schema")
        else:
            checked_semantic += 1
            if errors:
                failures.append(f"{case_id}: semantic control was already rejected at schema layer: {errors[0]}")
            else:
                found = semantic_errors(mutated, expected_registry_digest, known_scenario_ids)
                if not found:
                    failures.append(f"{case_id}: invalid mutation unexpectedly passed semantic checks")

    missing_ids = set(EXPECTED_NEGATIVE_CONTROL_LAYERS) - seen_case_ids
    if missing_ids:
        failures.append(f"missing required control IDs: {sorted(missing_ids)}")

    # Enforce the intended split as well as per-ID layer assignments so the suite
    # cannot become smaller or silently move semantic cases into JSON Schema.
    if checked_schema != 10:
        failures.append(f"expected 10 schema-level controls to be checked, got {checked_schema}")
    if checked_semantic != 12:
        failures.append(f"expected 12 semantic controls to be checked, got {checked_semantic}")

    if failures:
        print("FAIL: run receipt negative controls:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(
        "PASS: schema definition valid; positive fixture valid; "
        f"{checked_schema} schema-level and {checked_semantic} semantic negative controls rejected; "
        "required v1 control IDs intact; scenario digest and scenario ID are bound to the registry."
    )
    print("Claim boundary: synthetic deterministic research fixture only; no production or economic effects.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
