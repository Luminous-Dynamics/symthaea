#!/usr/bin/env python3
"""Independent known-answer validator for SEMI-PROC-001A.

Imports no Symthaea production code. It validates only the frozen synthetic
wafer/process evidence corpus from #5894 / PR #5895.

It does not model or execute semiconductor fabrication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA = "semi-proc-001a-synthetic-corpus-v1"
EXPECTED_SHA256 = (
    "ec2e05fdf668d7bb0a98688de5de44eae6642f0bd96662e8fa4300feea2b9c85"
)
EXPECTED_CASES = 16
EXPECTED_AUTHORITY = "representation_only_no_execution_authority"

ALLOWED_DISPOSITIONS = {
    "PlannedOnly",
    "CapabilityUnresolved",
    "AttemptedObservationIncomplete",
    "ObservedOutOfProfile",
    "ObservedCandidateState",
    "AcceptedUnderProfile",
    "RejectedUnderProfile",
    "ReworkRequired",
    "ExecutionNotAuthorized",
}


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def canonical_string(value: Any, field: str, case_id: str) -> str:
    require(
        isinstance(value, str) and value and value.strip() == value,
        f"{case_id}: {field} must be a canonical non-empty string",
    )
    return value


def unique(values: list[str], field: str, case_id: str) -> None:
    require(len(values) == len(set(values)), f"{case_id}: duplicate {field}")


def evaluate(case: dict[str, Any]) -> tuple[str, str]:
    case_id = canonical_string(case.get("id"), "id", "<unknown>")

    if case_id == "plan_without_equipment_capability":
        require(
            case.get("equipment_capability_ref") is None,
            f"{case_id}: equipment capability must be absent",
        )
        require(case.get("attempt_ref") is None, f"{case_id}: attempt must be absent")
        return "disposition", "CapabilityUnresolved"

    if case_id == "capability_without_attempt":
        canonical_string(
            case.get("equipment_capability_ref"),
            "equipment_capability_ref",
            case_id,
        )
        require(case.get("attempt_ref") is None, f"{case_id}: attempt must be absent")
        return "disposition", "PlannedOnly"

    if case_id == "attempt_without_metrology":
        canonical_string(case.get("attempt_ref"), "attempt_ref", case_id)
        require(case.get("metrology_ref") is None, f"{case_id}: metrology must be absent")
        require(
            case.get("observed_output_state_ref") is None,
            f"{case_id}: observed output must be absent",
        )
        return "disposition", "AttemptedObservationIncomplete"

    if case_id == "observed_candidate_not_yet_accepted":
        canonical_string(case.get("attempt_ref"), "attempt_ref", case_id)
        canonical_string(case.get("metrology_ref"), "metrology_ref", case_id)
        canonical_string(
            case.get("observed_output_state_ref"),
            "observed_output_state_ref",
            case_id,
        )
        require(
            case.get("acceptance_profile_ref") is None,
            f"{case_id}: acceptance profile must be absent",
        )
        return "disposition", "ObservedCandidateState"

    if case_id == "accepted_under_exact_profile":
        require(
            case.get("profile_evaluation") == "within_profile",
            f"{case_id}: expected within_profile",
        )
        canonical_string(
            case.get("acceptance_profile_ref"),
            "acceptance_profile_ref",
            case_id,
        )
        canonical_string(
            case.get("observed_output_state_ref"),
            "observed_output_state_ref",
            case_id,
        )
        return "disposition", "AcceptedUnderProfile"

    if case_id == "rejected_under_exact_profile":
        require(
            case.get("profile_evaluation") == "outside_profile",
            f"{case_id}: expected outside_profile",
        )
        canonical_string(
            case.get("acceptance_profile_ref"),
            "acceptance_profile_ref",
            case_id,
        )
        canonical_string(
            case.get("observed_output_state_ref"),
            "observed_output_state_ref",
            case_id,
        )
        return "disposition", "RejectedUnderProfile"

    if case_id == "wrong_input_state_lineage":
        declared = canonical_string(
            case.get("declared_input_state_ref"),
            "declared_input_state_ref",
            case_id,
        )
        actual = canonical_string(
            case.get("attempt_input_state_ref"),
            "attempt_input_state_ref",
            case_id,
        )
        require(declared != actual, f"{case_id}: fixture must contain a mismatch")
        return "error", "InputStateLineageMismatch"

    if case_id == "downstream_consumes_intended_not_observed":
        intended = canonical_string(
            case.get("upstream_intended_output_state_ref"),
            "upstream_intended_output_state_ref",
            case_id,
        )
        downstream = canonical_string(
            case.get("downstream_input_state_ref"),
            "downstream_input_state_ref",
            case_id,
        )
        require(
            case.get("upstream_observed_output_state_ref") is None,
            f"{case_id}: observed state must be absent",
        )
        require(
            downstream == intended,
            f"{case_id}: downstream must consume the intended state",
        )
        return "error", "UnobservedStateCannotSatisfyDownstreamInput"

    if case_id == "rework_creates_new_branch":
        source = canonical_string(case.get("source_state_ref"), "source_state_ref", case_id)
        rejected = canonical_string(
            case.get("rejected_state_ref"), "rejected_state_ref", case_id
        )
        reworked = canonical_string(
            case.get("reworked_state_ref"), "reworked_state_ref", case_id
        )
        require(
            case.get("history_rule") == "rejected_state_preserved",
            f"{case_id}: rejected history must remain preserved",
        )
        require(
            len({source, rejected, reworked}) == 3,
            f"{case_id}: rework must create a distinct state branch",
        )
        return "disposition", "ReworkRequired"

    if case_id == "missing_material_input_ref":
        required = case.get("required_material_refs")
        present = case.get("present_material_refs")
        require(
            isinstance(required, list) and bool(required),
            f"{case_id}: required materials must be a non-empty list",
        )
        require(isinstance(present, list), f"{case_id}: present materials must be a list")
        unique(required, "required material refs", case_id)
        unique(present, "present material refs", case_id)
        require(
            bool(set(required) - set(present)),
            f"{case_id}: fixture must omit a required material",
        )
        return "error", "RequiredInputReferenceMissing"

    if case_id == "stale_metrology_blocks_acceptance":
        require(
            case.get("metrology_currentness") == "stale",
            f"{case_id}: metrology must be stale",
        )
        canonical_string(case.get("metrology_ref"), "metrology_ref", case_id)
        canonical_string(
            case.get("acceptance_profile_ref"),
            "acceptance_profile_ref",
            case_id,
        )
        return "disposition", "AttemptedObservationIncomplete"

    if case_id == "process_profile_change_changes_identity":
        base = canonical_string(
            case.get("base_process_family_ref"), "base_process_family_ref", case_id
        )
        changed = canonical_string(
            case.get("changed_process_family_ref"),
            "changed_process_family_ref",
            case_id,
        )
        require(base != changed, f"{case_id}: process profiles must differ")
        return "identity", "distinct"

    if case_id == "equipment_configuration_change_changes_execution_subject":
        base = canonical_string(
            case.get("base_equipment_ref"), "base_equipment_ref", case_id
        )
        changed = canonical_string(
            case.get("changed_equipment_ref"), "changed_equipment_ref", case_id
        )
        require(base != changed, f"{case_id}: equipment configurations must differ")
        return "identity", "distinct"

    if case_id == "nonsemantic_reference_order_is_canonical":
        require(
            case.get("declared_order_semantics") == "set",
            f"{case_id}: ordering must be declared non-semantic",
        )
        a = case.get("material_refs_a")
        b = case.get("material_refs_b")
        require(
            isinstance(a, list) and isinstance(b, list),
            f"{case_id}: material refs must be lists",
        )
        unique(a, "material_refs_a", case_id)
        unique(b, "material_refs_b", case_id)
        require(a != b, f"{case_id}: source order must actually differ")
        require(sorted(a) == sorted(b), f"{case_id}: canonical sets must match")
        return "identity", "same"

    if case_id == "duplicate_reference_rejected":
        refs = case.get("material_refs")
        require(isinstance(refs, list), f"{case_id}: material_refs must be a list")
        require(
            len(refs) != len(set(refs)),
            f"{case_id}: fixture must actually contain a duplicate",
        )
        return "error", "DuplicateReference"

    if case_id == "later_success_does_not_erase_failure_or_mint_authority":
        attempts = case.get("historical_attempts")
        require(
            isinstance(attempts, list) and len(attempts) == 2,
            f"{case_id}: expected exactly two historical attempts",
        )
        dispositions = [entry.get("disposition") for entry in attempts]
        require(
            dispositions == ["RejectedUnderProfile", "AcceptedUnderProfile"],
            f"{case_id}: expected fail-then-pass history",
        )
        require(
            case.get("execution_authority") is False,
            f"{case_id}: execution authority must remain false",
        )
        require(
            case.get("expected_history") == "both_preserved",
            f"{case_id}: both historical outcomes must remain preserved",
        )
        return "history-authority", "both_preserved:none"

    fail(f"unknown fixture id {case_id!r}")
    raise AssertionError("unreachable")


def expected(case: dict[str, Any]) -> tuple[str, str]:
    if "expected_disposition" in case:
        disposition = case["expected_disposition"]
        require(
            disposition in ALLOWED_DISPOSITIONS,
            f"{case['id']}: unknown disposition {disposition!r}",
        )
        return "disposition", disposition
    if "expected_error" in case:
        return "error", canonical_string(
            case["expected_error"], "expected_error", case["id"]
        )
    if "expected_identity_relation" in case:
        return "identity", canonical_string(
            case["expected_identity_relation"],
            "expected_identity_relation",
            case["id"],
        )
    if (
        case.get("expected_history") == "both_preserved"
        and case.get("expected_authority") == "none"
    ):
        return "history-authority", "both_preserved:none"
    fail(f"{case['id']}: no expected result encoded")
    raise AssertionError("unreachable")


def validate(path: Path) -> None:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(
        digest == EXPECTED_SHA256,
        f"fixture digest drift: expected {EXPECTED_SHA256}, got {digest}",
    )

    try:
        corpus = json.loads(raw)
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON: {exc}")

    require(
        corpus.get("schema") == EXPECTED_SCHEMA,
        f"unexpected schema {corpus.get('schema')!r}",
    )
    require(
        corpus.get("authority") == EXPECTED_AUTHORITY,
        "authority boundary drift",
    )
    purpose = corpus.get("purpose")
    require(
        isinstance(purpose, str)
        and "no physical process recipe parameters" in purpose.lower(),
        "purpose must preserve the no-recipe claim ceiling",
    )

    cases = corpus.get("cases")
    require(isinstance(cases, list), "cases must be a list")
    require(
        len(cases) == EXPECTED_CASES,
        f"expected {EXPECTED_CASES} cases, found {len(cases)}",
    )

    ids = [
        canonical_string(case.get("id"), "id", f"case[{index}]")
        for index, case in enumerate(cases)
    ]
    unique(ids, "case ids", "corpus")

    for case in cases:
        actual = evaluate(case)
        wanted = expected(case)
        require(actual == wanted, f"{case['id']}: expected {wanted}, got {actual}")

    print(f"ok fixtures={len(cases)} digest={digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fixture",
        nargs="?",
        type=Path,
        default=Path(
            "docs/release/evidence/semi-proc-001a-synthetic-corpus-v1.json"
        ),
    )
    args = parser.parse_args()
    validate(args.fixture)


if __name__ == "__main__":
    main()
