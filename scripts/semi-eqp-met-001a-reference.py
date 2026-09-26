#!/usr/bin/env python3
"""Independent known-answer validator for SEMI-EQP-MET-001A.

Imports no Symthaea production code. It validates only the frozen synthetic
inspection/positioning/calibration corpus from #5914 / PR #5915.

It does not control hardware or model semiconductor fabrication.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA = "semi-eqp-met-001a-synthetic-corpus-v1"
EXPECTED_AUTHORITY = "representation_only_no_physical_execution_authority"
EXPECTED_SHA256 = "969508f750aedf3650b3466f463ed501b4120c7bd763a79f4157129bc05d2d02"
EXPECTED_CASES = 16

EXPECTED_IDS = {
    "sharp_image_without_scale_reference",
    "current_scale_reference_candidate_measurement",
    "commanded_vs_observed_position_discrepancy",
    "repeatable_without_absolute_reference",
    "stale_scale_calibration",
    "center_field_only_edge_uncharacterized",
    "remount_changes_transform_identity",
    "software_enhancement_preserves_raw_physical_source",
    "local_clean_region_not_whole_sample",
    "derived_measurement_without_raw_observation",
    "unordered_evidence_refs_canonicalize",
    "duplicate_evidence_ref_rejected",
    "recalibration_preserves_stale_history",
    "imported_subsystems_do_not_imply_productive_closure",
    "commissioned_bench_not_process_capability",
    "benchmark_cannot_mint_execution_authority",
}

ALLOWED_DISPOSITIONS = {
    "QualitativeOnly",
    "RepeatabilityOnly",
    "CandidateMeasurement",
    "MeasurementQualifiedUnderProfile",
    "CalibrationStaleOrMissing",
    "CoverageLimited",
    "DiscrepancyObserved",
    "EvidenceIncomplete",
    "LineageInvalid",
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


def unique_strings(values: Any, field: str, case_id: str) -> list[str]:
    require(isinstance(values, list), f"{case_id}: {field} must be a list")
    parsed = [canonical_string(v, field, case_id) for v in values]
    require(len(parsed) == len(set(parsed)), f"{case_id}: duplicate {field}")
    return parsed


def evaluate(case: dict[str, Any]) -> tuple[str, str]:
    case_id = canonical_string(case.get("id"), "id", "<unknown>")

    if case_id == "sharp_image_without_scale_reference":
        canonical_string(case.get("raw_observation_ref"), "raw_observation_ref", case_id)
        require(case.get("scale_reference_ref") is None, f"{case_id}: scale reference must be absent")
        return "disposition", "QualitativeOnly"

    if case_id == "current_scale_reference_candidate_measurement":
        canonical_string(case.get("raw_observation_ref"), "raw_observation_ref", case_id)
        canonical_string(case.get("scale_reference_ref"), "scale_reference_ref", case_id)
        canonical_string(case.get("derived_measurement_ref"), "derived_measurement_ref", case_id)
        require(case.get("calibration_currentness") == "current", f"{case_id}: calibration must be current")
        canonical_string(case.get("coverage_profile_ref"), "coverage_profile_ref", case_id)
        return "disposition", "CandidateMeasurement"

    if case_id == "commanded_vs_observed_position_discrepancy":
        commanded = canonical_string(case.get("commanded_position_ref"), "commanded_position_ref", case_id)
        observed = canonical_string(case.get("observed_position_ref"), "observed_position_ref", case_id)
        require(commanded != observed, f"{case_id}: fixture must contain a discrepancy")
        return "disposition", "DiscrepancyObserved"

    if case_id == "repeatable_without_absolute_reference":
        revisits = unique_strings(case.get("revisit_observation_refs"), "revisit_observation_refs", case_id)
        require(len(revisits) >= 2, f"{case_id}: at least two revisit observations required")
        require(case.get("repeatability_relation") == "consistent_under_profile", f"{case_id}: repeatability must be explicit")
        require(case.get("absolute_reference_ref") is None, f"{case_id}: absolute reference must be absent")
        return "disposition", "RepeatabilityOnly"

    if case_id == "stale_scale_calibration":
        canonical_string(case.get("raw_observation_ref"), "raw_observation_ref", case_id)
        canonical_string(case.get("scale_reference_ref"), "scale_reference_ref", case_id)
        canonical_string(case.get("derived_measurement_ref"), "derived_measurement_ref", case_id)
        require(case.get("calibration_currentness") == "stale", f"{case_id}: calibration must be stale")
        return "disposition", "CalibrationStaleOrMissing"

    if case_id == "center_field_only_edge_uncharacterized":
        canonical_string(case.get("raw_observation_ref"), "raw_observation_ref", case_id)
        canonical_string(case.get("scale_reference_ref"), "scale_reference_ref", case_id)
        require(case.get("calibration_currentness") == "current", f"{case_id}: calibration must be current")
        require(case.get("coverage_profile_ref") == "coverage:center-only", f"{case_id}: expected center-only coverage")
        require(case.get("edge_characterization_ref") is None, f"{case_id}: edge characterization must be absent")
        return "disposition", "CoverageLimited"

    if case_id == "remount_changes_transform_identity":
        before = canonical_string(case.get("pre_remount_transform_ref"), "pre_remount_transform_ref", case_id)
        after = canonical_string(case.get("post_remount_transform_ref"), "post_remount_transform_ref", case_id)
        require(before != after, f"{case_id}: remount transform identities must differ")
        return "identity", "distinct"

    if case_id == "software_enhancement_preserves_raw_physical_source":
        raw = canonical_string(case.get("raw_observation_ref"), "raw_observation_ref", case_id)
        source = canonical_string(case.get("physical_source_ref"), "physical_source_ref", case_id)
        derived = canonical_string(case.get("derived_image_ref"), "derived_image_ref", case_id)
        require(raw == source, f"{case_id}: derived image must point to the same raw physical source")
        require(derived != raw, f"{case_id}: derived image must have its own identity")
        return "lineage", "derived_from_raw_no_new_physical_observation"

    if case_id == "local_clean_region_not_whole_sample":
        canonical_string(case.get("raw_observation_ref"), "raw_observation_ref", case_id)
        require(case.get("coverage_profile_ref") == "coverage:local-region", f"{case_id}: expected local coverage")
        require(case.get("whole_sample_claim_requested") is True, f"{case_id}: fixture must request a whole-sample claim")
        canonical_string(case.get("local_region_disposition"), "local_region_disposition", case_id)
        return "disposition", "CoverageLimited"

    if case_id == "derived_measurement_without_raw_observation":
        require(case.get("raw_observation_ref") is None, f"{case_id}: raw observation must be absent")
        canonical_string(case.get("derived_measurement_ref"), "derived_measurement_ref", case_id)
        return "error", "LineageInvalid"

    if case_id == "unordered_evidence_refs_canonicalize":
        require(case.get("declared_order_semantics") == "set", f"{case_id}: refs must be declared set-like")
        a = unique_strings(case.get("evidence_refs_a"), "evidence_refs_a", case_id)
        b = unique_strings(case.get("evidence_refs_b"), "evidence_refs_b", case_id)
        require(a != b, f"{case_id}: source order must differ")
        require(sorted(a) == sorted(b), f"{case_id}: canonical sets must match")
        return "identity", "same"

    if case_id == "duplicate_evidence_ref_rejected":
        refs = case.get("evidence_refs")
        require(isinstance(refs, list), f"{case_id}: evidence_refs must be a list")
        parsed = [canonical_string(v, "evidence_refs", case_id) for v in refs]
        require(len(parsed) != len(set(parsed)), f"{case_id}: fixture must contain a duplicate")
        return "error", "DuplicateReference"

    if case_id == "recalibration_preserves_stale_history":
        events = case.get("historical_calibration_events")
        require(isinstance(events, list) and len(events) == 2, f"{case_id}: expected two calibration events")
        old, new = events
        old_ref = canonical_string(old.get("calibration_ref"), "old.calibration_ref", case_id)
        new_ref = canonical_string(new.get("calibration_ref"), "new.calibration_ref", case_id)
        require(old_ref != new_ref, f"{case_id}: recalibration must create a distinct calibration identity")
        require(old.get("currentness_at_use") == "stale", f"{case_id}: old event must remain stale")
        require(new.get("currentness_at_use") == "current", f"{case_id}: new event must be current")
        return "history", "both_preserved"

    if case_id == "imported_subsystems_do_not_imply_productive_closure":
        require(case.get("instrument_status") == "operational_under_profile", f"{case_id}: instrument must be operational")
        imported = unique_strings(case.get("imported_subsystem_refs"), "imported_subsystem_refs", case_id)
        require(bool(imported), f"{case_id}: at least one imported subsystem is required")
        return "closure", "operational_but_import_dependent"

    if case_id == "commissioned_bench_not_process_capability":
        canonical_string(case.get("commissioning_ref"), "commissioning_ref", case_id)
        require(case.get("requested_process_capability") == "semiconductor-processing", f"{case_id}: expected process capability request")
        return "error", "ProcessCapabilityNotEstablished"

    if case_id == "benchmark_cannot_mint_execution_authority":
        canonical_string(case.get("benchmark_result_ref"), "benchmark_result_ref", case_id)
        require(case.get("execution_authority") is False, f"{case_id}: execution authority must remain false")
        return "authority", "none"

    fail(f"unknown fixture id {case_id!r}")
    raise AssertionError("unreachable")


def expected(case: dict[str, Any]) -> tuple[str, str]:
    case_id = case["id"]
    if "expected_disposition" in case:
        value = canonical_string(case["expected_disposition"], "expected_disposition", case_id)
        require(value in ALLOWED_DISPOSITIONS, f"{case_id}: unknown disposition {value!r}")
        return "disposition", value
    if "expected_error" in case:
        return "error", canonical_string(case["expected_error"], "expected_error", case_id)
    if "expected_identity_relation" in case:
        return "identity", canonical_string(case["expected_identity_relation"], "expected_identity_relation", case_id)
    if "expected_lineage_relation" in case:
        return "lineage", canonical_string(case["expected_lineage_relation"], "expected_lineage_relation", case_id)
    if "expected_history" in case:
        return "history", canonical_string(case["expected_history"], "expected_history", case_id)
    if "expected_closure_relation" in case:
        return "closure", canonical_string(case["expected_closure_relation"], "expected_closure_relation", case_id)
    if "expected_authority" in case:
        return "authority", canonical_string(case["expected_authority"], "expected_authority", case_id)
    fail(f"{case_id}: no expected result encoded")
    raise AssertionError("unreachable")


def validate(path: Path) -> None:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == EXPECTED_SHA256, f"fixture digest drift: expected {EXPECTED_SHA256}, got {digest}")

    try:
        corpus = json.loads(raw)
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON: {exc}")

    require(corpus.get("schema") == EXPECTED_SCHEMA, f"unexpected schema {corpus.get('schema')!r}")
    require(corpus.get("authority") == EXPECTED_AUTHORITY, "authority boundary drift")

    purpose = corpus.get("purpose")
    require(
        isinstance(purpose, str)
        and "no hardware construction or semiconductor process parameters" in purpose.lower(),
        "purpose must preserve the no-construction/no-process claim ceiling",
    )

    cases = corpus.get("cases")
    require(isinstance(cases, list), "cases must be a list")
    require(len(cases) == EXPECTED_CASES, f"expected {EXPECTED_CASES} cases, found {len(cases)}")

    ids = [canonical_string(case.get("id"), "id", f"case[{index}]") for index, case in enumerate(cases)]
    require(len(ids) == len(set(ids)), "duplicate fixture id")
    require(set(ids) == EXPECTED_IDS, f"fixture ID set drift: {sorted(set(ids) ^ EXPECTED_IDS)}")

    for case in cases:
        got = evaluate(case)
        want = expected(case)
        require(got == want, f"{case['id']}: expected {want}, derived {got}")

    print(f"ok fixtures={len(cases)} digest={digest}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    validate(root / "docs/release/evidence/semi-eqp-met-001a-synthetic-corpus-v1.json")
