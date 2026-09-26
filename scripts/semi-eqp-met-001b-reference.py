#!/usr/bin/env python3
"""Independent validator for SEMI-EQP-MET-001B lineage semantics.

Imports no Symthaea production code. It validates only the frozen synthetic
configuration-binding/raw-observation-lineage corpus.

It does not control hardware or model semiconductor fabrication.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_SCHEMA = "semi-eqp-met-001b-lineage-corpus-v1"
EXPECTED_AUTHORITY = "representation_only_no_physical_execution_authority"
EXPECTED_CASES = 16
EXPECTED_SHA256 = "df745926fb6e8eecb220f622ccd7cf995c80309a24667f22d7e14d4eae14235f"

CORPUS = Path("docs/release/evidence/semi-eqp-met-001b-lineage-corpus-v1.json")


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def cstr(value: Any, field: str, case_id: str) -> str:
    require(
        isinstance(value, str) and bool(value) and value.strip() == value,
        f"{case_id}: {field} must be a canonical non-empty string",
    )
    return value


def unique(values: list[str], field: str, case_id: str) -> None:
    require(len(values) == len(set(values)), f"{case_id}: duplicate {field}")


def evaluate(case: dict[str, Any]) -> tuple[str, str]:
    case_id = cstr(case.get("id"), "id", "<unknown>")

    if case_id == "same_name_different_device_ref":
        a = cstr(case.get("device_ref_a"), "device_ref_a", case_id)
        b = cstr(case.get("device_ref_b"), "device_ref_b", case_id)
        require(a != b, f"{case_id}: physical device refs must differ")
        cstr(case.get("friendly_name"), "friendly_name", case_id)
        return ("identity", "distinct")

    if case_id == "same_imager_changed_optics_mount":
        cstr(case.get("imager_ref"), "imager_ref", case_id)
        oa = cstr(case.get("optics_ref_a"), "optics_ref_a", case_id)
        ob = cstr(case.get("optics_ref_b"), "optics_ref_b", case_id)
        ma = cstr(case.get("mount_ref_a"), "mount_ref_a", case_id)
        mb = cstr(case.get("mount_ref_b"), "mount_ref_b", case_id)
        require((oa, ma) != (ob, mb), f"{case_id}: configuration must actually change")
        return ("identity", "distinct")

    if case_id == "remount_transform_changes_context":
        cstr(case.get("physical_device_ref"), "physical_device_ref", case_id)
        a = cstr(case.get("transform_ref_a"), "transform_ref_a", case_id)
        b = cstr(case.get("transform_ref_b"), "transform_ref_b", case_id)
        require(a != b, f"{case_id}: remount transforms must differ")
        return ("identity", "distinct")

    if case_id == "parser_change_preserves_raw_root":
        cstr(case.get("raw_observation_ref"), "raw_observation_ref", case_id)
        cstr(case.get("raw_artifact_digest"), "raw_artifact_digest", case_id)
        a = cstr(case.get("parser_ref_a"), "parser_ref_a", case_id)
        b = cstr(case.get("parser_ref_b"), "parser_ref_b", case_id)
        require(a != b, f"{case_id}: parser refs must differ")
        return ("parser-change", "same-raw-root:distinct-derived")

    if case_id == "derived_enhancement_same_physical_root":
        cstr(case.get("raw_observation_ref"), "raw_observation_ref", case_id)
        cstr(case.get("raw_artifact_digest"), "raw_artifact_digest", case_id)
        cstr(case.get("derived_artifact_ref"), "derived_artifact_ref", case_id)
        cstr(case.get("transform_ref"), "transform_ref", case_id)
        return ("physical-witnesses", "1:derived_from_raw")

    if case_id == "orphan_derived_measurement_rejected":
        require(case.get("raw_observation_ref") is None, f"{case_id}: raw observation must be absent")
        require(case.get("raw_artifact_digest") is None, f"{case_id}: raw artifact must be absent")
        cstr(case.get("derived_artifact_ref"), "derived_artifact_ref", case_id)
        return ("error", "LineageInvalid")

    if case_id == "calibration_epoch_change_preserves_raw":
        cstr(case.get("raw_observation_ref"), "raw_observation_ref", case_id)
        a = cstr(case.get("calibration_ref_a"), "calibration_ref_a", case_id)
        b = cstr(case.get("calibration_ref_b"), "calibration_ref_b", case_id)
        require(a != b, f"{case_id}: calibration refs must differ")
        return ("calibration-change", "same-raw:distinct-measurement-context")

    if case_id == "stale_calibration_limits_claim":
        cstr(case.get("raw_observation_ref"), "raw_observation_ref", case_id)
        cstr(case.get("calibration_ref"), "calibration_ref", case_id)
        require(case.get("calibration_currentness") == "stale", f"{case_id}: calibration must be stale")
        return ("disposition", "CalibrationStaleOrMissing")

    if case_id == "source_receive_timestamps_preserved":
        sc = cstr(case.get("source_clock_ref"), "source_clock_ref", case_id)
        rc = cstr(case.get("receive_clock_ref"), "receive_clock_ref", case_id)
        st = cstr(case.get("source_timestamp"), "source_timestamp", case_id)
        rt = cstr(case.get("receive_timestamp"), "receive_timestamp", case_id)
        require((sc, st) != (rc, rt), f"{case_id}: source/receive timing contexts must differ")
        return ("timing", "distinct_preserved")

    if case_id == "replay_not_fresh_physical_observation":
        cstr(case.get("original_observation_ref"), "original_observation_ref", case_id)
        cstr(case.get("replay_artifact_ref"), "replay_artifact_ref", case_id)
        return ("replay", "SoftwareReplay:0-new-physical-witnesses")

    if case_id == "two_derivations_one_physical_witness":
        cstr(case.get("raw_observation_ref"), "raw_observation_ref", case_id)
        derived = case.get("derived_artifact_refs")
        require(isinstance(derived, list) and len(derived) == 2, f"{case_id}: expected two derivations")
        for item in derived:
            cstr(item, "derived_artifact_refs[]", case_id)
        unique(derived, "derived refs", case_id)
        return ("physical-witnesses", "1:2-derived")

    if case_id == "duplicate_reference_rejected":
        refs = case.get("evidence_refs")
        require(isinstance(refs, list), f"{case_id}: evidence_refs must be a list")
        require(case.get("declared_order_semantics") == "set", f"{case_id}: expected set semantics")
        require(len(refs) != len(set(refs)), f"{case_id}: fixture must contain duplicate")
        return ("error", "DuplicateReference")

    if case_id == "unordered_ancillary_refs_canonicalize":
        a = case.get("ancillary_refs_a")
        b = case.get("ancillary_refs_b")
        require(isinstance(a, list) and isinstance(b, list), f"{case_id}: ancillary refs must be lists")
        require(case.get("declared_order_semantics") == "set", f"{case_id}: expected set semantics")
        unique(a, "ancillary_refs_a", case_id)
        unique(b, "ancillary_refs_b", case_id)
        require(a != b, f"{case_id}: source order must differ")
        require(sorted(a) == sorted(b), f"{case_id}: canonical sets must match")
        return ("identity", "same")

    if case_id == "sample_article_mismatch_rejected":
        a = cstr(case.get("session_sample_ref"), "session_sample_ref", case_id)
        b = cstr(case.get("observation_sample_ref"), "observation_sample_ref", case_id)
        require(a != b, f"{case_id}: fixture must contain subject mismatch")
        return ("error", "SubjectLineageMismatch")

    if case_id == "imported_subsystem_separates_closure":
        require(
            case.get("operational_instrument_state") == "usable_under_profile",
            f"{case_id}: expected bounded operational state",
        )
        require(case.get("camera_closure") == "external_import", f"{case_id}: camera must be external")
        require(case.get("stage_closure") == "external_import", f"{case_id}: stage must be external")
        return ("operation-closure", "OperationalEvidencePossible:ImportDependent")

    if case_id == "binding_never_mints_execution_authority":
        require(case.get("binding_complete") is True, f"{case_id}: binding must be complete")
        require(case.get("observation_present") is True, f"{case_id}: observation must be present")
        require(case.get("calibration_current") is True, f"{case_id}: calibration must be current")
        return ("authority", "none")

    fail(f"unknown fixture id {case_id!r}")
    raise AssertionError("unreachable")


def expected(case: dict[str, Any]) -> tuple[str, str]:
    case_id = case["id"]

    if "expected_error" in case:
        return ("error", cstr(case["expected_error"], "expected_error", case_id))
    if "expected_disposition" in case:
        return ("disposition", cstr(case["expected_disposition"], "expected_disposition", case_id))
    if "expected_identity_relation" in case:
        return ("identity", cstr(case["expected_identity_relation"], "expected_identity_relation", case_id))

    if case_id == "parser_change_preserves_raw_root":
        require(case.get("expected_physical_root_relation") == "same", f"{case_id}: expected same raw root")
        require(case.get("expected_derived_identity_relation") == "distinct", f"{case_id}: expected distinct derived")
        return ("parser-change", "same-raw-root:distinct-derived")

    if case_id == "derived_enhancement_same_physical_root":
        require(case.get("expected_physical_witness_count") == 1, f"{case_id}: expected one physical witness")
        require(case.get("expected_lineage") == "derived_from_raw", f"{case_id}: expected derived lineage")
        return ("physical-witnesses", "1:derived_from_raw")

    if case_id == "calibration_epoch_change_preserves_raw":
        require(case.get("expected_raw_identity_relation") == "same", f"{case_id}: expected same raw identity")
        require(case.get("expected_measurement_context_relation") == "distinct", f"{case_id}: expected distinct context")
        return ("calibration-change", "same-raw:distinct-measurement-context")

    if case_id == "source_receive_timestamps_preserved":
        return ("timing", cstr(case.get("expected_timing_relation"), "expected_timing_relation", case_id))

    if case_id == "replay_not_fresh_physical_observation":
        require(case.get("expected_replay_class") == "SoftwareReplay", f"{case_id}: replay class drift")
        require(case.get("expected_new_physical_witnesses") == 0, f"{case_id}: replay cannot create physical witnesses")
        return ("replay", "SoftwareReplay:0-new-physical-witnesses")

    if case_id == "two_derivations_one_physical_witness":
        require(case.get("expected_physical_witness_count") == 1, f"{case_id}: physical witness count drift")
        require(case.get("expected_derived_artifact_count") == 2, f"{case_id}: derived count drift")
        return ("physical-witnesses", "1:2-derived")

    if case_id == "imported_subsystem_separates_closure":
        require(case.get("expected_operational_disposition") == "OperationalEvidencePossible", f"{case_id}: operation disposition drift")
        require(case.get("expected_closure_disposition") == "ImportDependent", f"{case_id}: closure disposition drift")
        return ("operation-closure", "OperationalEvidencePossible:ImportDependent")

    if case_id == "binding_never_mints_execution_authority":
        return ("authority", cstr(case.get("expected_authority"), "expected_authority", case_id))

    fail(f"{case_id}: no expected outcome encoded")
    raise AssertionError("unreachable")


def main() -> None:
    raw = CORPUS.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == EXPECTED_SHA256, f"fixture digest drift: expected {EXPECTED_SHA256}, got {digest}")

    corpus = json.loads(raw)
    require(corpus.get("schema") == EXPECTED_SCHEMA, "schema drift")
    require(corpus.get("authority") == EXPECTED_AUTHORITY, "authority boundary drift")

    purpose = corpus.get("purpose")
    require(isinstance(purpose, str) and "no hardware construction" in purpose.lower(), "purpose/claim ceiling drift")

    cases = corpus.get("cases")
    require(isinstance(cases, list), "cases must be a list")
    require(len(cases) == EXPECTED_CASES, f"expected {EXPECTED_CASES} cases, got {len(cases)}")

    ids = [cstr(case.get("id"), "id", f"case[{i}]") for i, case in enumerate(cases)]
    require(len(ids) == len(set(ids)), "duplicate fixture IDs")

    for case in cases:
        got = evaluate(case)
        want = expected(case)
        require(got == want, f"{case['id']}: expected {want}, got {got}")

    print(f"ok fixtures={len(cases)} digest={digest}")


if __name__ == "__main__":
    main()
