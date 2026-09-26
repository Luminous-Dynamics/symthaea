#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

EXPECTED_SHA256 = "8034e7d16f375fcfe69e210629bda9d6fdda6c29e5eb59878c5bbf93f3a3e64b"
EXPECTED_SCHEMA = "semi-eqp-met-001d-optical-corpus-v1"
EXPECTED_AUTHORITY = "representation_only_no_physical_execution_authority"
EXPECTED_IDS = (
    "sharp_image_without_scale_reference",
    "current_scale_reference_and_current_calibration",
    "nominal_magnification_without_physical_scale_reference",
    "center_field_calibrated_edges_uncharacterized",
    "current_calibration_with_stale_reference_artifact",
    "same_friendly_camera_changed_optics",
    "material_focus_or_zoom_change_requires_transfer_evidence",
    "remount_changes_optical_transform",
    "distortion_correction_is_derived",
    "software_superresolution_does_not_create_physical_resolution_witness",
    "crop_resample_one_raw_one_physical_witness",
    "local_scale_reference_requested_for_whole_field",
    "derived_scalar_without_raw_image",
    "recalibration_new_epoch_preserves_history",
    "software_replay_not_fresh_optical_acquisition",
    "optical_evidence_cannot_mint_execution_authority",
)


def expect(case, key, derived):
    expected_key = f"expected_{key}"
    expected = case.get(expected_key)
    if expected != derived:
        raise AssertionError(f"{case['id']}: {expected_key}={expected!r}, derived={derived!r}")


def derive(case):
    cid = case["id"]

    if cid == "sharp_image_without_scale_reference":
        expect(case, "disposition", "QualitativeOnly" if case.get("raw_observation_ref") and case.get("scale_reference_ref") is None else "Unexpected")
    elif cid == "current_scale_reference_and_current_calibration":
        expect(case, "disposition", "CandidateLocalDimensionalMeasurement" if case.get("raw_observation_ref") and case.get("scale_reference_ref") and case.get("reference_currentness") == "current" and case.get("calibration_ref") and case.get("calibration_currentness") == "current" and case.get("coverage_profile_ref") else "Unexpected")
    elif cid == "nominal_magnification_without_physical_scale_reference":
        expect(case, "disposition", "UncalibratedNominalScaleOnly" if case.get("nominal_magnification_ref") and case.get("scale_reference_ref") is None else "Unexpected")
    elif cid == "center_field_calibrated_edges_uncharacterized":
        expect(case, "disposition", "CoverageLimited" if case.get("calibration_ref") and case.get("calibration_currentness") == "current" and case.get("coverage_profile_ref") == "coverage:center-only" and case.get("edge_characterization_ref") is None else "Unexpected")
    elif cid == "current_calibration_with_stale_reference_artifact":
        expect(case, "disposition", "ReferenceStaleOrMissing" if case.get("calibration_currentness") == "current" and case.get("scale_reference_ref") and case.get("reference_currentness") != "current" else "Unexpected")
    elif cid == "same_friendly_camera_changed_optics":
        expect(case, "identity_relation", "distinct" if case.get("friendly_camera_name") and case.get("imager_ref") and case.get("optics_ref_a") != case.get("optics_ref_b") else "same")
    elif cid == "material_focus_or_zoom_change_requires_transfer_evidence":
        expect(case, "disposition", "CalibrationTransferUnestablished" if case.get("configuration_ref_a") != case.get("configuration_ref_b") and case.get("material_setting_change") and case.get("transfer_evidence_ref") is None else "Unexpected")
    elif cid == "remount_changes_optical_transform":
        expect(case, "identity_relation", "distinct" if case.get("pre_mount_ref") != case.get("post_mount_ref") and case.get("pre_transform_ref") != case.get("post_transform_ref") else "same")
    elif cid == "distortion_correction_is_derived":
        expect(case, "lineage_relation", "derived_from_raw_no_new_physical_observation" if case.get("raw_observation_ref") and case.get("distortion_transform_ref") and case.get("derived_image_ref") else "Unexpected")
    elif cid == "software_superresolution_does_not_create_physical_resolution_witness":
        witnesses = 1 if case.get("raw_observation_ref") and case.get("derived_superresolution_ref") else None
        expect(case, "physical_witness_count", witnesses)
        expect(case, "disposition", "DerivedEnhancementOnly" if witnesses == 1 else "Unexpected")
    elif cid == "crop_resample_one_raw_one_physical_witness":
        derived = case.get("derived_artifact_refs") or []
        witnesses = 1 if case.get("raw_observation_ref") and len(derived) >= 2 else None
        expect(case, "physical_witness_count", witnesses)
    elif cid == "local_scale_reference_requested_for_whole_field":
        expect(case, "disposition", "CoverageLimited" if case.get("scale_reference_ref") and case.get("scale_reference_scope") == "local_region" and case.get("requested_scope") == "whole_field" else "Unexpected")
    elif cid == "derived_scalar_without_raw_image":
        expect(case, "error", "LineageInvalid" if case.get("derived_measurement_ref") and case.get("raw_observation_ref") is None else "Unexpected")
    elif cid == "recalibration_new_epoch_preserves_history":
        expect(case, "history", "both_preserved" if case.get("historical_raw_observation_ref") and case.get("old_calibration_ref") != case.get("new_calibration_ref") else "Unexpected")
    elif cid == "software_replay_not_fresh_optical_acquisition":
        witnesses = 0 if case.get("raw_observation_ref") and case.get("replay_ref") else None
        expect(case, "fresh_physical_witnesses", witnesses)
        expect(case, "disposition", "SoftwareReplayOnly" if witnesses == 0 else "Unexpected")
    elif cid == "optical_evidence_cannot_mint_execution_authority":
        expect(case, "authority", "none" if case.get("optical_evidence_ref") and case.get("execution_authority") is False else "Unexpected")
    else:
        raise AssertionError(f"unknown fixture id: {cid}")


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "docs/release/evidence/semi-eqp-met-001d-optical-corpus-v1.json"
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise AssertionError(f"digest drift: {digest}")
    corpus = json.loads(raw)
    if corpus.get("schema") != EXPECTED_SCHEMA:
        raise AssertionError("schema drift")
    if corpus.get("authority") != EXPECTED_AUTHORITY:
        raise AssertionError("authority drift")
    cases = corpus.get("cases")
    if not isinstance(cases, list) or len(cases) != 16:
        raise AssertionError("case-count drift")
    ids = tuple(case.get("id") for case in cases)
    if ids != EXPECTED_IDS:
        raise AssertionError(f"fixture-id/order drift: {ids!r}")
    if len(set(ids)) != len(ids):
        raise AssertionError("duplicate fixture id")
    for case in cases:
        derive(case)
    print(f"ok fixtures={len(cases)} digest={digest}")


if __name__ == "__main__":
    main()
