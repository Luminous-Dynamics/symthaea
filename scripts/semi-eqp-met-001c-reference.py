#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

EXPECTED_SHA256 = "915225ff10857ce51e0d7a9e2dbd8c591d62d360e621b15a2f6d43a078b5686f"
EXPECTED_SCHEMA = "semi-eqp-met-001c-motion-corpus-v1"
EXPECTED_AUTHORITY = "representation_only_no_physical_execution_authority"
EXPECTED_IDS = (
    "command_without_physical_reference_observation",
    "command_and_controller_readback_agree_without_independent_reference",
    "command_differs_from_independent_observed_position",
    "clustered_revisits_under_same_configuration",
    "repeatability_without_absolute_reference",
    "current_absolute_reference_and_current_calibration",
    "stale_calibration_blocks_absolute_position_claim",
    "remount_changes_transform_identity",
    "remount_fresh_calibration_new_epoch_preserves_history",
    "correction_preserves_original_discrepancy",
    "same_friendly_stage_name_different_physical_configuration",
    "source_and_receive_timestamps_preserved_separately",
    "software_replay_is_not_fresh_repeatability_evidence",
    "local_reference_supports_local_position_claim_only",
    "multiple_corrected_derivations_from_one_raw_observation",
    "motion_evidence_cannot_mint_execution_authority",
)


def expect(case, key, derived):
    expected_key = f"expected_{key}"
    expected = case.get(expected_key)
    if expected != derived:
        raise AssertionError(
            f"{case['id']}: {expected_key}={expected!r}, derived={derived!r}"
        )


def derive(case):
    cid = case["id"]

    if cid == "command_without_physical_reference_observation":
        disposition = (
            "PositionUnobserved"
            if case.get("commanded_position_ref") and case.get("reference_observation_ref") is None
            else "Unexpected"
        )
        expect(case, "disposition", disposition)

    elif cid == "command_and_controller_readback_agree_without_independent_reference":
        disposition = (
            "ControllerConsistentOnly"
            if case.get("commanded_position_ref")
            and case.get("controller_readback_ref")
            and case.get("controller_relation") == "agrees_under_profile"
            and case.get("independent_reference_ref") is None
            else "Unexpected"
        )
        expect(case, "disposition", disposition)

    elif cid == "command_differs_from_independent_observed_position":
        disposition = (
            "DiscrepancyObserved"
            if case.get("commanded_position_ref")
            and case.get("independent_observed_position_ref")
            and case["commanded_position_ref"] != case["independent_observed_position_ref"]
            else "Unexpected"
        )
        expect(case, "disposition", disposition)

    elif cid == "clustered_revisits_under_same_configuration":
        refs = case.get("revisit_observation_refs") or []
        disposition = (
            "CandidateRepeatabilityEvidence"
            if case.get("configuration_ref")
            and case.get("revisit_relation") == "clustered_under_profile"
            and len(refs) >= 2
            and len(set(refs)) == len(refs)
            else "Unexpected"
        )
        expect(case, "disposition", disposition)

    elif cid == "repeatability_without_absolute_reference":
        disposition = (
            "RepeatabilityOnly"
            if case.get("configuration_ref")
            and case.get("revisit_relation") == "clustered_under_profile"
            and case.get("absolute_reference_ref") is None
            else "Unexpected"
        )
        expect(case, "disposition", disposition)

    elif cid == "current_absolute_reference_and_current_calibration":
        disposition = (
            "CandidateAbsolutePositionMeasurement"
            if case.get("absolute_reference_ref")
            and case.get("calibration_ref")
            and case.get("calibration_currentness") == "current"
            and case.get("configuration_ref")
            else "Unexpected"
        )
        expect(case, "disposition", disposition)

    elif cid == "stale_calibration_blocks_absolute_position_claim":
        disposition = (
            "CalibrationStaleOrMissing"
            if case.get("absolute_reference_ref")
            and case.get("calibration_ref")
            and case.get("calibration_currentness") != "current"
            else "Unexpected"
        )
        expect(case, "disposition", disposition)

    elif cid == "remount_changes_transform_identity":
        relation = (
            "distinct"
            if case.get("pre_remount_configuration_ref") != case.get("post_remount_configuration_ref")
            and case.get("pre_transform_ref") != case.get("post_transform_ref")
            else "same"
        )
        expect(case, "identity_relation", relation)

    elif cid == "remount_fresh_calibration_new_epoch_preserves_history":
        history = (
            "preserved_with_new_measurement_context"
            if case.get("pre_configuration_ref") != case.get("post_configuration_ref")
            and case.get("pre_calibration_ref") != case.get("post_calibration_ref")
            and case.get("historical_observation_ref")
            else "Unexpected"
        )
        expect(case, "history", history)

    elif cid == "correction_preserves_original_discrepancy":
        relation = (
            "derived_correction_preserves_original_discrepancy"
            if case.get("raw_observation_ref")
            and case.get("discrepancy_ref")
            and case.get("correction_transform_ref")
            and case.get("derived_corrected_position_ref")
            else "Unexpected"
        )
        expect(case, "lineage_relation", relation)

    elif cid == "same_friendly_stage_name_different_physical_configuration":
        relation = (
            "distinct"
            if case.get("friendly_name")
            and case.get("configuration_ref_a")
            and case.get("configuration_ref_b")
            and case["configuration_ref_a"] != case["configuration_ref_b"]
            else "same"
        )
        expect(case, "identity_relation", relation)

    elif cid == "source_and_receive_timestamps_preserved_separately":
        relation = (
            "preserve_distinct_no_simultaneity_inference"
            if case.get("source_clock_ref")
            and case.get("receive_clock_ref")
            and case.get("source_timestamp_ref")
            and case.get("receive_timestamp_ref")
            and (
                case["source_clock_ref"] != case["receive_clock_ref"]
                or case["source_timestamp_ref"] != case["receive_timestamp_ref"]
            )
            else "Unexpected"
        )
        expect(case, "timing_relation", relation)

    elif cid == "software_replay_is_not_fresh_repeatability_evidence":
        witnesses = 0 if case.get("original_observation_ref") and case.get("replay_ref") else None
        expect(case, "fresh_physical_witnesses", witnesses)
        disposition = "SoftwareReplayOnly" if witnesses == 0 else "Unexpected"
        expect(case, "disposition", disposition)

    elif cid == "local_reference_supports_local_position_claim_only":
        disposition = (
            "CoverageLimited"
            if case.get("reference_observation_ref")
            and case.get("reference_scope") == "local_region"
            and case.get("requested_scope") == "whole_sample"
            else "Unexpected"
        )
        expect(case, "disposition", disposition)

    elif cid == "multiple_corrected_derivations_from_one_raw_observation":
        derived = case.get("derived_corrected_refs") or []
        witnesses = 1 if case.get("raw_observation_ref") and len(derived) >= 2 else None
        expect(case, "physical_witness_count", witnesses)
        relation = (
            "multiple_derivations_one_physical_witness"
            if witnesses == 1
            else "Unexpected"
        )
        expect(case, "lineage_relation", relation)

    elif cid == "motion_evidence_cannot_mint_execution_authority":
        authority = (
            "none"
            if case.get("motion_evidence_ref")
            and case.get("calibration_ref")
            and case.get("execution_authority") is False
            else "Unexpected"
        )
        expect(case, "authority", authority)

    else:
        raise AssertionError(f"unknown fixture id: {cid}")


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "docs/release/evidence/semi-eqp-met-001c-motion-corpus-v1.json"
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
