#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

CORPUS = Path("docs/release/evidence/semi-eqp-met-001e-physical-campaign-corpus-v1.json")
EXPECTED_SHA256 = "b5bc685d53fef2afd6435f2ef53ff5c4348f9a56de822f01f7d03ec35cb13341"
EXPECTED_SCHEMA = "semi-eqp-met-001e-physical-campaign-corpus-v1"
EXPECTED_AUTHORITY = "representation_only_no_physical_execution_authority"
EXPECTED_IDS = {
    "preregistered_current_reference_candidate_claim_run",
    "exploratory_run_cannot_be_promoted_retroactively",
    "heldout_region_used_for_tuning_invalidates_claim",
    "reference_target_identity_mismatch",
    "stale_reference_artifact_narrows_measurement",
    "stale_calibration_narrows_measurement",
    "remount_without_transfer_evidence_blocks_prior_calibration",
    "remount_with_fresh_calibration_preserves_history",
    "aborted_run_is_retained_not_passed",
    "partial_coverage_cannot_satisfy_full_profile",
    "repeat_session_supports_repeatability_not_new_target_replication",
    "software_replay_not_fresh_campaign_run",
    "derived_outputs_do_not_multiply_physical_witnesses",
    "missing_session_custody_is_evidence_incomplete",
    "acceptance_criteria_changed_after_reveal_invalidates_claim",
    "campaign_evidence_mints_no_execution_authority",
}


def derive(case):
    cid = case["id"]

    if cid == "preregistered_current_reference_candidate_claim_run":
        assert case["campaign_mode"] == "claim_bearing"
        assert case["constitution_frozen_before_acquisition"] is True
        assert case["target_currentness"] == "current"
        assert case["calibration_currentness"] == "current"
        assert case["heldout_leakage"] is False
        assert case["raw_observation_ref"] and case["session_ref"]
        return "expected_disposition", "CandidateClaimRunUnderFrozenProfile"

    if cid == "exploratory_run_cannot_be_promoted_retroactively":
        assert case["campaign_mode"] == "exploratory"
        assert case["constitution_frozen_before_acquisition"] is False
        assert case["posthoc_promotion_requested"] is True
        return "expected_error", "ExploratoryCannotBecomeClaimBearingRetroactively"

    if cid == "heldout_region_used_for_tuning_invalidates_claim":
        assert case["campaign_mode"] == "claim_bearing"
        assert case["constitution_frozen_before_acquisition"] is True
        assert case["heldout_leakage"] is True
        return "expected_error", "HeldoutLeakage"

    if cid == "reference_target_identity_mismatch":
        assert case["campaign_target_ref"] != case["observation_target_ref"]
        return "expected_error", "TargetIdentityMismatch"

    if cid == "stale_reference_artifact_narrows_measurement":
        assert case["target_currentness"] != "current"
        assert case["calibration_currentness"] == "current"
        return "expected_disposition", "ReferenceCurrentnessInsufficient"

    if cid == "stale_calibration_narrows_measurement":
        assert case["target_currentness"] == "current"
        assert case["calibration_currentness"] != "current"
        return "expected_disposition", "CalibrationCurrentnessInsufficient"

    if cid == "remount_without_transfer_evidence_blocks_prior_calibration":
        assert case["pre_remount_configuration_ref"] != case["post_remount_configuration_ref"]
        assert case["transfer_evidence_ref"] is None
        return "expected_error", "CalibrationTransferUnsupported"

    if cid == "remount_with_fresh_calibration_preserves_history":
        assert case["pre_remount_configuration_ref"] != case["post_remount_configuration_ref"]
        assert case["historical_calibration_refs"]
        assert case["fresh_calibration_ref"]
        assert case["fresh_calibration_currentness"] == "current"
        assert case["fresh_calibration_ref"] not in case["historical_calibration_refs"]
        assert case["expected_history"] == "prior_and_fresh_preserved"
        return "expected_disposition", "NewCalibrationContext"

    if cid == "aborted_run_is_retained_not_passed":
        assert case["run_state"] == "aborted"
        assert case["partial_raw_observation_refs"]
        return "expected_disposition", "AbortedRetainedNotQualified"

    if cid == "partial_coverage_cannot_satisfy_full_profile":
        assert case["required_coverage_ref"] != case["observed_coverage_ref"]
        return "expected_disposition", "CoverageIncomplete"

    if cid == "repeat_session_supports_repeatability_not_new_target_replication":
        assert len(case["session_refs"]) >= 2
        assert case["same_physical_target"] is True
        return "expected_relation", "RepeatedSessionSameArticleNotIndependentTargetReplication"

    if cid == "software_replay_not_fresh_campaign_run":
        assert case["raw_artifact_digest"] == case["replay_of_raw_artifact_digest"]
        assert case["fresh_physical_acquisition"] is False
        return "expected_relation", "SoftwareReplayNoFreshPhysicalRun"

    if cid == "derived_outputs_do_not_multiply_physical_witnesses":
        assert case["raw_observation_ref"]
        assert len(case["derived_artifact_refs"]) >= 2
        return "expected_physical_witness_count", 1

    if cid == "missing_session_custody_is_evidence_incomplete":
        assert case["raw_observation_ref"]
        assert case["session_ref"] is None
        return "expected_disposition", "EvidenceIncomplete"

    if cid == "acceptance_criteria_changed_after_reveal_invalidates_claim":
        assert case["constitution_frozen_before_acquisition"] is True
        assert case["results_revealed"] is True
        assert case["acceptance_criteria_changed_after_reveal"] is True
        return "expected_error", "PostRevealCriteriaChange"

    if cid == "campaign_evidence_mints_no_execution_authority":
        assert case["benchmark_result_ref"] and case["physical_campaign_result_ref"]
        assert case["execution_authority"] is False
        return "expected_authority", "none"

    raise AssertionError(f"unknown fixture id: {cid}")


def main():
    raw = CORPUS.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit(f"digest mismatch: {digest}")

    data = json.loads(raw)
    if data.get("schema") != EXPECTED_SCHEMA:
        raise SystemExit("schema mismatch")
    if data.get("authority") != EXPECTED_AUTHORITY:
        raise SystemExit("authority mismatch")

    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) != 16:
        raise SystemExit("case-count mismatch")

    ids = [c.get("id") for c in cases]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate fixture id")
    if set(ids) != EXPECTED_IDS:
        raise SystemExit("fixture-id set mismatch")

    for case in cases:
        field, derived = derive(case)
        actual = case.get(field)
        if actual != derived:
            raise SystemExit(f"{case['id']}: {field}={actual!r}, derived={derived!r}")

    print(f"ok fixtures={len(cases)} digest={digest}")


if __name__ == "__main__":
    main()
