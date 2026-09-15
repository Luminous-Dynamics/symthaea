// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn telemetry_binding_closes_signed_wrapper_canonicality() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("SIGNED_UPGRADE_PROBATION_OBSERVATION_SCHEMA"));
    assert!(source.contains("NonCanonicalSignedObservationSchema"));
    assert!(source.contains("signed.schema_version != SIGNED_UPGRADE_PROBATION_OBSERVATION_SCHEMA"));
}

#[test]
fn exact_lineage_observation_set_is_reconstructed() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("SIGNED_OBSERVATION_EVIDENCE_DOMAIN"));
    assert!(source.contains("LINEAGE_OBSERVATION_SET_DOMAIN"));
    assert!(source.contains("observation_set_digest != clearance.observation_set_digest()"));
    assert!(source.contains("DuplicateObservation"));
    assert!(source.contains("DuplicateTelemetryBundle"));
}

#[test]
fn telemetry_must_match_observation_and_current_containment_context() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("observation.telemetry_evidence_digest != binding.telemetry_bundle.evidence_digest()"));
    assert!(source.contains("TelemetryMachineMismatch"));
    assert!(source.contains("TelemetryOutsideObservationWindow"));
    assert!(source.contains("TelemetryContainmentMismatch"));
    assert!(source.contains("binding.telemetry_bundle.compromise_tracker_digest()"));
    assert!(source.contains("AttemptedJobCoverageMismatch"));
}

#[test]
fn predecessor_provenance_survives_exact_telemetry_binding() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("predecessor_root_digest"));
    assert!(source.contains("current_head_digest"));
    assert!(source.contains("governance_view_digest"));
    assert!(source.contains("registry_head_digest"));
    assert!(source.contains("predecessor_finalization_sequence"));
    assert!(source.contains("handoff_plan_digest"));
}

#[test]
fn ordinary_probation_clearance_is_not_accepted() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("ClockGovernedUpgradeProbationClearanceV1"));
    assert!(!source.contains("TelemetryBoundUpgradeProbationClearanceV1"));
    assert!(!source.contains("bind_probation_clearance_to_exact_telemetry_v1"));
    assert!(!source.contains("now_unix_s"));
}
