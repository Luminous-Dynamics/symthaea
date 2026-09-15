// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn hardware_preserves_exact_lineage_probation_and_telemetry_provenance() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("probation.lineage_handoff_id() != handoff.id()"));
    assert!(source.contains("telemetry.clearance_id() != probation.id()"));
    assert!(source.contains("probation.predecessor_root_digest()"));
    assert!(source.contains("probation.current_head_digest()"));
    assert!(source.contains("probation.governance_view_digest()"));
    assert!(source.contains("probation.registry_head_digest()"));
    assert!(source.contains("predecessor_finalization_sequence"));
}

#[test]
fn hardware_requires_same_trust_containment_and_compromise_context() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("trust_snapshot_digest != probation.trust_snapshot_digest()"));
    assert!(source.contains("containment_state_digest != probation.containment_state_digest()"));
    assert!(source.contains("containment_state_digest != telemetry.containment_state_digest()"));
    assert!(source.contains("compromise_tracker_digest != telemetry.compromise_tracker_digest()"));
    assert!(source.contains("TrustSnapshotPostdatesStatement"));
}

#[test]
fn hardware_time_and_signature_authority_fail_closed() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("checked_mul(1_000)"));
    assert!(source.contains("StatementTooOld"));
    assert!(source.contains("StatementExpired"));
    assert!(source.contains("StatementOutlivesHandoff"));
    assert!(source.contains("SignerInvalidForStatementWindow"));
    assert!(source.contains("SignerCompromisedAcrossEnvelope"));
    assert!(source.contains("verify_hardware_reauthorization_signature"));
    assert!(source.contains("for provider in verification_providers"));
    assert!(!source.contains("saturating_mul"));
}

#[test]
fn ordered_clock_lineage_and_exact_evidence_are_committed() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("ClockLineageCommitment"));
    assert!(source.contains("bridge_basis_ids"));
    assert!(source.contains("clock_lineage_digest"));
    assert!(source.contains("SIGNED_HARDWARE_EVIDENCE_DOMAIN"));
    assert!(source.contains("statement_digest"));
    assert!(source.contains("verifier_set_digest"));
}

#[test]
fn ordinary_live_upgrade_authority_does_not_reenter_hardware_api() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("use symthaea_fabrication_upgrade_authority::"));
    assert!(!source.contains("ClockGovernedUpgradeProbationClearanceV1"));
    assert!(!source.contains("TelemetryBoundUpgradeProbationClearanceV1"));
    assert!(!source.contains("ClockGovernedHardwareReauthorizationV1"));
    assert!(!source.contains("AuthorizedUpgradeHandoff"));
    assert!(!source.contains("now_unix_s"));
    assert!(!source.contains("evaluation_time_unix_s"));
}
