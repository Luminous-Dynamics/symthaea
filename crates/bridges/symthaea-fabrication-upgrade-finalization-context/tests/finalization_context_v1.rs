// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn context_is_not_irreversible_finalization_authority() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("QualifiedUpgradeFinalizationContextV1"));
    assert!(source.contains("signing_payload_digest"));
    assert!(!source.contains("AuthorizedUpgradeFinalization"));
    assert!(!source.contains("FinalizedUpgrade"));
}

#[test]
fn one_registry_containment_retention_and_rollback_view_is_required() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("governance_view: &ContainmentCurrentWitnessRegistryHeadV1"));
    assert!(source.contains("registry_head: &QuorumObservedWitnessRegistryHeadV1"));
    assert!(source.contains("retention_head: &CurrentEvidenceRetentionHeadV1"));
    assert!(source.contains("no_rollback: &CurrentNoRollbackUpgradeAuthorityV1"));
    assert!(source.contains("retention_head.governance_view_id() != governance_view.id()"));
    assert!(source.contains("no_rollback.governance_view_id() != governance_view.id()"));
    assert!(source.contains("no_rollback.governance_checkpoint_digest() != governance_view.checkpoint_digest()"));
}

#[test]
fn exact_handoff_activation_probation_and_telemetry_chain_is_rebound() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("activation.handoff_id() != handoff.id()"));
    assert!(source.contains("no_rollback.activation_permit_id() != activation.id()"));
    assert!(source.contains("probation.handoff_id() != handoff.id()"));
    assert!(source.contains("probation.activation_permit_id() != activation.id()"));
    assert!(source.contains("telemetry_bound_probation.clearance_id() != probation.id()"));
    assert!(source.contains("no_rollback.probation_clearance_digest() != Some(probation.id().as_digest())"));
}

#[test]
fn operational_trust_and_key_continuity_end_at_current_snapshot() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("no_rollback.key_snapshot_sequence() != registry_head.trust_snapshot_sequence()"));
    assert!(source.contains("key_continuity.successor_snapshot_digest() != registry_head.trust_snapshot_digest()"));
    assert!(source.contains("key_continuity.successor_snapshot_sequence() != registry_head.trust_snapshot_sequence()"));
}

#[test]
fn hardware_authorities_are_current_single_view_capabilities() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("authority.handoff_id() != handoff.id()"));
    assert!(source.contains("authority.probation_clearance_id() != probation.id()"));
    assert!(source.contains("authority.telemetry_bound_clearance_id() != telemetry_bound_probation.id()"));
    assert!(source.contains("authority.trust_snapshot_digest() != registry_head.trust_snapshot_digest()"));
    assert!(source.contains("authority.containment_state_digest() != governance_view.containment_state_digest()"));
    assert!(source.contains("authority.compromise_tracker_digest() != governance_view.compromise_tracker_digest()"));
    assert!(source.contains("authority.current_operational_basis_id() != current_basis.id()"));
    assert!(source.contains("authority.current_clock_envelope_id() != current_clock.id()"));
}

#[test]
fn hardware_machine_scope_is_unique_and_matches_operational_count() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("seen_machine_ids.insert(machine_id.clone())"));
    assert!(source.contains("DuplicateHardwareMachine"));
    assert!(source.contains("no_rollback.reauthorized_machine_count() != supplied_hardware_count"));
    assert!(source.contains("OperationalHardwareCountMismatch"));
    assert!(source.contains("hardware_authority_set_digest"));
}

#[test]
fn all_time_sensitive_inputs_are_valid_across_one_exact_clock() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("current_basis.id() != governance_view.observation_operational_basis_id()"));
    assert!(source.contains("current_clock.id() != governance_view.observation_clock_envelope_id()"));
    assert!(source.contains("current_clock.upper_unix_ms() >= handoff.plan().finalization_deadline_unix_ms"));
    assert!(source.contains("current_clock.upper_unix_ms() >= probation.clearance_expires_at_unix_ms()"));
    assert!(source.contains("expires_at_unix_ms <= current_clock.upper_unix_ms()"));
}

#[test]
fn hardware_physical_identity_is_committed_not_only_machine_name() {
    let source = include_str!("../src/lib.rs");
    for required in [
        "hardware_identity_digest",
        "machine_profile_digest",
        "firmware_digest",
        "calibration_digest",
        "capability_digest",
        "statement_digest",
        "signed_evidence_digest",
    ] {
        assert!(source.contains(required), "missing hardware commitment {required}");
    }
}

#[test]
fn context_is_opaque_and_legacy_scalar_finalizer_is_absent() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct QualifiedUpgradeFinalizationContextV1"
    ));
    assert!(!source.contains("AuthorizedUpgradeHandoff"));
    assert!(!source.contains("HardwareReauthorizationTracker"));
    assert!(!source.contains("AuthorizedEvidenceRetentionPolicy"));
    assert!(!source.contains("VerifiedKeyContinuity"));
    assert!(!source.contains("VerifiedClockContinuity"));
    assert!(!source.contains("Option<&AutomaticRollbackTrigger>"));
    assert!(!source.contains("now_unix_s:"));
    assert!(!source.contains("authorized_at_unix_ms:"));
}
