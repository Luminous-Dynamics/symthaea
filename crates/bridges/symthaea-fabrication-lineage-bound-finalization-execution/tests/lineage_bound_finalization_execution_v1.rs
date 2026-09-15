// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn execution_requires_strict_log_and_time_advancement() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("fresh_log.verify_successor_of(authorized_log).is_err()"));
    assert!(source.contains("fresh_log.entries.len() <= authorized_log.entries.len()"));
    assert!(source.contains("fresh_clock.lower_unix_ms() <= authorized_clock.upper_unix_ms()"));
    assert!(source.contains("ExecutionClockNotDefinitelyLater"));
    assert!(source.contains("verify_clock_lineage("));
    assert!(!source.contains("now_unix_s"));
    assert!(!source.contains("evaluation_time_unix_s"));
}

#[test]
fn execution_freezes_governance_semantics_but_allows_new_view_ids() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("fresh_registry_head.activated_registry_id() != authorized_registry_head.activated_registry_id()"));
    assert!(source.contains("fresh_registry_head.registry_digest() != authorized_registry_head.registry_digest()"));
    assert!(source.contains("fresh_registry_head.trust_snapshot_digest() != authorized_registry_head.trust_snapshot_digest()"));
    assert!(source.contains("fresh_governance_view.containment_authority_id() != authorized_governance_view.containment_authority_id()"));
    assert!(source.contains("fresh_retention_head.retention_authority_id() != authorized_retention_head.retention_authority_id()"));
    assert!(source.contains("FreshCheckpointNotDifferent"));
}

#[test]
fn any_post_authorization_operational_head_for_this_handoff_forces_reauthorization() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("upgrade_operational_head_log_kind(context.handoff_plan_digest())"));
    assert!(source.contains("&fresh_log.entries[authorized_log.entries.len()..]"));
    assert!(source.contains("OperationalHeadAppended"));
}

#[test]
fn terminal_prepublication_fails_closed() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("finalized_upgrade_head_log_kind(context.handoff_plan_digest())"));
    assert!(source.contains("for entry in &fresh_log.entries"));
    assert!(source.contains("FinalizedHeadAlreadyPublished"));
}

#[test]
fn hardware_refresh_is_exact_evidence_requalification_not_replacement() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("fresh.statement() != authorized.statement()"));
    assert!(source.contains("fresh.signed_evidence_digest() != authorized.signed_evidence_digest()"));
    assert!(source.contains("fresh.hardware_policy_digest() != authorized.hardware_policy_digest()"));
    assert!(source.contains("fresh.verifier_set_digest() != authorized.verifier_set_digest()"));
    assert!(source.contains("fresh.current_operational_basis_id() != fresh_basis.id()"));
    assert!(source.contains("fresh.current_clock_envelope_id() != fresh_clock.id()"));
    assert!(source.contains("hardware_identity_digest"));
    assert!(source.contains("machine_profile_digest"));
    assert!(source.contains("firmware_digest"));
    assert!(source.contains("calibration_digest"));
    assert!(source.contains("capability_digest"));
}

#[test]
fn exact_authorized_hardware_set_is_rebound() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("context.hardware_authority_ids().to_vec()"));
    assert!(source.contains("context.machine_ids().to_vec()"));
    assert!(source.contains("AuthorizedHardwareSetMismatch"));
    assert!(source.contains("DuplicateAuthorizedHardwareAuthority"));
    assert!(source.contains("DuplicateFreshHardwareAuthority"));
}

#[test]
fn execution_does_not_rewrite_durable_operational_state() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("AuthorizedLineageBoundFinalizationV1"));
    assert!(source.contains("LineageBoundCurrentNoRollbackV1"));
    assert!(!source.contains("FabricationUpgradeOperationalState"));
    assert!(!source.contains("build_lineage_bound_no_rollback_operational_evidence_v1"));
    assert!(!source.contains("derive_lineage_bound_current_no_rollback"));
}

#[test]
fn execution_permit_remains_opaque_and_non_mutating() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("pub struct LineageBoundFinalizationExecutionPermitV1"));
    assert!(!source.contains("Serialize, Deserialize"));
    assert!(!source.contains("AuthorizedClockGovernedUpgradeFinalizationV1"));
    assert!(!source.contains("ClockGovernedUpgradeFinalizationExecutionPermitV1"));
    assert!(!source.contains("finalize_clock_governed_upgrade"));
    assert!(!source.contains("saturating_mul"));
    assert!(!source.contains(".expect("));
    assert!(!source.contains(".unwrap("));
}
