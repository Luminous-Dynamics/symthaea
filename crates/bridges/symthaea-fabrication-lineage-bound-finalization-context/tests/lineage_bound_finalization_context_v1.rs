// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn predecessor_view_and_current_finalization_view_remain_distinct() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("predecessor_governance_view_digest"));
    assert!(source.contains("current_checkpoint_digest"));
    assert!(source.contains("CurrentGovernanceMismatch"));
    assert!(source.contains("PredecessorProvenanceMismatch"));
    assert!(!source.contains("predecessor_governance_view_digest == governance_view.id().as_digest()"));
}

#[test]
fn exact_lineage_survives_to_the_finalization_payload() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("activation.predecessor_root_digest() == predecessor_root_digest"));
    assert!(source.contains("probation.predecessor_root_digest() == predecessor_root_digest"));
    assert!(source.contains("telemetry.predecessor_root_digest() == predecessor_root_digest"));
    assert!(source.contains("evidence_binding.predecessor_root_digest() == predecessor_root_digest"));
    assert!(source.contains("no_rollback.predecessor_root_digest() == predecessor_root_digest"));
    assert!(source.contains("predecessor_finalization_sequence"));
    assert!(source.contains("upgrade_cycle_sequence"));
}

#[test]
fn exact_hardware_authority_set_is_independently_recomputed() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("symthaea.fabrication.lineage-bound-hardware-authority-set.v1\\0"));
    assert!(source.contains("authority_id: authority.id().to_hex()"));
    assert!(source.contains("hardware_identity_digest"));
    assert!(source.contains("machine_profile_digest"));
    assert!(source.contains("firmware_digest"));
    assert!(source.contains("calibration_digest"));
    assert!(source.contains("capability_digest"));
    assert!(source.contains("hardware_authority_set_digest != evidence_binding.hardware_authority_set_digest()"));
    assert!(source.contains("hardware_authority_set_digest != no_rollback.hardware_authority_set_digest()"));
    assert!(source.contains("hardware_authority_ids"));
}

#[test]
fn current_governance_time_and_key_continuity_fail_closed() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("key_continuity.containment_state_digest() != governance_view.containment_state_digest()"));
    assert!(source.contains("key_continuity.compromise_tracker_digest() != governance_view.compromise_tracker_digest()"));
    assert!(source.contains("current_basis.id() != no_rollback.observation_operational_basis_id()"));
    assert!(source.contains("current_clock.upper_unix_ms() >= activation.finalization_deadline_unix_ms()"));
    assert!(source.contains("current_clock.upper_unix_ms() >= probation.clearance_expires_at_unix_ms()"));
}

#[test]
fn exact_hardware_must_still_be_fresh_on_the_context_clock() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("authority.current_operational_basis_id() != current_basis.id()"));
    assert!(source.contains("authority.current_clock_envelope_id() != current_clock.id()"));
    assert!(source.contains("HardwareStatementMayBeFuture"));
    assert!(source.contains("HardwareMayExpire"));
    assert!(source.contains("checked_mul(1_000)"));
    assert!(!source.contains("saturating_mul"));
}

#[test]
fn ordinary_finalization_authority_cannot_reenter_the_waist() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("use symthaea_fabrication_upgrade_authority::"));
    assert!(!source.contains("handoff: &ClockGovernedUpgradeHandoffV1"));
    assert!(!source.contains("ClockGovernedUpgradeActivationPermitV1"));
    assert!(!source.contains("ClockGovernedUpgradeProbationClearanceV1"));
    assert!(!source.contains("ClockGovernedHardwareReauthorizationV1"));
    assert!(!source.contains("CurrentNoRollbackUpgradeAuthorityV1"));
    assert!(!source.contains("QualifiedUpgradeFinalizationContextV1"));
    assert!(!source.contains("AuthorizedUpgradeHandoff"));
    assert!(!source.contains("now_unix_s"));
}

#[test]
fn context_is_opaque_non_executable_authority() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("pub struct LineageBoundUpgradeFinalizationContextV1"));
    assert!(source.contains("pub fn signing_payload_digest(&self)"));
    assert!(!source.contains("Serialize, Deserialize"));
    assert!(!source.contains("finalize_clock_governed_upgrade"));
}
