// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

const SOURCE: &str = include_str!("../src/lib.rs");

#[test]
fn execution_requires_strict_authenticated_log_extension() {
    assert!(SOURCE.contains("fresh_log.entries.len() <= authorized_log.entries.len()"));
    assert!(SOURCE.contains("fresh_log.verify_successor_of(authorized_log).is_err()"));
    assert!(SOURCE.contains("TransparencyLogNotStrictExtension"));
    assert!(SOURCE.contains("fresh_governance_view.checkpoint_digest() == context.governance_checkpoint_digest()"));
}

#[test]
fn execution_time_must_be_definitely_later_and_still_inside_all_live_windows() {
    assert!(SOURCE.contains("fresh_clock.lower_unix_ms() <= authorized_clock.upper_unix_ms()"));
    assert!(SOURCE.contains("ExecutionClockNotDefinitelyLater"));
    assert!(SOURCE.contains("fresh_clock.upper_unix_ms() >= context.finalization_deadline_unix_ms()"));
    assert!(SOURCE.contains("fresh_clock.upper_unix_ms() >= context.probation_clearance_expires_at_unix_ms()"));
    assert!(SOURCE.contains("expires_at_unix_ms <= fresh_clock.upper_unix_ms()"));
    assert!(SOURCE.contains("checked_mul(1_000)"));
    assert!(!SOURCE.contains("saturating_mul"));
}

#[test]
fn governance_semantics_are_frozen_across_the_newer_view() {
    for needle in [
        "RegistrySemanticsChanged",
        "TrustSnapshotChanged",
        "ContainmentSemanticsChanged",
        "RetentionSemanticsChanged",
        "OperationalSemanticsChanged",
        "authorized_registry_head.activated_registry_id() != fresh_registry_head.activated_registry_id()",
        "authorized_governance_view.containment_authority_id()",
        "authorized_retention_head.retention_authority_id()",
        "authorized_no_rollback.operational_lineage_digest()",
    ] {
        assert!(SOURCE.contains(needle), "missing semantic-freeze ratchet: {needle}");
    }
}

#[test]
fn original_hardware_set_is_rebound_and_every_machine_is_freshly_requalified() {
    for needle in [
        "AUTHORIZED_HARDWARE_SET_DOMAIN",
        "context.hardware_authority_set_digest()",
        "HardwareStatementChanged",
        "HardwareSignedEvidenceChanged",
        "HardwarePolicyChanged",
        "HardwareVerifierSetChanged",
        "HardwareTrustChanged",
        "HardwareContainmentChanged",
        "FreshHardwareClockMismatch",
        "fresh_authority_id",
    ] {
        assert!(SOURCE.contains(needle), "missing hardware refresh ratchet: {needle}");
    }
}

#[test]
fn execution_is_capability_only_and_has_no_scalar_current_time_api() {
    assert!(SOURCE.contains("ClockGovernedUpgradeFinalizationExecutionPermitV1"));
    assert!(!SOURCE.contains("AuthorizedUpgradeFinalization"));
    assert!(!SOURCE.contains("VerifiedClockContinuity"));
    assert!(!SOURCE.contains("VerifiedThresholdCeremony"));
    assert!(!SOURCE.contains("now_unix_s"));
    assert!(!SOURCE.contains("authorized_at_unix_ms"));
    assert!(!SOURCE.contains("execute_finalization"));
    assert!(!SOURCE.contains("retire_predecessor"));
    assert!(!SOURCE.contains("FabricationUpgradeState"));
}

#[test]
fn live_execution_capability_is_not_deserializable() {
    let declaration = SOURCE
        .split("pub struct ClockGovernedUpgradeFinalizationExecutionPermitV1")
        .next()
        .expect("execution permit declaration prefix");
    let derive_window = declaration
        .rsplit("#[derive(")
        .next()
        .expect("derive window");
    assert!(!derive_window.contains("Deserialize"));
}
