// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

const SOURCE: &str = include_str!("../src/lib.rs");

#[test]
fn execution_permit_must_name_the_exact_fresh_no_rollback_capability() {
    assert!(SOURCE.contains(
        "execution_permit.fresh_no_rollback_id() != fresh_no_rollback.id()"
    ));
    assert!(SOURCE.contains("NoRollbackPermitMismatch"));
}

#[test]
fn concrete_operational_state_must_match_the_fresh_currentness_digest() {
    assert!(SOURCE.contains("digest_upgrade_operational_state(operational_state)"));
    assert!(SOURCE.contains("operational_state_digest != fresh_no_rollback.state_digest()"));
    assert!(SOURCE.contains("operational_state.generation != fresh_no_rollback.state_generation()"));
    assert!(SOURCE.contains("operational_state.handoff_digest != fresh_no_rollback.handoff_plan_digest()"));
    assert!(SOURCE.contains("automatic_rollback_digest.is_some()"));
}

#[test]
fn concrete_upgrade_state_is_followed_through_the_operational_digest() {
    assert!(SOURCE.contains("digest_upgrade_state(upgrade_state)"));
    assert!(SOURCE.contains("operational_state.evidence.upgrade_state_digest != upgrade_state_digest"));
    assert!(SOURCE.contains("upgrade_state.active_stage != UpgradeStage::Activated"));
    assert!(SOURCE.contains("upgrade_state.evidence.handoff_digest != fresh_no_rollback.handoff_plan_digest()"));
}

#[test]
fn operational_evidence_counters_are_cross_checked_not_inferred() {
    for needle in [
        "probation_clearance_digest",
        "probation_sequence",
        "reauthorized_machine_count",
        "retention_policy_digest",
        "retention_policy_sequence",
        "key_snapshot_sequence",
        "clock_epoch",
    ] {
        assert!(SOURCE.contains(needle), "missing operational cross-check: {needle}");
    }
}

#[test]
fn binding_is_opaque_and_contains_no_scalar_time_or_mutation_api() {
    assert!(SOURCE.contains("ExecutionBoundUpgradeFinalizationStateV1"));
    assert!(!SOURCE.contains("now_unix_s"));
    assert!(!SOURCE.contains("authorized_at_unix_ms"));
    assert!(!SOURCE.contains("FabricationUpgradeState::successor"));
    assert!(!SOURCE.contains("UpgradeStage::Finalized"));
    assert!(!SOURCE.contains("Deserialize"));
}
