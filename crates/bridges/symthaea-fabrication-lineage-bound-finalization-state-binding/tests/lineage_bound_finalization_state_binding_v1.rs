// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn execution_and_authorization_are_exactly_rebound() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("execution_permit.authorization_id() != authorization.id()"));
    assert!(source.contains("execution_permit.context_id() != context.id()"));
    assert!(source.contains("execution_permit.authorized_no_rollback_id() != no_rollback.id()"));
    assert!(source.contains("context.activation_permit_id() != activation.id()"));
}

#[test]
fn concrete_upgrade_state_is_followed_from_the_exact_operational_digest() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("digest_upgrade_operational_state(operational_state)"));
    assert!(source.contains("operational_state.evidence.upgrade_state_digest != upgrade_state_digest"));
    assert!(source.contains("digest_upgrade_state(upgrade_state)"));
    assert!(source.contains("UpgradeStateDigestMismatch"));
}

#[test]
fn activated_state_is_bound_to_the_exact_global_upgrade_cycle() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("upgrade_state.active_stage != UpgradeStage::Activated"));
    assert!(source.contains("upgrade_state.evidence.handoff_digest != context.handoff_plan_digest()"));
    assert!(source.contains("upgrade_state.handoff_sequence != context.upgrade_cycle_sequence()"));
    assert!(source.contains("UpgradeStateSequenceMismatch"));
}

#[test]
fn state_commit_times_are_causally_ordered() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("upgrade_state.committed_at_unix_ms < activation.activates_at_unix_ms()"));
    assert!(source.contains("upgrade_state.committed_at_unix_ms > operational_state.committed_at_unix_ms"));
    assert!(source.contains("UpgradeStateBeforeActivation"));
    assert!(source.contains("UpgradeStateAfterOperationalCommit"));
}

#[test]
fn predecessor_provenance_survives_into_state_binding() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("execution_permit.predecessor_root_digest() != context.predecessor_root_digest()"));
    assert!(source.contains("activation.predecessor_root_digest() != context.predecessor_root_digest()"));
    assert!(source.contains("predecessor_current_head_digest"));
    assert!(source.contains("predecessor_finalization_sequence"));
    assert!(source.contains("upgrade_cycle_sequence"));
}

#[test]
fn no_rollback_state_is_exact_and_still_rollback_free() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("no_rollback.state_digest() != execution_permit.operational_state_digest()"));
    assert!(source.contains("no_rollback.operational_lineage_digest() != execution_permit.operational_lineage_digest()"));
    assert!(source.contains("operational_state.evidence.automatic_rollback_digest.is_some()"));
}

#[test]
fn binding_is_opaque_and_non_mutating() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("pub struct LineageBoundFinalizationStateBindingV1"));
    assert!(!source.contains("Serialize, Deserialize"));
    assert!(!source.contains("UpgradeStage::Finalized"));
    assert!(!source.contains("finalize_clock_governed_upgrade"));
    assert!(!source.contains("saturating_mul"));
    assert!(!source.contains(".expect("));
    assert!(!source.contains(".unwrap("));
}
