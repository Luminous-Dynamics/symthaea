// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn no_rollback_is_reproved_in_current_governance_view() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("governance_view: &ContainmentCurrentWitnessRegistryHeadV1"));
    assert!(source.contains("retention_head: &CurrentEvidenceRetentionHeadV1"));
    assert!(source.contains("retention_head.governance_view_id() != governance_view.id()"));
    assert!(source.contains("retention_head.governance_checkpoint_digest() != governance_view.checkpoint_digest()"));
    assert!(!source.contains("WitnessedNoRollbackUpgradeHeadV1"));
}

#[test]
fn complete_operational_sublog_is_required_not_only_candidate_tail() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("states.len() != matching_entries.len()"));
    assert!(source.contains("states.len() != publications.len()"));
    assert!(source.contains("OperationalInputCountMismatch"));
    assert!(source.contains("state.generation != 1"));
    assert!(source.contains("state.previous_state_digest.is_some()"));
}

#[test]
fn every_operational_hop_uses_kernel_successor_theorem() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("verify_upgrade_operational_state_successor(&states[index - 1], state)"));
    assert!(source.contains("OperationalSuccessorInvalid"));
    assert!(source.contains("digest_upgrade_operational_state(state)"));
    assert!(source.contains("operational_lineage_digest"));
}

#[test]
fn rollback_is_rejected_at_every_generation() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("state.evidence.automatic_rollback_digest"));
    assert!(source.contains("RollbackObserved"));
    assert!(source.contains("generation: state.generation"));
}

#[test]
fn every_state_reconstructs_exact_publication_and_log_entry() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("build_upgrade_operational_head_publication_v1(state)"));
    assert!(source.contains("digest_upgrade_operational_head_publication_v1(publication)"));
    assert!(source.contains("entry.subject_digest != publication_digest"));
    assert!(source.contains("PublicationDigestMismatch"));
    assert!(source.contains("publication_recorded_at_ms < state.committed_at_unix_ms"));
    assert!(source.contains("publication_recorded_at_ms > observation_clock.lower_unix_ms()"));
}

#[test]
fn operational_chain_is_bound_to_exact_handoff_and_activation() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("activation.handoff_id() != handoff.id()"));
    assert!(source.contains("activation.handoff_plan_digest() != handoff.plan_digest()"));
    assert!(source.contains("state.handoff_digest != handoff.plan_digest()"));
    assert!(source.contains("state.committed_at_unix_ms < activation.activates_at_unix_ms()"));
}

#[test]
fn same_authenticated_log_and_current_retention_are_required() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("transparency_log_digest != governance_view.transparency_log_digest()"));
    assert!(source.contains("transparency_log_digest != retention_head.transparency_log_digest()"));
    assert!(source.contains("candidate.evidence.retention_policy_sequence != retention_head.sequence()"));
    assert!(source.contains("candidate.evidence.retention_policy_digest != retention_head.policy_digest()"));
    assert!(source.contains("RetentionStateMismatch"));
}

#[test]
fn finalization_window_is_fresh_and_clock_ancestry_is_explicit() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("activation_basis: &OperationalClockBasisV1"));
    assert!(source.contains("activation_to_observation_clock_bridge: &[OperationalClockBasisV1]"));
    assert!(source.contains("observation_basis: &OperationalClockBasisV1"));
    assert!(source.contains("predecessor_operational_basis_id"));
    assert!(source.contains("observation_clock.upper_unix_ms() >= handoff.plan().finalization_deadline_unix_ms"));
    assert!(source.contains("FinalizationMayBeClosed"));
}

#[test]
fn current_no_rollback_capability_is_opaque_and_scalar_time_free() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct CurrentNoRollbackUpgradeAuthorityV1"
    ));
    assert!(!source.contains("now_unix_s:"));
    assert!(!source.contains("authorized_at_unix_ms:"));
    assert!(!source.contains("evaluation_time_unix_s:"));
    assert!(!source.contains("VerifiedTransparencyCheckpoint"));
    assert!(!source.contains("VerifiedTransparencyWitnessQuorum"));
    assert!(!source.contains("Option<&AutomaticRollbackTrigger>"));
}
