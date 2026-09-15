// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn durable_operational_evidence_binds_lineage_probation_telemetry_and_hardware() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("probation_clearance_digest: Some(binding.probation_clearance_id.as_digest())"));
    assert!(source.contains("probation_tracker_digest: binding.probation_tracker_digest"));
    assert!(source.contains("hardware_reauthorization_tracker_digest: binding.hardware_authority_set_digest"));
    assert!(source.contains("reauthorized_machine_count"));
    assert!(source.contains("telemetry_binding_set_digest"));
}

#[test]
fn durable_clock_semantics_are_separate_from_fresh_observation_lineage() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("durable_clock_continuity_digest: activation.clock_lineage_digest()"));
    assert!(source.contains("durable_clock_epoch: activation_basis.epoch()"));
    assert!(source.contains("observation_clock_lineage_digest"));
    assert!(source.contains("digest_observation_clock_lineage"));
    assert!(source.contains("bridge_basis_ids"));
}

#[test]
fn current_no_rollback_requires_complete_exact_publication_chain() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("states.len() != publications.len() || states.len() != matching_entries.len()"));
    assert!(source.contains("verify_upgrade_operational_state_successor"));
    assert!(source.contains("build_upgrade_operational_head_publication_v1"));
    assert!(source.contains("entry.subject_digest != publication_digest"));
    assert!(source.contains("RollbackObserved"));
    assert!(source.contains("PublicationMayBeFuture"));
}

#[test]
fn candidate_state_must_commit_exact_hardened_current_evidence() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("CandidateEvidenceMismatch"));
    assert!(source.contains("binding.retention_policy_digest"));
    assert!(source.contains("binding.key_continuity_id.as_digest()"));
    assert!(source.contains("binding.hardware_authority_set_digest"));
    assert!(source.contains("Some(binding.upgrade_cycle_sequence)"));
    assert!(source.contains("CandidateCommittedBeforeEvidence"));
}

#[test]
fn operational_states_cannot_predate_activation() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("state.committed_at_unix_ms < binding.activates_at_unix_ms"));
    assert!(source.contains("StateBeforeActivation"));
}

#[test]
fn ordinary_live_upgrade_authority_does_not_reenter_the_api() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("use symthaea_fabrication_upgrade_authority::"));
    assert!(!source.contains("ClockGovernedUpgradeActivationPermitV1"));
    assert!(!source.contains("ClockGovernedUpgradeProbationClearanceV1"));
    assert!(!source.contains("ClockGovernedHardwareReauthorizationV1"));
    assert!(!source.contains("CurrentNoRollbackUpgradeAuthorityV1"));
    assert!(!source.contains("AuthorizedUpgradeHandoff"));
    assert!(!source.contains("now_unix_s"));
    assert!(!source.contains("evaluation_time_unix_s"));
}

#[test]
fn live_authority_types_are_not_serializable_constructors() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("pub struct LineageBoundOperationalEvidenceBindingV1"));
    assert!(source.contains("pub struct LineageBoundCurrentNoRollbackV1"));
    assert!(!source.contains("Serialize, Deserialize"));
}
