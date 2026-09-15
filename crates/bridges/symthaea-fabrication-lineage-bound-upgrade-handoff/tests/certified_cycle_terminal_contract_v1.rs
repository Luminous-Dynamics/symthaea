// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lockfile-neutral ratchets for the terminal half of one certified lineage upgrade cycle.
//!
//! These tests intentionally inspect exact sibling source contracts instead of adding upward Cargo
//! dependencies to the handoff crate. They prove that predecessor provenance survives finalization
//! authorization, fresh execution, concrete-state binding, deterministic terminalization, witnessed
//! finalized-head currentness, global sequence currentness, and next-cycle predecessor-root derivation.

#[test]
fn finalization_authorization_signs_the_exact_lineage_context() {
    let authorization = include_str!(
        "../../symthaea-fabrication-lineage-bound-finalization-authorization/src/lib.rs"
    );

    assert!(authorization.contains("pub fn prepare_lineage_bound_finalization_authorization_v1"));
    assert!(authorization.contains("context: LineageBoundUpgradeFinalizationContextV1"));
    assert!(authorization.contains("pub fn authorize_lineage_bound_finalization_v1"));
    assert!(authorization.contains("LINEAGE_BOUND_UPGRADE_FINALIZATION_PURPOSE"));
    assert!(authorization.contains("predecessor_root_digest: context.predecessor_root_digest().to_hex()"));
    assert!(authorization.contains("predecessor_current_head_digest: context.predecessor_current_head_digest().to_hex()"));
    assert!(authorization.contains("upgrade_cycle_sequence: context.upgrade_cycle_sequence()"));
    assert!(authorization.contains("operational_lineage_digest: context.operational_lineage_digest().to_hex()"));
    assert!(authorization.contains("hardware_authority_set_digest: context.hardware_authority_set_digest().to_hex()"));
}

#[test]
fn fresh_execution_refuses_post_authorization_state_drift() {
    let execution = include_str!(
        "../../symthaea-fabrication-lineage-bound-finalization-execution/src/lib.rs"
    );

    assert!(execution.contains("authorization: &AuthorizedLineageBoundFinalizationV1"));
    assert!(execution.contains("authorized_no_rollback: &LineageBoundCurrentNoRollbackV1"));
    assert!(execution.contains("fresh_governance_view: &ContainmentCurrentWitnessRegistryHeadV1"));
    assert!(execution.contains("hardware_refreshes: &[LineageBoundFinalizationHardwareRefreshInputV1<'_>]"));
    assert!(execution.contains("TransparencyLogNotStrictExtension"));
    assert!(execution.contains("OperationalHeadAppended"));
    assert!(execution.contains("FinalizedHeadAlreadyPublished"));
    assert!(execution.contains("ExecutionClockNotDefinitelyLater"));
    assert!(execution.contains("HardwareStatementChanged"));
    assert!(execution.contains("HardwarePolicyChanged"));
    assert!(execution.contains("HardwareVerifierSetChanged"));
}

#[test]
fn exact_state_binding_follows_the_authorized_operational_state_digest() {
    let binding = include_str!(
        "../../symthaea-fabrication-lineage-bound-finalization-state-binding/src/lib.rs"
    );

    assert!(binding.contains("pub fn bind_lineage_bound_finalization_state_v1"));
    assert!(binding.contains("execution_permit: &LineageBoundFinalizationExecutionPermitV1"));
    assert!(binding.contains("no_rollback: &LineageBoundCurrentNoRollbackV1"));
    assert!(binding.contains("activation: &LineageBoundUpgradeActivationPermitV1"));
    assert!(binding.contains("operational_state.evidence.upgrade_state_digest != upgrade_state_digest"));
    assert!(binding.contains("upgrade_state.active_stage != UpgradeStage::Activated"));
    assert!(binding.contains("upgrade_state.evidence.handoff_digest != context.handoff_plan_digest()"));
    assert!(binding.contains("upgrade_state.committed_at_unix_ms < activation.activates_at_unix_ms()"));
    assert!(binding.contains("upgrade_state.committed_at_unix_ms > operational_state.committed_at_unix_ms"));
}

#[test]
fn deterministic_terminalization_keeps_predecessor_provenance_and_full_successor_endpoint() {
    let finalized = include_str!(
        "../../symthaea-fabrication-lineage-bound-finalized-state/src/lib.rs"
    );

    assert!(finalized.contains("pub fn finalize_lineage_bound_upgrade_v1"));
    assert!(finalized.contains("authorization: &AuthorizedLineageBoundFinalizationV1"));
    assert!(finalized.contains("execution_permit: &LineageBoundFinalizationExecutionPermitV1"));
    assert!(finalized.contains("state_binding: &LineageBoundFinalizationStateBindingV1"));
    assert!(finalized.contains("handoff: &LineageBoundClockGovernedUpgradeHandoffV1"));
    assert!(finalized.contains("handoff.predecessor_root_id().as_digest() != context.predecessor_root_digest()"));
    assert!(finalized.contains("predecessor_finalization_sequence()"));
    assert!(finalized.contains(".checked_add(1)"));
    assert!(finalized.contains("successor_endpoint: handoff.plan().successor.clone()"));
    assert!(finalized.contains("terminal_stage: UpgradeStage::Finalized"));
    assert!(finalized.contains("fresh_checkpoint_digest: execution_permit.fresh_checkpoint_digest()"));
    assert!(finalized.contains("fresh_transparency_log_digest: execution_permit.fresh_transparency_log_digest()"));
}

#[test]
fn handoff_scoped_finalized_head_requires_append_only_later_observation() {
    let head = include_str!(
        "../../symthaea-fabrication-lineage-bound-finalized-head/src/lib.rs"
    );

    assert!(head.contains("pub fn build_lineage_bound_finalized_upgrade_head_publication_v1"));
    assert!(head.contains("pub fn qualify_current_lineage_bound_finalized_upgrade_head_v1"));
    assert!(head.contains("LogNotStrictExtension"));
    assert!(head.contains("ObservationClockNotDefinitelyLater"));
    assert!(head.contains("LineageHeadPrepublished"));
    assert!(head.contains("ConflictingFinalizationPublished"));
    assert!(head.contains("publication_count: usize"));
    assert!(head.contains("current_transparency_log_digest: Sha256Digest"));
    assert!(head.contains("current_checkpoint_digest: Sha256Digest"));
}

#[test]
fn global_currentness_replays_state_lineage_and_rejects_sequence_ambiguity() {
    let global = include_str!(
        "../../symthaea-fabrication-lineage-bound-global-upgrade-head/src/lib.rs"
    );

    assert!(global.contains("pub fn verify_lineage_bound_finalized_state_lineage_v1"));
    assert!(global.contains("verify_upgrade_state_successor"));
    assert!(global.contains("pub fn qualify_global_current_lineage_bound_finalized_head_v1"));
    assert!(global.contains("FinalizedSequenceEquivocation"));
    assert!(global.contains("FinalizedSequenceGap"));
    assert!(global.contains("HigherFinalizedSequencePublished"));
    assert!(global.contains("pub fn derive_lineage_bound_finalized_predecessor_root_v1"));
    assert!(global.contains("endpoint: UpgradeEndpoint"));
    assert!(global.contains("rollback_target_digest: Sha256Digest"));
    assert!(global.contains("evidence_checkpoint_digest: Sha256Digest"));
    assert!(global.contains("transparency_log_digest: Sha256Digest"));
}

#[test]
fn terminal_global_root_is_exactly_the_type_re_notarized_for_the_next_cycle() {
    let global = include_str!(
        "../../symthaea-fabrication-lineage-bound-global-upgrade-head/src/lib.rs"
    );
    let mint = include_str!(
        "../../symthaea-fabrication-lineage-predecessor-certificate-mint/src/lib.rs"
    );

    assert!(global.contains("pub struct LineageBoundFinalizedPredecessorRootV1"));
    assert!(global.contains("pub fn derive_lineage_bound_finalized_predecessor_root_v1"));
    assert!(mint.contains("root: &LineageBoundFinalizedPredecessorRootV1"));
    assert!(mint.contains("root.endpoint().clone()"));
    assert!(mint.contains("root.finalization_sequence()"));
    assert!(mint.contains("root.evidence_checkpoint_digest()"));
    assert!(mint.contains("root.transparency_log_digest()"));
}

#[test]
fn terminal_half_does_not_reintroduce_scalar_or_legacy_live_authority() {
    let sources = [
        include_str!("../../symthaea-fabrication-lineage-bound-finalization-authorization/src/lib.rs"),
        include_str!("../../symthaea-fabrication-lineage-bound-finalization-execution/src/lib.rs"),
        include_str!("../../symthaea-fabrication-lineage-bound-finalization-state-binding/src/lib.rs"),
        include_str!("../../symthaea-fabrication-lineage-bound-finalized-state/src/lib.rs"),
        include_str!("../../symthaea-fabrication-lineage-bound-finalized-head/src/lib.rs"),
        include_str!("../../symthaea-fabrication-lineage-bound-global-upgrade-head/src/lib.rs"),
    ];

    for source in sources {
        assert!(!source.contains("VerifiedClockWindow"));
        assert!(!source.contains("AuthorizedUpgradeHandoff"));
        assert!(!source.contains("saturating_mul"));
    }
}
