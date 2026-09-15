// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn complete_concrete_state_lineage_is_required_before_global_currentness() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("if states.is_empty()"));
    assert!(source.contains("verify_upgrade_state_successor"));
    assert!(source.contains("genesis.generation != 1"));
    assert!(source.contains("activated.active_stage != UpgradeStage::Activated"));
    assert!(source.contains("activated_digest != record.predecessor_upgrade_state_digest"));
    assert!(source.contains("activated.evidence.handoff_digest != record.handoff_plan_digest"));
}

#[test]
fn global_head_scans_every_lineage_finalized_publication_in_the_exact_log() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("entry.kind.starts_with(LINEAGE_BOUND_FINALIZED_UPGRADE_HEAD_LOG_KIND_PREFIX)"));
    assert!(source.contains("matching_entries.len() != publications.len()"));
    assert!(source.contains("entry.kind != expected_kind"));
    assert!(source.contains("entry.subject_digest != publication_digest"));
    assert!(source.contains("log_digest != current_head.current_transparency_log_digest()"));
}

#[test]
fn finalized_sequences_are_non_equivocating_and_contiguous() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("BTreeMap::<u64, Sha256Digest>::new()"));
    assert!(source.contains("FinalizedSequenceEquivocation"));
    assert!(source.contains("let sequences = sequence_digests.keys().copied().collect::<Vec<_>>()"));
    assert!(source.contains("for pair in sequences.windows(2)"));
    assert!(source.contains("pair[0].checked_add(1)"));
    assert!(source.contains("FinalizedSequenceGap"));
}

#[test]
fn candidate_must_be_present_and_no_higher_sequence_may_exist() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("candidate_publication_count == 0"));
    assert!(source.contains("CandidatePublicationMissing"));
    assert!(source.contains("highest_sequence > record.finalization_sequence"));
    assert!(source.contains("HigherFinalizedSequencePublished"));
    assert!(source.contains("highest_finalization_sequence: highest_sequence"));
}

#[test]
fn migration_can_start_above_sequence_one_but_cannot_skip_afterwards() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("let Some(lowest_sequence) = sequences.first().copied()"));
    assert!(source.contains("lowest_finalization_sequence: lowest_sequence"));
    assert!(!source.contains("lowest_sequence != 1"));
    assert!(source.contains("FinalizedSequenceGap"));
}

#[test]
fn count_and_sequence_bookkeeping_use_checked_arithmetic() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("candidate_publication_count.checked_add(1)"));
    assert!(source.contains("FinalizedSequenceOverflow"));
    assert!(source.contains("CountOverflow"));
    assert!(!source.contains("saturating_add"));
    assert!(!source.contains("saturating_mul"));
}

#[test]
fn predecessor_root_is_derived_from_terminal_record_not_producing_handoff_input() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("pub fn derive_lineage_bound_finalized_predecessor_root_v1("));
    assert!(source.contains("let endpoint = record.successor_endpoint.clone()"));
    assert!(source.contains("digest_upgrade_endpoint(&endpoint)"));
    assert!(source.contains("endpoint_digest != record.successor_endpoint_digest"));
    assert!(source.contains("rollback_target_digest: record.successor_endpoint.durable_state_digest"));
    assert!(!source.contains("producing_handoff"));
}

#[test]
fn predecessor_root_commits_the_exact_distributed_view() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("evidence_checkpoint_digest: global_head.checkpoint_digest()"));
    assert!(source.contains("transparency_log_digest: global_head.transparency_log_digest()"));
    assert!(source.contains("clock_envelope_id: global_head.clock_envelope_id()"));
    assert!(source.contains("operational_basis_id: global_head.operational_basis_id()"));
    assert!(source.contains("prior_predecessor_root_digest: record.predecessor_root_digest"));
}

#[test]
fn live_global_head_and_predecessor_root_are_opaque_and_lineage_native() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("pub struct GlobalCurrentLineageBoundFinalizedHeadV1"));
    assert!(source.contains("pub struct LineageBoundFinalizedPredecessorRootV1"));
    assert!(!source.contains("Serialize, Deserialize)]\npub struct GlobalCurrentLineageBoundFinalizedHeadV1"));
    assert!(!source.contains("Serialize, Deserialize)]\npub struct LineageBoundFinalizedPredecessorRootV1"));
    assert!(!source.contains("ClockGovernedFinalizedUpgradeV1"));
    assert!(!source.contains("CurrentFinalizedUpgradeHeadV1"));
}

#[test]
fn live_authority_has_no_scalar_current_time_or_panic_style_extraction() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("now_unix_s"));
    assert!(!source.contains("evaluation_time_unix_s"));
    assert!(!source.contains(".expect("));
    assert!(!source.contains(".unwrap("));
}
