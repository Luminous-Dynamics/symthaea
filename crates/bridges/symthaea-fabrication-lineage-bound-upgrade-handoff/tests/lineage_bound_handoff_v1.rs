// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn bootstrap_plan_still_derives_predecessor_rollback_and_checkpoint_from_old_root() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("predecessor_root.endpoint().clone()"));
    assert!(source.contains("predecessor_root.rollback_target_digest()"));
    assert!(source.contains("predecessor_root.evidence_checkpoint_digest()"));
    assert!(source.contains("LineageBoundPredecessorAuthorityKindV1::BootstrapV1"));
}

#[test]
fn lineage_native_predecessor_is_requalified_from_the_exact_current_log() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("pub fn qualify_lineage_native_predecessor_authority_v1("));
    assert!(source.contains("digest_transparency_log(current_log)"));
    assert!(source.contains("log_digest != governance_view.transparency_log_digest()"));
    assert!(source.contains("entry.kind.starts_with(LINEAGE_NATIVE_FINALIZED_HEAD_LOG_KIND_PREFIX)"));
    assert!(source.contains("matching_entries.len() != publications.len()"));
    assert!(source.contains("entry.kind != expected_kind"));
    assert!(source.contains("entry.subject_digest != publication_digest"));
    assert!(source.contains("recorded_at_unix_ms > observation_clock.lower_unix_ms()"));
}

#[test]
fn lineage_native_publication_bytes_and_log_kind_match_the_terminal_namespace() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("symthaea.fabrication.lineage-bound-finalized-upgrade-head-publication.v1\\0"));
    assert!(source.contains("symthaea.fabrication.lineage-bound-finalized-upgrade-head-kind.v1\\0"));
    assert!(source.contains("lineage-finalized-upgrade-head-v1:"));
    assert!(source.contains("valid_lineage_native_publication"));
    assert!(source.contains("canonical_hex_id"));
}

#[test]
fn global_lineage_sequences_are_non_equivocating_and_contiguous() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("BTreeMap::<u64, Sha256Digest>::new()"));
    assert!(source.contains("NativeSequenceEquivocation"));
    assert!(source.contains("let sequences = sequence_digests.keys().copied().collect::<Vec<_>>()"));
    assert!(source.contains("for pair in sequences.windows(2)"));
    assert!(source.contains("pair[0].checked_add(1)"));
    assert!(source.contains("NativeSequenceGap"));
    assert!(!source.contains("saturating_add"));
}

#[test]
fn candidate_endpoint_must_be_the_highest_visible_finalized_sequence() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("let Some(highest_sequence) = sequences.last().copied()"));
    assert!(source.contains("publication.finalization_sequence == highest_sequence"));
    assert!(source.contains("publication.successor_endpoint_digest == endpoint_digest"));
    assert!(source.contains("candidate_publication_count == 0"));
    assert!(source.contains("NativeCandidateMissing"));
}

#[test]
fn native_predecessor_authority_is_opaque_and_carries_exact_governance_view() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("pub struct LineageNativePredecessorAuthorityV1"));
    assert!(!source.contains("Serialize, Deserialize)]\npub struct LineageNativePredecessorAuthorityV1"));
    assert!(source.contains("governance_view_id: governance_view.id()"));
    assert!(source.contains("checkpoint_digest: governance_view.checkpoint_digest()"));
    assert!(source.contains("transparency_log_digest: log_digest"));
    assert!(source.contains("clock_envelope_id: observation_clock.id()"));
    assert!(source.contains("operational_basis_id: observation_basis.id()"));
}

#[test]
fn bootstrap_and_lineage_native_handoff_commitments_use_distinct_domains() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("LineageBoundPredecessorAuthorityKindV1::BootstrapV1 => PREPARED_DOMAIN"));
    assert!(source.contains("LineageBoundPredecessorAuthorityKindV1::BootstrapV1 => AUTHORIZED_DOMAIN"));
    assert!(source.contains("LineageBoundPredecessorAuthorityKindV1::LineageNativeV1 => PREPARED_LINEAGE_NATIVE_DOMAIN"));
    assert!(source.contains("LineageBoundPredecessorAuthorityKindV1::LineageNativeV1 => AUTHORIZED_LINEAGE_NATIVE_DOMAIN"));
}

#[test]
fn native_handoff_uses_only_the_opaque_requalified_predecessor() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("pub fn build_lineage_native_upgrade_handoff_plan_v1("));
    assert!(source.contains("pub fn prepare_lineage_native_upgrade_handoff_v1("));
    assert!(source.contains("LineageBoundPredecessorRootRefIdV1(predecessor.id().as_digest())"));
    assert!(source.contains("LineageBoundCurrentHeadRefIdV1(predecessor.current_head_ref_digest())"));
    assert!(source.contains("predecessor.governance_view_id() != governance_view.id()"));
}

#[test]
fn downstream_live_handoff_keeps_digest_refs_and_hides_inner_authority() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("predecessor_root_id: LineageBoundPredecessorRootRefIdV1"));
    assert!(source.contains("current_head_id: LineageBoundCurrentHeadRefIdV1"));
    assert!(source.contains("pub fn predecessor_root_digest(&self) -> Sha256Digest"));
    assert!(source.contains("pub fn current_head_digest(&self) -> Sha256Digest"));
    assert!(source.contains("inner_handoff: ClockGovernedUpgradeHandoffV1"));
    assert!(!source.contains("pub fn inner_handoff(&self)"));
}

#[test]
fn no_caller_implementable_authority_or_legacy_scalar_time_reenters() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("pub trait LineageBoundPredecessorRootViewV1"));
    assert!(!source.contains("pub trait LineageBoundCurrentHeadViewV1"));
    assert!(!source.contains("AuthorizedUpgradeHandoff"));
    assert!(!source.contains("VerifiedClockWindow"));
    assert!(!source.contains("now_unix_s"));
    assert!(!source.contains("prepared_at_unix_ms"));
    assert!(!source.contains("saturating_mul"));
    assert!(!source.contains(".unwrap("));
    assert!(!source.contains(".expect("));
}
