// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn lineage_terminal_namespace_is_distinct_from_the_ordinary_head_namespace() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("lineage-finalized-upgrade-head-v1:"));
    assert!(source.contains("lineage-bound-finalized-upgrade-head-kind.v1\\0"));
    assert!(!source.contains("const LOG_KIND_DOMAIN: &[u8] = b\"symthaea.fabrication.finalized-upgrade-head-kind.v1\\0\""));
}

#[test]
fn publication_is_exactly_reconstructed_from_the_opaque_terminal_authority() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("build_lineage_bound_finalized_upgrade_head_publication_v1(finalized)"));
    assert!(source.contains("publication != &expected_publication"));
    assert!(source.contains("finalized.execution_permit_id() != execution_permit.id()"));
    assert!(source.contains("record.execution_permit_id != execution_permit.id().to_hex()"));
}

#[test]
fn currentness_requires_strict_append_only_advancement_and_later_time() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("current_log.verify_successor_of(execution_log).is_err()"));
    assert!(source.contains("current_log.entries.len() <= execution_log.entries.len()"));
    assert!(source.contains("governance_view.checkpoint_digest() == execution_permit.fresh_checkpoint_digest()"));
    assert!(source.contains("observation_clock.lower_unix_ms() <= execution_clock.upper_unix_ms()"));
    assert!(source.contains("verify_clock_lineage("));
}

#[test]
fn lineage_head_prepublication_is_independently_rejected() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("for entry in &execution_log.entries"));
    assert!(source.contains("if entry.kind == log_kind"));
    assert!(source.contains("LineageHeadPrepublished"));
}

#[test]
fn exact_duplicate_publication_is_idempotent_but_conflict_fails_closed() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("publication_count: matching_entries.len()"));
    assert!(source.contains("if entry.subject_digest != publication_digest"));
    assert!(source.contains("ConflictingFinalizationPublished"));
    assert!(source.contains("latest_publication_entry_sequence"));
}

#[test]
fn publication_must_be_after_execution_and_not_future_at_observation() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("recorded_at_unix_ms <= execution_clock.upper_unix_ms()"));
    assert!(source.contains("recorded_at_unix_ms > observation_clock.lower_unix_ms()"));
    assert!(source.contains("PublicationBeforeExecution"));
    assert!(source.contains("PublicationMayBeFuture"));
    assert!(source.contains("checked_mul(1_000)"));
}

#[test]
fn portable_publication_is_canonical_and_sequence_adjacent() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("fn canonical_hex_id(value: &str) -> bool"));
    assert!(source.contains("predecessor_finalization_sequence.checked_add(1)"));
    assert!(source.contains("predecessor_upgrade_state_generation.checked_add(1)"));
    assert!(source.contains("successor_endpoint_digest"));
}

#[test]
fn live_current_head_is_opaque_and_uses_no_scalar_current_time() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("pub struct CurrentLineageBoundFinalizedUpgradeHeadV1"));
    assert!(!source.contains("Serialize, Deserialize)]\npub struct CurrentLineageBoundFinalizedUpgradeHeadV1"));
    assert!(!source.contains("now_unix_s"));
    assert!(!source.contains("evaluation_time_unix_s"));
    assert!(!source.contains("saturating_mul"));
    assert!(!source.contains(".expect("));
    assert!(!source.contains(".unwrap("));
}
