// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

const SOURCE: &str = include_str!("../src/lib.rs");

#[test]
fn publication_log_must_strictly_extend_execution_log() {
    assert!(SOURCE.contains("current_log.entries.len() <= execution_log.entries.len()"));
    assert!(SOURCE.contains("current_log.verify_successor_of(execution_log).is_err()"));
    assert!(SOURCE.contains("LogNotStrictExtension"));
    assert!(SOURCE.contains("governance_view.checkpoint_digest() == record.fresh_checkpoint_digest"));
}

#[test]
fn publication_clock_is_explicitly_descendant_and_definitely_later() {
    assert!(SOURCE.contains("execution_to_observation_clock_bridge"));
    assert!(SOURCE.contains("observation_clock.lower_unix_ms() <= execution_clock.upper_unix_ms()"));
    assert!(SOURCE.contains("PublicationClockNotDefinitelyLater"));
    assert!(SOURCE.contains("checked_mul(1_000)"));
    assert!(!SOURCE.contains("saturating_mul"));
}

#[test]
fn conflicting_finalizations_for_one_handoff_fail_closed() {
    assert!(SOURCE.contains("finalized_upgrade_head_log_kind(record.handoff_plan_digest)"));
    assert!(SOURCE.contains("entry.subject_digest != publication_digest"));
    assert!(SOURCE.contains("ConflictingFinalizationPublished"));
    assert!(SOURCE.contains("publication_count: matching_entries.len()"));
}

#[test]
fn every_matching_publication_must_be_after_execution_and_not_future() {
    assert!(SOURCE.contains("for entry in &matching_entries"));
    assert!(SOURCE.contains("recorded_at_unix_ms < execution_clock.upper_unix_ms()"));
    assert!(SOURCE.contains("PublicationBeforeExecution"));
    assert!(SOURCE.contains("recorded_at_unix_ms > observation_clock.lower_unix_ms()"));
    assert!(SOURCE.contains("PublicationMayBeFuture"));
}

#[test]
fn live_current_head_is_opaque_and_scalar_time_is_not_an_api_input() {
    assert!(SOURCE.contains("CurrentFinalizedUpgradeHeadV1"));
    assert!(!SOURCE.contains("now_unix_s"));
    assert!(!SOURCE.contains("authorized_at_unix_ms"));
    assert!(!SOURCE.contains("caller_time"));
    let prefix = SOURCE
        .split("pub struct CurrentFinalizedUpgradeHeadV1")
        .next()
        .expect("current head declaration prefix");
    let derive_window = prefix.rsplit("#[derive(").next().expect("derive window");
    assert!(!derive_window.contains("Deserialize"));
}
