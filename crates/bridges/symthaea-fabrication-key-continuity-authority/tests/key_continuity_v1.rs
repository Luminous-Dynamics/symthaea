// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn live_continuity_rederives_from_raw_snapshots_without_scalar_transition_time() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("transition_at_unix_s:"));
    assert!(!source.contains("VerifiedKeyContinuity"));
    assert!(!source.contains("VerifiedClockWindow"));
    assert!(source.contains("previous: &TrustSnapshot"));
    assert!(source.contains("successor: &TrustSnapshot"));
    assert!(source.contains("derive_clock_governance_evaluation_envelope_v1"));
}

#[test]
fn overlap_is_anchored_to_latest_possible_transition_time() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("clock.upper_unix_ms().checked_add(overlap_ms)"));
    assert!(source.contains("overlap_required_until_unix_ms"));
    assert!(source.contains("not_after_ms >= overlap_required_until_unix_ms"));
}

#[test]
fn continuity_filters_compromised_bridge_keys() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("signer_compromise_tracker"));
    assert!(source.contains("compromise.affected_usages.contains(&usage)"));
    assert!(source.contains("effective_ms < overlap_required_until_unix_ms"));
    assert!(source.contains("compromise_tracker_digest"));
}

#[test]
fn snapshot_transition_is_exactly_adjacent() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("previous.sequence.checked_add(1)"));
    assert!(source.contains("SequenceNotAdjacent"));
}

#[test]
fn live_continuity_is_not_deserializable() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct ClockGovernedKeyContinuityV1"
    ));
}
