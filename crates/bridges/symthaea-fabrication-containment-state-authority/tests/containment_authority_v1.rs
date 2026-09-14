// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn genesis_is_exact_kernel_genesis_and_witness_bound() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("FabricationContainmentState::genesis("));
    assert!(source.contains("is_ok_and(|canonical| &canonical == state)"));
    assert!(source.contains("trust_head.containment_state_digest() != state_digest"));
    assert!(source.contains("trust_head.compromise_tracker_digest() != tracker_digest"));
    assert!(source.contains("TrustHeadContainmentMismatch"));
    assert!(source.contains("TrustHeadCompromiseTrackerMismatch"));
}

#[test]
fn successor_requires_strict_kernel_lineage_and_previous_witness_binding() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains(
        "verify_containment_state_successor(previous.state(), proposed)"
    ));
    assert!(source.contains(
        "trust_head.containment_state_digest() != previous.state_digest()"
    ));
    assert!(source.contains(
        "trust_head.compromise_tracker_digest() != previous.compromise_tracker_digest()"
    ));
    assert!(source.contains("Some(previous.id())"));
    assert!(source.contains("Some(previous.state_digest())"));
}

#[test]
fn trust_snapshot_authority_cannot_regress_or_substitute_same_sequence() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains(
        "trust_head.snapshot_sequence() < previous.trust_snapshot_sequence()"
    ));
    assert!(source.contains("TrustSnapshotSequenceRollback"));
    assert!(source.contains("TrustSnapshotSameSequenceSubstitution"));
    assert!(source.contains(
        "trust_head.snapshot_digest() != previous.trust_snapshot_digest()"
    ));
}

#[test]
fn ceremony_is_qualified_against_previous_not_proposed_compromise_state() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains(
        "ceremony.compromise_tracker_digest() != prepared.previous_compromise_tracker_digest"
    ));
    assert!(source.contains("previous_compromise_tracker_digest"));
    assert!(source.contains("proposed_compromise_tracker_digest"));
    assert!(source.contains("CompromiseTrackerDigestMismatch"));
}

#[test]
fn proposed_state_cannot_self_invalidate_authorizing_threshold_signer() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("prepared.proposed_state"));
    assert!(source.contains("KeyUsage::ThresholdCeremony"));
    assert!(source.contains("effective_at_ms <= prepared.clock_upper_unix_ms"));
    assert!(source.contains("CeremonySignerInvalidatedByProposedState"));
}

#[test]
fn authority_rebinds_exact_witnessed_clock_and_threshold_context() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains(
        "observation_basis.id() != trust_head.observation_operational_basis_id()"
    ));
    assert!(source.contains("trust_head.observation_clock_envelope_id()"));
    assert!(source.contains("threshold_policy.key_usage != KeyUsage::ThresholdCeremony"));
    assert!(source.contains("ceremony.trust_snapshot_digest() != prepared.trust_snapshot_digest"));
    assert!(source.contains("ceremony.clock_envelope_id() != prepared.clock_envelope_id"));
}

#[test]
fn proposal_commits_previous_and_proposed_state_authority_context() {
    let source = include_str!("../src/lib.rs");
    for required in [
        "proposed_state_digest",
        "proposed_generation",
        "previous_authority_id",
        "previous_state_digest",
        "previous_compromise_tracker_digest",
        "proposed_compromise_tracker_digest",
        "trust_head_id",
        "trust_snapshot_digest",
        "trust_snapshot_sequence",
        "threshold_policy_digest",
        "clock_envelope_id",
        "operational_basis_id",
        "clock_lower_unix_ms",
        "clock_upper_unix_ms",
    ] {
        assert!(source.contains(required), "missing commitment field {required}");
    }
}

#[test]
fn no_scalar_time_or_serde_live_construction() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("now_unix_s:"));
    assert!(!source.contains("evaluation_time_unix_s:"));
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct ClockGovernedContainmentStateV1"
    ));
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct PreparedContainmentStateAuthorityV1"
    ));
}
