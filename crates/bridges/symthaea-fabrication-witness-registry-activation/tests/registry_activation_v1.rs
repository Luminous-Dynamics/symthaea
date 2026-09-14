// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn activation_consumes_only_authorized_transition_and_exact_previous_registry() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("transition: &AuthorizedWitnessRegistryTransitionV1"));
    assert!(source.contains("previous_registry: &GovernedWitnessAuthorityRegistryV1"));
    assert!(source.contains("transition.previous_registry_id() != previous_registry.id()"));
    assert!(source.contains("transition.proposed_sequence() != expected_sequence"));
    assert!(source.contains("ProposedRegistryDigestMismatch"));
}

#[test]
fn activation_proves_both_clock_and_containment_ancestry() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("clock_bridge: &[OperationalClockBasisV1]"));
    assert!(source.contains("containment_authority_bridge: &[ClockGovernedContainmentStateV1]"));
    assert!(source.contains("predecessor_operational_basis_id"));
    assert!(source.contains("current.previous_authority_id() != Some(previous.id())"));
    assert!(source.contains("current.previous_state_digest() != Some(previous.state_digest())"));
    assert!(source.contains("previous.generation().checked_add(1)"));
    assert!(source.contains("clock_lineage_digest"));
    assert!(source.contains("containment_lineage_digest"));
}

#[test]
fn old_registry_remains_witness_authority_until_activation() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("trust_head.witness_registry_digest() != previous_registry.registry_digest()"));
    assert!(source.contains("containment_head.witness_registry_digest() != previous_registry.registry_digest()"));
    assert!(source.contains("CurrentHeadsObservedUnderDifferentRegistry"));
}

#[test]
fn activation_requires_fresh_coherent_trust_and_containment_heads() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("containment_head.authority_id() != current_containment_authority.id()"));
    assert!(source.contains("containment_head.trust_head_id() != trust_head.id()"));
    assert!(source.contains("trust_snapshot_digest != trust_head.snapshot_digest()"));
    assert!(source.contains("TrustSnapshotSequenceRollback"));
    assert!(source.contains("TrustSnapshotSameSequenceSubstitution"));
}

#[test]
fn activation_is_certain_under_fresh_trusted_time() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("transition.activates_at_unix_ms() > current_clock.lower_unix_ms()"));
    assert!(source.contains("ActivationNotYetCertain"));
    assert!(source.contains("require_valid_across_seconds_window"));
}

#[test]
fn proposed_witness_keys_are_requalified_at_activation_not_only_authorization() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("requalify_witness_key("));
    assert!(source.contains("KeyLifecycleStatus::Active"));
    assert!(source.contains("KeyUsage::TransparencyWitness"));
    assert!(source.contains("current_containment_authority"));
    assert!(source.contains("WitnessKeyCompromisedAcrossEnvelope"));
}

#[test]
fn activated_registry_is_opaque_and_commits_fresh_authority_paths() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct ActivatedWitnessAuthorityRegistryV1"
    ));
    for required in [
        "transition_id",
        "previous_registry_id",
        "registry_digest",
        "activation_clock_envelope_id",
        "clock_lineage_digest",
        "activation_containment_authority_id",
        "activation_compromise_tracker_digest",
        "containment_lineage_digest",
        "trust_head_id",
        "containment_head_id",
    ] {
        assert!(source.contains(required), "missing activated commitment {required}");
    }
}

#[test]
fn no_scalar_now_enters_activation() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("now_unix_s:"));
    assert!(!source.contains("evaluation_time_unix_s:"));
}
