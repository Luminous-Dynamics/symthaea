// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn legacy_registry_is_only_an_opaque_genesis_root() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("adopt_legacy_witness_authority_registry_genesis_v1"));
    assert!(source.contains("registry: &WitnessAuthorityRegistryV1"));
    assert!(source.contains("registry.sequence() != 1"));
    assert!(source.contains("registry_digest != registry.registry_digest()"));
    assert!(source.contains("legacy_genesis_ceremony_id"));
}

#[test]
fn successor_authorization_requires_current_heads_observed_under_previous_registry() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("trust_head.witness_registry_digest() != previous.registry_digest"));
    assert!(source.contains("trust_head.witness_registry_sequence() != previous.sequence"));
    assert!(source.contains("containment_head.witness_registry_digest() != previous.registry_digest"));
    assert!(source.contains("containment_head.witness_registry_sequence() != previous.sequence"));
    assert!(source.contains("TrustHeadObservedUnderDifferentRegistry"));
    assert!(source.contains("ContainmentHeadObservedUnderDifferentRegistry"));
}

#[test]
fn trust_and_containment_views_must_be_coherent() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("containment_head.trust_head_id() != trust_head.id()"));
    assert!(source.contains("containment_head.trust_snapshot_digest() != trust_head.snapshot_digest()"));
    assert!(source.contains("containment_head.authority_id() != containment_authority.id()"));
    assert!(source.contains("containment_head.compromise_tracker_digest() != containment_authority.compromise_tracker_digest()"));
}

#[test]
fn proposed_witnesses_are_real_current_witness_keys() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("KeyUsage::TransparencyWitness"));
    assert!(source.contains("KeyLifecycleStatus::Active"));
    assert!(source.contains("require_valid_across_optional_seconds_window"));
    assert!(source.contains("WitnessKeyCompromisedAcrossEnvelope"));
    assert!(source.contains("containment_authority.compromise_tracker()"));
}

#[test]
fn transition_is_adjacent_changed_and_future_scheduled() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("previous.sequence.checked_add(1)"));
    assert!(source.contains("NoRegistryChange"));
    assert!(source.contains("activates_at_unix_ms < clock.upper_unix_ms()"));
    assert!(source.contains("clock.lower_unix_ms().checked_add(maximum_delay_ms)"));
    assert!(source.contains("ActivationMayBePast"));
    assert!(source.contains("ActivationMayBeTooLate"));
}

#[test]
fn threshold_governance_is_bound_to_current_compromise_state() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("threshold_policy.key_usage != KeyUsage::ThresholdCeremony"));
    assert!(source.contains("ceremony.compromise_tracker_digest() != prepared.compromise_tracker_digest"));
    assert!(source.contains("ceremony.trust_snapshot_digest() != prepared.trust_snapshot_digest"));
    assert!(source.contains("ceremony.clock_envelope_id() != prepared.clock_envelope_id"));
    assert!(source.contains("WITNESS_REGISTRY_TRANSITION_PURPOSE"));
}

#[test]
fn authorization_is_not_activation() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("AuthorizedWitnessRegistryTransitionV1"));
    assert!(!source.contains("activate_witness_authority_registry_transition_v1"));
    assert!(!source.contains("pub fn active_successor"));
}

#[test]
fn live_authority_has_no_scalar_now_or_serde_constructor() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("now_unix_s:"));
    assert!(!source.contains("evaluation_time_unix_s:"));
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct AuthorizedWitnessRegistryTransitionV1"
    ));
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct GovernedWitnessAuthorityRegistryV1"
    ));
}
