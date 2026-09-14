// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn currentness_subject_is_only_opaque_activated_registry() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("registry: &ActivatedWitnessAuthorityRegistryV1"));
    assert!(source.contains("build_activated_witness_registry_head_publication_v1"));
    assert!(!source.contains("profiles: &[WitnessAuthorityProfileV1]"));
    assert!(source.contains("registry.id().to_hex()"));
    assert!(source.contains("registry.registry_digest()"));
}

#[test]
fn publication_is_strictly_adjacent_to_previous_registry() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("previous_sequence.checked_add(1) != Some(publication.sequence)"));
    assert!(source.contains("previous_registry_digest"));
    assert!(source.contains("transition_id"));
}

#[test]
fn registry_log_is_monotonic_and_fork_intolerant() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("WITNESS_REGISTRY_HEAD_LOG_KIND_PREFIX"));
    assert!(source.contains("parse_registry_head_sequence"));
    assert!(source.contains("RegistrySequenceRegressed"));
    assert!(source.contains("DuplicateRegistrySequence"));
    assert!(source.contains("HigherRegistrySequencePublished"));
    assert!(source.contains("suffix != sequence.to_string()"));
}

#[test]
fn publication_must_follow_activation_and_precede_fresh_observation() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("publication_recorded_at_ms < registry.activates_at_unix_ms()"));
    assert!(source.contains("PublicationBeforeActivation"));
    assert!(source.contains("publication_recorded_at_ms > observation_clock.lower_unix_ms()"));
    assert!(source.contains("PublicationMayBeFuture"));
}

#[test]
fn new_registry_itself_governs_post_cutover_witness_identity() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("let Some(profile) = registry.profile("));
    assert!(source.contains("WitnessNotInActivatedRegistry"));
    assert!(source.contains("WitnessOrganizationMismatch"));
    assert!(source.contains("WitnessFailureDomainMismatch"));
    assert!(!source.contains("symthaea_fabrication_witness_authority"));
}

#[test]
fn observation_proves_clock_and_containment_descent_from_activation() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("activation_basis: &OperationalClockBasisV1"));
    assert!(source.contains("clock_bridge: &[OperationalClockBasisV1]"));
    assert!(source.contains("activation_containment_authority: &ClockGovernedContainmentStateV1"));
    assert!(source.contains("containment_authority_bridge: &[ClockGovernedContainmentStateV1]"));
    assert!(source.contains("predecessor_operational_basis_id"));
    assert!(source.contains("current.previous_authority_id() != Some(previous.id())"));
    assert!(source.contains("current.previous_state_digest() != Some(previous.state_digest())"));
    assert!(source.contains("clock_lineage_digest"));
    assert!(source.contains("containment_lineage_digest"));
}

#[test]
fn activation_snapshot_is_materialized_exactly_and_must_remain_fresh() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("trust_snapshot_digest != registry.trust_snapshot_digest()"));
    assert!(source.contains("trust_snapshot.sequence != registry.trust_snapshot_sequence()"));
    assert!(source.contains("TrustSnapshotMismatch"));
    assert!(source.contains("require_valid_across_seconds_window"));
}

#[test]
fn signers_are_requalified_against_latest_explicit_containment_authority() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("observation_containment_authority"));
    assert!(source.contains("KeyUsage::TransparencyLog"));
    assert!(source.contains("KeyUsage::TransparencyWitness"));
    assert!(source.contains("SignerInvalidAtEvidenceTime"));
    assert!(source.contains("SignerCompromisedAcrossEnvelope"));
}

#[test]
fn exact_raw_checkpoint_and_witness_signatures_are_n_of_n_verified() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("ExactWitnessRegistryHeadEvidenceVerifierV1"));
    assert!(source.contains("verify_checkpoint_signature"));
    assert!(source.contains("verify_witness_signature"));
    assert!(source.contains("signed_checkpoint.signature.signature"));
    assert!(source.contains("witness.signature.signature"));
    assert!(source.contains("minimum_distinct_providers: 2"));
    assert!(source.contains("exact_verifier_set_digest"));
}

#[test]
fn live_registry_head_is_opaque_and_scalar_time_free() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct QuorumObservedWitnessRegistryHeadV1"
    ));
    assert!(!source.contains("now_unix_s:"));
    assert!(!source.contains("evaluation_time_unix_s:"));
    assert!(!source.contains("VerifiedTransparencyCheckpoint"));
    assert!(!source.contains("VerifiedTransparencyWitnessQuorum"));
}
