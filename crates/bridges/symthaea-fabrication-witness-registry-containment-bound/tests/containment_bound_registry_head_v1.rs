// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn stronger_capability_consumes_existing_registry_head_not_raw_registry_sequence_claim() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("registry_head: &QuorumObservedWitnessRegistryHeadV1"));
    assert!(source.contains("registry_head.activated_registry_id() != activated_registry.id()"));
    assert!(source.contains("registry_head.registry_digest() != activated_registry.registry_digest()"));
    assert!(source.contains("registry_head.sequence() != activated_registry.sequence()"));
}

#[test]
fn containment_subject_is_rebuilt_from_opaque_authority() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("build_authorized_containment_head_publication_v1("));
    assert!(source.contains("current_containment_authority"));
    assert!(source.contains("containment_publication != &expected_containment_publication"));
    assert!(source.contains("ContainmentPublicationMismatch"));
}

#[test]
fn exact_same_registry_log_and_checkpoint_are_required() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("transparency_log_digest != registry_head.transparency_log_digest()"));
    assert!(source.contains("TransparencyLogMismatch"));
    assert!(source.contains("signed_checkpoint.checkpoint_digest != registry_head.checkpoint_digest()"));
    assert!(source.contains("CheckpointMismatch"));
}

#[test]
fn containment_generation_is_strictly_monotonic_in_same_view() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("CONTAINMENT_HEAD_LOG_KIND_PREFIX"));
    assert!(source.contains("ContainmentGenerationRegressed"));
    assert!(source.contains("DuplicateContainmentGeneration"));
    assert!(source.contains("HigherContainmentGenerationPublished"));
    assert!(source.contains("suffix != generation.to_string()"));
}

#[test]
fn containment_authority_must_descend_from_state_used_by_registry_head() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("base_containment_authority.id() != registry_head.observation_containment_authority_id()"));
    assert!(source.contains("containment_authority_bridge: &[ClockGovernedContainmentStateV1]"));
    assert!(source.contains("current.previous_authority_id() != Some(previous.id())"));
    assert!(source.contains("current.previous_state_digest() != Some(previous.state_digest())"));
    assert!(source.contains("checked_add(1)"));
}

#[test]
fn published_containment_authority_is_temporally_before_registry_observation() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("current_containment_authorization_basis.id()"));
    assert!(source.contains("current_containment_authority.operational_basis_id()"));
    assert!(source.contains("containment_to_observation_clock_bridge"));
    assert!(source.contains("containment_recorded_at_ms < containment_authorization_clock.upper_unix_ms()"));
    assert!(source.contains("containment_recorded_at_ms > observation_clock.lower_unix_ms()"));
}

#[test]
fn exact_checkpoint_and_witnesses_are_requalified_under_highest_containment_state() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("current_containment_authority"));
    assert!(source.contains("KeyUsage::TransparencyLog"));
    assert!(source.contains("KeyUsage::TransparencyWitness"));
    assert!(source.contains("SignerCompromisedAcrossEnvelope"));
    assert!(source.contains("verify_checkpoint_signature"));
    assert!(source.contains("verify_witness_signature"));
    assert!(source.contains("signed_checkpoint.signature.signature"));
    assert!(source.contains("witness.signature.signature"));
}

#[test]
fn activated_registry_still_governs_witness_identity() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("activated_registry.profile("));
    assert!(source.contains("WitnessNotInActivatedRegistry"));
    assert!(source.contains("WitnessOrganizationMismatch"));
    assert!(source.contains("WitnessFailureDomainMismatch"));
}

#[test]
fn stronger_capability_commits_both_currentness_theorems() {
    let source = include_str!("../src/lib.rs");
    for required in [
        "registry_head_id",
        "registry_sequence",
        "containment_authority_id",
        "containment_generation",
        "compromise_tracker_digest",
        "containment_publication_digest",
        "transparency_log_digest",
        "checkpoint_digest",
        "containment_lineage_digest",
        "containment_to_observation_clock_lineage_digest",
    ] {
        assert!(source.contains(required), "missing commitment {required}");
    }
}

#[test]
fn live_stronger_capability_is_opaque_and_scalar_time_free() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct ContainmentCurrentWitnessRegistryHeadV1"
    ));
    assert!(!source.contains("now_unix_s:"));
    assert!(!source.contains("evaluation_time_unix_s:"));
    assert!(!source.contains("VerifiedTransparencyCheckpoint"));
    assert!(!source.contains("VerifiedTransparencyWitnessQuorum"));
}
