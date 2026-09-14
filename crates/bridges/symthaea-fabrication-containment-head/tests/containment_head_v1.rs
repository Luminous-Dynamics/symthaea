// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn live_currentness_subject_is_opaque_containment_authority() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("authority: &ClockGovernedContainmentStateV1"));
    assert!(source.contains("build_authorized_containment_head_publication_v1("));
    assert!(!source.contains("state: &FabricationContainmentState"));
    assert!(source.contains("authority.state_digest()"));
    assert!(source.contains("authority.compromise_tracker()"));
}

#[test]
fn monotonic_log_rejects_regression_duplicate_and_higher_generation() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("CONTAINMENT_HEAD_LOG_KIND_PREFIX"));
    assert!(source.contains("parse_containment_head_generation"));
    assert!(source.contains("ContainmentGenerationRegressed"));
    assert!(source.contains("DuplicateContainmentGeneration"));
    assert!(source.contains("HigherContainmentGenerationPublished"));
    assert!(source.contains("suffix != generation.to_string()"));
}

#[test]
fn publication_is_independent_of_observing_trust_head() {
    let source = include_str!("../src/lib.rs");
    let start = source
        .find("pub struct AuthorizedContainmentHeadPublicationV1")
        .unwrap();
    let end = source[start..].find("}\n\npub trait").unwrap() + start;
    let publication = &source[start..end];
    assert!(publication.contains("authorization_trust_head_id"));
    assert!(!publication.contains("trust_anchor_mode"));
    assert!(!publication.contains("witness_registry_id"));
    assert!(!publication.contains("observation_clock_envelope_id"));
}

#[test]
fn observing_trust_must_be_authorization_head_or_state_bound_refresh() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("ContainmentHeadTrustAnchorModeV1::AuthorizationHead"));
    assert!(source.contains("ContainmentHeadTrustAnchorModeV1::StateBoundRefresh"));
    assert!(source.contains("trust_head.id() == authority.trust_head_id()"));
    assert!(source.contains(
        "trust_head.containment_state_digest() == authority.state_digest()"
    ));
    assert!(source.contains(
        "trust_head.compromise_tracker_digest() == authority.compromise_tracker_digest()"
    ));
    assert!(source.contains("TrustHeadNotAnchoredToAuthority"));
}

#[test]
fn raw_snapshot_is_materialization_not_authority() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("trust_snapshot: &TrustSnapshot"));
    assert!(source.contains("trust_snapshot_digest != trust_head.snapshot_digest()"));
    assert!(source.contains("trust_snapshot.sequence != trust_head.snapshot_sequence()"));
    assert!(source.contains("TrustSnapshotHeadMismatch"));
}

#[test]
fn both_authorization_and_current_trust_clock_lineages_are_explicit() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains(
        "authorization_to_trust_clock_bridge: &[OperationalClockBasisV1]"
    ));
    assert!(source.contains("trust_to_observation_clock_bridge: &[OperationalClockBasisV1]"));
    assert!(source.contains("authorization_to_trust_clock_lineage_digest"));
    assert!(source.contains("trust_to_observation_clock_lineage_digest"));
    assert!(source.contains("predecessor_operational_basis_id"));
    assert!(source.contains("MAX_CONTAINMENT_HEAD_CLOCK_HOPS"));
}

#[test]
fn publication_must_be_definitely_after_authorization_and_before_observation() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains(
        "publication_recorded_at_ms < authorization_clock.upper_unix_ms()"
    ));
    assert!(source.contains("PublicationBeforeAuthorization"));
    assert!(source.contains(
        "publication_recorded_at_ms > observation_clock.lower_unix_ms()"
    ));
    assert!(source.contains("PublicationMayBeFuture"));
}

#[test]
fn checkpoint_and_witness_signers_use_current_authorized_compromise_state() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("authority.compromise_tracker()"));
    assert!(source.contains("KeyUsage::TransparencyLog"));
    assert!(source.contains("KeyUsage::TransparencyWitness"));
    assert!(source.contains("SignerInvalidAtEvidenceTime"));
    assert!(source.contains("SignerCompromisedAcrossEnvelope"));
}

#[test]
fn raw_checkpoint_and_witness_signature_bytes_are_verified_directly() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("ExactContainmentHeadEvidenceVerifierV1"));
    assert!(source.contains("verify_checkpoint_signature"));
    assert!(source.contains("verify_witness_signature"));
    assert!(source.contains("signed_checkpoint.signature.signature"));
    assert!(source.contains("witness.signature.signature"));
    assert!(source.contains("minimum_distinct_providers: 2"));
    assert!(source.contains("exact_verification_policy_digest"));
    assert!(source.contains("exact_verifier_set_digest"));
}

#[test]
fn legacy_scalar_verified_wrappers_and_scalar_now_are_absent() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("VerifiedTransparencyCheckpoint"));
    assert!(!source.contains("VerifiedTransparencyWitnessQuorum"));
    assert!(!source.contains("now_unix_s:"));
    assert!(!source.contains("evaluation_time_unix_s:"));
}

#[test]
fn live_head_is_opaque_and_commits_full_view() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct QuorumObservedContainmentHeadV1"
    ));
    for required in [
        "authority_id",
        "state_digest",
        "generation",
        "compromise_tracker_digest",
        "publication_digest",
        "transparency_log_digest",
        "transparency_root_digest",
        "checkpoint_digest",
        "signed_checkpoint_evidence_digest",
        "witness_set_evidence_digest",
        "witness_registry_id",
        "trust_head_id",
        "trust_anchor_mode",
        "exact_verifier_set_digest",
        "authorization_to_trust_clock_lineage_digest",
        "trust_to_observation_clock_lineage_digest",
    ] {
        assert!(source.contains(required), "missing head commitment {required}");
    }
}
