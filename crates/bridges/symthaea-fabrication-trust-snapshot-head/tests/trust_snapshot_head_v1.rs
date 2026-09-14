// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[test]
fn live_head_requires_legitimate_rotation_and_activation_capabilities() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("rotation: &ClockGovernedTrustRotationV1"));
    assert!(source.contains(
        "activation: &ClockGovernedTrustSnapshotActivationPermitV1"
    ));
    assert!(source.contains("require_rotation_activation_match"));
    assert!(source.contains("rotation.proposed_snapshot()"));
}

#[test]
fn currentness_is_monotonic_sequence_aware_not_last_digest_only() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("TRUST_SNAPSHOT_HEAD_LOG_KIND_PREFIX"));
    assert!(source.contains("parse_trust_snapshot_head_sequence"));
    assert!(source.contains("TrustSnapshotSequenceRegressed"));
    assert!(source.contains("DuplicateTrustSnapshotSequence"));
    assert!(source.contains("HigherSnapshotSequencePublished"));
    assert!(source.contains("suffix != sequence.to_string()"));
}

#[test]
fn observation_clock_must_descend_from_activation_clock() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("activation_basis: &OperationalClockBasisV1"));
    assert!(source.contains("observation_clock_bridge: &[OperationalClockBasisV1]"));
    assert!(source.contains("observation_basis: &OperationalClockBasisV1"));
    assert!(source.contains("predecessor_operational_basis_id"));
    assert!(source.contains("observation_clock_lineage_digest"));
    assert!(source.contains("MAX_TRUST_SNAPSHOT_HEAD_CLOCK_HOPS"));
}

#[test]
fn scalar_verified_transparency_wrappers_are_not_live_authority_inputs() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("VerifiedTransparencyCheckpoint,"));
    assert!(!source.contains("VerifiedTransparencyWitnessQuorum,"));
    assert!(!source.contains("verified_checkpoint:"));
    assert!(!source.contains("verified_witness_quorum:"));
    assert!(!source.contains("now_unix_s:"));
    assert!(!source.contains("evaluation_time_unix_s:"));
}

#[test]
fn checkpoint_and_witness_raw_bytes_are_verified_directly() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("ExactTrustSnapshotHeadEvidenceVerifierV1"));
    assert!(source.contains("verify_checkpoint_signature"));
    assert!(source.contains("verify_witness_signature"));
    assert!(source.contains("signed_checkpoint.signature.signature"));
    assert!(source.contains("witness.signature.signature"));
    assert!(source.contains("minimum_distinct_providers: 2"));
    assert!(source.contains("exact_verification_policy_digest"));
    assert!(source.contains("exact_verifier_set_digest"));
}

#[test]
fn activated_snapshot_itself_is_the_signer_authority() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("let snapshot = rotation.proposed_snapshot();"));
    assert!(source.contains("KeyUsage::TransparencyLog"));
    assert!(source.contains("KeyUsage::TransparencyWitness"));
    assert!(source.contains("SnapshotNotValidAcrossObservationEnvelope"));
    assert!(source.contains("SignerInvalidAtEvidenceTime"));
    assert!(source.contains("SignerCompromisedAcrossEnvelope"));
}

#[test]
fn trust_head_binds_governed_witness_identity_and_exact_view() {
    let source = include_str!("../src/lib.rs");
    for required in [
        "witness_registry_id",
        "witness_registry_digest",
        "witness_registry_sequence",
        "publication_digest",
        "publication_entry_sequence",
        "transparency_log_digest",
        "transparency_root_digest",
        "checkpoint_digest",
        "signed_checkpoint_evidence_digest",
        "witness_set_evidence_digest",
        "containment_state_digest",
        "compromise_tracker_digest",
    ] {
        assert!(source.contains(required), "missing head commitment {required}");
    }
    assert!(source.contains("profile.organization != signed.statement.witness_organization"));
    assert!(source.contains("profile.failure_domain != signed.statement.witness_region"));
}

#[test]
fn live_head_is_opaque_and_non_deserializable() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("pub struct QuorumObservedTrustSnapshotHeadV1"));
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct QuorumObservedTrustSnapshotHeadV1"
    ));
}
