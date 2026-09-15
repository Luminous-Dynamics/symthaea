// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_fabrication_kernel::authority_epoch::AuthorityEpochVector;
use symthaea_fabrication_kernel::crypto_digest::{Sha256Digest, sha256};
use symthaea_fabrication_kernel::upgrade_handoff::{
    UPGRADE_ENDPOINT_SCHEMA, UpgradeEndpoint, digest_upgrade_endpoint,
};
use symthaea_fabrication_lineage_predecessor_certificate::{
    LINEAGE_FINALIZED_PREDECESSOR_CERTIFICATE_SCHEMA,
    LineageFinalizedPredecessorCertificateV1, LineagePredecessorCertificateError,
    digest_lineage_finalized_predecessor_certificate_v1,
    validate_lineage_finalized_predecessor_certificate_v1,
};

fn digest(label: &[u8]) -> Sha256Digest {
    sha256(label)
}

fn valid_endpoint() -> UpgradeEndpoint {
    UpgradeEndpoint {
        schema_version: UPGRADE_ENDPOINT_SCHEMA.into(),
        software_version: "2.0.0".into(),
        source_tree_digest: digest(b"source-tree"),
        executable_digest: digest(b"executable"),
        durable_state_digest: digest(b"durable-state"),
        replay_contract_digest: digest(b"replay-contract"),
        authority_epoch: AuthorityEpochVector::new(2, 2, 1, 1, 1, 5, 3, 2)
            .expect("test authority epoch is valid"),
    }
}

fn valid_certificate() -> LineageFinalizedPredecessorCertificateV1 {
    let endpoint = valid_endpoint();
    let endpoint_digest = digest_upgrade_endpoint(&endpoint).expect("test endpoint is digestible");
    LineageFinalizedPredecessorCertificateV1 {
        schema_version: LINEAGE_FINALIZED_PREDECESSOR_CERTIFICATE_SCHEMA.into(),
        predecessor_root_id: digest(b"predecessor-root").to_hex(),
        global_head_id: digest(b"global-head").to_hex(),
        current_head_id: digest(b"current-head").to_hex(),
        finalized_upgrade_id: digest(b"finalized-upgrade").to_hex(),
        finalization_record_digest: digest(b"finalization-record"),
        prior_predecessor_root_digest: digest(b"prior-predecessor-root"),
        finalization_sequence: 7,
        endpoint: endpoint.clone(),
        endpoint_digest,
        rollback_target_digest: endpoint.durable_state_digest,
        evidence_checkpoint_digest: digest(b"checkpoint"),
        transparency_log_digest: digest(b"transparency-log"),
        governance_view_id: digest(b"governance-view").to_hex(),
        registry_head_id: digest(b"registry-head").to_hex(),
        registry_digest: digest(b"registry"),
        registry_sequence: 3,
        trust_snapshot_digest: digest(b"trust-snapshot"),
        containment_state_digest: digest(b"containment-state"),
        compromise_tracker_digest: digest(b"compromise-tracker"),
        containment_generation: 4,
        clock_envelope_id: digest(b"clock-envelope").to_hex(),
        operational_basis_id: digest(b"operational-basis").to_hex(),
    }
}

#[test]
fn valid_certificate_passes_behavioral_validation() {
    let certificate = valid_certificate();
    assert_eq!(
        validate_lineage_finalized_predecessor_certificate_v1(&certificate),
        Ok(())
    );
    assert!(digest_lineage_finalized_predecessor_certificate_v1(&certificate).is_ok());
}

#[test]
fn noncanonical_id_is_rejected() {
    let mut certificate = valid_certificate();
    certificate.predecessor_root_id = "A".repeat(64);
    assert_eq!(
        validate_lineage_finalized_predecessor_certificate_v1(&certificate),
        Err(LineagePredecessorCertificateError::InvalidCanonicalId(
            "predecessor_root_id"
        ))
    );
}

#[test]
fn zero_registry_sequence_is_rejected() {
    let mut certificate = valid_certificate();
    certificate.registry_sequence = 0;
    assert_eq!(
        validate_lineage_finalized_predecessor_certificate_v1(&certificate),
        Err(LineagePredecessorCertificateError::InvalidRegistrySequence)
    );
}

#[test]
fn zero_containment_generation_is_rejected() {
    let mut certificate = valid_certificate();
    certificate.containment_generation = 0;
    assert_eq!(
        validate_lineage_finalized_predecessor_certificate_v1(&certificate),
        Err(LineagePredecessorCertificateError::InvalidContainmentGeneration)
    );
}

#[test]
fn endpoint_digest_tamper_is_rejected() {
    let mut certificate = valid_certificate();
    certificate.endpoint_digest = digest(b"tampered-endpoint-digest");
    assert_eq!(
        validate_lineage_finalized_predecessor_certificate_v1(&certificate),
        Err(LineagePredecessorCertificateError::EndpointDigestMismatch)
    );
}

#[test]
fn rollback_target_tamper_is_rejected() {
    let mut certificate = valid_certificate();
    certificate.rollback_target_digest = digest(b"wrong-rollback-target");
    assert_eq!(
        validate_lineage_finalized_predecessor_certificate_v1(&certificate),
        Err(LineagePredecessorCertificateError::RollbackTargetMismatch)
    );
}

#[test]
fn governance_view_identity_is_digest_bound() {
    let mut certificate = valid_certificate();
    let before = digest_lineage_finalized_predecessor_certificate_v1(&certificate)
        .expect("valid certificate hashes");
    certificate.governance_view_id = digest(b"other-governance-view").to_hex();
    let after = digest_lineage_finalized_predecessor_certificate_v1(&certificate)
        .expect("mutated but still canonical certificate hashes");
    assert_ne!(before, after);
}

#[test]
fn serde_round_trip_preserves_certificate_digest() {
    let certificate = valid_certificate();
    let before = digest_lineage_finalized_predecessor_certificate_v1(&certificate)
        .expect("valid certificate hashes");
    let bytes = serde_json::to_vec(&certificate).expect("test certificate serializes");
    let decoded: LineageFinalizedPredecessorCertificateV1 =
        serde_json::from_slice(&bytes).expect("test certificate deserializes");
    let after = digest_lineage_finalized_predecessor_certificate_v1(&decoded)
        .expect("round-tripped certificate hashes");
    assert_eq!(certificate, decoded);
    assert_eq!(before, after);
}

#[test]
fn certificate_is_portable_but_verified_authority_is_opaque() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("pub struct LineageFinalizedPredecessorCertificateV1"));
    assert!(source.contains("Serialize, Deserialize"));
    assert!(source.contains("pub struct VerifiedLineageFinalizedPredecessorCertificateV1"));
    assert!(!source.contains("Serialize, Deserialize)]\npub struct VerifiedLineageFinalizedPredecessorCertificateV1"));
}

#[test]
fn certificate_quorum_has_distinct_non_replayable_purpose() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("lineage-finalized-predecessor-certificate-v1"));
    assert!(source.contains("certificate_ceremony.purpose() != LINEAGE_FINALIZED_PREDECESSOR_CERTIFICATE_PURPOSE"));
    assert!(source.contains("certificate_ceremony.payload_digest() != certificate_digest"));
}

#[test]
fn certificate_uses_exact_handoff_governance_context() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("prepared_handoff.trust_snapshot_digest() != certificate.trust_snapshot_digest"));
    assert!(source.contains("prepared_handoff.containment_state_digest() != certificate.containment_state_digest"));
    assert!(source.contains("prepared_handoff.compromise_tracker_digest() != certificate.compromise_tracker_digest"));
    assert!(source.contains("prepared_handoff.clock_envelope_id().to_hex() != certificate.clock_envelope_id"));
    assert!(source.contains("prepared_handoff.operational_basis_id().to_hex() != certificate.operational_basis_id"));
    assert!(source.contains("certificate_ceremony.policy_digest() != prepared_handoff.threshold_policy_digest()"));
    assert!(source.contains("certificate_ceremony.trust_snapshot_digest() != prepared_handoff.trust_snapshot_digest()"));
    assert!(source.contains("certificate_ceremony.compromise_tracker_digest() != prepared_handoff.compromise_tracker_digest()"));
    assert!(source.contains("certificate_ceremony.clock_envelope_id() != prepared_handoff.clock_envelope_id()"));
}

#[test]
fn certificate_carries_exact_witnessed_governance_snapshot() {
    let source = include_str!("../src/lib.rs");
    for field in [
        "pub governance_view_id: String",
        "pub registry_head_id: String",
        "pub registry_digest: Sha256Digest",
        "pub registry_sequence: u64",
        "pub trust_snapshot_digest: Sha256Digest",
        "pub containment_state_digest: Sha256Digest",
        "pub compromise_tracker_digest: Sha256Digest",
        "pub containment_generation: u64",
    ] {
        assert!(source.contains(field), "missing governance field: {field}");
    }
    assert!(source.contains("certificate.registry_sequence == 0"));
    assert!(source.contains("certificate.containment_generation == 0"));
}

#[test]
fn prepared_handoff_must_exactly_use_certified_root_facts() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("prepared_handoff.plan().predecessor != certificate.endpoint"));
    assert!(source.contains("prepared_handoff.plan().rollback_target_digest != certificate.rollback_target_digest"));
    assert!(source.contains("prepared_handoff.plan().evidence_checkpoint_digest != certificate.evidence_checkpoint_digest"));
    assert!(source.contains("prepared_handoff.operational_basis_id().to_hex() != certificate.operational_basis_id"));
}

#[test]
fn portable_certificate_is_canonical_and_endpoint_bound() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("canonical_hex_id"));
    assert!(source.contains("digest_upgrade_endpoint(&certificate.endpoint)"));
    assert!(source.contains("certificate.rollback_target_digest != certificate.endpoint.durable_state_digest"));
    assert!(source.contains("certificate.finalization_sequence == 0"));
}

#[test]
fn scalar_or_panic_authority_does_not_enter_certificate_verifier() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("now_unix"));
    assert!(!source.contains("saturating_"));
    assert!(!source.contains(".unwrap("));
    assert!(!source.contains(".expect("));
}
