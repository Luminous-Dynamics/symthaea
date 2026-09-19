// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::evidence_digest::{
    perceptual_collection_authenticity::CollectionVerifierIdentityV1,
    perceptual_enrollment_orchestrator::{
        EnrollmentWitnessAdapterFailureKindV1, EnrollmentWitnessProviderAdapterV1,
    },
    perceptual_enrollment_witness::{
        FrozenPerceptualEnrollmentWitnessPolicyV1,
        PERCEPTUAL_ENROLLMENT_WITNESS_POLICY_VERSION,
    },
    perceptual_enrollment_witness_provider::{
        collection_signing_request_commitment, witness_service_request_commitment,
        ExternalEnrollmentCollectionSigningRequestV1, ExternalEnrollmentWitnessServiceRequestV1,
        ENROLLMENT_WITNESS_PROVIDER_PROTOCOL_VERSION,
    },
    perceptual_enrollment_witness_provider_fixture::{
        DurableQualificationEnrollmentWitnessProviderV1, QualificationProviderFaultPointV1,
    },
};
use ed25519_dalek::SigningKey;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const COLLECTION_SEED: [u8; 32] = [7u8; 32];
const WITNESS_SEED: [u8; 32] = [9u8; 32];

fn identity(name: &str, seed: [u8; 32]) -> CollectionVerifierIdentityV1 {
    let key = SigningKey::from_bytes(&seed);
    CollectionVerifierIdentityV1 {
        signer_id: name.into(),
        key_epoch: 1,
        verifying_key_bytes: key.verifying_key().to_bytes().to_vec(),
    }
}

fn policy() -> FrozenPerceptualEnrollmentWitnessPolicyV1 {
    FrozenPerceptualEnrollmentWitnessPolicyV1 {
        policy_version: PERCEPTUAL_ENROLLMENT_WITNESS_POLICY_VERSION.into(),
        enrollment_policy_sha256: "10".repeat(32),
        collection_authenticity_policy_sha256: "20".repeat(32),
        witness_log_id: "qualification-witness-log".into(),
        collection_signer: identity("qualification-collection", COLLECTION_SEED),
        witness_signer: identity("qualification-witness", WITNESS_SEED),
        per_allocation_external_witness_required: true,
        scored_authority_requires_verified_witness: true,
        raw_participant_token_prohibited: true,
        policy_sha256: "30".repeat(32),
    }
}

fn temp_roots(label: &str) -> (PathBuf, PathBuf, PathBuf) {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let base = std::env::temp_dir().join(format!(
        "mel003-provider-fixture-{label}-{}-{nonce}",
        std::process::id()
    ));
    (base.join("collection"), base.join("witness"), base)
}

fn provider(collection: &PathBuf, witness: &PathBuf) -> DurableQualificationEnrollmentWitnessProviderV1 {
    DurableQualificationEnrollmentWitnessProviderV1::new(
        collection,
        witness,
        &policy(),
        COLLECTION_SEED,
        WITNESS_SEED,
    )
    .unwrap()
}

fn collection_request() -> ExternalEnrollmentCollectionSigningRequestV1 {
    let mut request = ExternalEnrollmentCollectionSigningRequestV1 {
        protocol_version: ENROLLMENT_WITNESS_PROVIDER_PROTOCOL_VERSION.into(),
        request_sha256: "41".repeat(32),
        witness_policy_sha256: policy().policy_sha256,
        sequence: 0,
        witness_head_sha256: "42".repeat(32),
        signer_id: "qualification-collection".into(),
        key_epoch: 1,
        signing_transcript: b"qualification-collection-transcript".to_vec(),
        collection_request_sha256: String::new(),
    };
    request.collection_request_sha256 = collection_signing_request_commitment(&request).unwrap();
    request
}

fn witness_request() -> ExternalEnrollmentWitnessServiceRequestV1 {
    let mut request = ExternalEnrollmentWitnessServiceRequestV1 {
        protocol_version: ENROLLMENT_WITNESS_PROVIDER_PROTOCOL_VERSION.into(),
        request_sha256: "51".repeat(32),
        witness_policy_sha256: policy().policy_sha256,
        witness_log_id: "qualification-witness-log".into(),
        sequence: 0,
        witness_head_sha256: "52".repeat(32),
        evidence_signer_id: "qualification-collection".into(),
        evidence_key_epoch: 1,
        evidence_signature: vec![0x53; 64],
        collection_request_sha256: "54".repeat(32),
        witness_service_request_sha256: String::new(),
    };
    request.witness_service_request_sha256 = witness_service_request_commitment(&request).unwrap();
    request
}

#[cfg(target_os = "linux")]
#[test]
fn collection_accept_then_response_loss_replays_exact_response_after_restart() {
    let (collection, witness, base) = temp_roots("collection-recovery");
    let request = collection_request();

    let mut first = provider(&collection, &witness);
    first.inject_collection_fault(QualificationProviderFaultPointV1::AfterPersistUncertain);
    let error = first.sign_collection(&request).unwrap_err();
    assert_eq!(error.kind, EnrollmentWitnessAdapterFailureKindV1::Uncertain);
    drop(first);

    let mut restarted = provider(&collection, &witness);
    let recovered = restarted.sign_collection(&request).unwrap();
    let replayed = restarted.sign_collection(&request).unwrap();
    assert_eq!(recovered, replayed);
    assert_eq!(recovered.request_sha256, request.request_sha256);
    assert_eq!(recovered.collection_request_sha256, request.collection_request_sha256);

    fs::remove_dir_all(base).unwrap();
}

#[cfg(target_os = "linux")]
#[test]
fn witness_accept_then_response_loss_replays_same_anchor_and_signature_after_restart() {
    let (collection, witness, base) = temp_roots("witness-recovery");
    let request = witness_request();

    let mut first = provider(&collection, &witness);
    first.inject_witness_fault(QualificationProviderFaultPointV1::AfterPersistUncertain);
    let error = first.witness_enrollment(&request).unwrap_err();
    assert_eq!(error.kind, EnrollmentWitnessAdapterFailureKindV1::Uncertain);
    drop(first);

    let mut restarted = provider(&collection, &witness);
    let recovered = restarted.witness_enrollment(&request).unwrap();
    let replayed = restarted.witness_enrollment(&request).unwrap();
    assert_eq!(recovered, replayed);
    assert_eq!(
        recovered.witness_anchor_reference,
        format!("mel003-local-qualification-anchor-v1:{}", request.request_sha256)
    );
    assert_eq!(recovered.witness_signature, replayed.witness_signature);

    fs::remove_dir_all(base).unwrap();
}

#[cfg(target_os = "linux")]
#[test]
fn same_collection_request_id_with_changed_payload_is_rejected() {
    let (collection, witness, base) = temp_roots("collection-conflict");
    let request = collection_request();
    let mut fixture = provider(&collection, &witness);
    let _ = fixture.sign_collection(&request).unwrap();

    let mut conflicting = request.clone();
    conflicting.witness_head_sha256 = "61".repeat(32);
    conflicting.collection_request_sha256 = collection_signing_request_commitment(&conflicting).unwrap();
    let error = fixture.sign_collection(&conflicting).unwrap_err();
    assert_eq!(error.kind, EnrollmentWitnessAdapterFailureKindV1::Rejected);

    fs::remove_dir_all(base).unwrap();
}

#[cfg(target_os = "linux")]
#[test]
fn same_witness_request_id_with_changed_payload_is_rejected() {
    let (collection, witness, base) = temp_roots("witness-conflict");
    let request = witness_request();
    let mut fixture = provider(&collection, &witness);
    let _ = fixture.witness_enrollment(&request).unwrap();

    let mut conflicting = request.clone();
    conflicting.witness_head_sha256 = "62".repeat(32);
    conflicting.witness_service_request_sha256 = witness_service_request_commitment(&conflicting).unwrap();
    let error = fixture.witness_enrollment(&conflicting).unwrap_err();
    assert_eq!(error.kind, EnrollmentWitnessAdapterFailureKindV1::Rejected);

    fs::remove_dir_all(base).unwrap();
}
