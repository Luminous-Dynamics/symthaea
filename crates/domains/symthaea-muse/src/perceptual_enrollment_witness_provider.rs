// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1EILR-E: provider-neutral signing protocol for independently
//! witnessed enrollment allocations.
//!
//! This module externalizes the two signatures already verified by P1EILR-B
//! without changing their cryptographic transcripts. Providers receive only
//! commitment-level enrollment evidence; raw participant pseudonyms, private
//! randomization semantics and arm labels are absent from every wire DTO.
//!
//! Receipt/bundle assembly remains candidate evidence. Final authority still
//! requires the existing P1EILR-B bundle validator and P1EILR-D durable-state
//! reconstruction before P1EACR may consume the sealed verified authority.

use crate::evidence_digest::{
    canonical_json_bytes, canonical_json_sha256, decode_hex_32, sha256_hex,
    perceptual_collection_authenticity::CollectionVerifierIdentityV1,
    perceptual_enrollment_coordinator::{
        pending_witness_request_commitment, PendingEnrollmentWitnessRequestV1,
    },
    perceptual_enrollment_witness::{
        enrollment_witness_bundle_commitment, enrollment_witness_receipt_commitment,
        FrozenPerceptualEnrollmentWitnessBundleV1, FrozenPerceptualEnrollmentWitnessPolicyV1,
        WitnessedEnrollmentAllocationReceiptV1, PERCEPTUAL_ENROLLMENT_WITNESS_BUNDLE_VERSION,
        PERCEPTUAL_ENROLLMENT_WITNESS_RECEIPT_VERSION, ZERO_SHA256,
    },
};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const ENROLLMENT_WITNESS_PROVIDER_PROTOCOL_VERSION: &str =
    "mel003-enrollment-witness-provider-protocol-v1";
pub const ENROLLMENT_WITNESS_COLLECTION_SIGNATURE_DOMAIN_V1: &[u8] =
    b"symthaea.mel003.p1.enrollment-witness.v1/collection-evidence";
pub const ENROLLMENT_WITNESS_EXTERNAL_SIGNATURE_DOMAIN_V1: &[u8] =
    b"symthaea.mel003.p1.enrollment-witness.v1/external-witness";
const MAX_SIGNATURE_MESSAGE_BYTES: usize = 1 << 20;
const MAX_SIGNATURE_DOMAIN_BYTES: usize = 256;
const ED25519_PUBLIC_KEY_BYTES: usize = 32;
const ED25519_SIGNATURE_BYTES: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedExternalEnrollmentWitnessV1 {
    pub protocol_version: String,
    /// Stable P1EILR-C idempotency identity.
    pub request_sha256: String,
    pub witness_policy_sha256: String,
    pub witness_log_id: String,
    pub sequence: u32,
    pub enrollment_ledger_sha256: String,
    pub durable_state_sha256: String,
    pub allocation_head_sha256: String,
    pub enrollment_participant_commitment_sha256: String,
    pub participant_schedule_projection_sha256: String,
    pub previous_witness_head_sha256: String,
    /// Exact hash consumed by the P1EILR-B collection-signature domain.
    pub witness_head_sha256: String,
    pub prepared_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalEnrollmentCollectionSigningRequestV1 {
    pub protocol_version: String,
    pub request_sha256: String,
    pub witness_policy_sha256: String,
    pub sequence: u32,
    pub witness_head_sha256: String,
    pub signer_id: String,
    pub key_epoch: u64,
    /// Exact bytes the frozen collection identity must Ed25519-sign.
    pub signing_transcript: Vec<u8>,
    pub collection_request_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalEnrollmentCollectionSignatureV1 {
    pub protocol_version: String,
    pub request_sha256: String,
    pub witness_head_sha256: String,
    pub collection_request_sha256: String,
    pub signer_id: String,
    pub key_epoch: u64,
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalEnrollmentWitnessServiceRequestV1 {
    pub protocol_version: String,
    /// Stable idempotency key. Retrying this exact identity must recover the
    /// same accepted semantic witness operation.
    pub request_sha256: String,
    pub witness_policy_sha256: String,
    pub witness_log_id: String,
    pub sequence: u32,
    pub witness_head_sha256: String,
    pub evidence_signer_id: String,
    pub evidence_key_epoch: u64,
    pub evidence_signature: Vec<u8>,
    pub collection_request_sha256: String,
    pub witness_service_request_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalEnrollmentWitnessServiceResponseV1 {
    pub protocol_version: String,
    pub request_sha256: String,
    pub witness_head_sha256: String,
    pub witness_service_request_sha256: String,
    pub witness_anchor_reference: String,
    pub witness_signer_id: String,
    pub witness_key_epoch: u64,
    pub witness_signature: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualEnrollmentWitnessProviderIssueV1 {
    WrongProtocolVersion,
    PendingRequestDigestMismatch,
    PreparedDigestMismatch,
    PreparedFieldMismatch { field: String },
    InvalidDigest { field: String },
    EmptyWitnessLogId,
    CollectionRequestDigestMismatch,
    CollectionRequestMismatch { field: String },
    CollectionSignerIdentityMismatch,
    CollectionSignatureInvalid,
    WitnessServiceRequestDigestMismatch,
    WitnessServiceRequestMismatch { field: String },
    WitnessSignerIdentityMismatch,
    EmptyWitnessAnchor,
    WitnessSignatureInvalid,
    InvalidSignatureDomain,
    SignatureMessageTooLarge,
    InvalidVerifyingKey,
    CandidateBundleMalformed,
    CandidateSequenceMismatch { found: u32, expected: u32 },
    CandidatePreviousHeadMismatch,
    DuplicateWitnessAnchor { anchor: String },
    SerializationFailed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct CompatibleEnrollmentWitnessUnsignedEntryV1<'a> {
    witness_policy_sha256: &'a str,
    sequence: u32,
    enrollment_ledger_sha256: &'a str,
    durable_state_sha256: &'a str,
    allocation_head_sha256: &'a str,
    enrollment_participant_commitment_sha256: &'a str,
    participant_schedule_projection_sha256: &'a str,
    previous_witness_head_sha256: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct PreparedExternalEnrollmentWitnessCommitmentV1<'a> {
    protocol_version: &'a str,
    request_sha256: &'a str,
    witness_policy_sha256: &'a str,
    witness_log_id: &'a str,
    sequence: u32,
    enrollment_ledger_sha256: &'a str,
    durable_state_sha256: &'a str,
    allocation_head_sha256: &'a str,
    enrollment_participant_commitment_sha256: &'a str,
    participant_schedule_projection_sha256: &'a str,
    previous_witness_head_sha256: &'a str,
    witness_head_sha256: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct CollectionSigningRequestCommitmentV1<'a> {
    protocol_version: &'a str,
    request_sha256: &'a str,
    witness_policy_sha256: &'a str,
    sequence: u32,
    witness_head_sha256: &'a str,
    signer_id: &'a str,
    key_epoch: u64,
    signing_transcript_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct WitnessServiceRequestCommitmentV1<'a> {
    protocol_version: &'a str,
    request_sha256: &'a str,
    witness_policy_sha256: &'a str,
    witness_log_id: &'a str,
    sequence: u32,
    witness_head_sha256: &'a str,
    evidence_signer_id: &'a str,
    evidence_key_epoch: u64,
    evidence_signature_sha256: String,
    collection_request_sha256: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct CompatibleEnrollmentWitnessStatementV1<'a> {
    witness_log_id: &'a str,
    sequence: u32,
    witness_head_sha256: &'a str,
    witness_anchor_reference: &'a str,
}

pub fn prepare_external_enrollment_witness(
    pending: &PendingEnrollmentWitnessRequestV1,
) -> Result<PreparedExternalEnrollmentWitnessV1, Vec<PerceptualEnrollmentWitnessProviderIssueV1>> {
    let mut issues = validate_pending_request(pending);
    if !issues.is_empty() {
        return Err(issues);
    }
    let unsigned = CompatibleEnrollmentWitnessUnsignedEntryV1 {
        witness_policy_sha256: &pending.enrollment_witness_policy_sha256,
        sequence: pending.sequence,
        enrollment_ledger_sha256: &pending.enrollment_ledger_sha256,
        durable_state_sha256: &pending.durable_state_sha256,
        allocation_head_sha256: &pending.allocation_head_sha256,
        enrollment_participant_commitment_sha256: &pending
            .enrollment_participant_commitment_sha256,
        participant_schedule_projection_sha256: &pending.participant_schedule_projection_sha256,
        previous_witness_head_sha256: &pending.previous_witness_head_sha256,
    };
    let witness_head_sha256 = canonical_json_sha256(&unsigned).map_err(|_| {
        vec![PerceptualEnrollmentWitnessProviderIssueV1::SerializationFailed]
    })?;
    let mut prepared = PreparedExternalEnrollmentWitnessV1 {
        protocol_version: ENROLLMENT_WITNESS_PROVIDER_PROTOCOL_VERSION.into(),
        request_sha256: pending.request_sha256.clone(),
        witness_policy_sha256: pending.enrollment_witness_policy_sha256.clone(),
        witness_log_id: pending.witness_log_id.clone(),
        sequence: pending.sequence,
        enrollment_ledger_sha256: pending.enrollment_ledger_sha256.clone(),
        durable_state_sha256: pending.durable_state_sha256.clone(),
        allocation_head_sha256: pending.allocation_head_sha256.clone(),
        enrollment_participant_commitment_sha256: pending
            .enrollment_participant_commitment_sha256
            .clone(),
        participant_schedule_projection_sha256: pending
            .participant_schedule_projection_sha256
            .clone(),
        previous_witness_head_sha256: pending.previous_witness_head_sha256.clone(),
        witness_head_sha256,
        prepared_sha256: String::new(),
    };
    prepared.prepared_sha256 = prepared_external_witness_commitment(&prepared).map_err(|_| {
        vec![PerceptualEnrollmentWitnessProviderIssueV1::SerializationFailed]
    })?;
    issues.extend(validate_prepared_external_enrollment_witness(&prepared));
    if issues.is_empty() {
        Ok(prepared)
    } else {
        Err(issues)
    }
}

pub fn prepared_external_witness_commitment(
    prepared: &PreparedExternalEnrollmentWitnessV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&PreparedExternalEnrollmentWitnessCommitmentV1 {
        protocol_version: &prepared.protocol_version,
        request_sha256: &prepared.request_sha256,
        witness_policy_sha256: &prepared.witness_policy_sha256,
        witness_log_id: &prepared.witness_log_id,
        sequence: prepared.sequence,
        enrollment_ledger_sha256: &prepared.enrollment_ledger_sha256,
        durable_state_sha256: &prepared.durable_state_sha256,
        allocation_head_sha256: &prepared.allocation_head_sha256,
        enrollment_participant_commitment_sha256: &prepared
            .enrollment_participant_commitment_sha256,
        participant_schedule_projection_sha256: &prepared
            .participant_schedule_projection_sha256,
        previous_witness_head_sha256: &prepared.previous_witness_head_sha256,
        witness_head_sha256: &prepared.witness_head_sha256,
    })
}

pub fn validate_prepared_external_enrollment_witness(
    prepared: &PreparedExternalEnrollmentWitnessV1,
) -> Vec<PerceptualEnrollmentWitnessProviderIssueV1> {
    let mut issues = Vec::new();
    if prepared.protocol_version != ENROLLMENT_WITNESS_PROVIDER_PROTOCOL_VERSION {
        issues.push(PerceptualEnrollmentWitnessProviderIssueV1::WrongProtocolVersion);
    }
    let pending = PendingEnrollmentWitnessRequestV1 {
        sequence: prepared.sequence,
        enrollment_witness_policy_sha256: prepared.witness_policy_sha256.clone(),
        witness_log_id: prepared.witness_log_id.clone(),
        enrollment_ledger_sha256: prepared.enrollment_ledger_sha256.clone(),
        durable_state_sha256: prepared.durable_state_sha256.clone(),
        allocation_head_sha256: prepared.allocation_head_sha256.clone(),
        enrollment_participant_commitment_sha256: prepared
            .enrollment_participant_commitment_sha256
            .clone(),
        participant_schedule_projection_sha256: prepared
            .participant_schedule_projection_sha256
            .clone(),
        previous_witness_head_sha256: prepared.previous_witness_head_sha256.clone(),
        request_sha256: prepared.request_sha256.clone(),
    };
    issues.extend(validate_pending_request(&pending));
    let unsigned = CompatibleEnrollmentWitnessUnsignedEntryV1 {
        witness_policy_sha256: &prepared.witness_policy_sha256,
        sequence: prepared.sequence,
        enrollment_ledger_sha256: &prepared.enrollment_ledger_sha256,
        durable_state_sha256: &prepared.durable_state_sha256,
        allocation_head_sha256: &prepared.allocation_head_sha256,
        enrollment_participant_commitment_sha256: &prepared
            .enrollment_participant_commitment_sha256,
        participant_schedule_projection_sha256: &prepared
            .participant_schedule_projection_sha256,
        previous_witness_head_sha256: &prepared.previous_witness_head_sha256,
    };
    match canonical_json_sha256(&unsigned) {
        Ok(value) if value == prepared.witness_head_sha256 => {}
        _ => issues.push(PerceptualEnrollmentWitnessProviderIssueV1::PreparedFieldMismatch {
            field: "witness_head_sha256".into(),
        }),
    }
    match prepared_external_witness_commitment(prepared) {
        Ok(value) if value == prepared.prepared_sha256 => {}
        _ => issues.push(PerceptualEnrollmentWitnessProviderIssueV1::PreparedDigestMismatch),
    }
    issues
}

pub fn enrollment_witness_signature_transcript_v1(
    domain: &[u8],
    message: &[u8],
) -> Result<Vec<u8>, PerceptualEnrollmentWitnessProviderIssueV1> {
    if domain.is_empty() || domain.len() > MAX_SIGNATURE_DOMAIN_BYTES {
        return Err(PerceptualEnrollmentWitnessProviderIssueV1::InvalidSignatureDomain);
    }
    if message.is_empty() || message.len() > MAX_SIGNATURE_MESSAGE_BYTES {
        return Err(PerceptualEnrollmentWitnessProviderIssueV1::SignatureMessageTooLarge);
    }
    let mut transcript = Vec::with_capacity(8 + domain.len() + message.len());
    transcript.extend_from_slice(&(domain.len() as u64).to_le_bytes());
    transcript.extend_from_slice(domain);
    transcript.extend_from_slice(message);
    Ok(transcript)
}

pub fn collection_signature_transcript(
    prepared: &PreparedExternalEnrollmentWitnessV1,
) -> Result<Vec<u8>, Vec<PerceptualEnrollmentWitnessProviderIssueV1>> {
    let issues = validate_prepared_external_enrollment_witness(prepared);
    if !issues.is_empty() {
        return Err(issues);
    }
    let head = decode_hex_32(&prepared.witness_head_sha256).ok_or_else(|| {
        vec![PerceptualEnrollmentWitnessProviderIssueV1::InvalidDigest {
            field: "witness_head_sha256".into(),
        }]
    })?;
    enrollment_witness_signature_transcript_v1(
        ENROLLMENT_WITNESS_COLLECTION_SIGNATURE_DOMAIN_V1,
        &head,
    )
    .map_err(|issue| vec![issue])
}

pub fn build_collection_signing_request(
    prepared: &PreparedExternalEnrollmentWitnessV1,
    policy: &FrozenPerceptualEnrollmentWitnessPolicyV1,
) -> Result<ExternalEnrollmentCollectionSigningRequestV1, Vec<PerceptualEnrollmentWitnessProviderIssueV1>> {
    let mut issues = validate_prepared_against_policy(prepared, policy);
    if !issues.is_empty() {
        return Err(issues);
    }
    let signing_transcript = collection_signature_transcript(prepared)?;
    let mut request = ExternalEnrollmentCollectionSigningRequestV1 {
        protocol_version: ENROLLMENT_WITNESS_PROVIDER_PROTOCOL_VERSION.into(),
        request_sha256: prepared.request_sha256.clone(),
        witness_policy_sha256: prepared.witness_policy_sha256.clone(),
        sequence: prepared.sequence,
        witness_head_sha256: prepared.witness_head_sha256.clone(),
        signer_id: policy.collection_signer.signer_id.clone(),
        key_epoch: policy.collection_signer.key_epoch,
        signing_transcript,
        collection_request_sha256: String::new(),
    };
    request.collection_request_sha256 = collection_signing_request_commitment(&request).map_err(|_| {
        vec![PerceptualEnrollmentWitnessProviderIssueV1::SerializationFailed]
    })?;
    issues.extend(validate_collection_signing_request(&request, prepared, policy));
    if issues.is_empty() {
        Ok(request)
    } else {
        Err(issues)
    }
}

pub fn collection_signing_request_commitment(
    request: &ExternalEnrollmentCollectionSigningRequestV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&CollectionSigningRequestCommitmentV1 {
        protocol_version: &request.protocol_version,
        request_sha256: &request.request_sha256,
        witness_policy_sha256: &request.witness_policy_sha256,
        sequence: request.sequence,
        witness_head_sha256: &request.witness_head_sha256,
        signer_id: &request.signer_id,
        key_epoch: request.key_epoch,
        signing_transcript_sha256: sha256_hex(&request.signing_transcript),
    })
}

pub fn build_witness_service_request(
    prepared: &PreparedExternalEnrollmentWitnessV1,
    collection_request: &ExternalEnrollmentCollectionSigningRequestV1,
    collection_response: &ExternalEnrollmentCollectionSignatureV1,
    policy: &FrozenPerceptualEnrollmentWitnessPolicyV1,
) -> Result<ExternalEnrollmentWitnessServiceRequestV1, Vec<PerceptualEnrollmentWitnessProviderIssueV1>> {
    let mut issues = validate_collection_signing_request(collection_request, prepared, policy);
    issues.extend(validate_collection_signature_response(
        collection_response,
        collection_request,
        prepared,
        policy,
    ));
    if !issues.is_empty() {
        return Err(issues);
    }
    let mut request = ExternalEnrollmentWitnessServiceRequestV1 {
        protocol_version: ENROLLMENT_WITNESS_PROVIDER_PROTOCOL_VERSION.into(),
        request_sha256: prepared.request_sha256.clone(),
        witness_policy_sha256: prepared.witness_policy_sha256.clone(),
        witness_log_id: prepared.witness_log_id.clone(),
        sequence: prepared.sequence,
        witness_head_sha256: prepared.witness_head_sha256.clone(),
        evidence_signer_id: collection_response.signer_id.clone(),
        evidence_key_epoch: collection_response.key_epoch,
        evidence_signature: collection_response.signature.clone(),
        collection_request_sha256: collection_request.collection_request_sha256.clone(),
        witness_service_request_sha256: String::new(),
    };
    request.witness_service_request_sha256 = witness_service_request_commitment(&request)
        .map_err(|_| vec![PerceptualEnrollmentWitnessProviderIssueV1::SerializationFailed])?;
    Ok(request)
}

pub fn witness_service_request_commitment(
    request: &ExternalEnrollmentWitnessServiceRequestV1,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&WitnessServiceRequestCommitmentV1 {
        protocol_version: &request.protocol_version,
        request_sha256: &request.request_sha256,
        witness_policy_sha256: &request.witness_policy_sha256,
        witness_log_id: &request.witness_log_id,
        sequence: request.sequence,
        witness_head_sha256: &request.witness_head_sha256,
        evidence_signer_id: &request.evidence_signer_id,
        evidence_key_epoch: request.evidence_key_epoch,
        evidence_signature_sha256: sha256_hex(&request.evidence_signature),
        collection_request_sha256: &request.collection_request_sha256,
    })
}

pub fn external_witness_signature_transcript(
    prepared: &PreparedExternalEnrollmentWitnessV1,
    witness_anchor_reference: &str,
) -> Result<Vec<u8>, Vec<PerceptualEnrollmentWitnessProviderIssueV1>> {
    let issues = validate_prepared_external_enrollment_witness(prepared);
    if !issues.is_empty() {
        return Err(issues);
    }
    if witness_anchor_reference.trim().is_empty() {
        return Err(vec![PerceptualEnrollmentWitnessProviderIssueV1::EmptyWitnessAnchor]);
    }
    let statement = CompatibleEnrollmentWitnessStatementV1 {
        witness_log_id: &prepared.witness_log_id,
        sequence: prepared.sequence,
        witness_head_sha256: &prepared.witness_head_sha256,
        witness_anchor_reference,
    };
    let message = canonical_json_bytes(&statement).map_err(|_| {
        vec![PerceptualEnrollmentWitnessProviderIssueV1::SerializationFailed]
    })?;
    enrollment_witness_signature_transcript_v1(
        ENROLLMENT_WITNESS_EXTERNAL_SIGNATURE_DOMAIN_V1,
        &message,
    )
    .map_err(|issue| vec![issue])
}

pub fn assemble_external_enrollment_witness_receipt(
    prepared: &PreparedExternalEnrollmentWitnessV1,
    collection_request: &ExternalEnrollmentCollectionSigningRequestV1,
    collection_response: &ExternalEnrollmentCollectionSignatureV1,
    witness_request: &ExternalEnrollmentWitnessServiceRequestV1,
    witness_response: &ExternalEnrollmentWitnessServiceResponseV1,
    policy: &FrozenPerceptualEnrollmentWitnessPolicyV1,
) -> Result<WitnessedEnrollmentAllocationReceiptV1, Vec<PerceptualEnrollmentWitnessProviderIssueV1>> {
    let mut issues = validate_collection_signing_request(collection_request, prepared, policy);
    issues.extend(validate_collection_signature_response(
        collection_response,
        collection_request,
        prepared,
        policy,
    ));
    issues.extend(validate_witness_service_request(
        witness_request,
        prepared,
        collection_request,
        collection_response,
        policy,
    ));
    issues.extend(validate_witness_service_response(
        witness_response,
        witness_request,
        prepared,
        policy,
    ));
    if !issues.is_empty() {
        return Err(issues);
    }

    let mut receipt = WitnessedEnrollmentAllocationReceiptV1 {
        receipt_version: PERCEPTUAL_ENROLLMENT_WITNESS_RECEIPT_VERSION.into(),
        witness_policy_sha256: prepared.witness_policy_sha256.clone(),
        sequence: prepared.sequence,
        enrollment_ledger_sha256: prepared.enrollment_ledger_sha256.clone(),
        durable_state_sha256: prepared.durable_state_sha256.clone(),
        allocation_head_sha256: prepared.allocation_head_sha256.clone(),
        enrollment_participant_commitment_sha256: prepared
            .enrollment_participant_commitment_sha256
            .clone(),
        participant_schedule_projection_sha256: prepared
            .participant_schedule_projection_sha256
            .clone(),
        previous_witness_head_sha256: prepared.previous_witness_head_sha256.clone(),
        witness_head_sha256: prepared.witness_head_sha256.clone(),
        evidence_signer_id: collection_response.signer_id.clone(),
        evidence_key_epoch: collection_response.key_epoch,
        evidence_signature: collection_response.signature.clone(),
        witness_anchor_reference: witness_response.witness_anchor_reference.clone(),
        witness_signer_id: witness_response.witness_signer_id.clone(),
        witness_key_epoch: witness_response.witness_key_epoch,
        witness_signature: witness_response.witness_signature.clone(),
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = enrollment_witness_receipt_commitment(&receipt).map_err(|_| {
        vec![PerceptualEnrollmentWitnessProviderIssueV1::SerializationFailed]
    })?;
    Ok(receipt)
}

pub fn new_external_enrollment_witness_candidate_bundle(
    policy: &FrozenPerceptualEnrollmentWitnessPolicyV1,
) -> Result<FrozenPerceptualEnrollmentWitnessBundleV1, serde_json::Error> {
    let mut bundle = FrozenPerceptualEnrollmentWitnessBundleV1 {
        bundle_version: PERCEPTUAL_ENROLLMENT_WITNESS_BUNDLE_VERSION.into(),
        witness_policy_sha256: policy.policy_sha256.clone(),
        entries: Vec::new(),
        final_witness_head_sha256: ZERO_SHA256.into(),
        bundle_sha256: String::new(),
    };
    bundle.bundle_sha256 = enrollment_witness_bundle_commitment(&bundle)?;
    Ok(bundle)
}

/// Append already assembled provider evidence to a *candidate* bundle.
/// This checks chain/identity structure only. The result becomes authority only
/// after the full P1EILR-B validator and P1EILR-D durable reconstruction.
pub fn append_external_enrollment_witness_candidate(
    prior: &FrozenPerceptualEnrollmentWitnessBundleV1,
    receipt: WitnessedEnrollmentAllocationReceiptV1,
    policy: &FrozenPerceptualEnrollmentWitnessPolicyV1,
) -> Result<FrozenPerceptualEnrollmentWitnessBundleV1, Vec<PerceptualEnrollmentWitnessProviderIssueV1>> {
    let mut issues = validate_candidate_bundle_structure(prior, policy);
    let expected_sequence = prior.entries.len() as u32;
    if receipt.sequence != expected_sequence {
        issues.push(PerceptualEnrollmentWitnessProviderIssueV1::CandidateSequenceMismatch {
            found: receipt.sequence,
            expected: expected_sequence,
        });
    }
    if receipt.witness_policy_sha256 != policy.policy_sha256 {
        issues.push(PerceptualEnrollmentWitnessProviderIssueV1::PreparedFieldMismatch {
            field: "receipt.witness_policy_sha256".into(),
        });
    }
    if receipt.previous_witness_head_sha256 != prior.final_witness_head_sha256 {
        issues.push(PerceptualEnrollmentWitnessProviderIssueV1::CandidatePreviousHeadMismatch);
    }
    if prior
        .entries
        .iter()
        .any(|entry| entry.witness_anchor_reference == receipt.witness_anchor_reference)
    {
        issues.push(PerceptualEnrollmentWitnessProviderIssueV1::DuplicateWitnessAnchor {
            anchor: receipt.witness_anchor_reference.clone(),
        });
    }
    match enrollment_witness_receipt_commitment(&receipt) {
        Ok(value) if value == receipt.receipt_sha256 => {}
        _ => issues.push(PerceptualEnrollmentWitnessProviderIssueV1::CandidateBundleMalformed),
    }
    if !issues.is_empty() {
        return Err(issues);
    }
    let final_witness_head_sha256 = receipt.witness_head_sha256.clone();
    let mut next = prior.clone();
    next.entries.push(receipt);
    next.final_witness_head_sha256 = final_witness_head_sha256;
    next.bundle_sha256 = enrollment_witness_bundle_commitment(&next).map_err(|_| {
        vec![PerceptualEnrollmentWitnessProviderIssueV1::SerializationFailed]
    })?;
    Ok(next)
}

fn validate_pending_request(
    pending: &PendingEnrollmentWitnessRequestV1,
) -> Vec<PerceptualEnrollmentWitnessProviderIssueV1> {
    let mut issues = Vec::new();
    if pending.witness_log_id.trim().is_empty() {
        issues.push(PerceptualEnrollmentWitnessProviderIssueV1::EmptyWitnessLogId);
    }
    for (field, digest) in [
        ("request_sha256", pending.request_sha256.as_str()),
        (
            "enrollment_witness_policy_sha256",
            pending.enrollment_witness_policy_sha256.as_str(),
        ),
        ("enrollment_ledger_sha256", pending.enrollment_ledger_sha256.as_str()),
        ("durable_state_sha256", pending.durable_state_sha256.as_str()),
        ("allocation_head_sha256", pending.allocation_head_sha256.as_str()),
        (
            "enrollment_participant_commitment_sha256",
            pending.enrollment_participant_commitment_sha256.as_str(),
        ),
        (
            "participant_schedule_projection_sha256",
            pending.participant_schedule_projection_sha256.as_str(),
        ),
        (
            "previous_witness_head_sha256",
            pending.previous_witness_head_sha256.as_str(),
        ),
    ] {
        if decode_hex_32(digest).is_none() {
            issues.push(PerceptualEnrollmentWitnessProviderIssueV1::InvalidDigest {
                field: field.into(),
            });
        }
    }
    match pending_witness_request_commitment(pending) {
        Ok(value) if value == pending.request_sha256 => {}
        _ => issues.push(PerceptualEnrollmentWitnessProviderIssueV1::PendingRequestDigestMismatch),
    }
    issues
}

fn validate_prepared_against_policy(
    prepared: &PreparedExternalEnrollmentWitnessV1,
    policy: &FrozenPerceptualEnrollmentWitnessPolicyV1,
) -> Vec<PerceptualEnrollmentWitnessProviderIssueV1> {
    let mut issues = validate_prepared_external_enrollment_witness(prepared);
    if prepared.witness_policy_sha256 != policy.policy_sha256 {
        issues.push(PerceptualEnrollmentWitnessProviderIssueV1::PreparedFieldMismatch {
            field: "witness_policy_sha256".into(),
        });
    }
    if prepared.witness_log_id != policy.witness_log_id {
        issues.push(PerceptualEnrollmentWitnessProviderIssueV1::PreparedFieldMismatch {
            field: "witness_log_id".into(),
        });
    }
    issues
}

fn validate_collection_signing_request(
    request: &ExternalEnrollmentCollectionSigningRequestV1,
    prepared: &PreparedExternalEnrollmentWitnessV1,
    policy: &FrozenPerceptualEnrollmentWitnessPolicyV1,
) -> Vec<PerceptualEnrollmentWitnessProviderIssueV1> {
    let mut issues = validate_prepared_against_policy(prepared, policy);
    if request.protocol_version != ENROLLMENT_WITNESS_PROVIDER_PROTOCOL_VERSION {
        issues.push(PerceptualEnrollmentWitnessProviderIssueV1::WrongProtocolVersion);
    }
    for (field, matches) in [
        ("request_sha256", request.request_sha256 == prepared.request_sha256),
        (
            "witness_policy_sha256",
            request.witness_policy_sha256 == prepared.witness_policy_sha256,
        ),
        ("sequence", request.sequence == prepared.sequence),
        (
            "witness_head_sha256",
            request.witness_head_sha256 == prepared.witness_head_sha256,
        ),
    ] {
        if !matches {
            issues.push(PerceptualEnrollmentWitnessProviderIssueV1::CollectionRequestMismatch {
                field: field.into(),
            });
        }
    }
    if request.signer_id != policy.collection_signer.signer_id
        || request.key_epoch != policy.collection_signer.key_epoch
    {
        issues.push(PerceptualEnrollmentWitnessProviderIssueV1::CollectionSignerIdentityMismatch);
    }
    match collection_signature_transcript(prepared) {
        Ok(expected) if expected == request.signing_transcript => {}
        _ => issues.push(PerceptualEnrollmentWitnessProviderIssueV1::CollectionRequestMismatch {
            field: "signing_transcript".into(),
        }),
    }
    match collection_signing_request_commitment(request) {
        Ok(value) if value == request.collection_request_sha256 => {}
        _ => issues.push(PerceptualEnrollmentWitnessProviderIssueV1::CollectionRequestDigestMismatch),
    }
    issues
}

fn validate_collection_signature_response(
    response: &ExternalEnrollmentCollectionSignatureV1,
    request: &ExternalEnrollmentCollectionSigningRequestV1,
    prepared: &PreparedExternalEnrollmentWitnessV1,
    policy: &FrozenPerceptualEnrollmentWitnessPolicyV1,
) -> Vec<PerceptualEnrollmentWitnessProviderIssueV1> {
    let mut issues = Vec::new();
    if response.protocol_version != ENROLLMENT_WITNESS_PROVIDER_PROTOCOL_VERSION {
        issues.push(PerceptualEnrollmentWitnessProviderIssueV1::WrongProtocolVersion);
    }
    if response.request_sha256 != prepared.request_sha256
        || response.witness_head_sha256 != prepared.witness_head_sha256
        || response.collection_request_sha256 != request.collection_request_sha256
    {
        issues.push(PerceptualEnrollmentWitnessProviderIssueV1::CollectionRequestMismatch {
            field: "collection_signature_response_binding".into(),
        });
    }
    if response.signer_id != policy.collection_signer.signer_id
        || response.key_epoch != policy.collection_signer.key_epoch
    {
        issues.push(PerceptualEnrollmentWitnessProviderIssueV1::CollectionSignerIdentityMismatch);
    }
    match verify_signature(
        &policy.collection_signer,
        &request.signing_transcript,
        &response.signature,
    ) {
        Ok(()) => {}
        Err(_) => issues.push(PerceptualEnrollmentWitnessProviderIssueV1::CollectionSignatureInvalid),
    }
    issues
}

fn validate_witness_service_request(
    request: &ExternalEnrollmentWitnessServiceRequestV1,
    prepared: &PreparedExternalEnrollmentWitnessV1,
    collection_request: &ExternalEnrollmentCollectionSigningRequestV1,
    collection_response: &ExternalEnrollmentCollectionSignatureV1,
    policy: &FrozenPerceptualEnrollmentWitnessPolicyV1,
) -> Vec<PerceptualEnrollmentWitnessProviderIssueV1> {
    let mut issues = Vec::new();
    if request.protocol_version != ENROLLMENT_WITNESS_PROVIDER_PROTOCOL_VERSION {
        issues.push(PerceptualEnrollmentWitnessProviderIssueV1::WrongProtocolVersion);
    }
    for (field, matches) in [
        ("request_sha256", request.request_sha256 == prepared.request_sha256),
        (
            "witness_policy_sha256",
            request.witness_policy_sha256 == prepared.witness_policy_sha256,
        ),
        ("witness_log_id", request.witness_log_id == prepared.witness_log_id),
        ("sequence", request.sequence == prepared.sequence),
        (
            "witness_head_sha256",
            request.witness_head_sha256 == prepared.witness_head_sha256,
        ),
        (
            "evidence_signer_id",
            request.evidence_signer_id == collection_response.signer_id,
        ),
        (
            "evidence_key_epoch",
            request.evidence_key_epoch == collection_response.key_epoch,
        ),
        (
            "evidence_signature",
            request.evidence_signature == collection_response.signature,
        ),
        (
            "collection_request_sha256",
            request.collection_request_sha256 == collection_request.collection_request_sha256,
        ),
    ] {
        if !matches {
            issues.push(PerceptualEnrollmentWitnessProviderIssueV1::WitnessServiceRequestMismatch {
                field: field.into(),
            });
        }
    }
    if request.evidence_signer_id != policy.collection_signer.signer_id
        || request.evidence_key_epoch != policy.collection_signer.key_epoch
    {
        issues.push(PerceptualEnrollmentWitnessProviderIssueV1::CollectionSignerIdentityMismatch);
    }
    match witness_service_request_commitment(request) {
        Ok(value) if value == request.witness_service_request_sha256 => {}
        _ => issues.push(
            PerceptualEnrollmentWitnessProviderIssueV1::WitnessServiceRequestDigestMismatch,
        ),
    }
    issues
}

fn validate_witness_service_response(
    response: &ExternalEnrollmentWitnessServiceResponseV1,
    request: &ExternalEnrollmentWitnessServiceRequestV1,
    prepared: &PreparedExternalEnrollmentWitnessV1,
    policy: &FrozenPerceptualEnrollmentWitnessPolicyV1,
) -> Vec<PerceptualEnrollmentWitnessProviderIssueV1> {
    let mut issues = Vec::new();
    if response.protocol_version != ENROLLMENT_WITNESS_PROVIDER_PROTOCOL_VERSION {
        issues.push(PerceptualEnrollmentWitnessProviderIssueV1::WrongProtocolVersion);
    }
    if response.request_sha256 != prepared.request_sha256
        || response.witness_head_sha256 != prepared.witness_head_sha256
        || response.witness_service_request_sha256 != request.witness_service_request_sha256
    {
        issues.push(PerceptualEnrollmentWitnessProviderIssueV1::WitnessServiceRequestMismatch {
            field: "witness_service_response_binding".into(),
        });
    }
    if response.witness_anchor_reference.trim().is_empty() {
        issues.push(PerceptualEnrollmentWitnessProviderIssueV1::EmptyWitnessAnchor);
    }
    if response.witness_signer_id != policy.witness_signer.signer_id
        || response.witness_key_epoch != policy.witness_signer.key_epoch
    {
        issues.push(PerceptualEnrollmentWitnessProviderIssueV1::WitnessSignerIdentityMismatch);
    }
    match external_witness_signature_transcript(prepared, &response.witness_anchor_reference) {
        Ok(transcript) => match verify_signature(
            &policy.witness_signer,
            &transcript,
            &response.witness_signature,
        ) {
            Ok(()) => {}
            Err(_) => issues.push(PerceptualEnrollmentWitnessProviderIssueV1::WitnessSignatureInvalid),
        },
        Err(mut found) => issues.append(&mut found),
    }
    issues
}

fn validate_candidate_bundle_structure(
    bundle: &FrozenPerceptualEnrollmentWitnessBundleV1,
    policy: &FrozenPerceptualEnrollmentWitnessPolicyV1,
) -> Vec<PerceptualEnrollmentWitnessProviderIssueV1> {
    let mut issues = Vec::new();
    if bundle.bundle_version != PERCEPTUAL_ENROLLMENT_WITNESS_BUNDLE_VERSION
        || bundle.witness_policy_sha256 != policy.policy_sha256
    {
        issues.push(PerceptualEnrollmentWitnessProviderIssueV1::CandidateBundleMalformed);
    }
    let expected_final = bundle
        .entries
        .last()
        .map(|entry| entry.witness_head_sha256.as_str())
        .unwrap_or(ZERO_SHA256);
    if bundle.final_witness_head_sha256 != expected_final {
        issues.push(PerceptualEnrollmentWitnessProviderIssueV1::CandidateBundleMalformed);
    }
    let mut anchors = BTreeSet::new();
    for entry in &bundle.entries {
        if entry.witness_anchor_reference.trim().is_empty()
            || !anchors.insert(entry.witness_anchor_reference.as_str())
        {
            issues.push(PerceptualEnrollmentWitnessProviderIssueV1::CandidateBundleMalformed);
        }
    }
    match enrollment_witness_bundle_commitment(bundle) {
        Ok(value) if value == bundle.bundle_sha256 => {}
        _ => issues.push(PerceptualEnrollmentWitnessProviderIssueV1::CandidateBundleMalformed),
    }
    issues
}

fn verify_signature(
    identity: &CollectionVerifierIdentityV1,
    transcript: &[u8],
    signature_bytes: &[u8],
) -> Result<(), PerceptualEnrollmentWitnessProviderIssueV1> {
    if identity.verifying_key_bytes.len() != ED25519_PUBLIC_KEY_BYTES
        || identity.verifying_key_bytes.iter().all(|byte| *byte == 0)
    {
        return Err(PerceptualEnrollmentWitnessProviderIssueV1::InvalidVerifyingKey);
    }
    let key_bytes: [u8; ED25519_PUBLIC_KEY_BYTES] = identity
        .verifying_key_bytes
        .as_slice()
        .try_into()
        .map_err(|_| PerceptualEnrollmentWitnessProviderIssueV1::InvalidVerifyingKey)?;
    let key = VerifyingKey::from_bytes(&key_bytes)
        .map_err(|_| PerceptualEnrollmentWitnessProviderIssueV1::InvalidVerifyingKey)?;
    let signature_bytes: [u8; ED25519_SIGNATURE_BYTES] = signature_bytes
        .try_into()
        .map_err(|_| PerceptualEnrollmentWitnessProviderIssueV1::CollectionSignatureInvalid)?;
    let signature = Signature::from_bytes(&signature_bytes);
    key.verify(transcript, &signature)
        .map_err(|_| PerceptualEnrollmentWitnessProviderIssueV1::CollectionSignatureInvalid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn pending(marker: char) -> PendingEnrollmentWitnessRequestV1 {
        let mut value = PendingEnrollmentWitnessRequestV1 {
            sequence: 2,
            enrollment_witness_policy_sha256: marker.to_string().repeat(64),
            witness_log_id: "external-enrollment-witness-log".into(),
            enrollment_ledger_sha256: "b".repeat(64),
            durable_state_sha256: "c".repeat(64),
            allocation_head_sha256: "d".repeat(64),
            enrollment_participant_commitment_sha256: "e".repeat(64),
            participant_schedule_projection_sha256: "f".repeat(64),
            previous_witness_head_sha256: "1".repeat(64),
            request_sha256: String::new(),
        };
        value.request_sha256 = pending_witness_request_commitment(&value).unwrap();
        value
    }

    fn policy(pending: &PendingEnrollmentWitnessRequestV1) -> (
        FrozenPerceptualEnrollmentWitnessPolicyV1,
        SigningKey,
        SigningKey,
    ) {
        let collection = SigningKey::from_bytes(&[7u8; 32]);
        let witness = SigningKey::from_bytes(&[9u8; 32]);
        (
            FrozenPerceptualEnrollmentWitnessPolicyV1 {
                policy_version: "test-policy".into(),
                enrollment_policy_sha256: "2".repeat(64),
                collection_authenticity_policy_sha256: "3".repeat(64),
                witness_log_id: pending.witness_log_id.clone(),
                collection_signer: CollectionVerifierIdentityV1 {
                    signer_id: "collection".into(),
                    key_epoch: 1,
                    verifying_key_bytes: collection.verifying_key().to_bytes().to_vec(),
                },
                witness_signer: CollectionVerifierIdentityV1 {
                    signer_id: "witness".into(),
                    key_epoch: 1,
                    verifying_key_bytes: witness.verifying_key().to_bytes().to_vec(),
                },
                per_allocation_external_witness_required: true,
                scored_authority_requires_verified_witness: true,
                raw_participant_token_prohibited: true,
                policy_sha256: pending.enrollment_witness_policy_sha256.clone(),
            },
            collection,
            witness,
        )
    }

    #[test]
    fn prepared_head_changes_when_load_bearing_input_changes() {
        let left = prepare_external_enrollment_witness(&pending('a')).unwrap();
        let mut changed = pending('a');
        changed.durable_state_sha256 = "8".repeat(64);
        changed.request_sha256 = pending_witness_request_commitment(&changed).unwrap();
        let right = prepare_external_enrollment_witness(&changed).unwrap();
        assert_ne!(left.witness_head_sha256, right.witness_head_sha256);
        assert_ne!(left.request_sha256, right.request_sha256);
    }

    #[test]
    fn malformed_pending_request_digest_fails_closed() {
        let mut value = pending('a');
        value.request_sha256 = "9".repeat(64);
        assert!(prepare_external_enrollment_witness(&value).is_err());
    }

    #[test]
    fn provider_wire_dtos_exclude_raw_participant_and_arm_semantics() {
        let prepared = prepare_external_enrollment_witness(&pending('a')).unwrap();
        let encoded = serde_json::to_string(&prepared).unwrap();
        assert!(!encoded.contains("\"participant_token\""));
        assert!(!encoded.contains("arm"));
        assert!(!encoded.contains("baseline"));
        assert!(!encoded.contains("intervention"));
    }

    #[test]
    fn collection_and_witness_domains_are_not_replayable() {
        let prepared = prepare_external_enrollment_witness(&pending('a')).unwrap();
        let collection = collection_signature_transcript(&prepared).unwrap();
        let witness = external_witness_signature_transcript(&prepared, "anchor-1").unwrap();
        assert_ne!(collection, witness);
    }

    #[test]
    fn valid_provider_round_trip_assembles_candidate_receipt() {
        let pending = pending('a');
        let prepared = prepare_external_enrollment_witness(&pending).unwrap();
        let (policy, collection_key, witness_key) = policy(&pending);
        let collection_request = build_collection_signing_request(&prepared, &policy).unwrap();
        let collection_response = ExternalEnrollmentCollectionSignatureV1 {
            protocol_version: ENROLLMENT_WITNESS_PROVIDER_PROTOCOL_VERSION.into(),
            request_sha256: prepared.request_sha256.clone(),
            witness_head_sha256: prepared.witness_head_sha256.clone(),
            collection_request_sha256: collection_request.collection_request_sha256.clone(),
            signer_id: policy.collection_signer.signer_id.clone(),
            key_epoch: policy.collection_signer.key_epoch,
            signature: collection_key
                .sign(&collection_request.signing_transcript)
                .to_bytes()
                .to_vec(),
        };
        let witness_request = build_witness_service_request(
            &prepared,
            &collection_request,
            &collection_response,
            &policy,
        )
        .unwrap();
        let anchor = "external-anchor-0001";
        let witness_transcript = external_witness_signature_transcript(&prepared, anchor).unwrap();
        let witness_response = ExternalEnrollmentWitnessServiceResponseV1 {
            protocol_version: ENROLLMENT_WITNESS_PROVIDER_PROTOCOL_VERSION.into(),
            request_sha256: prepared.request_sha256.clone(),
            witness_head_sha256: prepared.witness_head_sha256.clone(),
            witness_service_request_sha256: witness_request.witness_service_request_sha256.clone(),
            witness_anchor_reference: anchor.into(),
            witness_signer_id: policy.witness_signer.signer_id.clone(),
            witness_key_epoch: policy.witness_signer.key_epoch,
            witness_signature: witness_key.sign(&witness_transcript).to_bytes().to_vec(),
        };
        let receipt = assemble_external_enrollment_witness_receipt(
            &prepared,
            &collection_request,
            &collection_response,
            &witness_request,
            &witness_response,
            &policy,
        )
        .unwrap();
        assert_eq!(receipt.sequence, pending.sequence);
        assert_eq!(receipt.witness_head_sha256, prepared.witness_head_sha256);
        assert_eq!(receipt.witness_anchor_reference, anchor);
    }

    #[test]
    fn wrong_collection_signer_identity_is_rejected_before_receipt_assembly() {
        let pending = pending('a');
        let prepared = prepare_external_enrollment_witness(&pending).unwrap();
        let (policy, _, _) = policy(&pending);
        let collection_request = build_collection_signing_request(&prepared, &policy).unwrap();
        let response = ExternalEnrollmentCollectionSignatureV1 {
            protocol_version: ENROLLMENT_WITNESS_PROVIDER_PROTOCOL_VERSION.into(),
            request_sha256: prepared.request_sha256.clone(),
            witness_head_sha256: prepared.witness_head_sha256.clone(),
            collection_request_sha256: collection_request.collection_request_sha256.clone(),
            signer_id: "wrong-signer".into(),
            key_epoch: 1,
            signature: vec![0; 64],
        };
        let issues = validate_collection_signature_response(
            &response,
            &collection_request,
            &prepared,
            &policy,
        );
        assert!(issues.contains(
            &PerceptualEnrollmentWitnessProviderIssueV1::CollectionSignerIdentityMismatch
        ));
    }
}
