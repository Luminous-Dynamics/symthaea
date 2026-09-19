// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1EILR-B: independently witness each durable enrollment allocation.
//!
//! This module joins the owner-local durable P1ENR allocator with the frozen
//! P1EIR collection/witness identities without merging enrollment events into
//! the scored-response authenticity log. The two logs have different authority
//! semantics, but the same exact collection and independent-witness identities
//! authenticate both.
//!
//! The persistent witness receipt contains no raw participant token. P1ENR's
//! allocation-reference commitment remains the enrollment identity. The
//! collection-domain participant commitment is derived only inside the opaque
//! verified runtime result so P1EACR can bind scored authority without turning
//! the cross-namespace linkage into another public artifact.

use crate::evidence_digest::{
    canonical_json_bytes, canonical_json_sha256, decode_hex_32,
    perceptual_collection_authenticity::{
        participant_token_commitment, CollectionVerifierIdentityV1,
        FrozenPerceptualCollectionAuthenticityPolicyV1,
    },
    perceptual_enrollment_lifecycle::{
        enrollment_allocation_ledger_commitment, participant_allocation_reference_commitment,
        FrozenPerceptualEnrollmentAllocationLedgerV1, FrozenPerceptualEnrollmentPolicyV1,
    },
    perceptual_enrollment_store::{
        DurableEnrollmentAllocationStoreV1, UnwitnessedDurableEnrollmentAllocationV1,
    },
    perceptual_participant_identity::{
        project_participant_schedule, FrozenParticipantIdentityBoundaryPolicyV1,
        FrozenParticipantScheduleProjectionV1,
        FrozenPerceptualParticipantTokenGenerationReceiptV1,
    },
    perceptual_participant_schedule::{
        PerceptualCohortSlotsV1, PerceptualParticipantScheduleBookV1,
    },
    perceptual_stimulus_pack::{
        FrozenC6fRenderSubjectBindingV1, FrozenPerceptualStimulusPackV1,
    },
    perceptual_study_protocol::FrozenPerceptualStudyProtocolV1,
};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const PERCEPTUAL_ENROLLMENT_WITNESS_POLICY_VERSION: &str =
    "mel003-perceptual-enrollment-witness-policy-v1";
pub const PERCEPTUAL_ENROLLMENT_WITNESS_RECEIPT_VERSION: &str =
    "mel003-perceptual-enrollment-witness-receipt-v1";
pub const PERCEPTUAL_ENROLLMENT_WITNESS_BUNDLE_VERSION: &str =
    "mel003-perceptual-enrollment-witness-bundle-v1";
pub const ZERO_SHA256: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

const COLLECTION_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.mel003.p1.enrollment-witness.v1/collection-evidence";
const EXTERNAL_WITNESS_DOMAIN: &[u8] =
    b"symthaea.mel003.p1.enrollment-witness.v1/external-witness";
const MAX_SIGNED_MESSAGE_BYTES: usize = 1 << 20;
const MAX_SIGNATURE_DOMAIN_BYTES: usize = 256;
const ED25519_PUBLIC_KEY_BYTES: usize = 32;
const ED25519_SIGNATURE_BYTES: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPerceptualEnrollmentWitnessPolicyV1 {
    pub policy_version: String,
    pub enrollment_policy_sha256: String,
    pub collection_authenticity_policy_sha256: String,
    pub witness_log_id: String,
    pub collection_signer: CollectionVerifierIdentityV1,
    pub witness_signer: CollectionVerifierIdentityV1,
    pub per_allocation_external_witness_required: bool,
    pub scored_authority_requires_verified_witness: bool,
    pub raw_participant_token_prohibited: bool,
    pub policy_sha256: String,
}

/// Signing-side helper. Secret seed material is runtime-only and never enters
/// serialized study evidence.
pub struct EnrollmentWitnessSigningKeyV1 {
    signer_id: String,
    key_epoch: u64,
    inner: SigningKey,
}

impl EnrollmentWitnessSigningKeyV1 {
    pub fn from_seed(
        signer_id: impl Into<String>,
        key_epoch: u64,
        seed: [u8; 32],
    ) -> Result<Self, PerceptualEnrollmentWitnessIssueV1> {
        let signer_id = signer_id.into();
        if signer_id.trim().is_empty() {
            return Err(PerceptualEnrollmentWitnessIssueV1::EmptySignerId);
        }
        if key_epoch == 0 {
            return Err(PerceptualEnrollmentWitnessIssueV1::InvalidKeyEpoch);
        }
        if seed == [0u8; 32] {
            return Err(PerceptualEnrollmentWitnessIssueV1::InvalidSigningSeed);
        }
        Ok(Self {
            signer_id,
            key_epoch,
            inner: SigningKey::from_bytes(&seed),
        })
    }

    pub fn verifier_identity(&self) -> CollectionVerifierIdentityV1 {
        CollectionVerifierIdentityV1 {
            signer_id: self.signer_id.clone(),
            key_epoch: self.key_epoch,
            verifying_key_bytes: self.inner.verifying_key().to_bytes().to_vec(),
        }
    }

    fn sign_domain(
        &self,
        domain: &[u8],
        message: &[u8],
    ) -> Result<Vec<u8>, PerceptualEnrollmentWitnessIssueV1> {
        validate_domain_message(domain, message)?;
        Ok(self
            .inner
            .sign(&domain_separated_transcript(domain, message))
            .to_bytes()
            .to_vec())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WitnessedEnrollmentAllocationReceiptV1 {
    pub receipt_version: String,
    pub witness_policy_sha256: String,
    pub sequence: u32,
    /// Exact P1ENR ledger identity immediately after this allocation.
    pub enrollment_ledger_sha256: String,
    /// Exact owner-local durable-state identity observed before signing. This is
    /// correlation evidence, not an anti-rollback root by itself.
    pub durable_state_sha256: String,
    pub allocation_head_sha256: String,
    /// P1ENR-domain participant allocation reference. This is deliberately not
    /// the P1EIR collection-domain participant commitment.
    pub enrollment_participant_commitment_sha256: String,
    pub participant_schedule_projection_sha256: String,
    pub previous_witness_head_sha256: String,
    pub witness_head_sha256: String,
    pub evidence_signer_id: String,
    pub evidence_key_epoch: u64,
    pub evidence_signature: Vec<u8>,
    pub witness_anchor_reference: String,
    pub witness_signer_id: String,
    pub witness_key_epoch: u64,
    pub witness_signature: Vec<u8>,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPerceptualEnrollmentWitnessBundleV1 {
    pub bundle_version: String,
    pub witness_policy_sha256: String,
    /// Append order is evidence. Do not sort.
    pub entries: Vec<WitnessedEnrollmentAllocationReceiptV1>,
    pub final_witness_head_sha256: String,
    pub bundle_sha256: String,
}

/// Runtime-only proof that one exact durable allocation crossed both frozen
/// P1EIR signing identities. This type is not serializable and has no public
/// constructor. P1EACR should consume this type rather than raw P1ENR state.
#[derive(Debug)]
pub struct VerifiedWitnessedEnrollmentAllocationV1 {
    receipt: WitnessedEnrollmentAllocationReceiptV1,
    participant_schedule_projection: FrozenParticipantScheduleProjectionV1,
    collection_participant_commitment_sha256: String,
}

impl VerifiedWitnessedEnrollmentAllocationV1 {
    pub fn receipt(&self) -> &WitnessedEnrollmentAllocationReceiptV1 {
        &self.receipt
    }

    pub fn participant_schedule_projection(&self) -> &FrozenParticipantScheduleProjectionV1 {
        &self.participant_schedule_projection
    }

    pub fn collection_participant_commitment_sha256(&self) -> &str {
        &self.collection_participant_commitment_sha256
    }

    pub fn current_enrollment_ledger_sha256(&self) -> &str {
        &self.receipt.enrollment_ledger_sha256
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualEnrollmentWitnessIssueV1 {
    WrongPolicyVersion,
    EnrollmentPolicyDigestMismatch,
    CollectionAuthenticityPolicyDigestMismatch,
    SignerIdentityMismatch { role: String },
    SignerWitnessNotDistinct,
    EmptySignerId,
    InvalidKeyEpoch,
    InvalidSigningSeed,
    InvalidVerifyingKey { signer_id: String },
    EmptyWitnessLogId,
    MissingPolicyProtection { field: String },
    InvalidPolicyDigest,
    PolicyDigestMismatch,
    DurableStateUnavailable,
    DurableStateDigestMismatch,
    DurableLedgerDigestMismatch,
    AllocationNotCurrentTail,
    AllocationSequenceMismatch { found: u32, expected: u32 },
    AllocationReceiptMismatch,
    EnrollmentParticipantCommitmentMismatch,
    ProjectionDigestMismatch,
    ProjectionMismatch,
    CollectionParticipantCommitmentFailed,
    EmptyWitnessAnchor,
    DuplicateWitnessAnchor { anchor: String },
    WrongReceiptVersion { index: usize },
    ReceiptPolicyMismatch { index: usize },
    SequenceMismatch { index: usize },
    PreviousWitnessHeadMismatch { index: usize },
    WitnessHeadMismatch { index: usize },
    LedgerPrefixDigestMismatch { index: usize },
    AllocationHeadMismatch { index: usize },
    ParticipantCommitmentMismatch { index: usize },
    ParticipantProjectionMismatch { index: usize },
    ReceiptSignerMismatch { index: usize, role: String },
    EvidenceSignatureInvalid { index: usize },
    WitnessSignatureInvalid { index: usize },
    InvalidArtifactDigest { index: usize, field: String },
    ReceiptDigestMismatch { index: usize },
    WrongBundleVersion,
    BundlePolicyMismatch,
    WrongEntryCount { found: usize, expected: usize },
    FinalWitnessHeadMismatch,
    BundleDigestMismatch,
    SerializationFailed,
    SignatureMessageTooLarge,
    InvalidSignatureDomain,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct EnrollmentWitnessUnsignedEntryV1<'a> {
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
struct EnrollmentWitnessStatementV1<'a> {
    witness_log_id: &'a str,
    sequence: u32,
    witness_head_sha256: &'a str,
    witness_anchor_reference: &'a str,
}

pub struct PerceptualEnrollmentWitnessLogBuilderV1 {
    policy: FrozenPerceptualEnrollmentWitnessPolicyV1,
    entries: Vec<WitnessedEnrollmentAllocationReceiptV1>,
}

impl PerceptualEnrollmentWitnessLogBuilderV1 {
    pub fn begin(
        policy: FrozenPerceptualEnrollmentWitnessPolicyV1,
        enrollment_policy: &FrozenPerceptualEnrollmentPolicyV1,
        authenticity_policy: &FrozenPerceptualCollectionAuthenticityPolicyV1,
    ) -> Result<Self, Vec<PerceptualEnrollmentWitnessIssueV1>> {
        let issues = validate_enrollment_witness_policy(
            &policy,
            enrollment_policy,
            authenticity_policy,
        );
        if !issues.is_empty() {
            return Err(issues);
        }
        Ok(Self {
            policy,
            entries: Vec::new(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn append_durable_allocation(
        &mut self,
        protocol: &FrozenPerceptualStudyProtocolV1,
        stimulus_pack: &FrozenPerceptualStimulusPackV1,
        render_binding: &FrozenC6fRenderSubjectBindingV1,
        cohort: &PerceptualCohortSlotsV1,
        token_receipt: &FrozenPerceptualParticipantTokenGenerationReceiptV1,
        identity_policy: &FrozenParticipantIdentityBoundaryPolicyV1,
        schedule: &PerceptualParticipantScheduleBookV1,
        enrollment_policy: &FrozenPerceptualEnrollmentPolicyV1,
        authenticity_policy: &FrozenPerceptualCollectionAuthenticityPolicyV1,
        store: &DurableEnrollmentAllocationStoreV1,
        unwitnessed: UnwitnessedDurableEnrollmentAllocationV1,
        collection_signer: &EnrollmentWitnessSigningKeyV1,
        witness_signer: &EnrollmentWitnessSigningKeyV1,
        witness_anchor_reference: &str,
    ) -> Result<VerifiedWitnessedEnrollmentAllocationV1, Vec<PerceptualEnrollmentWitnessIssueV1>> {
        let mut issues = validate_enrollment_witness_policy(
            &self.policy,
            enrollment_policy,
            authenticity_policy,
        );
        validate_signing_key_matches(
            "collection_signer",
            collection_signer,
            &self.policy.collection_signer,
            &mut issues,
        );
        validate_signing_key_matches(
            "witness_signer",
            witness_signer,
            &self.policy.witness_signer,
            &mut issues,
        );
        if witness_anchor_reference.trim().is_empty() {
            issues.push(PerceptualEnrollmentWitnessIssueV1::EmptyWitnessAnchor);
        }
        if self
            .entries
            .iter()
            .any(|entry| entry.witness_anchor_reference == witness_anchor_reference)
        {
            issues.push(PerceptualEnrollmentWitnessIssueV1::DuplicateWitnessAnchor {
                anchor: witness_anchor_reference.into(),
            });
        }

        let state = match store.inspect_current(
            protocol,
            stimulus_pack,
            render_binding,
            cohort,
            token_receipt,
            identity_policy,
            schedule,
            enrollment_policy,
            &unwitnessed.current_ledger_sha256,
        ) {
            Ok(state) => Some(state),
            Err(_) => {
                issues.push(PerceptualEnrollmentWitnessIssueV1::DurableStateUnavailable);
                None
            }
        };

        if let Some(state) = state.as_ref() {
            if state.state_sha256 != unwitnessed.current_state_sha256 {
                issues.push(PerceptualEnrollmentWitnessIssueV1::DurableStateDigestMismatch);
            }
            if state.ledger.ledger_sha256 != unwitnessed.current_ledger_sha256 {
                issues.push(PerceptualEnrollmentWitnessIssueV1::DurableLedgerDigestMismatch);
            }
            match state.ledger.allocations.last() {
                Some(allocation) if allocation == &unwitnessed.allocation => {}
                _ => issues.push(PerceptualEnrollmentWitnessIssueV1::AllocationNotCurrentTail),
            }
        }

        let expected_sequence = self.entries.len() as u32;
        if unwitnessed.allocation.sequence != expected_sequence {
            issues.push(PerceptualEnrollmentWitnessIssueV1::AllocationSequenceMismatch {
                found: unwitnessed.allocation.sequence,
                expected: expected_sequence,
            });
        }
        if unwitnessed.allocation.enrollment_policy_sha256 != enrollment_policy.policy_sha256 {
            issues.push(PerceptualEnrollmentWitnessIssueV1::AllocationReceiptMismatch);
        }
        if unwitnessed.allocation.participant_schedule_projection_sha256
            != unwitnessed.participant_schedule_projection.projection_sha256
        {
            issues.push(PerceptualEnrollmentWitnessIssueV1::ProjectionDigestMismatch);
        }
        let participant_token = &unwitnessed
            .participant_schedule_projection
            .schedule
            .participant_token;
        match participant_allocation_reference_commitment(participant_token) {
            Ok(commitment)
                if commitment == unwitnessed.allocation.participant_token_commitment_sha256 => {}
            _ => issues.push(
                PerceptualEnrollmentWitnessIssueV1::EnrollmentParticipantCommitmentMismatch,
            ),
        }
        let expected_projection = project_participant_schedule(
            protocol,
            stimulus_pack,
            render_binding,
            cohort,
            token_receipt,
            identity_policy,
            schedule,
            participant_token,
        );
        match expected_projection {
            Ok(projection) if projection == unwitnessed.participant_schedule_projection => {}
            _ => issues.push(PerceptualEnrollmentWitnessIssueV1::ProjectionMismatch),
        }
        let collection_participant_commitment_sha256 =
            match participant_token_commitment(participant_token) {
                Ok(value) => Some(value),
                Err(_) => {
                    issues.push(
                        PerceptualEnrollmentWitnessIssueV1::CollectionParticipantCommitmentFailed,
                    );
                    None
                }
            };

        if !issues.is_empty() {
            return Err(issues);
        }

        let previous_witness_head_sha256 = self
            .entries
            .last()
            .map(|entry| entry.witness_head_sha256.clone())
            .unwrap_or_else(|| ZERO_SHA256.into());
        let unsigned = EnrollmentWitnessUnsignedEntryV1 {
            witness_policy_sha256: &self.policy.policy_sha256,
            sequence: expected_sequence,
            enrollment_ledger_sha256: &unwitnessed.current_ledger_sha256,
            durable_state_sha256: &unwitnessed.current_state_sha256,
            allocation_head_sha256: &unwitnessed.allocation.allocation_head_sha256,
            enrollment_participant_commitment_sha256: &unwitnessed
                .allocation
                .participant_token_commitment_sha256,
            participant_schedule_projection_sha256: &unwitnessed
                .allocation
                .participant_schedule_projection_sha256,
            previous_witness_head_sha256: &previous_witness_head_sha256,
        };
        let witness_head_sha256 = canonical_json_sha256(&unsigned).map_err(|_| {
            vec![PerceptualEnrollmentWitnessIssueV1::SerializationFailed]
        })?;
        let witness_head_bytes = decode_hex_32(&witness_head_sha256)
            .expect("canonical SHA-256 commitment must decode");
        let evidence_signature = collection_signer
            .sign_domain(COLLECTION_EVIDENCE_DOMAIN, &witness_head_bytes)
            .map_err(|issue| vec![issue])?;
        let witness_statement = EnrollmentWitnessStatementV1 {
            witness_log_id: &self.policy.witness_log_id,
            sequence: expected_sequence,
            witness_head_sha256: &witness_head_sha256,
            witness_anchor_reference,
        };
        let witness_message = canonical_json_bytes(&witness_statement).map_err(|_| {
            vec![PerceptualEnrollmentWitnessIssueV1::SerializationFailed]
        })?;
        let witness_signature = witness_signer
            .sign_domain(EXTERNAL_WITNESS_DOMAIN, &witness_message)
            .map_err(|issue| vec![issue])?;

        let mut receipt = WitnessedEnrollmentAllocationReceiptV1 {
            receipt_version: PERCEPTUAL_ENROLLMENT_WITNESS_RECEIPT_VERSION.into(),
            witness_policy_sha256: self.policy.policy_sha256.clone(),
            sequence: expected_sequence,
            enrollment_ledger_sha256: unwitnessed.current_ledger_sha256.clone(),
            durable_state_sha256: unwitnessed.current_state_sha256.clone(),
            allocation_head_sha256: unwitnessed.allocation.allocation_head_sha256.clone(),
            enrollment_participant_commitment_sha256: unwitnessed
                .allocation
                .participant_token_commitment_sha256
                .clone(),
            participant_schedule_projection_sha256: unwitnessed
                .allocation
                .participant_schedule_projection_sha256
                .clone(),
            previous_witness_head_sha256,
            witness_head_sha256,
            evidence_signer_id: self.policy.collection_signer.signer_id.clone(),
            evidence_key_epoch: self.policy.collection_signer.key_epoch,
            evidence_signature,
            witness_anchor_reference: witness_anchor_reference.into(),
            witness_signer_id: self.policy.witness_signer.signer_id.clone(),
            witness_key_epoch: self.policy.witness_signer.key_epoch,
            witness_signature,
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = enrollment_witness_receipt_commitment(&receipt).map_err(|_| {
            vec![PerceptualEnrollmentWitnessIssueV1::SerializationFailed]
        })?;
        self.entries.push(receipt.clone());

        Ok(VerifiedWitnessedEnrollmentAllocationV1 {
            receipt,
            participant_schedule_projection: unwitnessed.participant_schedule_projection,
            collection_participant_commitment_sha256: collection_participant_commitment_sha256
                .expect("validated collection participant commitment must exist"),
        })
    }

    pub fn finish(
        self,
        enrollment_policy: &FrozenPerceptualEnrollmentPolicyV1,
        authenticity_policy: &FrozenPerceptualCollectionAuthenticityPolicyV1,
        final_ledger: &FrozenPerceptualEnrollmentAllocationLedgerV1,
    ) -> Result<FrozenPerceptualEnrollmentWitnessBundleV1, Vec<PerceptualEnrollmentWitnessIssueV1>> {
        let mut bundle = FrozenPerceptualEnrollmentWitnessBundleV1 {
            bundle_version: PERCEPTUAL_ENROLLMENT_WITNESS_BUNDLE_VERSION.into(),
            witness_policy_sha256: self.policy.policy_sha256.clone(),
            final_witness_head_sha256: self
                .entries
                .last()
                .map(|entry| entry.witness_head_sha256.clone())
                .unwrap_or_else(|| ZERO_SHA256.into()),
            entries: self.entries,
            bundle_sha256: String::new(),
        };
        bundle.bundle_sha256 = enrollment_witness_bundle_commitment(&bundle).map_err(|_| {
            vec![PerceptualEnrollmentWitnessIssueV1::SerializationFailed]
        })?;
        let issues = validate_enrollment_witness_bundle(
            enrollment_policy,
            authenticity_policy,
            &self.policy,
            final_ledger,
            &bundle,
        );
        if issues.is_empty() {
            Ok(bundle)
        } else {
            Err(issues)
        }
    }
}

pub fn seal_enrollment_witness_policy(
    policy: &mut FrozenPerceptualEnrollmentWitnessPolicyV1,
) -> Result<(), serde_json::Error> {
    policy.policy_sha256 = enrollment_witness_policy_commitment(policy)?;
    Ok(())
}

pub fn enrollment_witness_policy_commitment(
    policy: &FrozenPerceptualEnrollmentWitnessPolicyV1,
) -> Result<String, serde_json::Error> {
    let mut unsigned = policy.clone();
    unsigned.policy_sha256.clear();
    canonical_json_sha256(&unsigned)
}

pub fn enrollment_witness_receipt_commitment(
    receipt: &WitnessedEnrollmentAllocationReceiptV1,
) -> Result<String, serde_json::Error> {
    let mut unsigned = receipt.clone();
    unsigned.receipt_sha256.clear();
    canonical_json_sha256(&unsigned)
}

pub fn enrollment_witness_bundle_commitment(
    bundle: &FrozenPerceptualEnrollmentWitnessBundleV1,
) -> Result<String, serde_json::Error> {
    let mut unsigned = bundle.clone();
    unsigned.bundle_sha256.clear();
    canonical_json_sha256(&unsigned)
}

pub fn validate_enrollment_witness_policy(
    policy: &FrozenPerceptualEnrollmentWitnessPolicyV1,
    enrollment_policy: &FrozenPerceptualEnrollmentPolicyV1,
    authenticity_policy: &FrozenPerceptualCollectionAuthenticityPolicyV1,
) -> Vec<PerceptualEnrollmentWitnessIssueV1> {
    let mut issues = Vec::new();
    if policy.policy_version != PERCEPTUAL_ENROLLMENT_WITNESS_POLICY_VERSION {
        issues.push(PerceptualEnrollmentWitnessIssueV1::WrongPolicyVersion);
    }
    if policy.enrollment_policy_sha256 != enrollment_policy.policy_sha256 {
        issues.push(PerceptualEnrollmentWitnessIssueV1::EnrollmentPolicyDigestMismatch);
    }
    if policy.collection_authenticity_policy_sha256 != authenticity_policy.policy_sha256 {
        issues.push(
            PerceptualEnrollmentWitnessIssueV1::CollectionAuthenticityPolicyDigestMismatch,
        );
    }
    if policy.collection_signer != authenticity_policy.collection_signer {
        issues.push(PerceptualEnrollmentWitnessIssueV1::SignerIdentityMismatch {
            role: "collection_signer".into(),
        });
    }
    if policy.witness_signer != authenticity_policy.witness_signer {
        issues.push(PerceptualEnrollmentWitnessIssueV1::SignerIdentityMismatch {
            role: "witness_signer".into(),
        });
    }
    validate_verifier_identity(&policy.collection_signer, &mut issues);
    validate_verifier_identity(&policy.witness_signer, &mut issues);
    if policy.collection_signer.signer_id == policy.witness_signer.signer_id
        || policy.collection_signer.verifying_key_bytes == policy.witness_signer.verifying_key_bytes
    {
        issues.push(PerceptualEnrollmentWitnessIssueV1::SignerWitnessNotDistinct);
    }
    if policy.witness_log_id.trim().is_empty() {
        issues.push(PerceptualEnrollmentWitnessIssueV1::EmptyWitnessLogId);
    }
    for (field, enabled) in [
        (
            "per_allocation_external_witness_required",
            policy.per_allocation_external_witness_required,
        ),
        (
            "scored_authority_requires_verified_witness",
            policy.scored_authority_requires_verified_witness,
        ),
        (
            "raw_participant_token_prohibited",
            policy.raw_participant_token_prohibited,
        ),
    ] {
        if !enabled {
            issues.push(PerceptualEnrollmentWitnessIssueV1::MissingPolicyProtection {
                field: field.into(),
            });
        }
    }
    if decode_hex_32(&policy.policy_sha256).is_none() {
        issues.push(PerceptualEnrollmentWitnessIssueV1::InvalidPolicyDigest);
    }
    match enrollment_witness_policy_commitment(policy) {
        Ok(value) if value == policy.policy_sha256 => {}
        Ok(_) => issues.push(PerceptualEnrollmentWitnessIssueV1::PolicyDigestMismatch),
        Err(_) => issues.push(PerceptualEnrollmentWitnessIssueV1::SerializationFailed),
    }
    issues
}

pub fn validate_enrollment_witness_bundle(
    enrollment_policy: &FrozenPerceptualEnrollmentPolicyV1,
    authenticity_policy: &FrozenPerceptualCollectionAuthenticityPolicyV1,
    witness_policy: &FrozenPerceptualEnrollmentWitnessPolicyV1,
    final_ledger: &FrozenPerceptualEnrollmentAllocationLedgerV1,
    bundle: &FrozenPerceptualEnrollmentWitnessBundleV1,
) -> Vec<PerceptualEnrollmentWitnessIssueV1> {
    let mut issues = validate_enrollment_witness_policy(
        witness_policy,
        enrollment_policy,
        authenticity_policy,
    );
    if bundle.bundle_version != PERCEPTUAL_ENROLLMENT_WITNESS_BUNDLE_VERSION {
        issues.push(PerceptualEnrollmentWitnessIssueV1::WrongBundleVersion);
    }
    if bundle.witness_policy_sha256 != witness_policy.policy_sha256 {
        issues.push(PerceptualEnrollmentWitnessIssueV1::BundlePolicyMismatch);
    }
    if bundle.entries.len() != final_ledger.allocations.len() {
        issues.push(PerceptualEnrollmentWitnessIssueV1::WrongEntryCount {
            found: bundle.entries.len(),
            expected: final_ledger.allocations.len(),
        });
    }

    let mut previous = ZERO_SHA256.to_string();
    let mut seen_anchors = BTreeSet::new();
    for (index, entry) in bundle.entries.iter().enumerate() {
        if entry.receipt_version != PERCEPTUAL_ENROLLMENT_WITNESS_RECEIPT_VERSION {
            issues.push(PerceptualEnrollmentWitnessIssueV1::WrongReceiptVersion { index });
        }
        if entry.witness_policy_sha256 != witness_policy.policy_sha256 {
            issues.push(PerceptualEnrollmentWitnessIssueV1::ReceiptPolicyMismatch { index });
        }
        if entry.sequence != index as u32 {
            issues.push(PerceptualEnrollmentWitnessIssueV1::SequenceMismatch { index });
        }
        if entry.previous_witness_head_sha256 != previous {
            issues.push(PerceptualEnrollmentWitnessIssueV1::PreviousWitnessHeadMismatch {
                index,
            });
        }
        for (field, digest) in [
            ("enrollment_ledger_sha256", entry.enrollment_ledger_sha256.as_str()),
            ("durable_state_sha256", entry.durable_state_sha256.as_str()),
            ("allocation_head_sha256", entry.allocation_head_sha256.as_str()),
            (
                "enrollment_participant_commitment_sha256",
                entry.enrollment_participant_commitment_sha256.as_str(),
            ),
            (
                "participant_schedule_projection_sha256",
                entry.participant_schedule_projection_sha256.as_str(),
            ),
            ("previous_witness_head_sha256", entry.previous_witness_head_sha256.as_str()),
            ("witness_head_sha256", entry.witness_head_sha256.as_str()),
            ("receipt_sha256", entry.receipt_sha256.as_str()),
        ] {
            if decode_hex_32(digest).is_none() {
                issues.push(PerceptualEnrollmentWitnessIssueV1::InvalidArtifactDigest {
                    index,
                    field: field.into(),
                });
            }
        }

        let Some(allocation) = final_ledger.allocations.get(index) else {
            previous = entry.witness_head_sha256.clone();
            continue;
        };
        if allocation.allocation_head_sha256 != entry.allocation_head_sha256 {
            issues.push(PerceptualEnrollmentWitnessIssueV1::AllocationHeadMismatch { index });
        }
        if allocation.participant_token_commitment_sha256
            != entry.enrollment_participant_commitment_sha256
        {
            issues.push(PerceptualEnrollmentWitnessIssueV1::ParticipantCommitmentMismatch {
                index,
            });
        }
        if allocation.participant_schedule_projection_sha256
            != entry.participant_schedule_projection_sha256
        {
            issues.push(PerceptualEnrollmentWitnessIssueV1::ParticipantProjectionMismatch {
                index,
            });
        }
        let mut prefix = final_ledger.clone();
        prefix.allocations.truncate(index + 1);
        prefix.final_allocation_head_sha256 = allocation.allocation_head_sha256.clone();
        prefix.ledger_sha256.clear();
        match enrollment_allocation_ledger_commitment(&prefix) {
            Ok(value) if value == entry.enrollment_ledger_sha256 => {}
            _ => issues.push(PerceptualEnrollmentWitnessIssueV1::LedgerPrefixDigestMismatch {
                index,
            }),
        }

        let unsigned = EnrollmentWitnessUnsignedEntryV1 {
            witness_policy_sha256: &entry.witness_policy_sha256,
            sequence: entry.sequence,
            enrollment_ledger_sha256: &entry.enrollment_ledger_sha256,
            durable_state_sha256: &entry.durable_state_sha256,
            allocation_head_sha256: &entry.allocation_head_sha256,
            enrollment_participant_commitment_sha256: &entry
                .enrollment_participant_commitment_sha256,
            participant_schedule_projection_sha256: &entry
                .participant_schedule_projection_sha256,
            previous_witness_head_sha256: &entry.previous_witness_head_sha256,
        };
        match canonical_json_sha256(&unsigned) {
            Ok(value) if value == entry.witness_head_sha256 => {}
            _ => issues.push(PerceptualEnrollmentWitnessIssueV1::WitnessHeadMismatch { index }),
        }
        if entry.evidence_signer_id != witness_policy.collection_signer.signer_id
            || entry.evidence_key_epoch != witness_policy.collection_signer.key_epoch
        {
            issues.push(PerceptualEnrollmentWitnessIssueV1::ReceiptSignerMismatch {
                index,
                role: "collection_signer".into(),
            });
        }
        if entry.witness_signer_id != witness_policy.witness_signer.signer_id
            || entry.witness_key_epoch != witness_policy.witness_signer.key_epoch
        {
            issues.push(PerceptualEnrollmentWitnessIssueV1::ReceiptSignerMismatch {
                index,
                role: "witness_signer".into(),
            });
        }
        if entry.witness_anchor_reference.trim().is_empty() {
            issues.push(PerceptualEnrollmentWitnessIssueV1::EmptyWitnessAnchor);
        } else if !seen_anchors.insert(entry.witness_anchor_reference.clone()) {
            issues.push(PerceptualEnrollmentWitnessIssueV1::DuplicateWitnessAnchor {
                anchor: entry.witness_anchor_reference.clone(),
            });
        }
        if let Some(head) = decode_hex_32(&entry.witness_head_sha256) {
            if verify_domain_signature(
                &witness_policy.collection_signer,
                COLLECTION_EVIDENCE_DOMAIN,
                &head,
                &entry.evidence_signature,
            )
            .is_err()
            {
                issues.push(PerceptualEnrollmentWitnessIssueV1::EvidenceSignatureInvalid {
                    index,
                });
            }
        } else {
            issues.push(PerceptualEnrollmentWitnessIssueV1::WitnessHeadMismatch { index });
        }
        let statement = EnrollmentWitnessStatementV1 {
            witness_log_id: &witness_policy.witness_log_id,
            sequence: entry.sequence,
            witness_head_sha256: &entry.witness_head_sha256,
            witness_anchor_reference: &entry.witness_anchor_reference,
        };
        match canonical_json_bytes(&statement) {
            Ok(message)
                if verify_domain_signature(
                    &witness_policy.witness_signer,
                    EXTERNAL_WITNESS_DOMAIN,
                    &message,
                    &entry.witness_signature,
                )
                .is_ok() => {}
            _ => issues.push(PerceptualEnrollmentWitnessIssueV1::WitnessSignatureInvalid {
                index,
            }),
        }
        match enrollment_witness_receipt_commitment(entry) {
            Ok(value) if value == entry.receipt_sha256 => {}
            _ => issues.push(PerceptualEnrollmentWitnessIssueV1::ReceiptDigestMismatch { index }),
        }
        previous = entry.witness_head_sha256.clone();
    }

    if bundle.final_witness_head_sha256 != previous {
        issues.push(PerceptualEnrollmentWitnessIssueV1::FinalWitnessHeadMismatch);
    }
    match enrollment_witness_bundle_commitment(bundle) {
        Ok(value) if value == bundle.bundle_sha256 => {}
        _ => issues.push(PerceptualEnrollmentWitnessIssueV1::BundleDigestMismatch),
    }
    issues
}

fn validate_signing_key_matches(
    role: &str,
    signing_key: &EnrollmentWitnessSigningKeyV1,
    expected: &CollectionVerifierIdentityV1,
    issues: &mut Vec<PerceptualEnrollmentWitnessIssueV1>,
) {
    if signing_key.verifier_identity() != *expected {
        issues.push(PerceptualEnrollmentWitnessIssueV1::SignerIdentityMismatch {
            role: role.into(),
        });
    }
}

fn validate_verifier_identity(
    identity: &CollectionVerifierIdentityV1,
    issues: &mut Vec<PerceptualEnrollmentWitnessIssueV1>,
) {
    if identity.signer_id.trim().is_empty() {
        issues.push(PerceptualEnrollmentWitnessIssueV1::EmptySignerId);
    }
    if identity.key_epoch == 0 {
        issues.push(PerceptualEnrollmentWitnessIssueV1::InvalidKeyEpoch);
    }
    if verifier_from_identity(identity).is_err() {
        issues.push(PerceptualEnrollmentWitnessIssueV1::InvalidVerifyingKey {
            signer_id: identity.signer_id.clone(),
        });
    }
}

fn verify_domain_signature(
    identity: &CollectionVerifierIdentityV1,
    domain: &[u8],
    message: &[u8],
    signature_bytes: &[u8],
) -> Result<(), ()> {
    validate_domain_message(domain, message).map_err(|_| ())?;
    let verifier = verifier_from_identity(identity)?;
    let encoded: [u8; ED25519_SIGNATURE_BYTES] = signature_bytes.try_into().map_err(|_| ())?;
    let signature = Signature::from_bytes(&encoded);
    verifier
        .verify(&domain_separated_transcript(domain, message), &signature)
        .map_err(|_| ())
}

fn verifier_from_identity(identity: &CollectionVerifierIdentityV1) -> Result<VerifyingKey, ()> {
    if identity.verifying_key_bytes.len() != ED25519_PUBLIC_KEY_BYTES
        || identity.verifying_key_bytes.iter().all(|byte| *byte == 0)
    {
        return Err(());
    }
    let encoded: [u8; ED25519_PUBLIC_KEY_BYTES] = identity
        .verifying_key_bytes
        .as_slice()
        .try_into()
        .map_err(|_| ())?;
    VerifyingKey::from_bytes(&encoded).map_err(|_| ())
}

fn validate_domain_message(
    domain: &[u8],
    message: &[u8],
) -> Result<(), PerceptualEnrollmentWitnessIssueV1> {
    if domain.is_empty() || domain.len() > MAX_SIGNATURE_DOMAIN_BYTES {
        return Err(PerceptualEnrollmentWitnessIssueV1::InvalidSignatureDomain);
    }
    if message.is_empty() || message.len() > MAX_SIGNED_MESSAGE_BYTES {
        return Err(PerceptualEnrollmentWitnessIssueV1::SignatureMessageTooLarge);
    }
    Ok(())
}

fn domain_separated_transcript(domain: &[u8], message: &[u8]) -> Vec<u8> {
    let mut transcript = Vec::with_capacity(8 + domain.len() + message.len());
    transcript.extend_from_slice(&(domain.len() as u64).to_le_bytes());
    transcript.extend_from_slice(domain);
    transcript.extend_from_slice(message);
    transcript
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(name: &str, seed_byte: u8) -> EnrollmentWitnessSigningKeyV1 {
        EnrollmentWitnessSigningKeyV1::from_seed(name, 1, [seed_byte; 32]).unwrap()
    }

    #[test]
    fn enrollment_and_collection_participant_commitments_remain_distinct_namespaces() {
        let token = "high-entropy-participant-token-0123456789abcdef";
        let enrollment = participant_allocation_reference_commitment(token).unwrap();
        let collection = participant_token_commitment(token).unwrap();
        assert_ne!(enrollment, collection);
        assert_eq!(enrollment.len(), 64);
        assert_eq!(collection.len(), 64);
    }

    #[test]
    fn collection_signature_cannot_replay_as_external_witness_signature() {
        let signer = key("collection", 7);
        let identity = signer.verifier_identity();
        let message = [3u8; 32];
        let signature = signer
            .sign_domain(COLLECTION_EVIDENCE_DOMAIN, &message)
            .unwrap();
        assert!(
            verify_domain_signature(
                &identity,
                COLLECTION_EVIDENCE_DOMAIN,
                &message,
                &signature,
            )
            .is_ok()
        );
        assert!(
            verify_domain_signature(
                &identity,
                EXTERNAL_WITNESS_DOMAIN,
                &message,
                &signature,
            )
            .is_err()
        );
    }

    #[test]
    fn witness_head_binds_durable_state_and_allocation_identity() {
        let policy = "a".repeat(64);
        let ledger = "b".repeat(64);
        let durable = "c".repeat(64);
        let allocation = "d".repeat(64);
        let participant = "e".repeat(64);
        let projection = "f".repeat(64);
        let changed = "1".repeat(64);
        let left = EnrollmentWitnessUnsignedEntryV1 {
            witness_policy_sha256: &policy,
            sequence: 0,
            enrollment_ledger_sha256: &ledger,
            durable_state_sha256: &durable,
            allocation_head_sha256: &allocation,
            enrollment_participant_commitment_sha256: &participant,
            participant_schedule_projection_sha256: &projection,
            previous_witness_head_sha256: ZERO_SHA256,
        };
        let mut right = left.clone();
        right.durable_state_sha256 = &changed;
        assert_ne!(
            canonical_json_sha256(&left).unwrap(),
            canonical_json_sha256(&right).unwrap()
        );
    }
}
