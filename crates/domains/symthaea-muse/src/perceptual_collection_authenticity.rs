// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1EIR/P1EWR: authenticated, independently witnessed collection evidence.
//!
//! P1E hash-chains response records, but a privileged writer who controls the
//! store can rewrite a whole history and recompute its hashes. This layer adds
//! a different theorem: blinded collection commitments were signed under a
//! frozen evidence key and every append-log head was also signed by a distinct
//! witness identity before the next governed stage may be admitted.
//!
//! Participant withdrawal is explicit. A session observed before withdrawal
//! may disappear from the retained P1E dataset only when the pre-frozen
//! withdrawal policy permits minimal audit-commitment retention and a distinct
//! withdrawal authority signs a tombstone. Withdrawals before any authenticated
//! session use the same policy and a no-session tombstone. The final P1E close
//! withdrawal count must reconcile exactly with authenticated tombstones.
//!
//! Signatures authenticate content and append order. They are deliberately NOT
//! trusted timestamps; absolute-time claims belong to P1ER. Participant-token
//! commitments are pseudonymous/linkable commitments, not anonymity claims.

use crate::evidence_digest::{
    canonical_json_bytes, canonical_json_sha256, decode_hex_32,
    perceptual_collection_evidence::{
        FrozenPerceptualCollectionAuthorityV1, PerceptualCollectionCloseV1,
        PerceptualSessionEvidenceV1, RawPerceptualCollectionV1, validate_collection_authority,
        validate_collection_close, validate_raw_collection,
    },
    perceptual_participant_schedule::PerceptualParticipantScheduleBookV1,
    perceptual_stimulus_pack::{
        FrozenC6fRenderSubjectBindingV1, FrozenPerceptualStimulusPackV1,
    },
    perceptual_study_protocol::FrozenPerceptualStudyProtocolV1,
};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const PERCEPTUAL_COLLECTION_AUTHENTICITY_POLICY_VERSION: &str =
    "mel003-perceptual-collection-authenticity-policy-v1";
pub const PERCEPTUAL_COLLECTION_AUTHENTICITY_BUNDLE_VERSION: &str =
    "mel003-perceptual-collection-authenticity-bundle-v1";
pub const PERCEPTUAL_WITHDRAWAL_POLICY_VERSION: &str =
    "mel003-perceptual-withdrawal-policy-v1";
pub const PERCEPTUAL_WITHDRAWAL_TOMBSTONE_VERSION: &str =
    "mel003-perceptual-withdrawal-tombstone-v1";
pub const PERCEPTUAL_COLLECTION_STUDY_DOMAIN: &str =
    "symthaea.mel003.p1.collection-authenticity.v1";
pub const ZERO_SHA256: &str =
    "0000000000000000000000000000000000000000000000000000000000000000";

const AUTHORITY_DOMAIN: &[u8] =
    b"symthaea.mel003.p1.collection-authenticity.v1/authority";
const SESSION_DOMAIN: &[u8] = b"symthaea.mel003.p1.collection-authenticity.v1/session";
const WITHDRAWAL_APPEND_DOMAIN: &[u8] =
    b"symthaea.mel003.p1.collection-authenticity.v1/withdrawal-append";
const WITHDRAWAL_AUTHORITY_DOMAIN: &[u8] =
    b"symthaea.mel003.p1.collection-authenticity.v1/withdrawal-authority";
const RAW_DATASET_DOMAIN: &[u8] =
    b"symthaea.mel003.p1.collection-authenticity.v1/raw-dataset";
const COLLECTION_CLOSE_DOMAIN: &[u8] =
    b"symthaea.mel003.p1.collection-authenticity.v1/collection-close";
const WITNESS_DOMAIN: &[u8] =
    b"symthaea.mel003.p1.collection-authenticity.v1/external-witness";
const PARTICIPANT_TOKEN_COMMITMENT_DOMAIN: &str =
    "symthaea.mel003.p1.collection-authenticity.v1/participant-token";
const MAX_SIGNED_MESSAGE_BYTES: usize = 1 << 20;
const MAX_SIGNATURE_DOMAIN_BYTES: usize = 256;
const ED25519_PUBLIC_KEY_BYTES: usize = 32;
const ED25519_SIGNATURE_BYTES: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CollectionVerifierIdentityV1 {
    pub signer_id: String,
    pub key_epoch: u64,
    pub verifying_key_bytes: Vec<u8>,
}

/// Signing-side helper. Secret seed material never appears in serialized study
/// evidence; only [`CollectionVerifierIdentityV1`] does.
pub struct CollectionSigningKeyV1 {
    signer_id: String,
    key_epoch: u64,
    inner: SigningKey,
}

impl CollectionSigningKeyV1 {
    pub fn from_seed(
        signer_id: impl Into<String>,
        key_epoch: u64,
        seed: [u8; 32],
    ) -> Result<Self, PerceptualCollectionAuthenticityIssueV1> {
        let signer_id = signer_id.into();
        if signer_id.trim().is_empty() {
            return Err(PerceptualCollectionAuthenticityIssueV1::EmptySignerId);
        }
        if key_epoch == 0 {
            return Err(PerceptualCollectionAuthenticityIssueV1::InvalidKeyEpoch);
        }
        if seed == [0u8; 32] {
            return Err(PerceptualCollectionAuthenticityIssueV1::InvalidSigningSeed);
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
    ) -> Result<Vec<u8>, PerceptualCollectionAuthenticityIssueV1> {
        validate_domain_message(domain, message)?;
        let transcript = domain_separated_transcript(domain, message);
        Ok(self.inner.sign(&transcript).to_bytes().to_vec())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WithdrawalEvidenceRetentionModeV1 {
    /// The approved consent/review policy permits retaining only the minimum
    /// non-content commitments needed to prove an authenticated withdrawal.
    MinimalAuditCommitment,
    /// The approved policy requires erasing participant-linked commitments as
    /// well. This P1EIR bundle cannot prove that policy and fails closed until
    /// a separately qualified custodial/aggregate evidence path exists.
    FullErasureRequiresSeparateCustodialEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPerceptualWithdrawalPolicyV1 {
    pub policy_version: String,
    pub collection_authority_sha256: String,
    pub consent_form_sha256: String,
    pub participant_information_sha256: String,
    pub privacy_notice_sha256: String,
    pub human_study_review_evidence_sha256: String,
    /// Opaque commitment to the independently frozen P1ER chronology policy.
    pub collection_chronology_policy_sha256: String,
    pub retention_mode: WithdrawalEvidenceRetentionModeV1,
    pub withdrawal_allowed_until_collection_close: bool,
    pub raw_session_data_deleted_on_withdrawal: bool,
    pub raw_participant_token_prohibited_in_authenticity_log: bool,
    pub outcome_aware_withdrawal_processing_prohibited: bool,
    pub arm_label_access_prohibited: bool,
    pub withdrawal_authority: CollectionVerifierIdentityV1,
    pub policy_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPerceptualCollectionAuthenticityPolicyV1 {
    pub policy_version: String,
    pub study_domain: String,
    pub collection_authority_sha256: String,
    pub withdrawal_policy_sha256: String,
    pub collection_signer: CollectionVerifierIdentityV1,
    pub witness_signer: CollectionVerifierIdentityV1,
    pub witness_log_id: String,
    pub per_entry_external_witness_required: bool,
    pub unblinding_requires_verified_close_anchor: bool,
    pub policy_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WithdrawalSessionDispositionV1 {
    /// Withdrawal occurred before any scored session head had been authenticated.
    NoAuthenticatedSession,
    /// A scored session head had already been authenticated and its raw/session
    /// material was subsequently deleted under the frozen withdrawal policy.
    AuthenticatedSessionDeleted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthenticatedWithdrawalTombstoneV1 {
    pub tombstone_version: String,
    pub withdrawal_policy_sha256: String,
    pub participant_token_commitment_sha256: String,
    pub disposition: WithdrawalSessionDispositionV1,
    /// Required only for `AuthenticatedSessionDeleted`.
    pub prior_session_sha256: Option<String>,
    /// Commitment to the participant withdrawal/request evidence retained under
    /// the approved governance policy. It must not encode arm/outcome meaning.
    pub withdrawal_event_evidence_sha256: String,
    /// Commitment to a deletion/no-retained-data disposition receipt. This is
    /// evidence of the governed operation, not proof that no copy exists anywhere.
    pub data_disposition_receipt_sha256: String,
    /// Opaque commitment to the restricted chronology event used by P1ER.
    pub withdrawal_chronology_event_sha256: String,
    pub withdrawal_signer_id: String,
    pub withdrawal_key_epoch: u64,
    pub withdrawal_signature: Vec<u8>,
    pub tombstone_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthenticatedCollectionArtifactKindV1 {
    CollectionAuthority,
    SessionHead,
    WithdrawalTombstone,
    RawDatasetRoot,
    CollectionCloseRoot,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthenticatedCollectionAppendReceiptV1 {
    pub sequence: u64,
    pub artifact_kind: AuthenticatedCollectionArtifactKindV1,
    /// Present only for session/tombstone entries. This is a pseudonymous
    /// commitment to the high-entropy schedule token; the raw token is excluded.
    pub participant_token_commitment_sha256: Option<String>,
    pub artifact_sha256: String,
    pub previous_log_head_sha256: String,
    pub log_head_sha256: String,
    pub evidence_signer_id: String,
    pub evidence_key_epoch: u64,
    pub evidence_signature: Vec<u8>,
    /// Opaque identifier supplied by the independent witness/log service.
    pub witness_anchor_reference: String,
    pub witness_signer_id: String,
    pub witness_key_epoch: u64,
    pub witness_signature: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPerceptualCollectionAuthenticityBundleV1 {
    pub bundle_version: String,
    pub policy_sha256: String,
    pub withdrawal_policy_sha256: String,
    pub raw_dataset_sha256: String,
    pub collection_close_sha256: String,
    pub withdrawal_tombstones: Vec<AuthenticatedWithdrawalTombstoneV1>,
    /// Append order is evidence. Do not sort this vector.
    pub entries: Vec<AuthenticatedCollectionAppendReceiptV1>,
    pub final_log_head_sha256: String,
    pub bundle_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthenticatedUnblindingGateV1 {
    pub authenticity_bundle_sha256: String,
    pub withdrawal_policy_sha256: String,
    pub collection_close_sha256: String,
    pub verified_close_log_head_sha256: String,
    pub private_audit_access_may_begin: bool,
    pub randomization_key_reveal_may_begin: bool,
    pub gate_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualCollectionAuthenticityIssueV1 {
    InvalidCollectionAuthority,
    InvalidRawCollection,
    InvalidCollectionClose,
    WrongPolicyVersion,
    WrongWithdrawalPolicyVersion,
    WrongWithdrawalTombstoneVersion,
    WrongStudyDomain,
    AuthorityDigestMismatch,
    GovernanceDigestMismatch { field: String },
    InvalidPolicyDigest,
    InvalidWithdrawalPolicyDigest,
    WithdrawalPolicyDigestMismatch,
    UnsupportedFullErasureRetentionMode,
    EmptySignerId,
    InvalidKeyEpoch,
    InvalidSigningSeed,
    InvalidVerifyingKey { signer_id: String },
    SignerWitnessNotDistinct,
    WithdrawalAuthorityNotDistinct,
    EmptyWitnessLogId,
    MissingPerEntryWitnessRequirement,
    MissingUnblindingGateRequirement,
    MissingWithdrawalProtection { field: String },
    PolicySerializationFailed,
    PolicyDigestMismatch,
    WithdrawalPolicySerializationFailed,
    SigningIdentityMismatch { role: String },
    WrongAppendStage,
    InvalidArtifactDigest { sequence: u64 },
    EmptyParticipantCommitment { sequence: u64 },
    UnexpectedParticipantCommitment { sequence: u64 },
    InvalidParticipantCommitment { sequence: u64 },
    EmptyWitnessAnchor { sequence: u64 },
    DuplicateWitnessAnchor { anchor: String },
    SignatureMessageTooLarge,
    InvalidSignatureDomain,
    EvidenceSignatureInvalid { sequence: u64 },
    WitnessSignatureInvalid { sequence: u64 },
    WithdrawalSignatureInvalid { tombstone_sha256: String },
    WrongBundleVersion,
    BundlePolicyDigestMismatch,
    BundleWithdrawalPolicyDigestMismatch,
    RawDatasetDigestMismatch,
    CollectionCloseDigestMismatch,
    WrongEntryCount { found: usize, expected: usize },
    SequenceMismatch { index: usize },
    ArtifactOrderMismatch { index: usize },
    PreviousLogHeadMismatch { index: usize },
    LogHeadMismatch { index: usize },
    ReceiptSignerMismatch { index: usize, role: String },
    DuplicateSessionReceipt { participant_commitment: String },
    MissingSessionReceipt { participant_commitment: String },
    UnexpectedSessionReceipt { participant_commitment: String },
    SessionDigestMismatch { participant_commitment: String },
    DuplicateWithdrawalTombstone { participant_commitment: String },
    MissingWithdrawalTombstone { participant_commitment: String },
    UnexpectedWithdrawalTombstone { participant_commitment: String },
    WithdrawalTombstoneDigestMismatch { participant_commitment: String },
    WithdrawalDispositionMismatch { participant_commitment: String },
    WithdrawalSessionNotPreviouslyAuthenticated { participant_commitment: String },
    WithdrawalSessionStillRetained { participant_commitment: String },
    WithdrawalCountMismatch { found: usize, expected: usize },
    UnknownWithdrawalParticipant { participant_commitment: String },
    WithdrawalTombstoneNotAppended { participant_commitment: String },
    InvalidWithdrawalEvidenceDigest { field: String },
    FinalLogHeadMismatch,
    BundleSerializationFailed,
    BundleDigestMismatch,
    UnblindingGateSerializationFailed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ParticipantTokenCommitmentV1<'a> {
    domain: &'static str,
    participant_token: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct UnsignedAppendEntryV1<'a> {
    sequence: u64,
    artifact_kind: AuthenticatedCollectionArtifactKindV1,
    participant_token_commitment_sha256: Option<&'a str>,
    artifact_sha256: &'a str,
    previous_log_head_sha256: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct WitnessStatementV1<'a> {
    witness_log_id: &'a str,
    sequence: u64,
    log_head_sha256: &'a str,
    witness_anchor_reference: &'a str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct WithdrawalStatementV1<'a> {
    withdrawal_policy_sha256: &'a str,
    participant_token_commitment_sha256: &'a str,
    disposition: WithdrawalSessionDispositionV1,
    prior_session_sha256: Option<&'a str>,
    withdrawal_event_evidence_sha256: &'a str,
    data_disposition_receipt_sha256: &'a str,
    withdrawal_chronology_event_sha256: &'a str,
}

pub struct PerceptualCollectionAuthenticityLogBuilderV1 {
    policy: FrozenPerceptualCollectionAuthenticityPolicyV1,
    withdrawal_policy: FrozenPerceptualWithdrawalPolicyV1,
    entries: Vec<AuthenticatedCollectionAppendReceiptV1>,
    withdrawal_tombstones: Vec<AuthenticatedWithdrawalTombstoneV1>,
}

impl PerceptualCollectionAuthenticityLogBuilderV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn begin(
        policy: FrozenPerceptualCollectionAuthenticityPolicyV1,
        withdrawal_policy: FrozenPerceptualWithdrawalPolicyV1,
        authority: &FrozenPerceptualCollectionAuthorityV1,
        collection_signer: &CollectionSigningKeyV1,
        witness_signer: &CollectionSigningKeyV1,
        withdrawal_signer: &CollectionSigningKeyV1,
        witness_anchor_reference: &str,
    ) -> Result<Self, Vec<PerceptualCollectionAuthenticityIssueV1>> {
        let mut issues = validate_withdrawal_policy(&withdrawal_policy, authority);
        issues.extend(validate_authenticity_policy(
            &policy,
            &withdrawal_policy,
            authority,
        ));
        validate_signing_key_matches(
            "collection_signer",
            collection_signer,
            &policy.collection_signer,
            &mut issues,
        );
        validate_signing_key_matches(
            "witness_signer",
            witness_signer,
            &policy.witness_signer,
            &mut issues,
        );
        validate_signing_key_matches(
            "withdrawal_signer",
            withdrawal_signer,
            &withdrawal_policy.withdrawal_authority,
            &mut issues,
        );
        if !issues.is_empty() {
            return Err(issues);
        }
        let mut builder = Self {
            policy,
            withdrawal_policy,
            entries: Vec::new(),
            withdrawal_tombstones: Vec::new(),
        };
        builder.append_exact(
            AuthenticatedCollectionArtifactKindV1::CollectionAuthority,
            None,
            &authority.authority_sha256,
            collection_signer,
            witness_signer,
            witness_anchor_reference,
        )?;
        Ok(builder)
    }

    pub fn append_session(
        &mut self,
        session: &PerceptualSessionEvidenceV1,
        collection_signer: &CollectionSigningKeyV1,
        witness_signer: &CollectionSigningKeyV1,
        witness_anchor_reference: &str,
    ) -> Result<(), Vec<PerceptualCollectionAuthenticityIssueV1>> {
        let allowed = self.entries.last().is_some_and(|entry| {
            matches!(
                entry.artifact_kind,
                AuthenticatedCollectionArtifactKindV1::CollectionAuthority
                    | AuthenticatedCollectionArtifactKindV1::SessionHead
                    | AuthenticatedCollectionArtifactKindV1::WithdrawalTombstone
            )
        });
        if !allowed {
            return Err(vec![PerceptualCollectionAuthenticityIssueV1::WrongAppendStage]);
        }
        let participant_commitment = participant_token_commitment(&session.participant_token)
            .map_err(|_| {
                vec![PerceptualCollectionAuthenticityIssueV1::BundleSerializationFailed]
            })?;
        self.append_exact(
            AuthenticatedCollectionArtifactKindV1::SessionHead,
            Some(participant_commitment.as_str()),
            &session.session_sha256,
            collection_signer,
            witness_signer,
            witness_anchor_reference,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn append_withdrawal(
        &mut self,
        participant_token: &str,
        prior_session_sha256: Option<&str>,
        withdrawal_event_evidence_sha256: &str,
        data_disposition_receipt_sha256: &str,
        withdrawal_chronology_event_sha256: &str,
        withdrawal_signer: &CollectionSigningKeyV1,
        collection_signer: &CollectionSigningKeyV1,
        witness_signer: &CollectionSigningKeyV1,
        witness_anchor_reference: &str,
    ) -> Result<(), Vec<PerceptualCollectionAuthenticityIssueV1>> {
        let allowed = self.entries.last().is_some_and(|entry| {
            matches!(
                entry.artifact_kind,
                AuthenticatedCollectionArtifactKindV1::CollectionAuthority
                    | AuthenticatedCollectionArtifactKindV1::SessionHead
                    | AuthenticatedCollectionArtifactKindV1::WithdrawalTombstone
            )
        });
        if !allowed {
            return Err(vec![PerceptualCollectionAuthenticityIssueV1::WrongAppendStage]);
        }
        let mut issues = Vec::new();
        validate_signing_key_matches(
            "withdrawal_signer",
            withdrawal_signer,
            &self.withdrawal_policy.withdrawal_authority,
            &mut issues,
        );
        let participant_commitment = participant_token_commitment(participant_token).map_err(|_| {
            vec![PerceptualCollectionAuthenticityIssueV1::BundleSerializationFailed]
        })?;
        if self
            .withdrawal_tombstones
            .iter()
            .any(|tombstone| tombstone.participant_token_commitment_sha256 == participant_commitment)
        {
            issues.push(
                PerceptualCollectionAuthenticityIssueV1::DuplicateWithdrawalTombstone {
                    participant_commitment: participant_commitment.clone(),
                },
            );
        }
        let disposition = match prior_session_sha256 {
            Some(session_sha256) => {
                let prior_session_exists = self.entries.iter().any(|entry| {
                    entry.artifact_kind == AuthenticatedCollectionArtifactKindV1::SessionHead
                        && entry.participant_token_commitment_sha256.as_deref()
                            == Some(participant_commitment.as_str())
                        && entry.artifact_sha256 == session_sha256
                });
                if !prior_session_exists {
                    issues.push(
                        PerceptualCollectionAuthenticityIssueV1::WithdrawalSessionNotPreviouslyAuthenticated {
                            participant_commitment: participant_commitment.clone(),
                        },
                    );
                }
                WithdrawalSessionDispositionV1::AuthenticatedSessionDeleted
            }
            None => WithdrawalSessionDispositionV1::NoAuthenticatedSession,
        };
        for (field, digest) in [
            ("withdrawal_event_evidence_sha256", withdrawal_event_evidence_sha256),
            ("data_disposition_receipt_sha256", data_disposition_receipt_sha256),
            (
                "withdrawal_chronology_event_sha256",
                withdrawal_chronology_event_sha256,
            ),
        ] {
            if decode_hex_32(digest).is_none() {
                issues.push(
                    PerceptualCollectionAuthenticityIssueV1::InvalidWithdrawalEvidenceDigest {
                        field: field.into(),
                    },
                );
            }
        }
        if !issues.is_empty() {
            return Err(issues);
        }

        let statement = WithdrawalStatementV1 {
            withdrawal_policy_sha256: &self.withdrawal_policy.policy_sha256,
            participant_token_commitment_sha256: &participant_commitment,
            disposition,
            prior_session_sha256,
            withdrawal_event_evidence_sha256,
            data_disposition_receipt_sha256,
            withdrawal_chronology_event_sha256,
        };
        let statement_bytes = canonical_json_bytes(&statement).map_err(|_| {
            vec![PerceptualCollectionAuthenticityIssueV1::BundleSerializationFailed]
        })?;
        let withdrawal_signature = withdrawal_signer
            .sign_domain(WITHDRAWAL_AUTHORITY_DOMAIN, &statement_bytes)
            .map_err(|issue| vec![issue])?;
        let mut tombstone = AuthenticatedWithdrawalTombstoneV1 {
            tombstone_version: PERCEPTUAL_WITHDRAWAL_TOMBSTONE_VERSION.into(),
            withdrawal_policy_sha256: self.withdrawal_policy.policy_sha256.clone(),
            participant_token_commitment_sha256: participant_commitment.clone(),
            disposition,
            prior_session_sha256: prior_session_sha256.map(str::to_owned),
            withdrawal_event_evidence_sha256: withdrawal_event_evidence_sha256.into(),
            data_disposition_receipt_sha256: data_disposition_receipt_sha256.into(),
            withdrawal_chronology_event_sha256: withdrawal_chronology_event_sha256.into(),
            withdrawal_signer_id: self.withdrawal_policy.withdrawal_authority.signer_id.clone(),
            withdrawal_key_epoch: self.withdrawal_policy.withdrawal_authority.key_epoch,
            withdrawal_signature,
            tombstone_sha256: String::new(),
        };
        tombstone.tombstone_sha256 = withdrawal_tombstone_commitment(&tombstone).map_err(|_| {
            vec![PerceptualCollectionAuthenticityIssueV1::BundleSerializationFailed]
        })?;
        self.append_exact(
            AuthenticatedCollectionArtifactKindV1::WithdrawalTombstone,
            Some(participant_commitment.as_str()),
            &tombstone.tombstone_sha256,
            collection_signer,
            witness_signer,
            witness_anchor_reference,
        )?;
        self.withdrawal_tombstones.push(tombstone);
        Ok(())
    }

    pub fn append_raw_dataset(
        &mut self,
        collection: &RawPerceptualCollectionV1,
        collection_signer: &CollectionSigningKeyV1,
        witness_signer: &CollectionSigningKeyV1,
        witness_anchor_reference: &str,
    ) -> Result<(), Vec<PerceptualCollectionAuthenticityIssueV1>> {
        let allowed = self.entries.last().is_some_and(|entry| {
            matches!(
                entry.artifact_kind,
                AuthenticatedCollectionArtifactKindV1::CollectionAuthority
                    | AuthenticatedCollectionArtifactKindV1::SessionHead
                    | AuthenticatedCollectionArtifactKindV1::WithdrawalTombstone
            )
        });
        if !allowed {
            return Err(vec![PerceptualCollectionAuthenticityIssueV1::WrongAppendStage]);
        }
        self.append_exact(
            AuthenticatedCollectionArtifactKindV1::RawDatasetRoot,
            None,
            &collection.raw_dataset_sha256,
            collection_signer,
            witness_signer,
            witness_anchor_reference,
        )
    }

    pub fn append_collection_close(
        &mut self,
        close: &PerceptualCollectionCloseV1,
        collection_signer: &CollectionSigningKeyV1,
        witness_signer: &CollectionSigningKeyV1,
        witness_anchor_reference: &str,
    ) -> Result<(), Vec<PerceptualCollectionAuthenticityIssueV1>> {
        if !self.entries.last().is_some_and(|entry| {
            entry.artifact_kind == AuthenticatedCollectionArtifactKindV1::RawDatasetRoot
        }) {
            return Err(vec![PerceptualCollectionAuthenticityIssueV1::WrongAppendStage]);
        }
        self.append_exact(
            AuthenticatedCollectionArtifactKindV1::CollectionCloseRoot,
            None,
            &close.close_sha256,
            collection_signer,
            witness_signer,
            witness_anchor_reference,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn finish(
        self,
        protocol: &FrozenPerceptualStudyProtocolV1,
        stimulus_pack: &FrozenPerceptualStimulusPackV1,
        render_binding: &FrozenC6fRenderSubjectBindingV1,
        schedule: &PerceptualParticipantScheduleBookV1,
        authority: &FrozenPerceptualCollectionAuthorityV1,
        collection: &RawPerceptualCollectionV1,
        close: &PerceptualCollectionCloseV1,
    ) -> Result<
        FrozenPerceptualCollectionAuthenticityBundleV1,
        Vec<PerceptualCollectionAuthenticityIssueV1>,
    > {
        if !self.entries.last().is_some_and(|entry| {
            entry.artifact_kind == AuthenticatedCollectionArtifactKindV1::CollectionCloseRoot
        }) {
            return Err(vec![PerceptualCollectionAuthenticityIssueV1::WrongAppendStage]);
        }
        let final_log_head_sha256 = self
            .entries
            .last()
            .map(|entry| entry.log_head_sha256.clone())
            .unwrap_or_else(|| ZERO_SHA256.into());
        let mut bundle = FrozenPerceptualCollectionAuthenticityBundleV1 {
            bundle_version: PERCEPTUAL_COLLECTION_AUTHENTICITY_BUNDLE_VERSION.into(),
            policy_sha256: self.policy.policy_sha256.clone(),
            withdrawal_policy_sha256: self.withdrawal_policy.policy_sha256.clone(),
            raw_dataset_sha256: collection.raw_dataset_sha256.clone(),
            collection_close_sha256: close.close_sha256.clone(),
            withdrawal_tombstones: self.withdrawal_tombstones,
            entries: self.entries,
            final_log_head_sha256,
            bundle_sha256: String::new(),
        };
        bundle.bundle_sha256 = authenticity_bundle_commitment(&bundle).map_err(|_| {
            vec![PerceptualCollectionAuthenticityIssueV1::BundleSerializationFailed]
        })?;
        let issues = validate_collection_authenticity_bundle(
            protocol,
            stimulus_pack,
            render_binding,
            schedule,
            authority,
            collection,
            close,
            &self.withdrawal_policy,
            &self.policy,
            &bundle,
        );
        if issues.is_empty() {
            Ok(bundle)
        } else {
            Err(issues)
        }
    }

    fn append_exact(
        &mut self,
        artifact_kind: AuthenticatedCollectionArtifactKindV1,
        participant_token_commitment_sha256: Option<&str>,
        artifact_sha256: &str,
        collection_signer: &CollectionSigningKeyV1,
        witness_signer: &CollectionSigningKeyV1,
        witness_anchor_reference: &str,
    ) -> Result<(), Vec<PerceptualCollectionAuthenticityIssueV1>> {
        let mut issues = Vec::new();
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
        let sequence = self.entries.len() as u64;
        if decode_hex_32(artifact_sha256).is_none() {
            issues.push(PerceptualCollectionAuthenticityIssueV1::InvalidArtifactDigest {
                sequence,
            });
        }
        match artifact_kind {
            AuthenticatedCollectionArtifactKindV1::SessionHead
            | AuthenticatedCollectionArtifactKindV1::WithdrawalTombstone => {
                match participant_token_commitment_sha256 {
                    None => issues.push(
                        PerceptualCollectionAuthenticityIssueV1::EmptyParticipantCommitment {
                            sequence,
                        },
                    ),
                    Some(value) if decode_hex_32(value).is_none() => issues.push(
                        PerceptualCollectionAuthenticityIssueV1::InvalidParticipantCommitment {
                            sequence,
                        },
                    ),
                    Some(_) => {}
                }
            }
            _ if participant_token_commitment_sha256.is_some() => {
                issues.push(
                    PerceptualCollectionAuthenticityIssueV1::UnexpectedParticipantCommitment {
                        sequence,
                    },
                );
            }
            _ => {}
        }
        if witness_anchor_reference.trim().is_empty() {
            issues.push(PerceptualCollectionAuthenticityIssueV1::EmptyWitnessAnchor { sequence });
        }
        if self
            .entries
            .iter()
            .any(|entry| entry.witness_anchor_reference == witness_anchor_reference)
        {
            issues.push(PerceptualCollectionAuthenticityIssueV1::DuplicateWitnessAnchor {
                anchor: witness_anchor_reference.into(),
            });
        }
        if !issues.is_empty() {
            return Err(issues);
        }

        let previous_log_head_sha256 = self
            .entries
            .last()
            .map(|entry| entry.log_head_sha256.clone())
            .unwrap_or_else(|| ZERO_SHA256.into());
        let unsigned = UnsignedAppendEntryV1 {
            sequence,
            artifact_kind,
            participant_token_commitment_sha256,
            artifact_sha256,
            previous_log_head_sha256: &previous_log_head_sha256,
        };
        let log_head_sha256 = canonical_json_sha256(&unsigned).map_err(|_| {
            vec![PerceptualCollectionAuthenticityIssueV1::BundleSerializationFailed]
        })?;
        let log_head_bytes =
            decode_hex_32(&log_head_sha256).expect("canonical SHA-256 commitment must decode");
        let evidence_signature = collection_signer
            .sign_domain(evidence_domain(artifact_kind), &log_head_bytes)
            .map_err(|issue| vec![issue])?;

        let witness_statement = WitnessStatementV1 {
            witness_log_id: &self.policy.witness_log_id,
            sequence,
            log_head_sha256: &log_head_sha256,
            witness_anchor_reference,
        };
        let witness_message = canonical_json_bytes(&witness_statement).map_err(|_| {
            vec![PerceptualCollectionAuthenticityIssueV1::BundleSerializationFailed]
        })?;
        let witness_signature = witness_signer
            .sign_domain(WITNESS_DOMAIN, &witness_message)
            .map_err(|issue| vec![issue])?;

        self.entries.push(AuthenticatedCollectionAppendReceiptV1 {
            sequence,
            artifact_kind,
            participant_token_commitment_sha256: participant_token_commitment_sha256
                .map(str::to_owned),
            artifact_sha256: artifact_sha256.into(),
            previous_log_head_sha256,
            log_head_sha256,
            evidence_signer_id: self.policy.collection_signer.signer_id.clone(),
            evidence_key_epoch: self.policy.collection_signer.key_epoch,
            evidence_signature,
            witness_anchor_reference: witness_anchor_reference.into(),
            witness_signer_id: self.policy.witness_signer.signer_id.clone(),
            witness_key_epoch: self.policy.witness_signer.key_epoch,
            witness_signature,
        });
        Ok(())
    }
}

pub fn participant_token_commitment(participant_token: &str) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ParticipantTokenCommitmentV1 {
        domain: PARTICIPANT_TOKEN_COMMITMENT_DOMAIN,
        participant_token,
    })
}

pub fn seal_withdrawal_policy(
    policy: &mut FrozenPerceptualWithdrawalPolicyV1,
) -> Result<(), serde_json::Error> {
    policy.policy_sha256 = withdrawal_policy_commitment(policy)?;
    Ok(())
}

pub fn withdrawal_policy_commitment(
    policy: &FrozenPerceptualWithdrawalPolicyV1,
) -> Result<String, serde_json::Error> {
    let mut unsigned = policy.clone();
    unsigned.policy_sha256.clear();
    canonical_json_sha256(&unsigned)
}

pub fn withdrawal_tombstone_commitment(
    tombstone: &AuthenticatedWithdrawalTombstoneV1,
) -> Result<String, serde_json::Error> {
    let mut unsigned = tombstone.clone();
    unsigned.tombstone_sha256.clear();
    canonical_json_sha256(&unsigned)
}

pub fn seal_authenticity_policy(
    policy: &mut FrozenPerceptualCollectionAuthenticityPolicyV1,
) -> Result<(), serde_json::Error> {
    policy.policy_sha256 = authenticity_policy_commitment(policy)?;
    Ok(())
}

pub fn authenticity_policy_commitment(
    policy: &FrozenPerceptualCollectionAuthenticityPolicyV1,
) -> Result<String, serde_json::Error> {
    let mut unsigned = policy.clone();
    unsigned.policy_sha256.clear();
    canonical_json_sha256(&unsigned)
}

pub fn authenticity_bundle_commitment(
    bundle: &FrozenPerceptualCollectionAuthenticityBundleV1,
) -> Result<String, serde_json::Error> {
    let mut unsigned = bundle.clone();
    unsigned.bundle_sha256.clear();
    canonical_json_sha256(&unsigned)
}

pub fn validate_withdrawal_policy(
    policy: &FrozenPerceptualWithdrawalPolicyV1,
    authority: &FrozenPerceptualCollectionAuthorityV1,
) -> Vec<PerceptualCollectionAuthenticityIssueV1> {
    let mut issues = Vec::new();
    if policy.policy_version != PERCEPTUAL_WITHDRAWAL_POLICY_VERSION {
        issues.push(PerceptualCollectionAuthenticityIssueV1::WrongWithdrawalPolicyVersion);
    }
    if policy.collection_authority_sha256 != authority.authority_sha256 {
        issues.push(PerceptualCollectionAuthenticityIssueV1::AuthorityDigestMismatch);
    }
    for (field, found, expected) in [
        (
            "consent_form_sha256",
            policy.consent_form_sha256.as_str(),
            authority.consent_form_sha256.as_str(),
        ),
        (
            "participant_information_sha256",
            policy.participant_information_sha256.as_str(),
            authority.participant_information_sha256.as_str(),
        ),
        (
            "privacy_notice_sha256",
            policy.privacy_notice_sha256.as_str(),
            authority.privacy_notice_sha256.as_str(),
        ),
        (
            "human_study_review_evidence_sha256",
            policy.human_study_review_evidence_sha256.as_str(),
            authority.human_study_review_evidence_sha256.as_str(),
        ),
    ] {
        if found != expected {
            issues.push(
                PerceptualCollectionAuthenticityIssueV1::GovernanceDigestMismatch {
                    field: field.into(),
                },
            );
        }
    }
    for (field, digest) in [
        (
            "collection_chronology_policy_sha256",
            policy.collection_chronology_policy_sha256.as_str(),
        ),
        ("policy_sha256", policy.policy_sha256.as_str()),
    ] {
        if decode_hex_32(digest).is_none() {
            issues.push(
                PerceptualCollectionAuthenticityIssueV1::InvalidWithdrawalEvidenceDigest {
                    field: field.into(),
                },
            );
        }
    }
    if policy.retention_mode
        == WithdrawalEvidenceRetentionModeV1::FullErasureRequiresSeparateCustodialEvidence
    {
        issues.push(PerceptualCollectionAuthenticityIssueV1::UnsupportedFullErasureRetentionMode);
    }
    for (field, enabled) in [
        (
            "withdrawal_allowed_until_collection_close",
            policy.withdrawal_allowed_until_collection_close,
        ),
        (
            "raw_session_data_deleted_on_withdrawal",
            policy.raw_session_data_deleted_on_withdrawal,
        ),
        (
            "raw_participant_token_prohibited_in_authenticity_log",
            policy.raw_participant_token_prohibited_in_authenticity_log,
        ),
        (
            "outcome_aware_withdrawal_processing_prohibited",
            policy.outcome_aware_withdrawal_processing_prohibited,
        ),
        ("arm_label_access_prohibited", policy.arm_label_access_prohibited),
    ] {
        if !enabled {
            issues.push(
                PerceptualCollectionAuthenticityIssueV1::MissingWithdrawalProtection {
                    field: field.into(),
                },
            );
        }
    }
    validate_verifier_identity(&policy.withdrawal_authority, &mut issues);
    if decode_hex_32(&policy.policy_sha256).is_none() {
        issues.push(PerceptualCollectionAuthenticityIssueV1::InvalidWithdrawalPolicyDigest);
    }
    match withdrawal_policy_commitment(policy) {
        Ok(value) if value == policy.policy_sha256 => {}
        Ok(_) => issues.push(PerceptualCollectionAuthenticityIssueV1::WithdrawalPolicyDigestMismatch),
        Err(_) => {
            issues.push(PerceptualCollectionAuthenticityIssueV1::WithdrawalPolicySerializationFailed)
        }
    }
    issues
}

pub fn validate_authenticity_policy(
    policy: &FrozenPerceptualCollectionAuthenticityPolicyV1,
    withdrawal_policy: &FrozenPerceptualWithdrawalPolicyV1,
    authority: &FrozenPerceptualCollectionAuthorityV1,
) -> Vec<PerceptualCollectionAuthenticityIssueV1> {
    let mut issues = Vec::new();
    if policy.policy_version != PERCEPTUAL_COLLECTION_AUTHENTICITY_POLICY_VERSION {
        issues.push(PerceptualCollectionAuthenticityIssueV1::WrongPolicyVersion);
    }
    if policy.study_domain != PERCEPTUAL_COLLECTION_STUDY_DOMAIN {
        issues.push(PerceptualCollectionAuthenticityIssueV1::WrongStudyDomain);
    }
    if policy.collection_authority_sha256 != authority.authority_sha256 {
        issues.push(PerceptualCollectionAuthenticityIssueV1::AuthorityDigestMismatch);
    }
    if policy.withdrawal_policy_sha256 != withdrawal_policy.policy_sha256 {
        issues.push(PerceptualCollectionAuthenticityIssueV1::WithdrawalPolicyDigestMismatch);
    }
    validate_verifier_identity(&policy.collection_signer, &mut issues);
    validate_verifier_identity(&policy.witness_signer, &mut issues);
    validate_identity_pair(&policy.collection_signer, &policy.witness_signer, &mut issues);
    validate_withdrawal_authority_distinct(
        &policy.collection_signer,
        &policy.witness_signer,
        &withdrawal_policy.withdrawal_authority,
        &mut issues,
    );
    if policy.witness_log_id.trim().is_empty() {
        issues.push(PerceptualCollectionAuthenticityIssueV1::EmptyWitnessLogId);
    }
    if !policy.per_entry_external_witness_required {
        issues.push(PerceptualCollectionAuthenticityIssueV1::MissingPerEntryWitnessRequirement);
    }
    if !policy.unblinding_requires_verified_close_anchor {
        issues.push(PerceptualCollectionAuthenticityIssueV1::MissingUnblindingGateRequirement);
    }
    if decode_hex_32(&policy.policy_sha256).is_none() {
        issues.push(PerceptualCollectionAuthenticityIssueV1::InvalidPolicyDigest);
    }
    match authenticity_policy_commitment(policy) {
        Ok(value) if value == policy.policy_sha256 => {}
        Ok(_) => issues.push(PerceptualCollectionAuthenticityIssueV1::PolicyDigestMismatch),
        Err(_) => issues.push(PerceptualCollectionAuthenticityIssueV1::PolicySerializationFailed),
    }
    issues
}

#[allow(clippy::too_many_arguments)]
pub fn validate_collection_authenticity_bundle(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    schedule: &PerceptualParticipantScheduleBookV1,
    authority: &FrozenPerceptualCollectionAuthorityV1,
    collection: &RawPerceptualCollectionV1,
    close: &PerceptualCollectionCloseV1,
    withdrawal_policy: &FrozenPerceptualWithdrawalPolicyV1,
    policy: &FrozenPerceptualCollectionAuthenticityPolicyV1,
    bundle: &FrozenPerceptualCollectionAuthenticityBundleV1,
) -> Vec<PerceptualCollectionAuthenticityIssueV1> {
    let mut issues = Vec::new();
    if !validate_collection_authority(protocol, stimulus_pack, render_binding, schedule, authority)
        .is_empty()
    {
        issues.push(PerceptualCollectionAuthenticityIssueV1::InvalidCollectionAuthority);
    }
    if !validate_raw_collection(
        protocol,
        stimulus_pack,
        render_binding,
        schedule,
        authority,
        collection,
    )
    .is_empty()
    {
        issues.push(PerceptualCollectionAuthenticityIssueV1::InvalidRawCollection);
    }
    if !validate_collection_close(
        protocol,
        stimulus_pack,
        render_binding,
        schedule,
        authority,
        collection,
        close,
    )
    .is_empty()
    {
        issues.push(PerceptualCollectionAuthenticityIssueV1::InvalidCollectionClose);
    }
    issues.extend(validate_withdrawal_policy(withdrawal_policy, authority));
    issues.extend(validate_authenticity_policy(
        policy,
        withdrawal_policy,
        authority,
    ));

    if bundle.bundle_version != PERCEPTUAL_COLLECTION_AUTHENTICITY_BUNDLE_VERSION {
        issues.push(PerceptualCollectionAuthenticityIssueV1::WrongBundleVersion);
    }
    if bundle.policy_sha256 != policy.policy_sha256 {
        issues.push(PerceptualCollectionAuthenticityIssueV1::BundlePolicyDigestMismatch);
    }
    if bundle.withdrawal_policy_sha256 != withdrawal_policy.policy_sha256 {
        issues.push(PerceptualCollectionAuthenticityIssueV1::BundleWithdrawalPolicyDigestMismatch);
    }
    if bundle.raw_dataset_sha256 != collection.raw_dataset_sha256 {
        issues.push(PerceptualCollectionAuthenticityIssueV1::RawDatasetDigestMismatch);
    }
    if bundle.collection_close_sha256 != close.close_sha256 {
        issues.push(PerceptualCollectionAuthenticityIssueV1::CollectionCloseDigestMismatch);
    }

    let scheduled_commitments: BTreeSet<String> = schedule
        .schedules
        .iter()
        .filter_map(|entry| participant_token_commitment(&entry.participant_token).ok())
        .collect();
    let retained_sessions: BTreeMap<String, &PerceptualSessionEvidenceV1> = collection
        .sessions
        .iter()
        .filter_map(|session| {
            participant_token_commitment(&session.participant_token)
                .ok()
                .map(|commitment| (commitment, session))
        })
        .collect();

    let mut tombstones_by_commitment = BTreeMap::new();
    let mut tombstones_by_digest = BTreeMap::new();
    let mut post_session_withdrawal_count = 0usize;
    for tombstone in &bundle.withdrawal_tombstones {
        validate_withdrawal_tombstone(tombstone, withdrawal_policy, &mut issues);
        let commitment = tombstone.participant_token_commitment_sha256.clone();
        if tombstones_by_commitment
            .insert(commitment.clone(), tombstone)
            .is_some()
        {
            issues.push(
                PerceptualCollectionAuthenticityIssueV1::DuplicateWithdrawalTombstone {
                    participant_commitment: commitment.clone(),
                },
            );
        }
        if tombstones_by_digest
            .insert(tombstone.tombstone_sha256.clone(), tombstone)
            .is_some()
        {
            issues.push(
                PerceptualCollectionAuthenticityIssueV1::WithdrawalTombstoneDigestMismatch {
                    participant_commitment: commitment.clone(),
                },
            );
        }
        if !scheduled_commitments.contains(&commitment) {
            issues.push(
                PerceptualCollectionAuthenticityIssueV1::UnknownWithdrawalParticipant {
                    participant_commitment: commitment.clone(),
                },
            );
        }
        if retained_sessions.contains_key(&commitment) {
            issues.push(
                PerceptualCollectionAuthenticityIssueV1::WithdrawalSessionStillRetained {
                    participant_commitment: commitment.clone(),
                },
            );
        }
        if tombstone.disposition == WithdrawalSessionDispositionV1::AuthenticatedSessionDeleted {
            post_session_withdrawal_count += 1;
        }
    }
    if bundle.withdrawal_tombstones.len() != close.withdrawn_and_deleted_sessions {
        issues.push(PerceptualCollectionAuthenticityIssueV1::WithdrawalCountMismatch {
            found: bundle.withdrawal_tombstones.len(),
            expected: close.withdrawn_and_deleted_sessions,
        });
    }

    let expected_entry_count = collection
        .sessions
        .len()
        .saturating_add(post_session_withdrawal_count)
        .saturating_add(bundle.withdrawal_tombstones.len())
        .saturating_add(3);
    if bundle.entries.len() != expected_entry_count {
        issues.push(PerceptualCollectionAuthenticityIssueV1::WrongEntryCount {
            found: bundle.entries.len(),
            expected: expected_entry_count,
        });
    }

    let mut session_receipts: BTreeMap<String, (usize, String)> = BTreeMap::new();
    let mut appended_tombstone_digests = BTreeSet::new();
    let mut seen_anchors = BTreeSet::new();
    let mut previous = ZERO_SHA256.to_string();
    let mut raw_root_seen = false;

    for (index, receipt) in bundle.entries.iter().enumerate() {
        if receipt.sequence != index as u64 {
            issues.push(PerceptualCollectionAuthenticityIssueV1::SequenceMismatch { index });
        }
        if receipt.previous_log_head_sha256 != previous {
            issues.push(PerceptualCollectionAuthenticityIssueV1::PreviousLogHeadMismatch { index });
        }
        if decode_hex_32(&receipt.artifact_sha256).is_none() {
            issues.push(PerceptualCollectionAuthenticityIssueV1::InvalidArtifactDigest {
                sequence: receipt.sequence,
            });
        }
        if index == 0
            && receipt.artifact_kind != AuthenticatedCollectionArtifactKindV1::CollectionAuthority
        {
            issues.push(PerceptualCollectionAuthenticityIssueV1::ArtifactOrderMismatch { index });
        }
        if raw_root_seen
            && index + 1 != bundle.entries.len()
            && receipt.artifact_kind != AuthenticatedCollectionArtifactKindV1::CollectionCloseRoot
        {
            issues.push(PerceptualCollectionAuthenticityIssueV1::ArtifactOrderMismatch { index });
        }

        match receipt.artifact_kind {
            AuthenticatedCollectionArtifactKindV1::CollectionAuthority => {
                if index != 0
                    || receipt.participant_token_commitment_sha256.is_some()
                    || receipt.artifact_sha256 != authority.authority_sha256
                {
                    issues.push(PerceptualCollectionAuthenticityIssueV1::ArtifactOrderMismatch {
                        index,
                    });
                }
            }
            AuthenticatedCollectionArtifactKindV1::SessionHead => {
                if raw_root_seen {
                    issues.push(PerceptualCollectionAuthenticityIssueV1::ArtifactOrderMismatch {
                        index,
                    });
                }
                match receipt.participant_token_commitment_sha256.as_deref() {
                    None => issues.push(
                        PerceptualCollectionAuthenticityIssueV1::EmptyParticipantCommitment {
                            sequence: receipt.sequence,
                        },
                    ),
                    Some(commitment) => {
                        if decode_hex_32(commitment).is_none() {
                            issues.push(
                                PerceptualCollectionAuthenticityIssueV1::InvalidParticipantCommitment {
                                    sequence: receipt.sequence,
                                },
                            );
                        }
                        if session_receipts
                            .insert(commitment.into(), (index, receipt.artifact_sha256.clone()))
                            .is_some()
                        {
                            issues.push(
                                PerceptualCollectionAuthenticityIssueV1::DuplicateSessionReceipt {
                                    participant_commitment: commitment.into(),
                                },
                            );
                        }
                    }
                }
            }
            AuthenticatedCollectionArtifactKindV1::WithdrawalTombstone => {
                if raw_root_seen {
                    issues.push(PerceptualCollectionAuthenticityIssueV1::ArtifactOrderMismatch {
                        index,
                    });
                }
                match receipt.participant_token_commitment_sha256.as_deref() {
                    None => issues.push(
                        PerceptualCollectionAuthenticityIssueV1::EmptyParticipantCommitment {
                            sequence: receipt.sequence,
                        },
                    ),
                    Some(commitment) => match tombstones_by_digest.get(&receipt.artifact_sha256) {
                        Some(tombstone)
                            if tombstone.participant_token_commitment_sha256 == commitment =>
                        {
                            appended_tombstone_digests.insert(receipt.artifact_sha256.clone());
                        }
                        _ => issues.push(
                            PerceptualCollectionAuthenticityIssueV1::WithdrawalTombstoneDigestMismatch {
                                participant_commitment: commitment.into(),
                            },
                        ),
                    },
                }
            }
            AuthenticatedCollectionArtifactKindV1::RawDatasetRoot => {
                if index + 2 != bundle.entries.len()
                    || receipt.participant_token_commitment_sha256.is_some()
                    || receipt.artifact_sha256 != collection.raw_dataset_sha256
                    || raw_root_seen
                {
                    issues.push(PerceptualCollectionAuthenticityIssueV1::RawDatasetDigestMismatch);
                }
                raw_root_seen = true;
            }
            AuthenticatedCollectionArtifactKindV1::CollectionCloseRoot => {
                if index + 1 != bundle.entries.len()
                    || receipt.participant_token_commitment_sha256.is_some()
                    || receipt.artifact_sha256 != close.close_sha256
                    || !raw_root_seen
                {
                    issues.push(
                        PerceptualCollectionAuthenticityIssueV1::CollectionCloseDigestMismatch,
                    );
                }
            }
        }

        let unsigned = UnsignedAppendEntryV1 {
            sequence: receipt.sequence,
            artifact_kind: receipt.artifact_kind,
            participant_token_commitment_sha256: receipt
                .participant_token_commitment_sha256
                .as_deref(),
            artifact_sha256: &receipt.artifact_sha256,
            previous_log_head_sha256: &receipt.previous_log_head_sha256,
        };
        match canonical_json_sha256(&unsigned) {
            Ok(value) if value == receipt.log_head_sha256 => {}
            _ => issues.push(PerceptualCollectionAuthenticityIssueV1::LogHeadMismatch { index }),
        }

        if receipt.evidence_signer_id != policy.collection_signer.signer_id
            || receipt.evidence_key_epoch != policy.collection_signer.key_epoch
        {
            issues.push(PerceptualCollectionAuthenticityIssueV1::ReceiptSignerMismatch {
                index,
                role: "collection_signer".into(),
            });
        }
        if receipt.witness_signer_id != policy.witness_signer.signer_id
            || receipt.witness_key_epoch != policy.witness_signer.key_epoch
        {
            issues.push(PerceptualCollectionAuthenticityIssueV1::ReceiptSignerMismatch {
                index,
                role: "witness_signer".into(),
            });
        }
        if receipt.witness_anchor_reference.trim().is_empty() {
            issues.push(PerceptualCollectionAuthenticityIssueV1::EmptyWitnessAnchor {
                sequence: receipt.sequence,
            });
        } else if !seen_anchors.insert(receipt.witness_anchor_reference.clone()) {
            issues.push(PerceptualCollectionAuthenticityIssueV1::DuplicateWitnessAnchor {
                anchor: receipt.witness_anchor_reference.clone(),
            });
        }

        if let Some(log_head_bytes) = decode_hex_32(&receipt.log_head_sha256) {
            if verify_domain_signature(
                &policy.collection_signer,
                evidence_domain(receipt.artifact_kind),
                &log_head_bytes,
                &receipt.evidence_signature,
            )
            .is_err()
            {
                issues.push(PerceptualCollectionAuthenticityIssueV1::EvidenceSignatureInvalid {
                    sequence: receipt.sequence,
                });
            }
        } else {
            issues.push(PerceptualCollectionAuthenticityIssueV1::LogHeadMismatch { index });
        }

        let witness_statement = WitnessStatementV1 {
            witness_log_id: &policy.witness_log_id,
            sequence: receipt.sequence,
            log_head_sha256: &receipt.log_head_sha256,
            witness_anchor_reference: &receipt.witness_anchor_reference,
        };
        match canonical_json_bytes(&witness_statement) {
            Ok(message)
                if verify_domain_signature(
                    &policy.witness_signer,
                    WITNESS_DOMAIN,
                    &message,
                    &receipt.witness_signature,
                )
                .is_ok() => {}
            _ => issues.push(PerceptualCollectionAuthenticityIssueV1::WitnessSignatureInvalid {
                sequence: receipt.sequence,
            }),
        }

        previous = receipt.log_head_sha256.clone();
    }

    for (commitment, session) in &retained_sessions {
        match session_receipts.get(commitment) {
            Some((_, digest)) if digest == &session.session_sha256 => {}
            Some(_) => issues.push(PerceptualCollectionAuthenticityIssueV1::SessionDigestMismatch {
                participant_commitment: commitment.clone(),
            }),
            None => issues.push(PerceptualCollectionAuthenticityIssueV1::MissingSessionReceipt {
                participant_commitment: commitment.clone(),
            }),
        }
        if tombstones_by_commitment.contains_key(commitment) {
            issues.push(
                PerceptualCollectionAuthenticityIssueV1::WithdrawalSessionStillRetained {
                    participant_commitment: commitment.clone(),
                },
            );
        }
    }

    for (commitment, tombstone) in &tombstones_by_commitment {
        if !appended_tombstone_digests.contains(&tombstone.tombstone_sha256) {
            issues.push(
                PerceptualCollectionAuthenticityIssueV1::WithdrawalTombstoneNotAppended {
                    participant_commitment: commitment.clone(),
                },
            );
        }
        match tombstone.disposition {
            WithdrawalSessionDispositionV1::NoAuthenticatedSession => {
                if tombstone.prior_session_sha256.is_some()
                    || session_receipts.contains_key(commitment)
                {
                    issues.push(
                        PerceptualCollectionAuthenticityIssueV1::WithdrawalDispositionMismatch {
                            participant_commitment: commitment.clone(),
                        },
                    );
                }
            }
            WithdrawalSessionDispositionV1::AuthenticatedSessionDeleted => {
                let Some(expected_session_sha) = tombstone.prior_session_sha256.as_deref() else {
                    issues.push(
                        PerceptualCollectionAuthenticityIssueV1::WithdrawalDispositionMismatch {
                            participant_commitment: commitment.clone(),
                        },
                    );
                    continue;
                };
                match session_receipts.get(commitment) {
                    Some((session_index, digest)) if digest == expected_session_sha => {
                        let tombstone_index = bundle.entries.iter().position(|entry| {
                            entry.artifact_kind
                                == AuthenticatedCollectionArtifactKindV1::WithdrawalTombstone
                                && entry.artifact_sha256 == tombstone.tombstone_sha256
                        });
                        if tombstone_index.is_none_or(|index| index <= *session_index) {
                            issues.push(
                                PerceptualCollectionAuthenticityIssueV1::WithdrawalSessionNotPreviouslyAuthenticated {
                                    participant_commitment: commitment.clone(),
                                },
                            );
                        }
                    }
                    _ => issues.push(
                        PerceptualCollectionAuthenticityIssueV1::WithdrawalSessionNotPreviouslyAuthenticated {
                            participant_commitment: commitment.clone(),
                        },
                    ),
                }
            }
        }
    }

    for commitment in session_receipts.keys() {
        if !retained_sessions.contains_key(commitment)
            && !tombstones_by_commitment.contains_key(commitment)
        {
            issues.push(PerceptualCollectionAuthenticityIssueV1::UnexpectedSessionReceipt {
                participant_commitment: commitment.clone(),
            });
        }
    }

    if bundle.final_log_head_sha256 != previous {
        issues.push(PerceptualCollectionAuthenticityIssueV1::FinalLogHeadMismatch);
    }
    match authenticity_bundle_commitment(bundle) {
        Ok(value) if value == bundle.bundle_sha256 => {}
        Ok(_) => issues.push(PerceptualCollectionAuthenticityIssueV1::BundleDigestMismatch),
        Err(_) => issues.push(PerceptualCollectionAuthenticityIssueV1::BundleSerializationFailed),
    }
    issues
}

fn validate_withdrawal_tombstone(
    tombstone: &AuthenticatedWithdrawalTombstoneV1,
    policy: &FrozenPerceptualWithdrawalPolicyV1,
    issues: &mut Vec<PerceptualCollectionAuthenticityIssueV1>,
) {
    if tombstone.tombstone_version != PERCEPTUAL_WITHDRAWAL_TOMBSTONE_VERSION {
        issues.push(PerceptualCollectionAuthenticityIssueV1::WrongWithdrawalTombstoneVersion);
    }
    if tombstone.withdrawal_policy_sha256 != policy.policy_sha256 {
        issues.push(PerceptualCollectionAuthenticityIssueV1::WithdrawalPolicyDigestMismatch);
    }
    if decode_hex_32(&tombstone.participant_token_commitment_sha256).is_none() {
        issues.push(PerceptualCollectionAuthenticityIssueV1::InvalidParticipantCommitment {
            sequence: u64::MAX,
        });
    }
    match tombstone.disposition {
        WithdrawalSessionDispositionV1::NoAuthenticatedSession => {
            if tombstone.prior_session_sha256.is_some() {
                issues.push(
                    PerceptualCollectionAuthenticityIssueV1::WithdrawalDispositionMismatch {
                        participant_commitment: tombstone.participant_token_commitment_sha256.clone(),
                    },
                );
            }
        }
        WithdrawalSessionDispositionV1::AuthenticatedSessionDeleted => {
            if tombstone
                .prior_session_sha256
                .as_deref()
                .and_then(decode_hex_32)
                .is_none()
            {
                issues.push(
                    PerceptualCollectionAuthenticityIssueV1::WithdrawalDispositionMismatch {
                        participant_commitment: tombstone.participant_token_commitment_sha256.clone(),
                    },
                );
            }
        }
    }
    for (field, digest) in [
        (
            "withdrawal_event_evidence_sha256",
            tombstone.withdrawal_event_evidence_sha256.as_str(),
        ),
        (
            "data_disposition_receipt_sha256",
            tombstone.data_disposition_receipt_sha256.as_str(),
        ),
        (
            "withdrawal_chronology_event_sha256",
            tombstone.withdrawal_chronology_event_sha256.as_str(),
        ),
        ("tombstone_sha256", tombstone.tombstone_sha256.as_str()),
    ] {
        if decode_hex_32(digest).is_none() {
            issues.push(
                PerceptualCollectionAuthenticityIssueV1::InvalidWithdrawalEvidenceDigest {
                    field: field.into(),
                },
            );
        }
    }
    if tombstone.withdrawal_signer_id != policy.withdrawal_authority.signer_id
        || tombstone.withdrawal_key_epoch != policy.withdrawal_authority.key_epoch
    {
        issues.push(PerceptualCollectionAuthenticityIssueV1::SigningIdentityMismatch {
            role: "withdrawal_signer".into(),
        });
    }
    let statement = WithdrawalStatementV1 {
        withdrawal_policy_sha256: &tombstone.withdrawal_policy_sha256,
        participant_token_commitment_sha256: &tombstone.participant_token_commitment_sha256,
        disposition: tombstone.disposition,
        prior_session_sha256: tombstone.prior_session_sha256.as_deref(),
        withdrawal_event_evidence_sha256: &tombstone.withdrawal_event_evidence_sha256,
        data_disposition_receipt_sha256: &tombstone.data_disposition_receipt_sha256,
        withdrawal_chronology_event_sha256: &tombstone.withdrawal_chronology_event_sha256,
    };
    match canonical_json_bytes(&statement) {
        Ok(message)
            if verify_domain_signature(
                &policy.withdrawal_authority,
                WITHDRAWAL_AUTHORITY_DOMAIN,
                &message,
                &tombstone.withdrawal_signature,
            )
            .is_ok() => {}
        _ => issues.push(PerceptualCollectionAuthenticityIssueV1::WithdrawalSignatureInvalid {
            tombstone_sha256: tombstone.tombstone_sha256.clone(),
        }),
    }
    match withdrawal_tombstone_commitment(tombstone) {
        Ok(value) if value == tombstone.tombstone_sha256 => {}
        _ => issues.push(
            PerceptualCollectionAuthenticityIssueV1::WithdrawalTombstoneDigestMismatch {
                participant_commitment: tombstone.participant_token_commitment_sha256.clone(),
            },
        ),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn authorize_unblinding_after_authenticated_close(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    schedule: &PerceptualParticipantScheduleBookV1,
    authority: &FrozenPerceptualCollectionAuthorityV1,
    collection: &RawPerceptualCollectionV1,
    close: &PerceptualCollectionCloseV1,
    withdrawal_policy: &FrozenPerceptualWithdrawalPolicyV1,
    policy: &FrozenPerceptualCollectionAuthenticityPolicyV1,
    bundle: &FrozenPerceptualCollectionAuthenticityBundleV1,
) -> Result<AuthenticatedUnblindingGateV1, Vec<PerceptualCollectionAuthenticityIssueV1>> {
    let issues = validate_collection_authenticity_bundle(
        protocol,
        stimulus_pack,
        render_binding,
        schedule,
        authority,
        collection,
        close,
        withdrawal_policy,
        policy,
        bundle,
    );
    if !issues.is_empty() {
        return Err(issues);
    }
    let Some(last) = bundle.entries.last() else {
        return Err(vec![PerceptualCollectionAuthenticityIssueV1::FinalLogHeadMismatch]);
    };
    if last.artifact_kind != AuthenticatedCollectionArtifactKindV1::CollectionCloseRoot
        || last.artifact_sha256 != close.close_sha256
    {
        return Err(vec![
            PerceptualCollectionAuthenticityIssueV1::CollectionCloseDigestMismatch,
        ]);
    }
    let mut gate = AuthenticatedUnblindingGateV1 {
        authenticity_bundle_sha256: bundle.bundle_sha256.clone(),
        withdrawal_policy_sha256: withdrawal_policy.policy_sha256.clone(),
        collection_close_sha256: close.close_sha256.clone(),
        verified_close_log_head_sha256: last.log_head_sha256.clone(),
        private_audit_access_may_begin: true,
        randomization_key_reveal_may_begin: true,
        gate_sha256: String::new(),
    };
    gate.gate_sha256 = unblinding_gate_commitment(&gate).map_err(|_| {
        vec![PerceptualCollectionAuthenticityIssueV1::UnblindingGateSerializationFailed]
    })?;
    Ok(gate)
}

pub fn unblinding_gate_commitment(
    gate: &AuthenticatedUnblindingGateV1,
) -> Result<String, serde_json::Error> {
    let mut unsigned = gate.clone();
    unsigned.gate_sha256.clear();
    canonical_json_sha256(&unsigned)
}

fn validate_signing_key_matches(
    role: &str,
    signing_key: &CollectionSigningKeyV1,
    expected: &CollectionVerifierIdentityV1,
    issues: &mut Vec<PerceptualCollectionAuthenticityIssueV1>,
) {
    if signing_key.verifier_identity() != *expected {
        issues.push(PerceptualCollectionAuthenticityIssueV1::SigningIdentityMismatch {
            role: role.into(),
        });
    }
}

fn validate_verifier_identity(
    identity: &CollectionVerifierIdentityV1,
    issues: &mut Vec<PerceptualCollectionAuthenticityIssueV1>,
) {
    if identity.signer_id.trim().is_empty() {
        issues.push(PerceptualCollectionAuthenticityIssueV1::EmptySignerId);
    }
    if identity.key_epoch == 0 {
        issues.push(PerceptualCollectionAuthenticityIssueV1::InvalidKeyEpoch);
    }
    if verifier_from_identity(identity).is_err() {
        issues.push(PerceptualCollectionAuthenticityIssueV1::InvalidVerifyingKey {
            signer_id: identity.signer_id.clone(),
        });
    }
}

fn validate_identity_pair(
    collection: &CollectionVerifierIdentityV1,
    witness: &CollectionVerifierIdentityV1,
    issues: &mut Vec<PerceptualCollectionAuthenticityIssueV1>,
) {
    if collection.signer_id == witness.signer_id
        || collection.verifying_key_bytes == witness.verifying_key_bytes
    {
        issues.push(PerceptualCollectionAuthenticityIssueV1::SignerWitnessNotDistinct);
    }
}

fn validate_withdrawal_authority_distinct(
    collection: &CollectionVerifierIdentityV1,
    witness: &CollectionVerifierIdentityV1,
    withdrawal: &CollectionVerifierIdentityV1,
    issues: &mut Vec<PerceptualCollectionAuthenticityIssueV1>,
) {
    if withdrawal.signer_id == collection.signer_id
        || withdrawal.signer_id == witness.signer_id
        || withdrawal.verifying_key_bytes == collection.verifying_key_bytes
        || withdrawal.verifying_key_bytes == witness.verifying_key_bytes
    {
        issues.push(PerceptualCollectionAuthenticityIssueV1::WithdrawalAuthorityNotDistinct);
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
    let transcript = domain_separated_transcript(domain, message);
    verifier.verify(&transcript, &signature).map_err(|_| ())
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
) -> Result<(), PerceptualCollectionAuthenticityIssueV1> {
    if domain.is_empty() || domain.len() > MAX_SIGNATURE_DOMAIN_BYTES {
        return Err(PerceptualCollectionAuthenticityIssueV1::InvalidSignatureDomain);
    }
    if message.is_empty() || message.len() > MAX_SIGNED_MESSAGE_BYTES {
        return Err(PerceptualCollectionAuthenticityIssueV1::SignatureMessageTooLarge);
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

fn evidence_domain(kind: AuthenticatedCollectionArtifactKindV1) -> &'static [u8] {
    match kind {
        AuthenticatedCollectionArtifactKindV1::CollectionAuthority => AUTHORITY_DOMAIN,
        AuthenticatedCollectionArtifactKindV1::SessionHead => SESSION_DOMAIN,
        AuthenticatedCollectionArtifactKindV1::WithdrawalTombstone => WITHDRAWAL_APPEND_DOMAIN,
        AuthenticatedCollectionArtifactKindV1::RawDatasetRoot => RAW_DATASET_DOMAIN,
        AuthenticatedCollectionArtifactKindV1::CollectionCloseRoot => COLLECTION_CLOSE_DOMAIN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(name: &str, seed_byte: u8) -> CollectionSigningKeyV1 {
        CollectionSigningKeyV1::from_seed(name, 1, [seed_byte; 32]).unwrap()
    }

    #[test]
    fn artifact_domains_are_not_replayable() {
        let signer = key("collection", 7);
        let message = [3u8; 32];
        let signature = signer.sign_domain(SESSION_DOMAIN, &message).unwrap();
        let identity = signer.verifier_identity();
        assert!(verify_domain_signature(&identity, SESSION_DOMAIN, &message, &signature).is_ok());
        assert!(
            verify_domain_signature(&identity, RAW_DATASET_DOMAIN, &message, &signature).is_err()
        );
    }

    #[test]
    fn participant_token_commitment_is_domain_separated_and_not_raw_token() {
        let token = "high-entropy-participant-token-0123456789abcdef";
        let commitment = participant_token_commitment(token).unwrap();
        assert_eq!(commitment.len(), 64);
        assert!(!commitment.contains(token));
        assert_ne!(commitment, canonical_json_sha256(&token).unwrap());
    }

    #[test]
    fn witness_signature_binds_anchor_reference() {
        let witness = key("witness", 9);
        let identity = witness.verifier_identity();
        let head = "a".repeat(64);
        let left = WitnessStatementV1 {
            witness_log_id: "log",
            sequence: 4,
            log_head_sha256: &head,
            witness_anchor_reference: "anchor-a",
        };
        let left_bytes = canonical_json_bytes(&left).unwrap();
        let signature = witness.sign_domain(WITNESS_DOMAIN, &left_bytes).unwrap();
        let right = WitnessStatementV1 {
            witness_log_id: "log",
            sequence: 4,
            log_head_sha256: &head,
            witness_anchor_reference: "anchor-b",
        };
        let right_bytes = canonical_json_bytes(&right).unwrap();
        assert!(
            verify_domain_signature(&identity, WITNESS_DOMAIN, &left_bytes, &signature).is_ok()
        );
        assert!(
            verify_domain_signature(&identity, WITNESS_DOMAIN, &right_bytes, &signature).is_err()
        );
    }

    #[test]
    fn all_three_authorities_must_be_distinct() {
        let collection = key("collection", 7).verifier_identity();
        let witness = key("witness", 9).verifier_identity();
        let withdrawal = key("withdrawal", 11).verifier_identity();
        let mut issues = Vec::new();
        validate_identity_pair(&collection, &witness, &mut issues);
        validate_withdrawal_authority_distinct(
            &collection,
            &witness,
            &withdrawal,
            &mut issues,
        );
        assert!(issues.is_empty());

        validate_withdrawal_authority_distinct(
            &collection,
            &witness,
            &collection,
            &mut issues,
        );
        assert!(issues.contains(
            &PerceptualCollectionAuthenticityIssueV1::WithdrawalAuthorityNotDistinct
        ));
    }

    #[test]
    fn withdrawal_signature_binds_disposition_and_session() {
        let withdrawal = key("withdrawal", 11);
        let identity = withdrawal.verifier_identity();
        let policy = "a".repeat(64);
        let participant = "b".repeat(64);
        let session = "c".repeat(64);
        let event = "d".repeat(64);
        let deletion = "e".repeat(64);
        let chronology = "f".repeat(64);
        let left = WithdrawalStatementV1 {
            withdrawal_policy_sha256: &policy,
            participant_token_commitment_sha256: &participant,
            disposition: WithdrawalSessionDispositionV1::AuthenticatedSessionDeleted,
            prior_session_sha256: Some(&session),
            withdrawal_event_evidence_sha256: &event,
            data_disposition_receipt_sha256: &deletion,
            withdrawal_chronology_event_sha256: &chronology,
        };
        let left_bytes = canonical_json_bytes(&left).unwrap();
        let signature = withdrawal
            .sign_domain(WITHDRAWAL_AUTHORITY_DOMAIN, &left_bytes)
            .unwrap();
        let right = WithdrawalStatementV1 {
            withdrawal_policy_sha256: &policy,
            participant_token_commitment_sha256: &participant,
            disposition: WithdrawalSessionDispositionV1::NoAuthenticatedSession,
            prior_session_sha256: None,
            withdrawal_event_evidence_sha256: &event,
            data_disposition_receipt_sha256: &deletion,
            withdrawal_chronology_event_sha256: &chronology,
        };
        let right_bytes = canonical_json_bytes(&right).unwrap();
        assert!(
            verify_domain_signature(
                &identity,
                WITHDRAWAL_AUTHORITY_DOMAIN,
                &left_bytes,
                &signature
            )
            .is_ok()
        );
        assert!(
            verify_domain_signature(
                &identity,
                WITHDRAWAL_AUTHORITY_DOMAIN,
                &right_bytes,
                &signature
            )
            .is_err()
        );
    }

    #[test]
    fn append_head_is_order_sensitive() {
        let artifact = "a".repeat(64);
        let previous = "b".repeat(64);
        let participant = "c".repeat(64);
        let left = UnsignedAppendEntryV1 {
            sequence: 1,
            artifact_kind: AuthenticatedCollectionArtifactKindV1::SessionHead,
            participant_token_commitment_sha256: Some(&participant),
            artifact_sha256: &artifact,
            previous_log_head_sha256: &previous,
        };
        let right = UnsignedAppendEntryV1 {
            sequence: 2,
            artifact_kind: AuthenticatedCollectionArtifactKindV1::SessionHead,
            participant_token_commitment_sha256: Some(&participant),
            artifact_sha256: &artifact,
            previous_log_head_sha256: &previous,
        };
        assert_ne!(
            canonical_json_sha256(&left).unwrap(),
            canonical_json_sha256(&right).unwrap()
        );
    }

    #[test]
    fn full_erasure_mode_is_explicitly_distinct() {
        assert_ne!(
            WithdrawalEvidenceRetentionModeV1::MinimalAuditCommitment,
            WithdrawalEvidenceRetentionModeV1::FullErasureRequiresSeparateCustodialEvidence
        );
    }

    #[test]
    fn policy_commitment_binds_withdrawal_policy_and_public_keys() {
        let collection = key("collection", 7);
        let witness = key("witness", 9);
        let mut base = FrozenPerceptualCollectionAuthenticityPolicyV1 {
            policy_version: PERCEPTUAL_COLLECTION_AUTHENTICITY_POLICY_VERSION.into(),
            study_domain: PERCEPTUAL_COLLECTION_STUDY_DOMAIN.into(),
            collection_authority_sha256: "a".repeat(64),
            withdrawal_policy_sha256: "b".repeat(64),
            collection_signer: collection.verifier_identity(),
            witness_signer: witness.verifier_identity(),
            witness_log_id: "external-witness-log-test".into(),
            per_entry_external_witness_required: true,
            unblinding_requires_verified_close_anchor: true,
            policy_sha256: String::new(),
        };
        seal_authenticity_policy(&mut base).unwrap();
        let base_digest = base.policy_sha256.clone();
        let mut changed = base.clone();
        changed.withdrawal_policy_sha256 = "c".repeat(64);
        changed.policy_sha256.clear();
        seal_authenticity_policy(&mut changed).unwrap();
        assert_ne!(base_digest, changed.policy_sha256);
    }
}
