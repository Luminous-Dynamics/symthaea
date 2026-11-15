// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Signed, pseudonymous human adjudication for musical edit references.
//! Signatures establish key possession and payload integrity, not identity,
//! independence, competence, legal rights, absence of bias, or musical truth.

use crate::music_edit_benchmark_adapter::{MusicEditOperationV1, MusicEditSplitV1};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const REVIEW_TASK_SCHEMA_V1: &str = "mel-music-reference-review-task-v1";
pub const REVIEW_CREDENTIAL_SCHEMA_V1: &str = "mel-music-reviewer-credential-v1";
pub const REVIEW_TRUST_SCHEMA_V1: &str = "mel-music-review-trust-bundle-v1";
pub const SIGNED_JUDGMENT_SCHEMA_V1: &str = "mel-music-signed-review-v1";
pub const REVIEW_POLICY_SCHEMA_V1: &str = "mel-music-review-panel-policy-v1";
pub const REVIEW_RECEIPT_SCHEMA_V1: &str = "mel-music-review-panel-receipt-v1";

const TASK_DOMAIN: &[u8] = b"luminous.mel.review-task.v1";
const CREDENTIAL_DOMAIN: &[u8] = b"luminous.mel.reviewer-credential.v1";
const TRUST_DOMAIN: &[u8] = b"luminous.mel.review-trust-bundle.v1";
const JUDGMENT_DOMAIN: &[u8] = b"luminous.mel.signed-judgment.v1";
const POLICY_DOMAIN: &[u8] = b"luminous.mel.review-policy.v1";
const RECEIPT_DOMAIN: &[u8] = b"luminous.mel.review-panel-receipt.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicReferenceReviewTaskV1 {
    pub schema_version: String,
    pub task_id: String,
    pub study_id: String,
    pub dataset_id: String,
    pub dataset_version: String,
    pub benchmark_track_id: String,
    pub sample_id: String,
    pub split: MusicEditSplitV1,
    pub source_lineage_sha256: String,
    pub operation: MusicEditOperationV1,
    pub instruction_sha256: String,
    pub target_selector_sha256: String,
    pub validity_envelope_sha256: String,
    pub reference_audio_sha256: String,
    pub renderer_profile_sha256: String,
    pub review_protocol_sha256: String,
    pub consent_scope_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicReviewerRoleV1 { ReferenceAdjudicator }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicReviewerCredentialPayloadV1 {
    pub schema_version: String,
    /// Opaque stable pseudonym. Real identity mapping remains with the issuer.
    pub credential_id: String,
    pub study_id: String,
    pub reviewer_role: MusicReviewerRoleV1,
    pub issuer_id: String,
    pub issuer_key_id: String,
    pub reviewer_public_key_hex: String,
    /// Issuer-attested group and conflict-screening commitments, not proof of independence.
    pub independence_group_sha256: String,
    pub conflict_screening_sha256: String,
    pub valid_from_unix: u64,
    pub valid_until_unix: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicReviewerCredentialV1 {
    pub payload: MusicReviewerCredentialPayloadV1,
    pub issuer_signature_hex: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicReviewTrustRootV1 {
    /// Pinned through study configuration, not provided by the trust bundle itself.
    pub root_id: String,
    pub key_id: String,
    pub public_key_hex: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicReviewIssuerStatusV1 { Active, Revoked }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrustedMusicReviewIssuerV1 {
    pub issuer_id: String,
    pub key_id: String,
    pub public_key_hex: String,
    pub status: MusicReviewIssuerStatusV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicReviewTrustBundlePayloadV1 {
    pub schema_version: String,
    pub bundle_id: String,
    pub issued_at_unix: u64,
    pub expires_at_unix: u64,
    /// Canonical order by issuer ID then key ID.
    pub trusted_issuers: Vec<TrustedMusicReviewIssuerV1>,
    pub revoked_credential_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedMusicReviewTrustBundleV1 {
    pub payload: MusicReviewTrustBundlePayloadV1,
    pub root_signature_hex: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicReferenceReviewAdequacyV1 { Validated, Ambiguous, Inaudible, Invalid }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicReferenceReviewJudgmentVectorV1 {
    pub edit_execution_1_to_5: u8,
    pub source_preservation_1_to_5: u8,
    pub musical_plausibility_1_to_5: u8,
    pub instruction_alignment_1_to_5: u8,
    pub artifact_absence_1_to_5: u8,
}

impl MusicReferenceReviewJudgmentVectorV1 {
    fn valid(self) -> bool {
        (1..=5).contains(&self.edit_execution_1_to_5)
            && (1..=5).contains(&self.source_preservation_1_to_5)
            && (1..=5).contains(&self.musical_plausibility_1_to_5)
            && (1..=5).contains(&self.instruction_alignment_1_to_5)
            && (1..=5).contains(&self.artifact_absence_1_to_5)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicReferenceReviewJudgmentPayloadV1 {
    pub schema_version: String,
    pub task_sha256: String,
    pub credential_sha256: String,
    pub credential_id: String,
    pub reviewer_key_fingerprint_sha256: String,
    pub adequacy: MusicReferenceReviewAdequacyV1,
    pub judgments: MusicReferenceReviewJudgmentVectorV1,
    pub confidence_0_to_100: u8,
    pub consent_scope_sha256: String,
    pub reviewed_at_unix: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedMusicReferenceReviewV1 {
    pub payload: MusicReferenceReviewJudgmentPayloadV1,
    pub reviewer_signature_hex: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicReviewPanelPolicyV1 {
    pub schema_version: String,
    pub policy_id: String,
    pub minimum_reviewers: u16,
    pub minimum_independence_groups: u16,
    pub required_role: MusicReviewerRoleV1,
    /// v1 does not majority-vote adequacy disagreement into Validated.
    pub require_unanimous_adequacy: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicReviewPanelReceiptV1 {
    pub schema_version: String,
    pub task_sha256: String,
    pub trust_root_id: String,
    pub trust_root_key_id: String,
    pub trust_bundle_sha256: String,
    pub policy_sha256: String,
    pub reviewer_count: u16,
    pub distinct_reviewer_key_count: u16,
    pub distinct_independence_group_count: u16,
    pub signed_review_digests: Vec<String>,
    pub reviewer_key_fingerprints: Vec<String>,
    pub independence_group_commitments: Vec<String>,
    pub adequacy: MusicReferenceReviewAdequacyV1,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MusicReferenceReviewIssueV1 {
    WrongSchema(&'static str), EmptyField(&'static str), InvalidDigest, InvalidHex,
    InvalidPublicKey, InvalidSignature, UnknownIssuer, RevokedIssuer, RevokedCredential,
    NonCanonicalIssuerOrder, NonCanonicalRevocationOrder, NonCanonicalCredentialOrder,
    CredentialOutsideValidity, CredentialWrongStudy, CredentialWrongRole, TaskMismatch,
    ConsentScopeMismatch, JudgmentOutsideCredentialValidity, FutureDatedReview,
    InvalidJudgmentVector, InvalidConfidence, DuplicateReviewerKey, TooManyReviewers,
    InsufficientReviewers, InsufficientIndependenceGroups, InvalidPolicy,
    TrustBundleExpiredOrNotYetValid, ReceiptMismatch, SerializationError,
}

impl std::fmt::Display for MusicReferenceReviewIssueV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{self:?}") }
}
impl std::error::Error for MusicReferenceReviewIssueV1 {}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(HEX[(byte >> 4) as usize] as char);
        result.push(HEX[(byte & 0x0f) as usize] as char);
    }
    result
}

fn decode_hex(value: &str) -> Result<Vec<u8>, MusicReferenceReviewIssueV1> {
    if value.is_empty() || value.len() % 2 != 0 {
        return Err(MusicReferenceReviewIssueV1::InvalidHex);
    }
    let mut result = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        let high = decode_nibble(pair[0])?;
        let low = decode_nibble(pair[1])?;
        result.push((high << 4) | low);
    }
    Ok(result)
}

fn decode_nibble(value: u8) -> Result<u8, MusicReferenceReviewIssueV1> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        _ => Err(MusicReferenceReviewIssueV1::InvalidHex),
    }
}

fn digest<T: Serialize>(domain: &[u8], value: &T) -> Result<String, MusicReferenceReviewIssueV1> {
    let bytes = serde_json::to_vec(value).map_err(|_| MusicReferenceReviewIssueV1::SerializationError)?;
    let mut hasher = Sha256::new();
    hasher.update(b"luminous.mel.signed-reference-review.v1\0");
    hasher.update(domain);
    hasher.update(b"\0");
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn message<T: Serialize>(domain: &[u8], value: &T) -> Result<Vec<u8>, MusicReferenceReviewIssueV1> {
    let bytes = serde_json::to_vec(value).map_err(|_| MusicReferenceReviewIssueV1::SerializationError)?;
    let mut result = Vec::with_capacity(domain.len() + bytes.len() + 64);
    result.extend_from_slice(b"luminous.mel.signed-reference-review.v1\0");
    result.extend_from_slice(domain);
    result.push(0);
    result.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    result.extend_from_slice(&bytes);
    Ok(result)
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn require(value: &str, field: &'static str) -> Result<(), MusicReferenceReviewIssueV1> {
    if value.trim().is_empty() { return Err(MusicReferenceReviewIssueV1::EmptyField(field)); }
    Ok(())
}

fn require_digest(value: &str) -> Result<(), MusicReferenceReviewIssueV1> {
    if !valid_digest(value) { return Err(MusicReferenceReviewIssueV1::InvalidDigest); }
    Ok(())
}

fn public_key(hex: &str) -> Result<VerifyingKey, MusicReferenceReviewIssueV1> {
    let bytes = decode_hex(hex)?;
    let array: [u8; 32] = bytes.try_into().map_err(|_| MusicReferenceReviewIssueV1::InvalidPublicKey)?;
    VerifyingKey::from_bytes(&array).map_err(|_| MusicReferenceReviewIssueV1::InvalidPublicKey)
}

fn verify_signature(key_hex: &str, data: &[u8], signature_hex: &str) -> Result<(), MusicReferenceReviewIssueV1> {
    let key = public_key(key_hex)?;
    let bytes = decode_hex(signature_hex)?;
    let signature = Signature::from_slice(&bytes).map_err(|_| MusicReferenceReviewIssueV1::InvalidSignature)?;
    key.verify(data, &signature).map_err(|_| MusicReferenceReviewIssueV1::InvalidSignature)
}

pub fn music_reference_review_task_digest_v1(
    task: &MusicReferenceReviewTaskV1,
) -> Result<String, MusicReferenceReviewIssueV1> {
    if task.schema_version != REVIEW_TASK_SCHEMA_V1 { return Err(MusicReferenceReviewIssueV1::WrongSchema("task")); }
    for (value, field) in [
        (task.task_id.as_str(), "task_id"), (task.study_id.as_str(), "study_id"),
        (task.dataset_id.as_str(), "dataset_id"), (task.dataset_version.as_str(), "dataset_version"),
        (task.benchmark_track_id.as_str(), "benchmark_track_id"), (task.sample_id.as_str(), "sample_id"),
    ] { require(value, field)?; }
    for hash in [
        &task.source_lineage_sha256, &task.instruction_sha256, &task.target_selector_sha256,
        &task.validity_envelope_sha256, &task.reference_audio_sha256, &task.renderer_profile_sha256,
        &task.review_protocol_sha256, &task.consent_scope_sha256,
    ] { require_digest(hash)?; }
    digest(TASK_DOMAIN, task)
}

pub fn music_reviewer_credential_digest_v1(
    credential: &MusicReviewerCredentialV1,
) -> Result<String, MusicReferenceReviewIssueV1> {
    let p = &credential.payload;
    if p.schema_version != REVIEW_CREDENTIAL_SCHEMA_V1 { return Err(MusicReferenceReviewIssueV1::WrongSchema("credential")); }
    for (value, field) in [
        (p.credential_id.as_str(), "credential_id"), (p.study_id.as_str(), "credential_study_id"),
        (p.issuer_id.as_str(), "issuer_id"), (p.issuer_key_id.as_str(), "issuer_key_id"),
    ] { require(value, field)?; }
    let _ = public_key(&p.reviewer_public_key_hex)?;
    require_digest(&p.independence_group_sha256)?;
    require_digest(&p.conflict_screening_sha256)?;
    if p.valid_until_unix <= p.valid_from_unix { return Err(MusicReferenceReviewIssueV1::CredentialOutsideValidity); }
    digest(CREDENTIAL_DOMAIN, &credential.payload)
}

fn validate_trust_bundle(
    bundle: &SignedMusicReviewTrustBundleV1,
    root: &MusicReviewTrustRootV1,
    now: u64,
) -> Result<String, MusicReferenceReviewIssueV1> {
    require(&root.root_id, "trust_root_id")?;
    require(&root.key_id, "trust_root_key_id")?;
    let p = &bundle.payload;
    if p.schema_version != REVIEW_TRUST_SCHEMA_V1 { return Err(MusicReferenceReviewIssueV1::WrongSchema("trust bundle")); }
    require(&p.bundle_id, "trust_bundle_id")?;
    if p.expires_at_unix <= p.issued_at_unix || now < p.issued_at_unix || now > p.expires_at_unix {
        return Err(MusicReferenceReviewIssueV1::TrustBundleExpiredOrNotYetValid);
    }
    if !p.trusted_issuers.windows(2).all(|x| (&x[0].issuer_id, &x[0].key_id) < (&x[1].issuer_id, &x[1].key_id)) {
        return Err(MusicReferenceReviewIssueV1::NonCanonicalIssuerOrder);
    }
    for issuer in &p.trusted_issuers {
        require(&issuer.issuer_id, "issuer_id")?;
        require(&issuer.key_id, "issuer_key_id")?;
        let _ = public_key(&issuer.public_key_hex)?;
    }
    if !p.revoked_credential_ids.windows(2).all(|x| x[0] < x[1]) {
        return Err(MusicReferenceReviewIssueV1::NonCanonicalRevocationOrder);
    }
    let signed = message(TRUST_DOMAIN, p)?;
    verify_signature(&root.public_key_hex, &signed, &bundle.root_signature_hex)?;
    digest(TRUST_DOMAIN, bundle)
}

fn verify_credential(
    credential: &MusicReviewerCredentialV1,
    task: &MusicReferenceReviewTaskV1,
    trust: &SignedMusicReviewTrustBundleV1,
    role: MusicReviewerRoleV1,
) -> Result<(String, String, String), MusicReferenceReviewIssueV1> {
    let p = &credential.payload;
    let credential_sha = music_reviewer_credential_digest_v1(credential)?;
    if p.study_id != task.study_id { return Err(MusicReferenceReviewIssueV1::CredentialWrongStudy); }
    if p.reviewer_role != role { return Err(MusicReferenceReviewIssueV1::CredentialWrongRole); }
    if trust.payload.revoked_credential_ids.binary_search(&p.credential_id).is_ok() {
        return Err(MusicReferenceReviewIssueV1::RevokedCredential);
    }
    let issuer = trust.payload.trusted_issuers.iter().find(|i| i.issuer_id == p.issuer_id && i.key_id == p.issuer_key_id)
        .ok_or(MusicReferenceReviewIssueV1::UnknownIssuer)?;
    if issuer.status == MusicReviewIssuerStatusV1::Revoked { return Err(MusicReferenceReviewIssueV1::RevokedIssuer); }
    let signed = message(CREDENTIAL_DOMAIN, p)?;
    verify_signature(&issuer.public_key_hex, &signed, &credential.issuer_signature_hex)?;
    let reviewer_key = public_key(&p.reviewer_public_key_hex)?;
    let fingerprint = format!("{:x}", Sha256::digest(reviewer_key.as_bytes()));
    Ok((credential_sha, fingerprint, p.independence_group_sha256.clone()))
}

fn validate_judgment(
    p: &MusicReferenceReviewJudgmentPayloadV1,
    task: &MusicReferenceReviewTaskV1,
    task_sha: &str,
    credential: &MusicReviewerCredentialPayloadV1,
    credential_sha: &str,
    fingerprint: &str,
    now: u64,
) -> Result<(), MusicReferenceReviewIssueV1> {
    if p.schema_version != SIGNED_JUDGMENT_SCHEMA_V1 { return Err(MusicReferenceReviewIssueV1::WrongSchema("judgment")); }
    if p.task_sha256 != task_sha || p.credential_sha256 != credential_sha
        || p.credential_id != credential.credential_id || p.reviewer_key_fingerprint_sha256 != fingerprint {
        return Err(MusicReferenceReviewIssueV1::TaskMismatch);
    }
    if p.consent_scope_sha256 != task.consent_scope_sha256 { return Err(MusicReferenceReviewIssueV1::ConsentScopeMismatch); }
    if p.reviewed_at_unix < credential.valid_from_unix || p.reviewed_at_unix > credential.valid_until_unix {
        return Err(MusicReferenceReviewIssueV1::JudgmentOutsideCredentialValidity);
    }
    if p.reviewed_at_unix > now { return Err(MusicReferenceReviewIssueV1::FutureDatedReview); }
    if !p.judgments.valid() { return Err(MusicReferenceReviewIssueV1::InvalidJudgmentVector); }
    if p.confidence_0_to_100 > 100 { return Err(MusicReferenceReviewIssueV1::InvalidConfidence); }
    Ok(())
}

/// Verify issuer-signed pseudonymous credentials and reviewer-signed judgments.
/// Adequacy disagreement becomes Ambiguous; v1 never majority-votes it away.
pub fn evaluate_music_reference_review_panel_v1(
    task: &MusicReferenceReviewTaskV1,
    credentials: &[MusicReviewerCredentialV1],
    judgments: &[SignedMusicReferenceReviewV1],
    trust_bundle: &SignedMusicReviewTrustBundleV1,
    trust_root: &MusicReviewTrustRootV1,
    policy: &MusicReviewPanelPolicyV1,
    now: u64,
) -> Result<MusicReviewPanelReceiptV1, Vec<MusicReferenceReviewIssueV1>> {
    let task_sha = music_reference_review_task_digest_v1(task).map_err(|e| vec![e])?;
    let trust_sha = validate_trust_bundle(trust_bundle, trust_root, now).map_err(|e| vec![e])?;
    if policy.schema_version != REVIEW_POLICY_SCHEMA_V1 { return Err(vec![MusicReferenceReviewIssueV1::WrongSchema("policy")]); }
    require(&policy.policy_id, "panel_policy_id").map_err(|e| vec![e])?;
    if policy.minimum_reviewers < 2 || policy.minimum_independence_groups < 2 || !policy.require_unanimous_adequacy {
        return Err(vec![MusicReferenceReviewIssueV1::InvalidPolicy]);
    }
    let policy_sha = digest(POLICY_DOMAIN, policy).map_err(|e| vec![e])?;
    if credentials.len() != judgments.len() || credentials.len() < policy.minimum_reviewers as usize {
        return Err(vec![MusicReferenceReviewIssueV1::InsufficientReviewers]);
    }
    if credentials.len() > u16::MAX as usize { return Err(vec![MusicReferenceReviewIssueV1::TooManyReviewers]); }
    if !credentials.windows(2).all(|p| p[0].payload.credential_id < p[1].payload.credential_id)
        || !judgments.windows(2).all(|p| p[0].payload.credential_id < p[1].payload.credential_id) {
        return Err(vec![MusicReferenceReviewIssueV1::NonCanonicalCredentialOrder]);
    }
    let mut keys = BTreeSet::new();
    let mut groups = BTreeSet::new();
    let mut review_digests = Vec::with_capacity(judgments.len());
    let mut fingerprints = Vec::with_capacity(judgments.len());
    let mut group_hashes = Vec::with_capacity(judgments.len());
    let mut adequacies = Vec::with_capacity(judgments.len());

    for (credential, judgment) in credentials.iter().zip(judgments) {
        if credential.payload.credential_id != judgment.payload.credential_id {
            return Err(vec![MusicReferenceReviewIssueV1::TaskMismatch]);
        }
        let (credential_sha, fingerprint, group) = verify_credential(
            credential, task, trust_bundle, policy.required_role,
        ).map_err(|e| vec![e])?;
        if !keys.insert(fingerprint.clone()) { return Err(vec![MusicReferenceReviewIssueV1::DuplicateReviewerKey]); }
        validate_judgment(&judgment.payload, task, &task_sha, &credential.payload,
            &credential_sha, &fingerprint, now).map_err(|e| vec![e])?;
        let signed = message(JUDGMENT_DOMAIN, &judgment.payload).map_err(|e| vec![e])?;
        verify_signature(&credential.payload.reviewer_public_key_hex, &signed,
            &judgment.reviewer_signature_hex).map_err(|e| vec![e])?;
        groups.insert(group.clone());
        review_digests.push(digest(JUDGMENT_DOMAIN, judgment).map_err(|e| vec![e])?);
        fingerprints.push(fingerprint);
        group_hashes.push(group);
        adequacies.push(judgment.payload.adequacy);
    }
    if groups.len() < policy.minimum_independence_groups as usize {
        return Err(vec![MusicReferenceReviewIssueV1::InsufficientIndependenceGroups]);
    }
    let first = adequacies[0];
    let adequacy = if adequacies.iter().all(|v| *v == first) { first } else { MusicReferenceReviewAdequacyV1::Ambiguous };
    let mut receipt = MusicReviewPanelReceiptV1 {
        schema_version: REVIEW_RECEIPT_SCHEMA_V1.to_owned(),
        task_sha256: task_sha,
        trust_root_id: trust_root.root_id.clone(),
        trust_root_key_id: trust_root.key_id.clone(),
        trust_bundle_sha256: trust_sha,
        policy_sha256: policy_sha,
        reviewer_count: judgments.len() as u16,
        distinct_reviewer_key_count: keys.len() as u16,
        distinct_independence_group_count: groups.len() as u16,
        signed_review_digests: review_digests,
        reviewer_key_fingerprints: fingerprints,
        independence_group_commitments: group_hashes,
        adequacy,
        receipt_sha256: String::new(),
    };
    receipt.receipt_sha256 = digest(RECEIPT_DOMAIN, &receipt).map_err(|e| vec![e])?;
    Ok(receipt)
}

pub fn verify_music_reference_review_panel_receipt_v1(
    receipt: &MusicReviewPanelReceiptV1,
    task: &MusicReferenceReviewTaskV1,
    credentials: &[MusicReviewerCredentialV1],
    judgments: &[SignedMusicReferenceReviewV1],
    trust_bundle: &SignedMusicReviewTrustBundleV1,
    trust_root: &MusicReviewTrustRootV1,
    policy: &MusicReviewPanelPolicyV1,
    now: u64,
) -> Vec<MusicReferenceReviewIssueV1> {
    match evaluate_music_reference_review_panel_v1(task, credentials, judgments, trust_bundle, trust_root, policy, now) {
        Ok(expected) if &expected == receipt => Vec::new(),
        Ok(_) => vec![MusicReferenceReviewIssueV1::ReceiptMismatch],
        Err(issues) => issues,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn hash(byte: u8) -> String { format!("{byte:02x}").repeat(32) }
    fn key(seed: u8) -> SigningKey { SigningKey::from_bytes(&[seed; 32]) }

    fn task() -> MusicReferenceReviewTaskV1 {
        MusicReferenceReviewTaskV1 {
            schema_version: REVIEW_TASK_SCHEMA_V1.into(),
            task_id: "task".into(), study_id: "study".into(), dataset_id: "dataset".into(),
            dataset_version: "v1".into(), benchmark_track_id: "dev".into(), sample_id: "item-1".into(),
            split: MusicEditSplitV1::Development, source_lineage_sha256: hash(1),
            operation: MusicEditOperationV1::ReplaceInstrument, instruction_sha256: hash(2),
            target_selector_sha256: hash(3), validity_envelope_sha256: hash(4),
            reference_audio_sha256: hash(5), renderer_profile_sha256: hash(6),
            review_protocol_sha256: hash(7), consent_scope_sha256: hash(8),
        }
    }

    fn credential(id: &str, reviewer_seed: u8, group: u8, issuer: &SigningKey) -> MusicReviewerCredentialV1 {
        let reviewer = key(reviewer_seed);
        let payload = MusicReviewerCredentialPayloadV1 {
            schema_version: REVIEW_CREDENTIAL_SCHEMA_V1.into(), credential_id: id.into(),
            study_id: "study".into(), reviewer_role: MusicReviewerRoleV1::ReferenceAdjudicator,
            issuer_id: "issuer".into(), issuer_key_id: "issuer-key".into(),
            reviewer_public_key_hex: encode_hex(&reviewer.verifying_key().to_bytes()),
            independence_group_sha256: hash(group), conflict_screening_sha256: hash(group.wrapping_add(30)),
            valid_from_unix: 1, valid_until_unix: 10_000,
        };
        let signed = message(CREDENTIAL_DOMAIN, &payload).unwrap();
        MusicReviewerCredentialV1 { payload, issuer_signature_hex: encode_hex(&issuer.sign(&signed).to_bytes()) }
    }

    fn judgment(task: &MusicReferenceReviewTaskV1, credential: &MusicReviewerCredentialV1,
        reviewer_seed: u8, adequacy: MusicReferenceReviewAdequacyV1) -> SignedMusicReferenceReviewV1 {
        let reviewer = key(reviewer_seed);
        let payload = MusicReferenceReviewJudgmentPayloadV1 {
            schema_version: SIGNED_JUDGMENT_SCHEMA_V1.into(),
            task_sha256: music_reference_review_task_digest_v1(task).unwrap(),
            credential_sha256: music_reviewer_credential_digest_v1(credential).unwrap(),
            credential_id: credential.payload.credential_id.clone(),
            reviewer_key_fingerprint_sha256: format!("{:x}", Sha256::digest(reviewer.verifying_key().as_bytes())),
            adequacy,
            judgments: MusicReferenceReviewJudgmentVectorV1 {
                edit_execution_1_to_5: 5, source_preservation_1_to_5: 5,
                musical_plausibility_1_to_5: 4, instruction_alignment_1_to_5: 5,
                artifact_absence_1_to_5: 4,
            },
            confidence_0_to_100: 90, consent_scope_sha256: task.consent_scope_sha256.clone(),
            reviewed_at_unix: 90,
        };
        let signed = message(JUDGMENT_DOMAIN, &payload).unwrap();
        SignedMusicReferenceReviewV1 { payload, reviewer_signature_hex: encode_hex(&reviewer.sign(&signed).to_bytes()) }
    }

    fn bundle(issuer: &SigningKey, root: &SigningKey, revoked: Vec<String>,
        status: MusicReviewIssuerStatusV1) -> SignedMusicReviewTrustBundleV1 {
        let payload = MusicReviewTrustBundlePayloadV1 {
            schema_version: REVIEW_TRUST_SCHEMA_V1.into(), bundle_id: "bundle-v1".into(),
            issued_at_unix: 1, expires_at_unix: 10_000,
            trusted_issuers: vec![TrustedMusicReviewIssuerV1 {
                issuer_id: "issuer".into(), key_id: "issuer-key".into(),
                public_key_hex: encode_hex(&issuer.verifying_key().to_bytes()), status,
            }],
            revoked_credential_ids: revoked,
        };
        let signed = message(TRUST_DOMAIN, &payload).unwrap();
        SignedMusicReviewTrustBundleV1 { payload, root_signature_hex: encode_hex(&root.sign(&signed).to_bytes()) }
    }

    fn fixture() -> (MusicReferenceReviewTaskV1, Vec<MusicReviewerCredentialV1>,
        Vec<SignedMusicReferenceReviewV1>, SignedMusicReviewTrustBundleV1,
        MusicReviewTrustRootV1, MusicReviewPanelPolicyV1) {
        let task = task();
        let issuer = key(90);
        let root = key(91);
        let mut creds = vec![credential("reviewer-a", 71, 31, &issuer), credential("reviewer-b", 72, 32, &issuer)];
        creds.sort_by(|a,b| a.payload.credential_id.cmp(&b.payload.credential_id));
        let mut reviews = vec![
            judgment(&task, &creds[0], 71, MusicReferenceReviewAdequacyV1::Validated),
            judgment(&task, &creds[1], 72, MusicReferenceReviewAdequacyV1::Validated),
        ];
        reviews.sort_by(|a,b| a.payload.credential_id.cmp(&b.payload.credential_id));
        let trust = bundle(&issuer, &root, Vec::new(), MusicReviewIssuerStatusV1::Active);
        let root_record = MusicReviewTrustRootV1 {
            root_id: "study-root".into(), key_id: "root-key".into(),
            public_key_hex: encode_hex(&root.verifying_key().to_bytes()),
        };
        let policy = MusicReviewPanelPolicyV1 {
            schema_version: REVIEW_POLICY_SCHEMA_V1.into(), policy_id: "unanimous-v1".into(),
            minimum_reviewers: 2, minimum_independence_groups: 2,
            required_role: MusicReviewerRoleV1::ReferenceAdjudicator, require_unanimous_adequacy: true,
        };
        (task, creds, reviews, trust, root_record, policy)
    }

    #[test]
    fn valid_signed_panel_receipt_replays() {
        let (t,c,r,b,root,p) = fixture();
        let receipt = evaluate_music_reference_review_panel_v1(&t,&c,&r,&b,&root,&p,100).unwrap();
        assert_eq!(receipt.adequacy, MusicReferenceReviewAdequacyV1::Validated);
        assert_eq!(receipt.distinct_reviewer_key_count, 2);
        assert!(verify_music_reference_review_panel_receipt_v1(&receipt,&t,&c,&r,&b,&root,&p,100).is_empty());
    }

    #[test]
    fn reviewer_disagreement_is_ambiguous_not_majority_voted() {
        let (t,c,mut r,b,root,p) = fixture();
        r[1] = judgment(&t,&c[1],72,MusicReferenceReviewAdequacyV1::Ambiguous);
        let result = evaluate_music_reference_review_panel_v1(&t,&c,&r,&b,&root,&p,100).unwrap();
        assert_eq!(result.adequacy, MusicReferenceReviewAdequacyV1::Ambiguous);
    }

    #[test]
    fn reference_or_task_substitution_rejects() {
        let (mut t,c,r,b,root,p) = fixture();
        t.reference_audio_sha256 = hash(99);
        assert!(matches!(evaluate_music_reference_review_panel_v1(&t,&c,&r,&b,&root,&p,100),
            Err(issues) if issues.contains(&MusicReferenceReviewIssueV1::TaskMismatch)));
    }

    #[test]
    fn signed_judgment_mutation_rejects() {
        let (t,c,mut r,b,root,p) = fixture();
        r[0].payload.judgments.edit_execution_1_to_5 = 1;
        assert!(matches!(evaluate_music_reference_review_panel_v1(&t,&c,&r,&b,&root,&p,100),
            Err(issues) if issues.contains(&MusicReferenceReviewIssueV1::InvalidSignature)));
    }

    #[test]
    fn revoked_credential_and_issuer_reject() {
        let (t,c,r,_b,root,p) = fixture();
        let revoked = bundle(&key(90),&key(91),vec!["reviewer-b".into()],MusicReviewIssuerStatusV1::Active);
        assert!(matches!(evaluate_music_reference_review_panel_v1(&t,&c,&r,&revoked,&root,&p,100),
            Err(issues) if issues.contains(&MusicReferenceReviewIssueV1::RevokedCredential)));
        let revoked_issuer = bundle(&key(90),&key(91),Vec::new(),MusicReviewIssuerStatusV1::Revoked);
        assert!(matches!(evaluate_music_reference_review_panel_v1(&t,&c,&r,&revoked_issuer,&root,&p,100),
            Err(issues) if issues.contains(&MusicReferenceReviewIssueV1::RevokedIssuer)));
    }

    #[test]
    fn duplicate_key_cannot_count_as_two_reviewers() {
        let (t,mut c,mut r,b,root,p) = fixture();
        let issuer = key(90);
        c[1].payload.reviewer_public_key_hex = c[0].payload.reviewer_public_key_hex.clone();
        let signed = message(CREDENTIAL_DOMAIN,&c[1].payload).unwrap();
        c[1].issuer_signature_hex = encode_hex(&issuer.sign(&signed).to_bytes());
        r[1] = judgment(&t,&c[1],71,MusicReferenceReviewAdequacyV1::Validated);
        assert!(matches!(evaluate_music_reference_review_panel_v1(&t,&c,&r,&b,&root,&p,100),
            Err(issues) if issues.contains(&MusicReferenceReviewIssueV1::DuplicateReviewerKey)));
    }

    #[test]
    fn wrong_consent_scope_and_independence_group_reject() {
        let (t,c,mut r,b,root,p) = fixture();
        let mut payload = r[0].payload.clone();
        payload.consent_scope_sha256 = hash(44);
        let signer = key(71);
        let signed = message(JUDGMENT_DOMAIN,&payload).unwrap();
        r[0] = SignedMusicReferenceReviewV1 { payload, reviewer_signature_hex: encode_hex(&signer.sign(&signed).to_bytes()) };
        assert!(matches!(evaluate_music_reference_review_panel_v1(&t,&c,&r,&b,&root,&p,100),
            Err(issues) if issues.contains(&MusicReferenceReviewIssueV1::ConsentScopeMismatch)));
        let (t,mut c,mut r,b,root,p) = fixture();
        let issuer = key(90);
        c[1].payload.independence_group_sha256 = hash(31);
        let signed = message(CREDENTIAL_DOMAIN,&c[1].payload).unwrap();
        c[1].issuer_signature_hex = encode_hex(&issuer.sign(&signed).to_bytes());
        r[1] = judgment(&t,&c[1],72,MusicReferenceReviewAdequacyV1::Validated);
        assert!(matches!(evaluate_music_reference_review_panel_v1(&t,&c,&r,&b,&root,&p,100),
            Err(issues) if issues.contains(&MusicReferenceReviewIssueV1::InsufficientIndependenceGroups)));
    }

    #[test]
    fn trust_root_and_bundle_expiry_reject() {
        let (t,c,r,b,mut root,p) = fixture();
        root.public_key_hex = encode_hex(&key(92).verifying_key().to_bytes());
        assert!(matches!(evaluate_music_reference_review_panel_v1(&t,&c,&r,&b,&root,&p,100),
            Err(issues) if issues.contains(&MusicReferenceReviewIssueV1::InvalidSignature)));
        let (t,c,r,mut b,root,p) = fixture();
        b.payload.expires_at_unix = 99;
        let root_signer = key(91);
        let signed = message(TRUST_DOMAIN,&b.payload).unwrap();
        b.root_signature_hex = encode_hex(&root_signer.sign(&signed).to_bytes());
        assert!(matches!(evaluate_music_reference_review_panel_v1(&t,&c,&r,&b,&root,&p,100),
            Err(issues) if issues.contains(&MusicReferenceReviewIssueV1::TrustBundleExpiredOrNotYetValid)));
    }

    #[test]
    fn future_review_and_invalid_rating_reject() {
        let (t,c,mut r,b,root,p) = fixture();
        let mut payload = r[0].payload.clone();
        payload.reviewed_at_unix = 200;
        let signer = key(71);
        let signed = message(JUDGMENT_DOMAIN,&payload).unwrap();
        r[0] = SignedMusicReferenceReviewV1 { payload, reviewer_signature_hex: encode_hex(&signer.sign(&signed).to_bytes()) };
        assert!(matches!(evaluate_music_reference_review_panel_v1(&t,&c,&r,&b,&root,&p,100),
            Err(issues) if issues.contains(&MusicReferenceReviewIssueV1::FutureDatedReview)));
        let (t,c,mut r,b,root,p) = fixture();
        let mut payload = r[0].payload.clone();
        payload.judgments.edit_execution_1_to_5 = 6;
        let signer = key(71);
        let signed = message(JUDGMENT_DOMAIN,&payload).unwrap();
        r[0] = SignedMusicReferenceReviewV1 { payload, reviewer_signature_hex: encode_hex(&signer.sign(&signed).to_bytes()) };
        assert!(matches!(evaluate_music_reference_review_panel_v1(&t,&c,&r,&b,&root,&p,100),
            Err(issues) if issues.contains(&MusicReferenceReviewIssueV1::InvalidJudgmentVector)));
    }

    #[test]
    fn policy_and_receipt_tampering_reject() {
        let (t,c,r,b,root,mut p) = fixture();
        p.minimum_reviewers = 1;
        assert!(matches!(evaluate_music_reference_review_panel_v1(&t,&c,&r,&b,&root,&p,100),
            Err(issues) if issues.contains(&MusicReferenceReviewIssueV1::InvalidPolicy)));
        let (t,c,r,b,root,p) = fixture();
        let receipt = evaluate_music_reference_review_panel_v1(&t,&c,&r,&b,&root,&p,100).unwrap();
        let mut changed = receipt.clone();
        changed.adequacy = MusicReferenceReviewAdequacyV1::Invalid;
        assert!(verify_music_reference_review_panel_receipt_v1(&changed,&t,&c,&r,&b,&root,&p,100)
            .contains(&MusicReferenceReviewIssueV1::ReceiptMismatch));
    }
}
