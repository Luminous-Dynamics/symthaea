// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Challenge-bound verifier-event semantics for runtime-backed TPM/reference evidence.
//!
//! This is a semantic successor to the first TPM/runtime adapter. It binds the
//! continuous runtime computation to the actual verifier event record rather than
//! attributing deterministic downstream adapter construction to the verifier.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use symthaea_domain_awareness_evidence::IndependentVerification;
use symthaea_evidence_verifier_runtime_continuity::ContinuousVerifierExecution;
use symthaea_formal_safety::{EvidenceKind, SafetyEvidenceReceipt};
use symthaea_tpm_reference_current_independence::TpmCurrentCrossStageQualification;
use symthaea_tpm_reference_evidence::TpmReferenceCandidateEvidence;

pub const TPM_VERIFIER_CHALLENGE_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.tpm-verifier-challenge-policy.v1";
pub const TPM_VERIFIER_CHALLENGE_SCHEMA_V1: &str =
    "symthaea.assurance.tpm-verifier-challenge.v1";
pub const TPM_RUNTIME_EVENT_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.tpm-runtime-event-policy.v1";
pub const TPM_RUNTIME_EVENT_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.tpm-runtime-event-report.v1";
pub const MAX_CHALLENGE_KEYS: usize = 1_024;
pub const MAX_EVIDENCE_REFS: usize = 128;
pub const MAX_TEXT_BYTES: usize = 512;

const CHALLENGE_POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm-verifier-challenge-policy.digest.v1\0";
const CHALLENGE_MESSAGE_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm-verifier-challenge.message.v1\0";
const CHALLENGE_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm-verifier-challenge.digest.v1\0";
const CAMPAIGN_SUBJECT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm-campaign-verification-subject.digest.v1\0";
const OBLIGATION_SUBJECT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm-obligation-verification-subject.digest.v1\0";
const INDEPENDENT_VERIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.independent-verification-event.digest.v1\0";
const STRICT_RECEIPT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.strict-safety-evidence-receipt.digest.v1\0";
const EVENT_POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm-runtime-event-policy.digest.v1\0";
const EVENT_REPORT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm-runtime-event-report.digest.v1\0";
const QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.challenge-bound-runtime-tpm-qualification.digest.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum VerifierChallengeRole {
    Campaign,
    Obligation,
}

impl VerifierChallengeRole {
    fn code(self) -> &'static str {
        match self {
            Self::Campaign => "campaign",
            Self::Obligation => "obligation",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierChallengeAuthorityKey {
    pub key_id: String,
    pub public_key_ed25519_hex: String,
    pub valid_from_ms: u64,
    pub valid_until_ms: Option<u64>,
    pub revoked_at_ms: Option<u64>,
    pub allowed_roles: Vec<VerifierChallengeRole>,
    pub evidence_refs: Vec<String>,
}

impl VerifierChallengeAuthorityKey {
    fn validate(&self) -> bool {
        canonical_text(&self.key_id)
            && lower_hex_exact(&self.public_key_ed25519_hex, 64)
            && self
                .valid_until_ms
                .map(|until| until > self.valid_from_ms)
                .unwrap_or(true)
            && self
                .revoked_at_ms
                .map(|revoked| revoked >= self.valid_from_ms)
                .unwrap_or(true)
            && !self.allowed_roles.is_empty()
            && unique(&self.allowed_roles)
            && valid_refs(&self.evidence_refs)
    }

    fn usable_for(&self, role: VerifierChallengeRole, at_ms: u64) -> bool {
        at_ms >= self.valid_from_ms
            && self
                .valid_until_ms
                .map(|until| at_ms < until)
                .unwrap_or(true)
            && self.revoked_at_ms.map(|at| at_ms < at).unwrap_or(true)
            && self.allowed_roles.contains(&role)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TpmVerifierChallengePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub sequence: u64,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub trusted_keys: Vec<VerifierChallengeAuthorityKey>,
    pub evidence_refs: Vec<String>,
}

impl TpmVerifierChallengePolicy {
    pub fn validate(&self) -> bool {
        if self.schema_version != TPM_VERIFIER_CHALLENGE_POLICY_SCHEMA_V1
            || !canonical_text(&self.policy_id)
            || self.sequence == 0
            || self.issued_at_ms >= self.expires_at_ms
            || self.trusted_keys.is_empty()
            || self.trusted_keys.len() > MAX_CHALLENGE_KEYS
            || !valid_refs(&self.evidence_refs)
        {
            return false;
        }
        let mut ids = BTreeSet::new();
        self.trusted_keys
            .iter()
            .all(|key| key.validate() && ids.insert(key.key_id.as_str()))
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(CHALLENGE_POLICY_DIGEST_DOMAIN);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.policy_id);
        push_u64(&mut hasher, self.sequence);
        push_u64(&mut hasher, self.issued_at_ms);
        push_u64(&mut hasher, self.expires_at_ms);
        let mut keys = self.trusted_keys.iter().collect::<Vec<_>>();
        keys.sort_by(|left, right| left.key_id.cmp(&right.key_id));
        for key in keys {
            push_field(&mut hasher, &key.key_id);
            push_field(&mut hasher, &key.public_key_ed25519_hex);
            push_u64(&mut hasher, key.valid_from_ms);
            push_optional_u64(&mut hasher, key.valid_until_ms);
            push_optional_u64(&mut hasher, key.revoked_at_ms);
            let mut roles = key.allowed_roles.clone();
            roles.sort();
            for role in roles {
                push_field(&mut hasher, role.code());
            }
            push_sorted_refs(&mut hasher, &key.evidence_refs);
        }
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TpmVerifierChallengeEnvelope {
    pub schema_version: String,
    pub challenge_id: String,
    pub role: VerifierChallengeRole,
    pub subject_digest: String,
    pub verifier_ref: String,
    pub runtime_policy_digest: String,
    pub nonce_blake3_hex: String,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub challenge_policy_digest: String,
    pub signer_key_id: String,
    pub signer_public_key_ed25519_hex: String,
    pub signature_ed25519_hex: String,
}

impl TpmVerifierChallengeEnvelope {
    pub fn validate(&self) -> bool {
        self.schema_version == TPM_VERIFIER_CHALLENGE_SCHEMA_V1
            && canonical_text(&self.challenge_id)
            && digest_text(&self.subject_digest)
            && canonical_text(&self.verifier_ref)
            && digest_text(&self.runtime_policy_digest)
            && lower_hex_exact(&self.nonce_blake3_hex, 64)
            && self.issued_at_ms < self.expires_at_ms
            && digest_text(&self.challenge_policy_digest)
            && canonical_text(&self.signer_key_id)
            && lower_hex_exact(&self.signer_public_key_ed25519_hex, 64)
            && lower_hex_exact(&self.signature_ed25519_hex, 128)
    }

    pub fn canonical_unsigned_bytes(&self) -> Option<Vec<u8>> {
        if !self.validate() {
            return None;
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(CHALLENGE_MESSAGE_DOMAIN);
        push_vec_field(&mut bytes, &self.schema_version);
        push_vec_field(&mut bytes, &self.challenge_id);
        push_vec_field(&mut bytes, self.role.code());
        push_vec_field(&mut bytes, &self.subject_digest);
        push_vec_field(&mut bytes, &self.verifier_ref);
        push_vec_field(&mut bytes, &self.runtime_policy_digest);
        push_vec_field(&mut bytes, &self.nonce_blake3_hex);
        bytes.extend_from_slice(&self.issued_at_ms.to_be_bytes());
        bytes.extend_from_slice(&self.expires_at_ms.to_be_bytes());
        push_vec_field(&mut bytes, &self.challenge_policy_digest);
        push_vec_field(&mut bytes, &self.signer_key_id);
        push_vec_field(&mut bytes, &self.signer_public_key_ed25519_hex);
        Some(bytes)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        let unsigned = self.canonical_unsigned_bytes()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(CHALLENGE_DIGEST_DOMAIN);
        hasher.update(&(unsigned.len() as u64).to_be_bytes());
        hasher.update(&unsigned);
        push_field(&mut hasher, &self.signature_ed25519_hex);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChallengeDisposition {
    Rejected,
    SignatureValidIssuerUntrusted,
    Accepted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChallengeIssue {
    InvalidPolicy,
    InvalidEnvelope,
    PolicyDigestMismatch,
    SignatureInvalid,
    IssuerUnknown,
    IssuerPublicKeyMismatch,
    PolicyNotEffectiveAtIssueTime,
    IssuerNotEffectiveAtIssueTime,
    IssuerRoleNotAuthorized,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChallengeVerificationReport {
    pub disposition: ChallengeDisposition,
    pub challenge_id: String,
    pub role: VerifierChallengeRole,
    pub subject_digest: String,
    pub verifier_ref: String,
    pub runtime_policy_digest: String,
    pub nonce_blake3_hex: String,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub policy_digest: Option<String>,
    pub challenge_digest: Option<String>,
    pub signer_key_id: String,
    pub signature_valid: bool,
    pub issuer_trusted_for_role: bool,
    pub issues: Vec<ChallengeIssue>,
}

impl ChallengeVerificationReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedTpmVerifierChallenge {
    challenge_digest: String,
    challenge_id: String,
    role: VerifierChallengeRole,
    subject_digest: String,
    verifier_ref: String,
    runtime_policy_digest: String,
    nonce_blake3_hex: String,
    issued_at_ms: u64,
    expires_at_ms: u64,
    policy_digest: String,
}

impl VerifiedTpmVerifierChallenge {
    pub fn challenge_digest(&self) -> &str { &self.challenge_digest }
    pub fn challenge_id(&self) -> &str { &self.challenge_id }
    pub const fn role(&self) -> VerifierChallengeRole { self.role }
    pub fn subject_digest(&self) -> &str { &self.subject_digest }
    pub fn verifier_ref(&self) -> &str { &self.verifier_ref }
    pub fn runtime_policy_digest(&self) -> &str { &self.runtime_policy_digest }
    pub fn nonce_blake3_hex(&self) -> &str { &self.nonce_blake3_hex }
    pub const fn issued_at_ms(&self) -> u64 { self.issued_at_ms }
    pub const fn expires_at_ms(&self) -> u64 { self.expires_at_ms }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChallengeVerification {
    pub report: ChallengeVerificationReport,
    verified: Option<VerifiedTpmVerifierChallenge>,
}

impl ChallengeVerification {
    pub fn verified(&self) -> Option<&VerifiedTpmVerifierChallenge> { self.verified.as_ref() }
    pub fn into_verified(self) -> Option<VerifiedTpmVerifierChallenge> { self.verified }
}

pub fn verify_tpm_verifier_challenge(
    policy: &TpmVerifierChallengePolicy,
    envelope: &TpmVerifierChallengeEnvelope,
) -> ChallengeVerification {
    let policy_digest = policy.canonical_digest();
    let challenge_digest = envelope.canonical_digest();
    let mut issues = Vec::new();
    if !policy.validate() {
        issues.push(ChallengeIssue::InvalidPolicy);
    }
    if !envelope.validate() {
        issues.push(ChallengeIssue::InvalidEnvelope);
    }
    if policy_digest.as_deref() != Some(envelope.challenge_policy_digest.as_str()) {
        issues.push(ChallengeIssue::PolicyDigestMismatch);
    }
    if !issues.is_empty() {
        return challenge_result(
            policy_digest,
            challenge_digest,
            envelope,
            false,
            false,
            ChallengeDisposition::Rejected,
            issues,
            None,
        );
    }

    let unsigned = envelope.canonical_unsigned_bytes().expect("validated envelope");
    let signature_valid = verify_ed25519(
        &envelope.signer_public_key_ed25519_hex,
        &unsigned,
        &envelope.signature_ed25519_hex,
    );
    if !signature_valid {
        issues.push(ChallengeIssue::SignatureInvalid);
        return challenge_result(
            policy_digest,
            challenge_digest,
            envelope,
            false,
            false,
            ChallengeDisposition::Rejected,
            issues,
            None,
        );
    }

    let Some(key) = policy
        .trusted_keys
        .iter()
        .find(|key| key.key_id == envelope.signer_key_id)
    else {
        issues.push(ChallengeIssue::IssuerUnknown);
        return challenge_result(
            policy_digest,
            challenge_digest,
            envelope,
            true,
            false,
            ChallengeDisposition::SignatureValidIssuerUntrusted,
            issues,
            None,
        );
    };

    if key.public_key_ed25519_hex != envelope.signer_public_key_ed25519_hex {
        issues.push(ChallengeIssue::IssuerPublicKeyMismatch);
    }
    if !(policy.issued_at_ms <= envelope.issued_at_ms
        && envelope.issued_at_ms < policy.expires_at_ms)
    {
        issues.push(ChallengeIssue::PolicyNotEffectiveAtIssueTime);
    }
    if !(envelope.issued_at_ms >= key.valid_from_ms
        && key
            .valid_until_ms
            .map(|until| envelope.issued_at_ms < until)
            .unwrap_or(true)
        && key
            .revoked_at_ms
            .map(|revoked| envelope.issued_at_ms < revoked)
            .unwrap_or(true))
    {
        issues.push(ChallengeIssue::IssuerNotEffectiveAtIssueTime);
    }
    if !key.usable_for(envelope.role, envelope.issued_at_ms) {
        issues.push(ChallengeIssue::IssuerRoleNotAuthorized);
    }

    if !issues.is_empty() {
        return challenge_result(
            policy_digest,
            challenge_digest,
            envelope,
            true,
            false,
            ChallengeDisposition::SignatureValidIssuerUntrusted,
            issues,
            None,
        );
    }

    let policy_digest_value = policy_digest.clone().expect("validated policy");
    let challenge_digest_value = challenge_digest.clone().expect("validated envelope");
    let verified = VerifiedTpmVerifierChallenge {
        challenge_digest: challenge_digest_value,
        challenge_id: envelope.challenge_id.clone(),
        role: envelope.role,
        subject_digest: envelope.subject_digest.clone(),
        verifier_ref: envelope.verifier_ref.clone(),
        runtime_policy_digest: envelope.runtime_policy_digest.clone(),
        nonce_blake3_hex: envelope.nonce_blake3_hex.clone(),
        issued_at_ms: envelope.issued_at_ms,
        expires_at_ms: envelope.expires_at_ms,
        policy_digest: policy_digest_value,
    };
    challenge_result(
        policy_digest,
        challenge_digest,
        envelope,
        true,
        true,
        ChallengeDisposition::Accepted,
        Vec::new(),
        Some(verified),
    )
}

#[allow(clippy::too_many_arguments)]
fn challenge_result(
    policy_digest: Option<String>,
    challenge_digest: Option<String>,
    envelope: &TpmVerifierChallengeEnvelope,
    signature_valid: bool,
    issuer_trusted_for_role: bool,
    disposition: ChallengeDisposition,
    issues: Vec<ChallengeIssue>,
    verified: Option<VerifiedTpmVerifierChallenge>,
) -> ChallengeVerification {
    ChallengeVerification {
        report: ChallengeVerificationReport {
            disposition,
            challenge_id: envelope.challenge_id.clone(),
            role: envelope.role,
            subject_digest: envelope.subject_digest.clone(),
            verifier_ref: envelope.verifier_ref.clone(),
            runtime_policy_digest: envelope.runtime_policy_digest.clone(),
            nonce_blake3_hex: envelope.nonce_blake3_hex.clone(),
            issued_at_ms: envelope.issued_at_ms,
            expires_at_ms: envelope.expires_at_ms,
            policy_digest,
            challenge_digest,
            signer_key_id: envelope.signer_key_id.clone(),
            signature_valid,
            issuer_trusted_for_role,
            issues,
        },
        verified,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TpmRuntimeEventBindingPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_challenge_policy_digest: String,
    pub expected_campaign_runtime_policy_digest: String,
    pub expected_obligation_runtime_policy_digest: String,
    pub evidence_refs: Vec<String>,
}

impl TpmRuntimeEventBindingPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == TPM_RUNTIME_EVENT_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && digest_text(&self.expected_challenge_policy_digest)
            && digest_text(&self.expected_campaign_runtime_policy_digest)
            && digest_text(&self.expected_obligation_runtime_policy_digest)
            && valid_refs(&self.evidence_refs)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(EVENT_POLICY_DIGEST_DOMAIN);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.policy_id);
        push_field(&mut hasher, &self.expected_challenge_policy_digest);
        push_field(&mut hasher, &self.expected_campaign_runtime_policy_digest);
        push_field(&mut hasher, &self.expected_obligation_runtime_policy_digest);
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

pub fn campaign_verification_subject_digest(
    candidate: &TpmReferenceCandidateEvidence,
) -> Option<String> {
    if !candidate.validate() {
        return None;
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(CAMPAIGN_SUBJECT_DIGEST_DOMAIN);
    push_field(&mut hasher, &candidate.campaign_digest);
    push_field(&mut hasher, &candidate.evidence_policy_digest);
    push_field(&mut hasher, &candidate.campaign_verifier_ref);
    push_u64(&mut hasher, candidate.observed_at_ms);
    Some(format!("blake3:{}", hasher.finalize().to_hex()))
}

pub fn obligation_verification_subject_digest(
    candidate: &TpmReferenceCandidateEvidence,
) -> Option<String> {
    if !candidate.validate() {
        return None;
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(OBLIGATION_SUBJECT_DIGEST_DOMAIN);
    push_field(&mut hasher, &candidate.evidence_digest);
    push_field(&mut hasher, &candidate.obligation_key());
    push_field(&mut hasher, &candidate.campaign_verification_digest);
    push_field(&mut hasher, &candidate.campaign_verifier_ref);
    push_u64(&mut hasher, candidate.campaign_verified_at_ms);
    Some(format!("blake3:{}", hasher.finalize().to_hex()))
}

pub fn independent_verification_digest(
    verification: &IndependentVerification,
) -> Option<String> {
    if !verification.validate() {
        return None;
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(INDEPENDENT_VERIFICATION_DIGEST_DOMAIN);
    push_field(&mut hasher, &verification.receipt_id);
    push_field(&mut hasher, &verification.verifier_ref);
    push_u64(&mut hasher, verification.verified_at_ms);
    Some(format!("blake3:{}", hasher.finalize().to_hex()))
}

pub fn strict_receipt_digest(receipt: &SafetyEvidenceReceipt) -> Option<String> {
    if !receipt.validate() {
        return None;
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(STRICT_RECEIPT_DIGEST_DOMAIN);
    push_field(&mut hasher, &receipt.receipt_id);
    push_field(&mut hasher, &receipt.obligation_key);
    push_field(&mut hasher, evidence_kind_code(receipt.evidence_kind));
    push_field(&mut hasher, &receipt.evidence_ref);
    push_field(&mut hasher, &receipt.evidence_digest);
    push_field(&mut hasher, &receipt.verifier_ref);
    push_u64(&mut hasher, receipt.verified_at_ms);
    Some(format!("blake3:{}", hasher.finalize().to_hex()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeEventBindingDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeEventBindingIssue {
    InvalidPolicy,
    InvalidCandidate,
    InvalidVerification,
    BaseReceiptFailed,
    ParentCampaignRoleMismatch,
    ParentObligationRoleMismatch,
    ParentCampaignTimeMismatch,
    ParentObligationTimeMismatch,
    CampaignRuntimeVerifierMismatch,
    ObligationRuntimeVerifierMismatch,
    CampaignRuntimeCompletionMismatch,
    ObligationRuntimeCompletionMismatch,
    RuntimeEvidenceAfterCurrentUse,
    SameRuntimeProcessInstance,
    CampaignRuntimePolicyMismatch,
    ObligationRuntimePolicyMismatch,
    CampaignChallengePolicyMismatch,
    ObligationChallengePolicyMismatch,
    CampaignChallengeRoleMismatch,
    ObligationChallengeRoleMismatch,
    CampaignChallengeSubjectMismatch,
    ObligationChallengeSubjectMismatch,
    CampaignChallengeVerifierMismatch,
    ObligationChallengeVerifierMismatch,
    CampaignChallengeRuntimePolicyMismatch,
    ObligationChallengeRuntimePolicyMismatch,
    CampaignChallengeNotProspective,
    ObligationChallengeNotProspective,
    CampaignChallengeExpiredBeforeComputation,
    ObligationChallengeExpiredBeforeComputation,
    DuplicateChallengeIdentity,
    CampaignRuntimeNonceMismatch,
    ObligationRuntimeNonceMismatch,
    CampaignRuntimeInputMismatch,
    ObligationRuntimeInputMismatch,
    CampaignRuntimeOutputMismatch,
    ObligationRuntimeOutputMismatch,
}

impl RuntimeEventBindingIssue {
    fn code(&self) -> &'static str {
        match self {
            Self::InvalidPolicy => "invalid-policy",
            Self::InvalidCandidate => "invalid-candidate",
            Self::InvalidVerification => "invalid-verification",
            Self::BaseReceiptFailed => "base-receipt-failed",
            Self::ParentCampaignRoleMismatch => "parent-campaign-role-mismatch",
            Self::ParentObligationRoleMismatch => "parent-obligation-role-mismatch",
            Self::ParentCampaignTimeMismatch => "parent-campaign-time-mismatch",
            Self::ParentObligationTimeMismatch => "parent-obligation-time-mismatch",
            Self::CampaignRuntimeVerifierMismatch => "campaign-runtime-verifier-mismatch",
            Self::ObligationRuntimeVerifierMismatch => "obligation-runtime-verifier-mismatch",
            Self::CampaignRuntimeCompletionMismatch => "campaign-runtime-completion-mismatch",
            Self::ObligationRuntimeCompletionMismatch => "obligation-runtime-completion-mismatch",
            Self::RuntimeEvidenceAfterCurrentUse => "runtime-evidence-after-current-use",
            Self::SameRuntimeProcessInstance => "same-runtime-process-instance",
            Self::CampaignRuntimePolicyMismatch => "campaign-runtime-policy-mismatch",
            Self::ObligationRuntimePolicyMismatch => "obligation-runtime-policy-mismatch",
            Self::CampaignChallengePolicyMismatch => "campaign-challenge-policy-mismatch",
            Self::ObligationChallengePolicyMismatch => "obligation-challenge-policy-mismatch",
            Self::CampaignChallengeRoleMismatch => "campaign-challenge-role-mismatch",
            Self::ObligationChallengeRoleMismatch => "obligation-challenge-role-mismatch",
            Self::CampaignChallengeSubjectMismatch => "campaign-challenge-subject-mismatch",
            Self::ObligationChallengeSubjectMismatch => "obligation-challenge-subject-mismatch",
            Self::CampaignChallengeVerifierMismatch => "campaign-challenge-verifier-mismatch",
            Self::ObligationChallengeVerifierMismatch => "obligation-challenge-verifier-mismatch",
            Self::CampaignChallengeRuntimePolicyMismatch => "campaign-challenge-runtime-policy-mismatch",
            Self::ObligationChallengeRuntimePolicyMismatch => "obligation-challenge-runtime-policy-mismatch",
            Self::CampaignChallengeNotProspective => "campaign-challenge-not-prospective",
            Self::ObligationChallengeNotProspective => "obligation-challenge-not-prospective",
            Self::CampaignChallengeExpiredBeforeComputation => "campaign-challenge-expired-before-computation",
            Self::ObligationChallengeExpiredBeforeComputation => "obligation-challenge-expired-before-computation",
            Self::DuplicateChallengeIdentity => "duplicate-challenge-identity",
            Self::CampaignRuntimeNonceMismatch => "campaign-runtime-nonce-mismatch",
            Self::ObligationRuntimeNonceMismatch => "obligation-runtime-nonce-mismatch",
            Self::CampaignRuntimeInputMismatch => "campaign-runtime-input-mismatch",
            Self::ObligationRuntimeInputMismatch => "obligation-runtime-input-mismatch",
            Self::CampaignRuntimeOutputMismatch => "campaign-runtime-output-mismatch",
            Self::ObligationRuntimeOutputMismatch => "obligation-runtime-output-mismatch",
        }
    }

    fn is_invalid(&self) -> bool {
        !matches!(
            self,
            Self::SameRuntimeProcessInstance
                | Self::CampaignRuntimePolicyMismatch
                | Self::ObligationRuntimePolicyMismatch
                | Self::CampaignChallengePolicyMismatch
                | Self::ObligationChallengePolicyMismatch
                | Self::CampaignChallengeExpiredBeforeComputation
                | Self::ObligationChallengeExpiredBeforeComputation
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeEventBindingReport {
    pub schema_version: String,
    pub disposition: RuntimeEventBindingDisposition,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub parent_tpm_qualification_digest: String,
    pub campaign_verifier_ref: String,
    pub obligation_verifier_ref: String,
    pub campaign_verified_at_ms: u64,
    pub obligation_verified_at_ms: u64,
    pub current_use_at_ms: u64,
    pub campaign_runtime_continuity_digest: String,
    pub obligation_runtime_continuity_digest: String,
    pub campaign_challenge_digest: String,
    pub obligation_challenge_digest: String,
    pub challenge_policy_digest: String,
    pub campaign_subject_digest: Option<String>,
    pub obligation_subject_digest: Option<String>,
    pub campaign_expected_output_digest: String,
    pub obligation_expected_output_digest: Option<String>,
    pub strict_receipt_digest: Option<String>,
    pub issues: Vec<RuntimeEventBindingIssue>,
    pub qualification_digest: Option<String>,
}

impl RuntimeEventBindingReport {
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChallengeBoundRuntimeTpmQualification {
    qualification_digest: String,
    parent_tpm_qualification_digest: String,
    campaign_runtime_continuity_digest: String,
    obligation_runtime_continuity_digest: String,
    campaign_challenge_digest: String,
    obligation_challenge_digest: String,
    challenge_policy_digest: String,
    campaign_verification_digest: String,
    obligation_verification_digest: String,
    strict_receipt_digest: String,
    current_use_at_ms: u64,
}

impl ChallengeBoundRuntimeTpmQualification {
    pub fn qualification_digest(&self) -> &str { &self.qualification_digest }
    pub fn parent_tpm_qualification_digest(&self) -> &str { &self.parent_tpm_qualification_digest }
    pub fn campaign_runtime_continuity_digest(&self) -> &str { &self.campaign_runtime_continuity_digest }
    pub fn obligation_runtime_continuity_digest(&self) -> &str { &self.obligation_runtime_continuity_digest }
    pub fn campaign_challenge_digest(&self) -> &str { &self.campaign_challenge_digest }
    pub fn obligation_challenge_digest(&self) -> &str { &self.obligation_challenge_digest }
    pub fn challenge_policy_digest(&self) -> &str { &self.challenge_policy_digest }
    pub fn campaign_verification_digest(&self) -> &str { &self.campaign_verification_digest }
    pub fn obligation_verification_digest(&self) -> &str { &self.obligation_verification_digest }
    pub fn strict_receipt_digest(&self) -> &str { &self.strict_receipt_digest }
    pub const fn current_use_at_ms(&self) -> u64 { self.current_use_at_ms }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeEventBindingAssessment {
    pub report: RuntimeEventBindingReport,
    pub receipt: Option<SafetyEvidenceReceipt>,
    qualification: Option<ChallengeBoundRuntimeTpmQualification>,
}

impl RuntimeEventBindingAssessment {
    pub fn qualification(&self) -> Option<&ChallengeBoundRuntimeTpmQualification> {
        self.qualification.as_ref()
    }

    pub fn into_qualification(self) -> Option<ChallengeBoundRuntimeTpmQualification> {
        self.qualification
    }
}

#[allow(clippy::too_many_arguments)]
pub fn assess_tpm_runtime_event_binding(
    candidate: &TpmReferenceCandidateEvidence,
    verification: &IndependentVerification,
    parent: &TpmCurrentCrossStageQualification,
    campaign_runtime: &ContinuousVerifierExecution,
    obligation_runtime: &ContinuousVerifierExecution,
    campaign_challenge: &VerifiedTpmVerifierChallenge,
    obligation_challenge: &VerifiedTpmVerifierChallenge,
    policy: &TpmRuntimeEventBindingPolicy,
) -> RuntimeEventBindingAssessment {
    let policy_digest = policy.canonical_digest();
    let campaign_subject = campaign_verification_subject_digest(candidate);
    let obligation_subject = obligation_verification_subject_digest(candidate);
    let obligation_output = independent_verification_digest(verification);
    let receipt = candidate.verify(verification).ok();
    let strict_receipt = receipt.as_ref().and_then(strict_receipt_digest);
    let mut issues = Vec::new();

    if !policy.validate() { issues.push(RuntimeEventBindingIssue::InvalidPolicy); }
    if !candidate.validate() { issues.push(RuntimeEventBindingIssue::InvalidCandidate); }
    if !verification.validate() { issues.push(RuntimeEventBindingIssue::InvalidVerification); }
    if receipt.is_none() || strict_receipt.is_none() {
        issues.push(RuntimeEventBindingIssue::BaseReceiptFailed);
    }

    if parent.campaign_verifier_ref() != candidate.campaign_verifier_ref {
        issues.push(RuntimeEventBindingIssue::ParentCampaignRoleMismatch);
    }
    if parent.obligation_verifier_ref() != verification.verifier_ref {
        issues.push(RuntimeEventBindingIssue::ParentObligationRoleMismatch);
    }
    if parent.campaign_verified_at_ms() != candidate.campaign_verified_at_ms {
        issues.push(RuntimeEventBindingIssue::ParentCampaignTimeMismatch);
    }
    if parent.obligation_verified_at_ms() != verification.verified_at_ms {
        issues.push(RuntimeEventBindingIssue::ParentObligationTimeMismatch);
    }

    if campaign_runtime.verifier_ref() != candidate.campaign_verifier_ref {
        issues.push(RuntimeEventBindingIssue::CampaignRuntimeVerifierMismatch);
    }
    if obligation_runtime.verifier_ref() != verification.verifier_ref {
        issues.push(RuntimeEventBindingIssue::ObligationRuntimeVerifierMismatch);
    }
    if campaign_runtime.completed_at_ms() != candidate.campaign_verified_at_ms {
        issues.push(RuntimeEventBindingIssue::CampaignRuntimeCompletionMismatch);
    }
    if obligation_runtime.completed_at_ms() != verification.verified_at_ms {
        issues.push(RuntimeEventBindingIssue::ObligationRuntimeCompletionMismatch);
    }
    if campaign_runtime.assessed_at_ms() > parent.current_use_at_ms()
        || obligation_runtime.assessed_at_ms() > parent.current_use_at_ms()
    {
        issues.push(RuntimeEventBindingIssue::RuntimeEvidenceAfterCurrentUse);
    }
    if campaign_runtime.process_instance_id() == obligation_runtime.process_instance_id() {
        issues.push(RuntimeEventBindingIssue::SameRuntimeProcessInstance);
    }
    if campaign_runtime.policy_digest() != policy.expected_campaign_runtime_policy_digest {
        issues.push(RuntimeEventBindingIssue::CampaignRuntimePolicyMismatch);
    }
    if obligation_runtime.policy_digest() != policy.expected_obligation_runtime_policy_digest {
        issues.push(RuntimeEventBindingIssue::ObligationRuntimePolicyMismatch);
    }

    for (challenge, expected_role, expected_policy, role_issue, policy_issue) in [
        (
            campaign_challenge,
            VerifierChallengeRole::Campaign,
            campaign_runtime.policy_digest(),
            RuntimeEventBindingIssue::CampaignChallengeRoleMismatch,
            RuntimeEventBindingIssue::CampaignChallengePolicyMismatch,
        ),
        (
            obligation_challenge,
            VerifierChallengeRole::Obligation,
            obligation_runtime.policy_digest(),
            RuntimeEventBindingIssue::ObligationChallengeRoleMismatch,
            RuntimeEventBindingIssue::ObligationChallengePolicyMismatch,
        ),
    ] {
        if challenge.role() != expected_role { issues.push(role_issue); }
        if challenge.policy_digest() != policy.expected_challenge_policy_digest {
            issues.push(policy_issue);
        }
        if challenge.runtime_policy_digest() != expected_policy {
            issues.push(match expected_role {
                VerifierChallengeRole::Campaign => RuntimeEventBindingIssue::CampaignChallengeRuntimePolicyMismatch,
                VerifierChallengeRole::Obligation => RuntimeEventBindingIssue::ObligationChallengeRuntimePolicyMismatch,
            });
        }
    }

    if campaign_challenge.subject_digest() != campaign_subject.as_deref().unwrap_or("") {
        issues.push(RuntimeEventBindingIssue::CampaignChallengeSubjectMismatch);
    }
    if obligation_challenge.subject_digest() != obligation_subject.as_deref().unwrap_or("") {
        issues.push(RuntimeEventBindingIssue::ObligationChallengeSubjectMismatch);
    }
    if campaign_challenge.verifier_ref() != candidate.campaign_verifier_ref {
        issues.push(RuntimeEventBindingIssue::CampaignChallengeVerifierMismatch);
    }
    if obligation_challenge.verifier_ref() != verification.verifier_ref {
        issues.push(RuntimeEventBindingIssue::ObligationChallengeVerifierMismatch);
    }
    if campaign_challenge.issued_at_ms() > campaign_runtime.started_at_ms() {
        issues.push(RuntimeEventBindingIssue::CampaignChallengeNotProspective);
    }
    if obligation_challenge.issued_at_ms() > obligation_runtime.started_at_ms() {
        issues.push(RuntimeEventBindingIssue::ObligationChallengeNotProspective);
    }
    if campaign_runtime.started_at_ms() >= campaign_challenge.expires_at_ms() {
        issues.push(RuntimeEventBindingIssue::CampaignChallengeExpiredBeforeComputation);
    }
    if obligation_runtime.started_at_ms() >= obligation_challenge.expires_at_ms() {
        issues.push(RuntimeEventBindingIssue::ObligationChallengeExpiredBeforeComputation);
    }
    if campaign_challenge.challenge_id() == obligation_challenge.challenge_id()
        || campaign_challenge.nonce_blake3_hex() == obligation_challenge.nonce_blake3_hex()
    {
        issues.push(RuntimeEventBindingIssue::DuplicateChallengeIdentity);
    }
    if campaign_runtime.request_nonce_blake3_hex() != campaign_challenge.nonce_blake3_hex() {
        issues.push(RuntimeEventBindingIssue::CampaignRuntimeNonceMismatch);
    }
    if obligation_runtime.request_nonce_blake3_hex() != obligation_challenge.nonce_blake3_hex() {
        issues.push(RuntimeEventBindingIssue::ObligationRuntimeNonceMismatch);
    }
    if campaign_runtime.input_digest() != campaign_subject.as_deref().unwrap_or("") {
        issues.push(RuntimeEventBindingIssue::CampaignRuntimeInputMismatch);
    }
    if obligation_runtime.input_digest() != obligation_subject.as_deref().unwrap_or("") {
        issues.push(RuntimeEventBindingIssue::ObligationRuntimeInputMismatch);
    }
    if campaign_runtime.output_digest() != candidate.campaign_verification_digest {
        issues.push(RuntimeEventBindingIssue::CampaignRuntimeOutputMismatch);
    }
    if obligation_runtime.output_digest() != obligation_output.as_deref().unwrap_or("") {
        issues.push(RuntimeEventBindingIssue::ObligationRuntimeOutputMismatch);
    }

    let disposition = if issues.iter().any(RuntimeEventBindingIssue::is_invalid) {
        RuntimeEventBindingDisposition::Invalid
    } else if issues.is_empty() {
        RuntimeEventBindingDisposition::Qualified
    } else {
        RuntimeEventBindingDisposition::Blocked
    };

    let qualification = if disposition == RuntimeEventBindingDisposition::Qualified {
        let policy_digest_value = policy_digest.clone().expect("validated policy");
        let strict_receipt_value = strict_receipt.clone().expect("qualified receipt");
        let obligation_output_value = obligation_output.clone().expect("validated verification");
        let qualification_digest = runtime_event_qualification_digest(
            parent,
            campaign_runtime,
            obligation_runtime,
            campaign_challenge,
            obligation_challenge,
            &policy_digest_value,
            &candidate.campaign_verification_digest,
            &obligation_output_value,
            &strict_receipt_value,
        );
        Some(ChallengeBoundRuntimeTpmQualification {
            qualification_digest,
            parent_tpm_qualification_digest: parent.qualification_digest().to_string(),
            campaign_runtime_continuity_digest: campaign_runtime.continuity_digest().to_string(),
            obligation_runtime_continuity_digest: obligation_runtime.continuity_digest().to_string(),
            campaign_challenge_digest: campaign_challenge.challenge_digest().to_string(),
            obligation_challenge_digest: obligation_challenge.challenge_digest().to_string(),
            challenge_policy_digest: campaign_challenge.policy_digest().to_string(),
            campaign_verification_digest: candidate.campaign_verification_digest.clone(),
            obligation_verification_digest: obligation_output_value,
            strict_receipt_digest: strict_receipt_value,
            current_use_at_ms: parent.current_use_at_ms(),
        })
    } else {
        None
    };

    let qualification_digest = qualification
        .as_ref()
        .map(|value| value.qualification_digest().to_string());
    RuntimeEventBindingAssessment {
        report: RuntimeEventBindingReport {
            schema_version: TPM_RUNTIME_EVENT_REPORT_SCHEMA_V1.into(),
            disposition,
            policy_id: policy.policy_id.clone(),
            policy_digest,
            parent_tpm_qualification_digest: parent.qualification_digest().to_string(),
            campaign_verifier_ref: candidate.campaign_verifier_ref.clone(),
            obligation_verifier_ref: verification.verifier_ref.clone(),
            campaign_verified_at_ms: candidate.campaign_verified_at_ms,
            obligation_verified_at_ms: verification.verified_at_ms,
            current_use_at_ms: parent.current_use_at_ms(),
            campaign_runtime_continuity_digest: campaign_runtime.continuity_digest().to_string(),
            obligation_runtime_continuity_digest: obligation_runtime.continuity_digest().to_string(),
            campaign_challenge_digest: campaign_challenge.challenge_digest().to_string(),
            obligation_challenge_digest: obligation_challenge.challenge_digest().to_string(),
            challenge_policy_digest: campaign_challenge.policy_digest().to_string(),
            campaign_subject_digest: campaign_subject,
            obligation_subject_digest: obligation_subject,
            campaign_expected_output_digest: candidate.campaign_verification_digest.clone(),
            obligation_expected_output_digest: obligation_output,
            strict_receipt_digest: strict_receipt,
            issues,
            qualification_digest,
        },
        receipt,
        qualification,
    }
}

#[allow(clippy::too_many_arguments)]
fn runtime_event_qualification_digest(
    parent: &TpmCurrentCrossStageQualification,
    campaign_runtime: &ContinuousVerifierExecution,
    obligation_runtime: &ContinuousVerifierExecution,
    campaign_challenge: &VerifiedTpmVerifierChallenge,
    obligation_challenge: &VerifiedTpmVerifierChallenge,
    policy_digest: &str,
    campaign_output_digest: &str,
    obligation_output_digest: &str,
    strict_receipt_digest: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_DIGEST_DOMAIN);
    for value in [
        parent.qualification_digest(),
        campaign_runtime.continuity_digest(),
        obligation_runtime.continuity_digest(),
        campaign_challenge.challenge_digest(),
        obligation_challenge.challenge_digest(),
        campaign_challenge.policy_digest(),
        policy_digest,
        campaign_output_digest,
        obligation_output_digest,
        strict_receipt_digest,
    ] {
        push_field(&mut hasher, value);
    }
    push_u64(&mut hasher, parent.campaign_verified_at_ms());
    push_u64(&mut hasher, parent.obligation_verified_at_ms());
    push_u64(&mut hasher, parent.current_use_at_ms());
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn verify_ed25519(public_key_hex: &str, message: &[u8], signature_hex: &str) -> bool {
    let Ok(public_key) = hex::decode(public_key_hex) else { return false; };
    let Ok(signature) = hex::decode(signature_hex) else { return false; };
    let Ok(public_key): Result<[u8; 32], _> = public_key.try_into() else { return false; };
    let Ok(signature): Result<[u8; 64], _> = signature.try_into() else { return false; };
    let Ok(key) = VerifyingKey::from_bytes(&public_key) else { return false; };
    key.verify(message, &Signature::from_bytes(&signature)).is_ok()
}

fn evidence_kind_code(kind: EvidenceKind) -> &'static str {
    match kind {
        EvidenceKind::FormalProof => "formal-proof",
        EvidenceKind::Simulation => "simulation",
        EvidenceKind::Test => "test",
        EvidenceKind::Telemetry => "telemetry",
        EvidenceKind::Standard => "standard",
    }
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_TEXT_BYTES
        && !value.chars().any(char::is_control)
}

fn digest_text(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|digest| {
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

fn lower_hex_exact(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_refs(refs: &[String]) -> bool {
    refs.len() <= MAX_EVIDENCE_REFS
        && refs.iter().all(|value| canonical_text(value))
        && unique(refs)
}

fn unique<T: Ord + Clone>(values: &[T]) -> bool {
    values.iter().cloned().collect::<BTreeSet<_>>().len() == values.len()
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn push_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_be_bytes());
}

fn push_optional_u64(hasher: &mut blake3::Hasher, value: Option<u64>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            push_u64(hasher, value);
        }
        None => hasher.update(&[0]),
    }
}

fn push_sorted_refs(hasher: &mut blake3::Hasher, refs: &[String]) {
    let mut refs = refs.to_vec();
    refs.sort();
    for reference in refs {
        push_field(hasher, &reference);
    }
}

fn push_vec_field(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn d(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn key() -> SigningKey {
        SigningKey::from_bytes(&[13u8; 32])
    }

    fn public_key() -> String {
        hex::encode(key().verifying_key().as_bytes())
    }

    fn challenge_policy() -> TpmVerifierChallengePolicy {
        TpmVerifierChallengePolicy {
            schema_version: TPM_VERIFIER_CHALLENGE_POLICY_SCHEMA_V1.into(),
            policy_id: "challenge-policy:1".into(),
            sequence: 1,
            issued_at_ms: 100,
            expires_at_ms: 10_000,
            trusted_keys: vec![VerifierChallengeAuthorityKey {
                key_id: "challenge-key:1".into(),
                public_key_ed25519_hex: public_key(),
                valid_from_ms: 50,
                valid_until_ms: Some(9_000),
                revoked_at_ms: None,
                allowed_roles: vec![VerifierChallengeRole::Campaign, VerifierChallengeRole::Obligation],
                evidence_refs: vec!["review:challenge-key".into()],
            }],
            evidence_refs: vec!["review:challenge-policy".into()],
        }
    }

    fn signed_challenge(role: VerifierChallengeRole, nonce_byte: &str) -> TpmVerifierChallengeEnvelope {
        let policy = challenge_policy();
        let mut envelope = TpmVerifierChallengeEnvelope {
            schema_version: TPM_VERIFIER_CHALLENGE_SCHEMA_V1.into(),
            challenge_id: format!("challenge:{}", role.code()),
            role,
            subject_digest: d("subject"),
            verifier_ref: match role {
                VerifierChallengeRole::Campaign => "verifier:campaign".into(),
                VerifierChallengeRole::Obligation => "verifier:obligation".into(),
            },
            runtime_policy_digest: d("runtime-policy"),
            nonce_blake3_hex: nonce_byte.repeat(32),
            issued_at_ms: 1_000,
            expires_at_ms: 2_000,
            challenge_policy_digest: policy.canonical_digest().unwrap(),
            signer_key_id: "challenge-key:1".into(),
            signer_public_key_ed25519_hex: public_key(),
            signature_ed25519_hex: "00".repeat(64),
        };
        let bytes = envelope.canonical_unsigned_bytes().unwrap();
        envelope.signature_ed25519_hex = hex::encode(key().sign(&bytes).to_bytes());
        envelope
    }

    #[test]
    fn trusted_signed_challenge_mints_opaque_verified_form() {
        let policy = challenge_policy();
        let envelope = signed_challenge(VerifierChallengeRole::Campaign, "11");
        let verified = verify_tpm_verifier_challenge(&policy, &envelope);
        assert_eq!(verified.report.disposition, ChallengeDisposition::Accepted);
        assert!(verified.verified().is_some());
        assert!(!verified.verified().unwrap().grants_physical_authority());
    }

    #[test]
    fn unknown_signer_is_not_trusted_even_with_valid_signature() {
        let mut policy = challenge_policy();
        policy.trusted_keys.clear();
        assert!(!policy.validate());
    }

    #[test]
    fn challenge_nonce_changes_content_identity() {
        let left = signed_challenge(VerifierChallengeRole::Campaign, "11");
        let right = signed_challenge(VerifierChallengeRole::Campaign, "22");
        assert_ne!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn independent_verification_identity_binds_receipt_verifier_and_time() {
        let base = IndependentVerification {
            receipt_id: "receipt:1".into(),
            verifier_ref: "verifier:obligation".into(),
            verified_at_ms: 4_000,
        };
        let digest = independent_verification_digest(&base).unwrap();
        let mut changed = base.clone();
        changed.receipt_id = "receipt:2".into();
        assert_ne!(digest, independent_verification_digest(&changed).unwrap());
        let mut changed_time = base;
        changed_time.verified_at_ms += 1;
        assert_ne!(digest, independent_verification_digest(&changed_time).unwrap());
    }

    #[test]
    fn challenge_policy_digest_is_key_order_independent() {
        let mut a = challenge_policy();
        let mut second = a.trusted_keys[0].clone();
        second.key_id = "challenge-key:2".into();
        second.public_key_ed25519_hex = "44".repeat(32);
        a.trusted_keys.push(second);
        let mut b = a.clone();
        b.trusted_keys.reverse();
        assert_eq!(a.canonical_digest(), b.canonical_digest());
    }
}
