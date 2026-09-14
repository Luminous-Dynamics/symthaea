// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Signed lifecycle-head currentness and anti-rollback tracking.
//!
//! An internally valid lifecycle ledger is not evidence that it is the latest
//! authoritative ledger. This crate binds an `ActiveVerifierProfile` to a
//! signed, sequenced lifecycle-head statement and a challenge-bound assertion
//! that the exact head is current at one exact use-time.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use symthaea_evidence_independence_lifecycle::ActiveVerifierProfile;

pub const HEAD_AUTHORITY_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.lifecycle-head-authority-policy.v1";
pub const LIFECYCLE_HEAD_STATEMENT_SCHEMA_V1: &str =
    "symthaea.assurance.lifecycle-head-statement.v1";
pub const LIFECYCLE_HEAD_CURRENTNESS_SCHEMA_V1: &str =
    "symthaea.assurance.lifecycle-head-currentness.v1";
pub const MAX_TRUSTED_KEYS: usize = 1_024;
pub const MAX_EVIDENCE_REFS: usize = 128;
pub const MAX_TEXT_BYTES: usize = 512;

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.lifecycle-head-authority-policy.digest.v1\0";
const HEAD_MESSAGE_DOMAIN: &[u8] =
    b"symthaea.assurance.lifecycle-head-statement.message.v1\0";
const HEAD_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.lifecycle-head-statement.digest.v1\0";
const CURRENTNESS_MESSAGE_DOMAIN: &[u8] =
    b"symthaea.assurance.lifecycle-head-currentness.message.v1\0";
const CURRENTNESS_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.lifecycle-head-currentness.digest.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LifecycleHeadClaimScope {
    HeadStatement,
    CurrentnessAttestation,
}

impl LifecycleHeadClaimScope {
    fn code(self) -> &'static str {
        match self {
            Self::HeadStatement => "head-statement",
            Self::CurrentnessAttestation => "currentness-attestation",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleHeadAuthorityKey {
    pub key_id: String,
    pub public_key_ed25519_hex: String,
    pub valid_from_ms: u64,
    pub valid_until_ms: Option<u64>,
    pub revoked_at_ms: Option<u64>,
    pub allowed_scopes: Vec<LifecycleHeadClaimScope>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleHeadAuthorityPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub sequence: u64,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub trusted_keys: Vec<LifecycleHeadAuthorityKey>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleHeadStatement {
    pub schema_version: String,
    pub profile_id: String,
    pub record_digest: String,
    pub lifecycle_digest: String,
    pub head_sequence: u64,
    pub previous_head_digest: Option<String>,
    pub issued_at_ms: u64,
    pub authority_policy_digest: String,
    pub signer_key_id: String,
    pub signer_public_key_ed25519_hex: String,
    pub evidence_refs: Vec<String>,
    pub signature_ed25519_hex: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleHeadCurrentnessAttestation {
    pub schema_version: String,
    pub head_statement_digest: String,
    pub profile_id: String,
    pub record_digest: String,
    pub lifecycle_digest: String,
    pub head_sequence: u64,
    pub authority_policy_digest: String,
    /// The authority asserts that `head_statement_digest` is its current head
    /// at exactly this time. Trusted-time provenance remains outside this crate.
    pub asserted_current_at_ms: u64,
    /// Caller challenge. The verifier supplies the expected value separately;
    /// a valid old attestation cannot satisfy a fresh challenge.
    pub challenge_nonce_blake3_hex: String,
    pub signer_key_id: String,
    pub signer_public_key_ed25519_hex: String,
    pub evidence_refs: Vec<String>,
    pub signature_ed25519_hex: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeadStatementDisposition {
    Invalid,
    Untrusted,
    Verified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HeadStatementIssue {
    InvalidPolicy,
    InvalidStatement,
    PolicyDigestMismatch,
    PolicyNotEffectiveAtStatementTime,
    SignerUnknown,
    SignerPublicKeyMismatch,
    SignerNotEffective,
    SignerRevoked,
    ScopeNotAuthorized,
    InvalidPublicKey,
    SignatureInvalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeadStatementReport {
    pub disposition: HeadStatementDisposition,
    pub profile_id: String,
    pub record_digest: String,
    pub lifecycle_digest: String,
    pub head_sequence: u64,
    pub statement_digest: Option<String>,
    pub policy_digest: Option<String>,
    pub signer_key_id: String,
    pub signature_valid: bool,
    pub issues: Vec<HeadStatementIssue>,
}

impl HeadStatementReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedLifecycleHeadStatement {
    profile_id: String,
    record_digest: String,
    lifecycle_digest: String,
    head_sequence: u64,
    previous_head_digest: Option<String>,
    issued_at_ms: u64,
    statement_digest: String,
    policy_digest: String,
    signer_key_id: String,
}

impl VerifiedLifecycleHeadStatement {
    pub fn profile_id(&self) -> &str { &self.profile_id }
    pub fn record_digest(&self) -> &str { &self.record_digest }
    pub fn lifecycle_digest(&self) -> &str { &self.lifecycle_digest }
    pub const fn head_sequence(&self) -> u64 { self.head_sequence }
    pub fn previous_head_digest(&self) -> Option<&str> { self.previous_head_digest.as_deref() }
    pub const fn issued_at_ms(&self) -> u64 { self.issued_at_ms }
    pub fn statement_digest(&self) -> &str { &self.statement_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub fn signer_key_id(&self) -> &str { &self.signer_key_id }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeadStatementVerification {
    pub report: HeadStatementReport,
    verified: Option<VerifiedLifecycleHeadStatement>,
}

impl HeadStatementVerification {
    pub fn verified(&self) -> Option<&VerifiedLifecycleHeadStatement> { self.verified.as_ref() }
    pub fn into_verified(self) -> Option<VerifiedLifecycleHeadStatement> { self.verified }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TrackedHeadState {
    sequence: u64,
    digest: String,
    issued_at_ms: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LifecycleHeadTracker {
    heads: BTreeMap<String, TrackedHeadState>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LifecycleHeadTrackingError {
    FirstHeadMustBeGenesis { proposed_sequence: u64 },
    FirstHeadHasPreviousDigest,
    SequenceRollback { latest: u64, proposed: u64 },
    SequenceCollision { sequence: u64 },
    SequenceGap { expected: u64, proposed: u64 },
    PreviousHeadDigestMismatch,
    IssuedAtRegressed { latest: u64, proposed: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackedLifecycleHead {
    profile_id: String,
    record_digest: String,
    lifecycle_digest: String,
    head_sequence: u64,
    statement_digest: String,
    policy_digest: String,
    issued_at_ms: u64,
}

impl TrackedLifecycleHead {
    pub fn profile_id(&self) -> &str { &self.profile_id }
    pub fn record_digest(&self) -> &str { &self.record_digest }
    pub fn lifecycle_digest(&self) -> &str { &self.lifecycle_digest }
    pub const fn head_sequence(&self) -> u64 { self.head_sequence }
    pub fn statement_digest(&self) -> &str { &self.statement_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub const fn issued_at_ms(&self) -> u64 { self.issued_at_ms }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

impl LifecycleHeadTracker {
    pub fn accept(
        &mut self,
        head: &VerifiedLifecycleHeadStatement,
    ) -> Result<TrackedLifecycleHead, LifecycleHeadTrackingError> {
        let profile_id = head.profile_id.clone();
        match self.heads.get(&profile_id) {
            None => {
                if head.head_sequence != 1 {
                    return Err(LifecycleHeadTrackingError::FirstHeadMustBeGenesis {
                        proposed_sequence: head.head_sequence,
                    });
                }
                if head.previous_head_digest.is_some() {
                    return Err(LifecycleHeadTrackingError::FirstHeadHasPreviousDigest);
                }
            }
            Some(latest) => {
                if head.head_sequence < latest.sequence {
                    return Err(LifecycleHeadTrackingError::SequenceRollback {
                        latest: latest.sequence,
                        proposed: head.head_sequence,
                    });
                }
                if head.head_sequence == latest.sequence {
                    if head.statement_digest == latest.digest {
                        return Ok(tracked(head));
                    }
                    return Err(LifecycleHeadTrackingError::SequenceCollision {
                        sequence: head.head_sequence,
                    });
                }
                let expected = latest.sequence.saturating_add(1);
                if head.head_sequence != expected {
                    return Err(LifecycleHeadTrackingError::SequenceGap {
                        expected,
                        proposed: head.head_sequence,
                    });
                }
                if head.previous_head_digest.as_deref() != Some(latest.digest.as_str()) {
                    return Err(LifecycleHeadTrackingError::PreviousHeadDigestMismatch);
                }
                if head.issued_at_ms < latest.issued_at_ms {
                    return Err(LifecycleHeadTrackingError::IssuedAtRegressed {
                        latest: latest.issued_at_ms,
                        proposed: head.issued_at_ms,
                    });
                }
            }
        }
        self.heads.insert(
            profile_id,
            TrackedHeadState {
                sequence: head.head_sequence,
                digest: head.statement_digest.clone(),
                issued_at_ms: head.issued_at_ms,
            },
        );
        Ok(tracked(head))
    }

    pub fn latest_sequence(&self, profile_id: &str) -> Option<u64> {
        self.heads.get(profile_id).map(|state| state.sequence)
    }

    pub fn latest_digest(&self, profile_id: &str) -> Option<&str> {
        self.heads.get(profile_id).map(|state| state.digest.as_str())
    }
}

fn tracked(head: &VerifiedLifecycleHeadStatement) -> TrackedLifecycleHead {
    TrackedLifecycleHead {
        profile_id: head.profile_id.clone(),
        record_digest: head.record_digest.clone(),
        lifecycle_digest: head.lifecycle_digest.clone(),
        head_sequence: head.head_sequence,
        statement_digest: head.statement_digest.clone(),
        policy_digest: head.policy_digest.clone(),
        issued_at_ms: head.issued_at_ms,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifecycleHeadCurrentnessDisposition {
    Invalid,
    Blocked,
    Current,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifecycleHeadCurrentnessIssue {
    InvalidPolicy,
    InvalidAttestation,
    PolicyDigestMismatch,
    HeadPolicyDigestMismatch,
    HeadStatementDigestMismatch,
    HeadProfileMismatch,
    HeadRecordMismatch,
    HeadLifecycleMismatch,
    HeadSequenceMismatch,
    ActiveRecordMismatch,
    ActiveLifecycleMismatch,
    ActiveAssessmentNotAtUseTime,
    CurrentnessNotAtUseTime,
    ChallengeNonceMismatch,
    PolicyNotEffectiveAtUseTime,
    SignerUnknown,
    SignerPublicKeyMismatch,
    SignerNotEffective,
    SignerRevoked,
    ScopeNotAuthorized,
    InvalidPublicKey,
    SignatureInvalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleHeadCurrentnessReport {
    pub disposition: LifecycleHeadCurrentnessDisposition,
    pub profile_id: String,
    pub record_digest: String,
    pub lifecycle_digest: String,
    pub head_sequence: u64,
    pub head_statement_digest: String,
    pub currentness_attestation_digest: Option<String>,
    pub policy_digest: Option<String>,
    pub use_at_ms: u64,
    pub signer_key_id: String,
    pub signature_valid: bool,
    pub issues: Vec<LifecycleHeadCurrentnessIssue>,
}

impl LifecycleHeadCurrentnessReport {
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentActiveVerifierProfile {
    profile_id: String,
    record_digest: String,
    graph_digest: String,
    relation_completeness_digest: String,
    provenance_attestation_digest: String,
    lifecycle_digest: String,
    lifecycle_head_statement_digest: String,
    currentness_attestation_digest: String,
    head_sequence: u64,
    verification_at_ms: u64,
    use_at_ms: u64,
}

impl CurrentActiveVerifierProfile {
    pub fn profile_id(&self) -> &str { &self.profile_id }
    pub fn record_digest(&self) -> &str { &self.record_digest }
    pub fn graph_digest(&self) -> &str { &self.graph_digest }
    pub fn relation_completeness_digest(&self) -> &str { &self.relation_completeness_digest }
    pub fn provenance_attestation_digest(&self) -> &str { &self.provenance_attestation_digest }
    pub fn lifecycle_digest(&self) -> &str { &self.lifecycle_digest }
    pub fn lifecycle_head_statement_digest(&self) -> &str { &self.lifecycle_head_statement_digest }
    pub fn currentness_attestation_digest(&self) -> &str { &self.currentness_attestation_digest }
    pub const fn head_sequence(&self) -> u64 { self.head_sequence }
    pub const fn verification_at_ms(&self) -> u64 { self.verification_at_ms }
    pub const fn use_at_ms(&self) -> u64 { self.use_at_ms }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifecycleHeadCurrentnessAssessment {
    pub report: LifecycleHeadCurrentnessReport,
    current: Option<CurrentActiveVerifierProfile>,
}

impl LifecycleHeadCurrentnessAssessment {
    pub fn current(&self) -> Option<&CurrentActiveVerifierProfile> { self.current.as_ref() }
    pub fn into_current(self) -> Option<CurrentActiveVerifierProfile> { self.current }
}

impl LifecycleHeadAuthorityKey {
    fn validate(&self) -> bool {
        canonical_text(&self.key_id)
            && lower_hex_exact(&self.public_key_ed25519_hex, 64)
            && self.valid_until_ms.map(|until| until > self.valid_from_ms).unwrap_or(true)
            && self.revoked_at_ms.map(|revoked| revoked >= self.valid_from_ms).unwrap_or(true)
            && !self.allowed_scopes.is_empty()
            && unique(&self.allowed_scopes)
            && valid_refs(&self.evidence_refs)
    }

    fn usable_at(&self, at_ms: u64) -> bool {
        at_ms >= self.valid_from_ms
            && self.valid_until_ms.map(|until| at_ms < until).unwrap_or(true)
    }

    fn revoked_at(&self, at_ms: u64) -> bool {
        self.revoked_at_ms.map(|revoked| at_ms >= revoked).unwrap_or(false)
    }
}

impl LifecycleHeadAuthorityPolicy {
    pub fn validate(&self) -> bool {
        if self.schema_version != HEAD_AUTHORITY_POLICY_SCHEMA_V1
            || !canonical_text(&self.policy_id)
            || self.sequence == 0
            || self.issued_at_ms >= self.expires_at_ms
            || self.trusted_keys.is_empty()
            || self.trusted_keys.len() > MAX_TRUSTED_KEYS
            || !valid_refs(&self.evidence_refs)
        {
            return false;
        }
        let mut ids = BTreeSet::new();
        self.trusted_keys.iter().all(|key| key.validate() && ids.insert(key.key_id.as_str()))
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() { return None; }
        let mut hasher = blake3::Hasher::new();
        hasher.update(POLICY_DIGEST_DOMAIN);
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
            let mut scopes = key.allowed_scopes.clone();
            scopes.sort();
            for scope in scopes { push_field(&mut hasher, scope.code()); }
            push_sorted_refs(&mut hasher, &key.evidence_refs);
        }
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

impl LifecycleHeadStatement {
    pub fn validate(&self) -> bool {
        self.schema_version == LIFECYCLE_HEAD_STATEMENT_SCHEMA_V1
            && canonical_text(&self.profile_id)
            && digest_text(&self.record_digest)
            && digest_text(&self.lifecycle_digest)
            && self.head_sequence > 0
            && if self.head_sequence == 1 { self.previous_head_digest.is_none() } else { self.previous_head_digest.as_deref().is_some_and(digest_text) }
            && digest_text(&self.authority_policy_digest)
            && canonical_text(&self.signer_key_id)
            && lower_hex_exact(&self.signer_public_key_ed25519_hex, 64)
            && valid_refs(&self.evidence_refs)
            && lower_hex_exact(&self.signature_ed25519_hex, 128)
    }

    pub fn canonical_unsigned_bytes(&self) -> Option<Vec<u8>> {
        if !self.validate() { return None; }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(HEAD_MESSAGE_DOMAIN);
        push_vec_field(&mut bytes, &self.schema_version);
        push_vec_field(&mut bytes, &self.profile_id);
        push_vec_field(&mut bytes, &self.record_digest);
        push_vec_field(&mut bytes, &self.lifecycle_digest);
        bytes.extend_from_slice(&self.head_sequence.to_be_bytes());
        push_vec_optional_string(&mut bytes, self.previous_head_digest.as_deref());
        bytes.extend_from_slice(&self.issued_at_ms.to_be_bytes());
        push_vec_field(&mut bytes, &self.authority_policy_digest);
        push_vec_field(&mut bytes, &self.signer_key_id);
        push_vec_field(&mut bytes, &self.signer_public_key_ed25519_hex);
        push_vec_sorted_refs(&mut bytes, &self.evidence_refs);
        Some(bytes)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        let unsigned = self.canonical_unsigned_bytes()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(HEAD_DIGEST_DOMAIN);
        hasher.update(&(unsigned.len() as u64).to_be_bytes());
        hasher.update(&unsigned);
        push_field(&mut hasher, &self.signature_ed25519_hex);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

impl LifecycleHeadCurrentnessAttestation {
    pub fn validate(&self) -> bool {
        self.schema_version == LIFECYCLE_HEAD_CURRENTNESS_SCHEMA_V1
            && digest_text(&self.head_statement_digest)
            && canonical_text(&self.profile_id)
            && digest_text(&self.record_digest)
            && digest_text(&self.lifecycle_digest)
            && self.head_sequence > 0
            && digest_text(&self.authority_policy_digest)
            && lower_hex_exact(&self.challenge_nonce_blake3_hex, 64)
            && canonical_text(&self.signer_key_id)
            && lower_hex_exact(&self.signer_public_key_ed25519_hex, 64)
            && valid_refs(&self.evidence_refs)
            && lower_hex_exact(&self.signature_ed25519_hex, 128)
    }

    pub fn canonical_unsigned_bytes(&self) -> Option<Vec<u8>> {
        if !self.validate() { return None; }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(CURRENTNESS_MESSAGE_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.head_statement_digest.as_str(),
            self.profile_id.as_str(),
            self.record_digest.as_str(),
            self.lifecycle_digest.as_str(),
        ] { push_vec_field(&mut bytes, value); }
        bytes.extend_from_slice(&self.head_sequence.to_be_bytes());
        push_vec_field(&mut bytes, &self.authority_policy_digest);
        bytes.extend_from_slice(&self.asserted_current_at_ms.to_be_bytes());
        push_vec_field(&mut bytes, &self.challenge_nonce_blake3_hex);
        push_vec_field(&mut bytes, &self.signer_key_id);
        push_vec_field(&mut bytes, &self.signer_public_key_ed25519_hex);
        push_vec_sorted_refs(&mut bytes, &self.evidence_refs);
        Some(bytes)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        let unsigned = self.canonical_unsigned_bytes()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(CURRENTNESS_DIGEST_DOMAIN);
        hasher.update(&(unsigned.len() as u64).to_be_bytes());
        hasher.update(&unsigned);
        push_field(&mut hasher, &self.signature_ed25519_hex);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

pub fn verify_lifecycle_head_statement(
    policy: &LifecycleHeadAuthorityPolicy,
    statement: &LifecycleHeadStatement,
) -> HeadStatementVerification {
    let policy_digest = policy.canonical_digest();
    let statement_digest = statement.canonical_digest();
    let mut issues = Vec::new();
    if !policy.validate() { issues.push(HeadStatementIssue::InvalidPolicy); }
    if !statement.validate() { issues.push(HeadStatementIssue::InvalidStatement); }
    if policy_digest.as_deref() != Some(statement.authority_policy_digest.as_str()) {
        issues.push(HeadStatementIssue::PolicyDigestMismatch);
    }
    if !issues.is_empty() {
        return head_result(policy_digest, statement_digest, statement, false, issues, None);
    }
    if !(policy.issued_at_ms <= statement.issued_at_ms && statement.issued_at_ms < policy.expires_at_ms) {
        issues.push(HeadStatementIssue::PolicyNotEffectiveAtStatementTime);
    }
    let key = policy.trusted_keys.iter().find(|key| key.key_id == statement.signer_key_id);
    let mut signature_valid = false;
    if let Some(key) = key {
        if key.public_key_ed25519_hex != statement.signer_public_key_ed25519_hex {
            issues.push(HeadStatementIssue::SignerPublicKeyMismatch);
        }
        if !key.usable_at(statement.issued_at_ms) { issues.push(HeadStatementIssue::SignerNotEffective); }
        if key.revoked_at(statement.issued_at_ms) { issues.push(HeadStatementIssue::SignerRevoked); }
        if !key.allowed_scopes.contains(&LifecycleHeadClaimScope::HeadStatement) {
            issues.push(HeadStatementIssue::ScopeNotAuthorized);
        }
        let bytes = statement.canonical_unsigned_bytes().expect("validated statement");
        match verify_signature(&statement.signer_public_key_ed25519_hex, &statement.signature_ed25519_hex, &bytes) {
            Ok(valid) => {
                signature_valid = valid;
                if !valid { issues.push(HeadStatementIssue::SignatureInvalid); }
            }
            Err(()) => issues.push(HeadStatementIssue::InvalidPublicKey),
        }
    } else {
        issues.push(HeadStatementIssue::SignerUnknown);
    }
    let untrusted = issues.iter().any(|issue| matches!(
        issue,
        HeadStatementIssue::PolicyNotEffectiveAtStatementTime
            | HeadStatementIssue::SignerUnknown
            | HeadStatementIssue::SignerPublicKeyMismatch
            | HeadStatementIssue::SignerNotEffective
            | HeadStatementIssue::SignerRevoked
            | HeadStatementIssue::ScopeNotAuthorized
    ));
    let rejected = issues.iter().any(|issue| matches!(
        issue,
        HeadStatementIssue::InvalidPublicKey | HeadStatementIssue::SignatureInvalid
    ));
    if untrusted || rejected {
        return head_result(policy_digest, statement_digest, statement, signature_valid, issues, None);
    }
    let verified = VerifiedLifecycleHeadStatement {
        profile_id: statement.profile_id.clone(),
        record_digest: statement.record_digest.clone(),
        lifecycle_digest: statement.lifecycle_digest.clone(),
        head_sequence: statement.head_sequence,
        previous_head_digest: statement.previous_head_digest.clone(),
        issued_at_ms: statement.issued_at_ms,
        statement_digest: statement_digest.clone().expect("validated statement"),
        policy_digest: policy_digest.clone().expect("validated policy"),
        signer_key_id: statement.signer_key_id.clone(),
    };
    head_result(policy_digest, statement_digest, statement, true, Vec::new(), Some(verified))
}

fn head_result(
    policy_digest: Option<String>,
    statement_digest: Option<String>,
    statement: &LifecycleHeadStatement,
    signature_valid: bool,
    issues: Vec<HeadStatementIssue>,
    verified: Option<VerifiedLifecycleHeadStatement>,
) -> HeadStatementVerification {
    let disposition = if verified.is_some() {
        HeadStatementDisposition::Verified
    } else if signature_valid {
        HeadStatementDisposition::Untrusted
    } else {
        HeadStatementDisposition::Invalid
    };
    HeadStatementVerification {
        report: HeadStatementReport {
            disposition,
            profile_id: statement.profile_id.clone(),
            record_digest: statement.record_digest.clone(),
            lifecycle_digest: statement.lifecycle_digest.clone(),
            head_sequence: statement.head_sequence,
            statement_digest,
            policy_digest,
            signer_key_id: statement.signer_key_id.clone(),
            signature_valid,
            issues,
        },
        verified,
    }
}

pub fn verify_current_active_profile(
    active: &ActiveVerifierProfile,
    head: &TrackedLifecycleHead,
    policy: &LifecycleHeadAuthorityPolicy,
    currentness: &LifecycleHeadCurrentnessAttestation,
    expected_challenge_nonce_blake3_hex: &str,
    use_at_ms: u64,
) -> LifecycleHeadCurrentnessAssessment {
    let policy_digest = policy.canonical_digest();
    let attestation_digest = currentness.canonical_digest();
    let mut issues = Vec::new();
    if !policy.validate() { issues.push(LifecycleHeadCurrentnessIssue::InvalidPolicy); }
    if !currentness.validate() { issues.push(LifecycleHeadCurrentnessIssue::InvalidAttestation); }
    if policy_digest.as_deref() != Some(currentness.authority_policy_digest.as_str()) {
        issues.push(LifecycleHeadCurrentnessIssue::PolicyDigestMismatch);
    }
    if head.policy_digest != currentness.authority_policy_digest {
        issues.push(LifecycleHeadCurrentnessIssue::HeadPolicyDigestMismatch);
    }
    if head.statement_digest != currentness.head_statement_digest {
        issues.push(LifecycleHeadCurrentnessIssue::HeadStatementDigestMismatch);
    }
    if head.profile_id != currentness.profile_id { issues.push(LifecycleHeadCurrentnessIssue::HeadProfileMismatch); }
    if head.record_digest != currentness.record_digest { issues.push(LifecycleHeadCurrentnessIssue::HeadRecordMismatch); }
    if head.lifecycle_digest != currentness.lifecycle_digest { issues.push(LifecycleHeadCurrentnessIssue::HeadLifecycleMismatch); }
    if head.head_sequence != currentness.head_sequence { issues.push(LifecycleHeadCurrentnessIssue::HeadSequenceMismatch); }
    if active.record_digest() != head.record_digest { issues.push(LifecycleHeadCurrentnessIssue::ActiveRecordMismatch); }
    if active.lifecycle_digest() != head.lifecycle_digest { issues.push(LifecycleHeadCurrentnessIssue::ActiveLifecycleMismatch); }
    if active.assessed_at_ms() != use_at_ms { issues.push(LifecycleHeadCurrentnessIssue::ActiveAssessmentNotAtUseTime); }
    if currentness.asserted_current_at_ms != use_at_ms { issues.push(LifecycleHeadCurrentnessIssue::CurrentnessNotAtUseTime); }
    if currentness.challenge_nonce_blake3_hex != expected_challenge_nonce_blake3_hex {
        issues.push(LifecycleHeadCurrentnessIssue::ChallengeNonceMismatch);
    }
    if !issues.is_empty() {
        return current_result(policy_digest, attestation_digest, head, currentness, use_at_ms, false, issues, None);
    }
    if !(policy.issued_at_ms <= use_at_ms && use_at_ms < policy.expires_at_ms) {
        issues.push(LifecycleHeadCurrentnessIssue::PolicyNotEffectiveAtUseTime);
    }
    let key = policy.trusted_keys.iter().find(|key| key.key_id == currentness.signer_key_id);
    let mut signature_valid = false;
    if let Some(key) = key {
        if key.public_key_ed25519_hex != currentness.signer_public_key_ed25519_hex {
            issues.push(LifecycleHeadCurrentnessIssue::SignerPublicKeyMismatch);
        }
        if !key.usable_at(use_at_ms) { issues.push(LifecycleHeadCurrentnessIssue::SignerNotEffective); }
        if key.revoked_at(use_at_ms) { issues.push(LifecycleHeadCurrentnessIssue::SignerRevoked); }
        if !key.allowed_scopes.contains(&LifecycleHeadClaimScope::CurrentnessAttestation) {
            issues.push(LifecycleHeadCurrentnessIssue::ScopeNotAuthorized);
        }
        let bytes = currentness.canonical_unsigned_bytes().expect("validated attestation");
        match verify_signature(&currentness.signer_public_key_ed25519_hex, &currentness.signature_ed25519_hex, &bytes) {
            Ok(valid) => {
                signature_valid = valid;
                if !valid { issues.push(LifecycleHeadCurrentnessIssue::SignatureInvalid); }
            }
            Err(()) => issues.push(LifecycleHeadCurrentnessIssue::InvalidPublicKey),
        }
    } else {
        issues.push(LifecycleHeadCurrentnessIssue::SignerUnknown);
    }
    if !issues.is_empty() {
        return current_result(policy_digest, attestation_digest, head, currentness, use_at_ms, signature_valid, issues, None);
    }
    let attestation_digest = attestation_digest.clone().expect("validated attestation");
    let current = CurrentActiveVerifierProfile {
        profile_id: head.profile_id.clone(),
        record_digest: active.record_digest().to_string(),
        graph_digest: active.graph_digest().to_string(),
        relation_completeness_digest: active.relation_completeness_digest().to_string(),
        provenance_attestation_digest: active.provenance_attestation_digest().to_string(),
        lifecycle_digest: active.lifecycle_digest().to_string(),
        lifecycle_head_statement_digest: head.statement_digest.clone(),
        currentness_attestation_digest: attestation_digest,
        head_sequence: head.head_sequence,
        verification_at_ms: active.verification_at_ms(),
        use_at_ms,
    };
    current_result(policy_digest, attestation_digest, head, currentness, use_at_ms, true, Vec::new(), Some(current))
}

#[allow(clippy::too_many_arguments)]
fn current_result(
    policy_digest: Option<String>,
    attestation_digest: Option<String>,
    head: &TrackedLifecycleHead,
    currentness: &LifecycleHeadCurrentnessAttestation,
    use_at_ms: u64,
    signature_valid: bool,
    issues: Vec<LifecycleHeadCurrentnessIssue>,
    current: Option<CurrentActiveVerifierProfile>,
) -> LifecycleHeadCurrentnessAssessment {
    let disposition = if current.is_some() {
        LifecycleHeadCurrentnessDisposition::Current
    } else if issues.iter().any(|issue| matches!(
        issue,
        LifecycleHeadCurrentnessIssue::InvalidPolicy
            | LifecycleHeadCurrentnessIssue::InvalidAttestation
            | LifecycleHeadCurrentnessIssue::PolicyDigestMismatch
            | LifecycleHeadCurrentnessIssue::HeadPolicyDigestMismatch
            | LifecycleHeadCurrentnessIssue::HeadStatementDigestMismatch
            | LifecycleHeadCurrentnessIssue::HeadProfileMismatch
            | LifecycleHeadCurrentnessIssue::HeadRecordMismatch
            | LifecycleHeadCurrentnessIssue::HeadLifecycleMismatch
            | LifecycleHeadCurrentnessIssue::HeadSequenceMismatch
            | LifecycleHeadCurrentnessIssue::ActiveRecordMismatch
            | LifecycleHeadCurrentnessIssue::ActiveLifecycleMismatch
            | LifecycleHeadCurrentnessIssue::InvalidPublicKey
            | LifecycleHeadCurrentnessIssue::SignatureInvalid
    )) {
        LifecycleHeadCurrentnessDisposition::Invalid
    } else {
        LifecycleHeadCurrentnessDisposition::Blocked
    };
    LifecycleHeadCurrentnessAssessment {
        report: LifecycleHeadCurrentnessReport {
            disposition,
            profile_id: head.profile_id.clone(),
            record_digest: head.record_digest.clone(),
            lifecycle_digest: head.lifecycle_digest.clone(),
            head_sequence: head.head_sequence,
            head_statement_digest: head.statement_digest.clone(),
            currentness_attestation_digest: attestation_digest,
            policy_digest,
            use_at_ms,
            signer_key_id: currentness.signer_key_id.clone(),
            signature_valid,
            issues,
        },
        current,
    }
}

fn verify_signature(public_key_hex: &str, signature_hex: &str, message: &[u8]) -> Result<bool, ()> {
    let public_key = hex::decode(public_key_hex).map_err(|_| ())?;
    let signature = hex::decode(signature_hex).map_err(|_| ())?;
    let public_key: [u8; 32] = public_key.try_into().map_err(|_| ())?;
    let signature: [u8; 64] = signature.try_into().map_err(|_| ())?;
    let key = VerifyingKey::from_bytes(&public_key).map_err(|_| ())?;
    let signature = Signature::from_bytes(&signature);
    Ok(key.verify(message, &signature).is_ok())
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty() && value.len() <= MAX_TEXT_BYTES && value.trim() == value
}

fn digest_text(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|digest| lower_hex_exact(digest, 64))
}

fn lower_hex_exact(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_refs(values: &[String]) -> bool {
    !values.is_empty()
        && values.len() <= MAX_EVIDENCE_REFS
        && values.iter().all(|value| canonical_text(value))
        && values.iter().collect::<BTreeSet<_>>().len() == values.len()
}

fn unique<T: Ord + Copy>(values: &[T]) -> bool {
    values.iter().copied().collect::<BTreeSet<_>>().len() == values.len()
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn push_u64(hasher: &mut blake3::Hasher, value: u64) { hasher.update(&value.to_be_bytes()); }

fn push_optional_u64(hasher: &mut blake3::Hasher, value: Option<u64>) {
    match value {
        Some(value) => { hasher.update(&[1]); push_u64(hasher, value); }
        None => { hasher.update(&[0]); }
    }
}

fn push_sorted_refs(hasher: &mut blake3::Hasher, values: &[String]) {
    let mut values = values.to_vec();
    values.sort();
    for value in values { push_field(hasher, &value); }
}

fn push_vec_field(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn push_vec_optional_string(bytes: &mut Vec<u8>, value: Option<&str>) {
    match value {
        Some(value) => { bytes.push(1); push_vec_field(bytes, value); }
        None => bytes.push(0),
    }
}

fn push_vec_sorted_refs(bytes: &mut Vec<u8>, values: &[String]) {
    let mut values = values.to_vec();
    values.sort();
    for value in values { push_vec_field(bytes, &value); }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn d(label: &str) -> String { format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex()) }
    fn nonce(label: &str) -> String { blake3::hash(label.as_bytes()).to_hex().to_string() }

    fn authority() -> (SigningKey, LifecycleHeadAuthorityPolicy) {
        let signing = SigningKey::from_bytes(&[17u8; 32]);
        let key = LifecycleHeadAuthorityKey {
            key_id: "head-authority:1".into(),
            public_key_ed25519_hex: hex::encode(signing.verifying_key().as_bytes()),
            valid_from_ms: 1,
            valid_until_ms: Some(10_000),
            revoked_at_ms: None,
            allowed_scopes: vec![LifecycleHeadClaimScope::HeadStatement, LifecycleHeadClaimScope::CurrentnessAttestation],
            evidence_refs: vec!["review:key".into()],
        };
        let policy = LifecycleHeadAuthorityPolicy {
            schema_version: HEAD_AUTHORITY_POLICY_SCHEMA_V1.into(),
            policy_id: "policy:head".into(),
            sequence: 1,
            issued_at_ms: 1,
            expires_at_ms: 10_000,
            trusted_keys: vec![key],
            evidence_refs: vec!["review:policy".into()],
        };
        (signing, policy)
    }

    fn signed_head(
        signing: &SigningKey,
        policy: &LifecycleHeadAuthorityPolicy,
        sequence: u64,
        previous: Option<String>,
        lifecycle: &str,
    ) -> LifecycleHeadStatement {
        let mut statement = LifecycleHeadStatement {
            schema_version: LIFECYCLE_HEAD_STATEMENT_SCHEMA_V1.into(),
            profile_id: "profile:a".into(),
            record_digest: d("record:a"),
            lifecycle_digest: lifecycle.into(),
            head_sequence: sequence,
            previous_head_digest: previous,
            issued_at_ms: 100 + sequence,
            authority_policy_digest: policy.canonical_digest().unwrap(),
            signer_key_id: "head-authority:1".into(),
            signer_public_key_ed25519_hex: hex::encode(signing.verifying_key().as_bytes()),
            evidence_refs: vec![format!("head:{sequence}")],
            signature_ed25519_hex: "00".repeat(64),
        };
        let bytes = statement.canonical_unsigned_bytes().unwrap();
        statement.signature_ed25519_hex = hex::encode(signing.sign(&bytes).to_bytes());
        statement
    }

    #[test]
    fn signed_genesis_head_verifies_and_tracks() {
        let (signing, policy) = authority();
        let statement = signed_head(&signing, &policy, 1, None, &d("ledger:1"));
        let verification = verify_lifecycle_head_statement(&policy, &statement);
        assert_eq!(verification.report.disposition, HeadStatementDisposition::Verified);
        let verified = verification.into_verified().unwrap();
        let mut tracker = LifecycleHeadTracker::default();
        let tracked = tracker.accept(&verified).unwrap();
        assert_eq!(tracked.head_sequence(), 1);
        assert_eq!(tracker.latest_digest("profile:a"), Some(tracked.statement_digest()));
    }

    #[test]
    fn tracker_requires_contiguous_hash_linked_heads() {
        let (signing, policy) = authority();
        let first = signed_head(&signing, &policy, 1, None, &d("ledger:1"));
        let first = verify_lifecycle_head_statement(&policy, &first).into_verified().unwrap();
        let mut tracker = LifecycleHeadTracker::default();
        let tracked = tracker.accept(&first).unwrap();
        let third = signed_head(&signing, &policy, 3, Some(tracked.statement_digest().into()), &d("ledger:3"));
        let third = verify_lifecycle_head_statement(&policy, &third).into_verified().unwrap();
        assert!(matches!(tracker.accept(&third), Err(LifecycleHeadTrackingError::SequenceGap { .. })));
        let second = signed_head(&signing, &policy, 2, Some(d("wrong-head")), &d("ledger:2"));
        let second = verify_lifecycle_head_statement(&policy, &second).into_verified().unwrap();
        assert_eq!(tracker.accept(&second), Err(LifecycleHeadTrackingError::PreviousHeadDigestMismatch));
    }

    #[test]
    fn tracker_rejects_fresh_bootstrap_at_non_genesis_sequence() {
        let (signing, policy) = authority();
        let statement = signed_head(&signing, &policy, 7, Some(d("prior")), &d("ledger:7"));
        let verified = verify_lifecycle_head_statement(&policy, &statement).into_verified().unwrap();
        let mut tracker = LifecycleHeadTracker::default();
        assert!(matches!(tracker.accept(&verified), Err(LifecycleHeadTrackingError::FirstHeadMustBeGenesis { proposed_sequence: 7 })));
    }

    #[test]
    fn statement_digest_changes_with_lifecycle_head() {
        let (signing, policy) = authority();
        let left = signed_head(&signing, &policy, 1, None, &d("ledger:a"));
        let right = signed_head(&signing, &policy, 1, None, &d("ledger:b"));
        assert_ne!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn currentness_challenge_is_content_bound() {
        let (signing, policy) = authority();
        let statement = signed_head(&signing, &policy, 1, None, &d("ledger:1"));
        let statement_digest = statement.canonical_digest().unwrap();
        let mut currentness = LifecycleHeadCurrentnessAttestation {
            schema_version: LIFECYCLE_HEAD_CURRENTNESS_SCHEMA_V1.into(),
            head_statement_digest: statement_digest,
            profile_id: "profile:a".into(),
            record_digest: d("record:a"),
            lifecycle_digest: d("ledger:1"),
            head_sequence: 1,
            authority_policy_digest: policy.canonical_digest().unwrap(),
            asserted_current_at_ms: 500,
            challenge_nonce_blake3_hex: nonce("challenge:a"),
            signer_key_id: "head-authority:1".into(),
            signer_public_key_ed25519_hex: hex::encode(signing.verifying_key().as_bytes()),
            evidence_refs: vec!["currentness:1".into()],
            signature_ed25519_hex: "00".repeat(64),
        };
        let before = currentness.canonical_unsigned_bytes().unwrap();
        currentness.signature_ed25519_hex = hex::encode(signing.sign(&before).to_bytes());
        let digest_a = currentness.canonical_digest().unwrap();
        currentness.challenge_nonce_blake3_hex = nonce("challenge:b");
        let digest_b = currentness.canonical_digest().unwrap();
        assert_ne!(digest_a, digest_b);
    }
}
