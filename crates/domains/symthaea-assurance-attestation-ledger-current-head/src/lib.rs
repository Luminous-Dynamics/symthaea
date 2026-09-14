// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Signed current-head and anti-rollback assurance for attestation acceptance ledgers.

#![deny(unsafe_code)]

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const LEDGER_HEAD_AUTHORITY_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.attestation-ledger-head-authority-policy.v1";
pub const LEDGER_HEAD_STATEMENT_SCHEMA_V1: &str =
    "symthaea.assurance.attestation-ledger-head-statement.v1";
pub const LEDGER_HEAD_CURRENTNESS_SCHEMA_V1: &str =
    "symthaea.assurance.attestation-ledger-head-currentness.v1";

pub const MAX_TRUSTED_KEYS: usize = 1_024;
pub const MAX_EVIDENCE_REFS: usize = 128;
pub const MAX_TEXT_BYTES: usize = 512;

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.attestation-ledger-head-authority-policy.digest.v1\0";
const HEAD_MESSAGE_DOMAIN: &[u8] =
    b"symthaea.assurance.attestation-ledger-head-statement.message.v1\0";
const HEAD_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.attestation-ledger-head-statement.digest.v1\0";
const CURRENTNESS_MESSAGE_DOMAIN: &[u8] =
    b"symthaea.assurance.attestation-ledger-head-currentness.message.v1\0";
const CURRENTNESS_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.attestation-ledger-head-currentness.digest.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LedgerHeadClaimScope {
    HeadStatement,
    CurrentnessAttestation,
}

impl LedgerHeadClaimScope {
    fn code(self) -> &'static str {
        match self {
            Self::HeadStatement => "head-statement",
            Self::CurrentnessAttestation => "currentness-attestation",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerHeadAuthorityKey {
    pub key_id: String,
    pub public_key_ed25519_hex: String,
    pub valid_from_ms: u64,
    pub valid_until_ms: Option<u64>,
    pub revoked_at_ms: Option<u64>,
    pub allowed_scopes: Vec<LedgerHeadClaimScope>,
    pub evidence_refs: Vec<String>,
}

impl LedgerHeadAuthorityKey {
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
            && !self.allowed_scopes.is_empty()
            && unique(&self.allowed_scopes)
            && valid_refs(&self.evidence_refs)
    }

    fn usable_at(&self, at_ms: u64) -> bool {
        at_ms >= self.valid_from_ms
            && self
                .valid_until_ms
                .map(|until| at_ms < until)
                .unwrap_or(true)
    }

    fn revoked_at(&self, at_ms: u64) -> bool {
        self.revoked_at_ms
            .map(|revoked| at_ms >= revoked)
            .unwrap_or(false)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerHeadAuthorityPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub sequence: u64,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub trusted_keys: Vec<LedgerHeadAuthorityKey>,
    pub evidence_refs: Vec<String>,
}

impl LedgerHeadAuthorityPolicy {
    pub fn validate(&self) -> bool {
        if self.schema_version != LEDGER_HEAD_AUTHORITY_POLICY_SCHEMA_V1
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
        self.trusted_keys
            .iter()
            .all(|key| key.validate() && ids.insert(key.key_id.as_str()))
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(POLICY_DIGEST_DOMAIN);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.policy_id);
        push_u64(&mut hasher, self.sequence);
        push_u64(&mut hasher, self.issued_at_ms);
        push_u64(&mut hasher, self.expires_at_ms);

        let mut keys = self.trusted_keys.iter().collect::<Vec<_>>();
        keys.sort_by(|left, right| left.key_id.cmp(&right.key_id));
        hasher.update(&(keys.len() as u64).to_le_bytes());
        for key in keys {
            push_field(&mut hasher, &key.key_id);
            push_field(&mut hasher, &key.public_key_ed25519_hex);
            push_u64(&mut hasher, key.valid_from_ms);
            push_optional_u64(&mut hasher, key.valid_until_ms);
            push_optional_u64(&mut hasher, key.revoked_at_ms);
            let mut scopes = key.allowed_scopes.clone();
            scopes.sort();
            hasher.update(&(scopes.len() as u64).to_le_bytes());
            for scope in scopes {
                push_field(&mut hasher, scope.code());
            }
            push_sorted_refs(&mut hasher, &key.evidence_refs);
        }
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcceptanceLedgerHeadStatement {
    pub schema_version: String,
    pub ledger_id: String,
    /// The authoritative head sequence is the acceptance revision itself.
    pub acceptance_revision: u64,
    pub acceptance_digest: String,
    pub previous_head_statement_digest: Option<String>,
    pub issued_at_ms: u64,
    pub authority_policy_digest: String,
    pub signer_key_id: String,
    pub signer_public_key_ed25519_hex: String,
    pub evidence_refs: Vec<String>,
    pub signature_ed25519_hex: String,
}

impl AcceptanceLedgerHeadStatement {
    pub fn validate(&self) -> bool {
        self.schema_version == LEDGER_HEAD_STATEMENT_SCHEMA_V1
            && canonical_text(&self.ledger_id)
            && self.acceptance_revision > 0
            && digest_text(&self.acceptance_digest)
            && if self.acceptance_revision == 1 {
                self.previous_head_statement_digest.is_none()
            } else {
                self.previous_head_statement_digest
                    .as_deref()
                    .is_some_and(digest_text)
            }
            && self.issued_at_ms > 0
            && digest_text(&self.authority_policy_digest)
            && canonical_text(&self.signer_key_id)
            && lower_hex_exact(&self.signer_public_key_ed25519_hex, 64)
            && valid_refs(&self.evidence_refs)
            && lower_hex_exact(&self.signature_ed25519_hex, 128)
    }

    pub fn canonical_unsigned_bytes(&self) -> Option<Vec<u8>> {
        if !self.validate() {
            return None;
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(HEAD_MESSAGE_DOMAIN);
        push_vec_field(&mut bytes, &self.schema_version);
        push_vec_field(&mut bytes, &self.ledger_id);
        bytes.extend_from_slice(&self.acceptance_revision.to_be_bytes());
        push_vec_field(&mut bytes, &self.acceptance_digest);
        push_vec_optional_string(&mut bytes, self.previous_head_statement_digest.as_deref());
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcceptanceLedgerHeadCurrentnessAttestation {
    pub schema_version: String,
    pub head_statement_digest: String,
    pub ledger_id: String,
    pub acceptance_revision: u64,
    pub acceptance_digest: String,
    pub authority_policy_digest: String,
    /// The authority asserts that this exact head is current at this exact time.
    /// Trusted-time provenance remains outside this crate.
    pub asserted_current_at_ms: u64,
    /// BLAKE3 digest bytes rendered as 64 lowercase hex characters. The caller
    /// supplies the expected value separately to prevent replay across uses.
    pub challenge_nonce_blake3_hex: String,
    pub signer_key_id: String,
    pub signer_public_key_ed25519_hex: String,
    pub evidence_refs: Vec<String>,
    pub signature_ed25519_hex: String,
}

impl AcceptanceLedgerHeadCurrentnessAttestation {
    pub fn validate(&self) -> bool {
        self.schema_version == LEDGER_HEAD_CURRENTNESS_SCHEMA_V1
            && digest_text(&self.head_statement_digest)
            && canonical_text(&self.ledger_id)
            && self.acceptance_revision > 0
            && digest_text(&self.acceptance_digest)
            && digest_text(&self.authority_policy_digest)
            && self.asserted_current_at_ms > 0
            && lower_hex_exact(&self.challenge_nonce_blake3_hex, 64)
            && canonical_text(&self.signer_key_id)
            && lower_hex_exact(&self.signer_public_key_ed25519_hex, 64)
            && valid_refs(&self.evidence_refs)
            && lower_hex_exact(&self.signature_ed25519_hex, 128)
    }

    pub fn canonical_unsigned_bytes(&self) -> Option<Vec<u8>> {
        if !self.validate() {
            return None;
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(CURRENTNESS_MESSAGE_DOMAIN);
        push_vec_field(&mut bytes, &self.schema_version);
        push_vec_field(&mut bytes, &self.head_statement_digest);
        push_vec_field(&mut bytes, &self.ledger_id);
        bytes.extend_from_slice(&self.acceptance_revision.to_be_bytes());
        push_vec_field(&mut bytes, &self.acceptance_digest);
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
    pub ledger_id: String,
    pub acceptance_revision: u64,
    pub acceptance_digest: String,
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
pub struct VerifiedAcceptanceLedgerHeadStatement {
    ledger_id: String,
    acceptance_revision: u64,
    acceptance_digest: String,
    previous_head_statement_digest: Option<String>,
    issued_at_ms: u64,
    statement_digest: String,
    policy_digest: String,
    signer_key_id: String,
}

impl VerifiedAcceptanceLedgerHeadStatement {
    pub fn ledger_id(&self) -> &str {
        &self.ledger_id
    }
    pub const fn acceptance_revision(&self) -> u64 {
        self.acceptance_revision
    }
    pub fn acceptance_digest(&self) -> &str {
        &self.acceptance_digest
    }
    pub fn previous_head_statement_digest(&self) -> Option<&str> {
        self.previous_head_statement_digest.as_deref()
    }
    pub const fn issued_at_ms(&self) -> u64 {
        self.issued_at_ms
    }
    pub fn statement_digest(&self) -> &str {
        &self.statement_digest
    }
    pub fn policy_digest(&self) -> &str {
        &self.policy_digest
    }
    pub fn signer_key_id(&self) -> &str {
        &self.signer_key_id
    }
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeadStatementVerification {
    pub report: HeadStatementReport,
    verified: Option<VerifiedAcceptanceLedgerHeadStatement>,
}

impl HeadStatementVerification {
    pub fn verified(&self) -> Option<&VerifiedAcceptanceLedgerHeadStatement> {
        self.verified.as_ref()
    }
    pub fn into_verified(self) -> Option<VerifiedAcceptanceLedgerHeadStatement> {
        self.verified
    }
}

pub fn verify_acceptance_ledger_head_statement(
    policy: &LedgerHeadAuthorityPolicy,
    statement: &AcceptanceLedgerHeadStatement,
) -> HeadStatementVerification {
    let policy_digest = policy.canonical_digest();
    let statement_digest = statement.canonical_digest();
    let mut issues = Vec::new();
    if !policy.validate() {
        issues.push(HeadStatementIssue::InvalidPolicy);
    }
    if !statement.validate() {
        issues.push(HeadStatementIssue::InvalidStatement);
    }
    if policy_digest.as_deref() != Some(statement.authority_policy_digest.as_str()) {
        issues.push(HeadStatementIssue::PolicyDigestMismatch);
    }
    if !issues.is_empty() {
        return head_result(
            policy_digest,
            statement_digest,
            statement,
            false,
            issues,
            None,
        );
    }
    if !(policy.issued_at_ms <= statement.issued_at_ms
        && statement.issued_at_ms < policy.expires_at_ms)
    {
        issues.push(HeadStatementIssue::PolicyNotEffectiveAtStatementTime);
    }

    let key = policy
        .trusted_keys
        .iter()
        .find(|key| key.key_id == statement.signer_key_id);
    let mut signature_valid = false;
    if let Some(key) = key {
        if key.public_key_ed25519_hex != statement.signer_public_key_ed25519_hex {
            issues.push(HeadStatementIssue::SignerPublicKeyMismatch);
        }
        if !key.usable_at(statement.issued_at_ms) {
            issues.push(HeadStatementIssue::SignerNotEffective);
        }
        if key.revoked_at(statement.issued_at_ms) {
            issues.push(HeadStatementIssue::SignerRevoked);
        }
        if !key.allowed_scopes.contains(&LedgerHeadClaimScope::HeadStatement) {
            issues.push(HeadStatementIssue::ScopeNotAuthorized);
        }
        let bytes = statement
            .canonical_unsigned_bytes()
            .expect("validated statement has canonical bytes");
        match verify_signature(
            &statement.signer_public_key_ed25519_hex,
            &statement.signature_ed25519_hex,
            &bytes,
        ) {
            Ok(valid) => {
                signature_valid = valid;
                if !valid {
                    issues.push(HeadStatementIssue::SignatureInvalid);
                }
            }
            Err(()) => issues.push(HeadStatementIssue::InvalidPublicKey),
        }
    } else {
        issues.push(HeadStatementIssue::SignerUnknown);
    }

    if !issues.is_empty() {
        return head_result(
            policy_digest,
            statement_digest,
            statement,
            signature_valid,
            issues,
            None,
        );
    }

    let verified = VerifiedAcceptanceLedgerHeadStatement {
        ledger_id: statement.ledger_id.clone(),
        acceptance_revision: statement.acceptance_revision,
        acceptance_digest: statement.acceptance_digest.clone(),
        previous_head_statement_digest: statement.previous_head_statement_digest.clone(),
        issued_at_ms: statement.issued_at_ms,
        statement_digest: statement_digest.clone().expect("validated statement has digest"),
        policy_digest: policy_digest.clone().expect("validated policy has digest"),
        signer_key_id: statement.signer_key_id.clone(),
    };
    head_result(
        policy_digest,
        statement_digest,
        statement,
        true,
        Vec::new(),
        Some(verified),
    )
}

fn head_result(
    policy_digest: Option<String>,
    statement_digest: Option<String>,
    statement: &AcceptanceLedgerHeadStatement,
    signature_valid: bool,
    issues: Vec<HeadStatementIssue>,
    verified: Option<VerifiedAcceptanceLedgerHeadStatement>,
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
            ledger_id: statement.ledger_id.clone(),
            acceptance_revision: statement.acceptance_revision,
            acceptance_digest: statement.acceptance_digest.clone(),
            statement_digest,
            policy_digest,
            signer_key_id: statement.signer_key_id.clone(),
            signature_valid,
            issues,
        },
        verified,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TrackedHeadState {
    revision: u64,
    statement_digest: String,
    issued_at_ms: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AcceptanceLedgerHeadTracker {
    heads: BTreeMap<String, TrackedHeadState>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcceptanceLedgerHeadTrackingError {
    FirstHeadMustBeGenesis { proposed_revision: u64 },
    FirstHeadHasPreviousDigest,
    RevisionRollback { latest: u64, proposed: u64 },
    RevisionCollision { revision: u64 },
    RevisionGap { expected: u64, proposed: u64 },
    PreviousHeadDigestMismatch,
    IssuedAtRegressed { latest: u64, proposed: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackedAcceptanceLedgerHead {
    ledger_id: String,
    acceptance_revision: u64,
    acceptance_digest: String,
    statement_digest: String,
    policy_digest: String,
    issued_at_ms: u64,
}

impl TrackedAcceptanceLedgerHead {
    pub fn ledger_id(&self) -> &str {
        &self.ledger_id
    }
    pub const fn acceptance_revision(&self) -> u64 {
        self.acceptance_revision
    }
    pub fn acceptance_digest(&self) -> &str {
        &self.acceptance_digest
    }
    pub fn statement_digest(&self) -> &str {
        &self.statement_digest
    }
    pub fn policy_digest(&self) -> &str {
        &self.policy_digest
    }
    pub const fn issued_at_ms(&self) -> u64 {
        self.issued_at_ms
    }
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

impl AcceptanceLedgerHeadTracker {
    pub fn accept(
        &mut self,
        head: &VerifiedAcceptanceLedgerHeadStatement,
    ) -> Result<TrackedAcceptanceLedgerHead, AcceptanceLedgerHeadTrackingError> {
        let ledger_id = head.ledger_id.clone();
        match self.heads.get(&ledger_id) {
            None => {
                if head.acceptance_revision != 1 {
                    return Err(
                        AcceptanceLedgerHeadTrackingError::FirstHeadMustBeGenesis {
                            proposed_revision: head.acceptance_revision,
                        },
                    );
                }
                if head.previous_head_statement_digest.is_some() {
                    return Err(AcceptanceLedgerHeadTrackingError::FirstHeadHasPreviousDigest);
                }
            }
            Some(latest) => {
                if head.acceptance_revision < latest.revision {
                    return Err(AcceptanceLedgerHeadTrackingError::RevisionRollback {
                        latest: latest.revision,
                        proposed: head.acceptance_revision,
                    });
                }
                if head.acceptance_revision == latest.revision {
                    if head.statement_digest == latest.statement_digest {
                        return Ok(tracked(head));
                    }
                    return Err(AcceptanceLedgerHeadTrackingError::RevisionCollision {
                        revision: head.acceptance_revision,
                    });
                }
                let expected = latest.revision.saturating_add(1);
                if head.acceptance_revision != expected {
                    return Err(AcceptanceLedgerHeadTrackingError::RevisionGap {
                        expected,
                        proposed: head.acceptance_revision,
                    });
                }
                if head.previous_head_statement_digest.as_deref()
                    != Some(latest.statement_digest.as_str())
                {
                    return Err(AcceptanceLedgerHeadTrackingError::PreviousHeadDigestMismatch);
                }
                if head.issued_at_ms < latest.issued_at_ms {
                    return Err(AcceptanceLedgerHeadTrackingError::IssuedAtRegressed {
                        latest: latest.issued_at_ms,
                        proposed: head.issued_at_ms,
                    });
                }
            }
        }

        self.heads.insert(
            ledger_id,
            TrackedHeadState {
                revision: head.acceptance_revision,
                statement_digest: head.statement_digest.clone(),
                issued_at_ms: head.issued_at_ms,
            },
        );
        Ok(tracked(head))
    }

    pub fn latest_revision(&self, ledger_id: &str) -> Option<u64> {
        self.heads.get(ledger_id).map(|state| state.revision)
    }

    pub fn latest_statement_digest(&self, ledger_id: &str) -> Option<&str> {
        self.heads
            .get(ledger_id)
            .map(|state| state.statement_digest.as_str())
    }

    pub fn is_latest(&self, head: &TrackedAcceptanceLedgerHead) -> bool {
        self.heads.get(&head.ledger_id).is_some_and(|latest| {
            latest.revision == head.acceptance_revision
                && latest.statement_digest == head.statement_digest
        })
    }
}

fn tracked(head: &VerifiedAcceptanceLedgerHeadStatement) -> TrackedAcceptanceLedgerHead {
    TrackedAcceptanceLedgerHead {
        ledger_id: head.ledger_id.clone(),
        acceptance_revision: head.acceptance_revision,
        acceptance_digest: head.acceptance_digest.clone(),
        statement_digest: head.statement_digest.clone(),
        policy_digest: head.policy_digest.clone(),
        issued_at_ms: head.issued_at_ms,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LedgerHeadCurrentnessDisposition {
    Invalid,
    Blocked,
    Current,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LedgerHeadCurrentnessIssue {
    InvalidPolicy,
    InvalidAttestation,
    InvalidExpectedChallenge,
    PolicyDigestMismatch,
    HeadPolicyDigestMismatch,
    HeadNotLatestInTracker,
    HeadStatementDigestMismatch,
    HeadLedgerMismatch,
    HeadRevisionMismatch,
    HeadAcceptanceDigestMismatch,
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
pub struct LedgerHeadCurrentnessReport {
    pub disposition: LedgerHeadCurrentnessDisposition,
    pub ledger_id: String,
    pub acceptance_revision: u64,
    pub acceptance_digest: String,
    pub head_statement_digest: String,
    pub currentness_attestation_digest: Option<String>,
    pub policy_digest: Option<String>,
    pub use_at_ms: u64,
    pub signer_key_id: String,
    pub signature_valid: bool,
    pub issues: Vec<LedgerHeadCurrentnessIssue>,
}

impl LedgerHeadCurrentnessReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentAcceptanceLedgerHead {
    ledger_id: String,
    acceptance_revision: u64,
    acceptance_digest: String,
    head_statement_digest: String,
    currentness_attestation_digest: String,
    authority_policy_digest: String,
    use_at_ms: u64,
}

impl CurrentAcceptanceLedgerHead {
    pub fn ledger_id(&self) -> &str {
        &self.ledger_id
    }
    pub const fn acceptance_revision(&self) -> u64 {
        self.acceptance_revision
    }
    pub fn acceptance_digest(&self) -> &str {
        &self.acceptance_digest
    }
    pub fn head_statement_digest(&self) -> &str {
        &self.head_statement_digest
    }
    pub fn currentness_attestation_digest(&self) -> &str {
        &self.currentness_attestation_digest
    }
    pub fn authority_policy_digest(&self) -> &str {
        &self.authority_policy_digest
    }
    pub const fn use_at_ms(&self) -> u64 {
        self.use_at_ms
    }
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerHeadCurrentnessAssessment {
    pub report: LedgerHeadCurrentnessReport,
    current: Option<CurrentAcceptanceLedgerHead>,
}

impl LedgerHeadCurrentnessAssessment {
    pub fn current(&self) -> Option<&CurrentAcceptanceLedgerHead> {
        self.current.as_ref()
    }
    pub fn into_current(self) -> Option<CurrentAcceptanceLedgerHead> {
        self.current
    }
}

pub fn verify_current_acceptance_ledger_head(
    tracker: &AcceptanceLedgerHeadTracker,
    head: &TrackedAcceptanceLedgerHead,
    policy: &LedgerHeadAuthorityPolicy,
    currentness: &AcceptanceLedgerHeadCurrentnessAttestation,
    expected_challenge_nonce_blake3_hex: &str,
    use_at_ms: u64,
) -> LedgerHeadCurrentnessAssessment {
    let policy_digest = policy.canonical_digest();
    let attestation_digest = currentness.canonical_digest();
    let mut issues = Vec::new();

    if !policy.validate() {
        issues.push(LedgerHeadCurrentnessIssue::InvalidPolicy);
    }
    if !currentness.validate() {
        issues.push(LedgerHeadCurrentnessIssue::InvalidAttestation);
    }
    if !lower_hex_exact(expected_challenge_nonce_blake3_hex, 64) {
        issues.push(LedgerHeadCurrentnessIssue::InvalidExpectedChallenge);
    }
    if policy_digest.as_deref() != Some(currentness.authority_policy_digest.as_str()) {
        issues.push(LedgerHeadCurrentnessIssue::PolicyDigestMismatch);
    }
    if head.policy_digest != currentness.authority_policy_digest {
        issues.push(LedgerHeadCurrentnessIssue::HeadPolicyDigestMismatch);
    }
    if !tracker.is_latest(head) {
        issues.push(LedgerHeadCurrentnessIssue::HeadNotLatestInTracker);
    }
    if head.statement_digest != currentness.head_statement_digest {
        issues.push(LedgerHeadCurrentnessIssue::HeadStatementDigestMismatch);
    }
    if head.ledger_id != currentness.ledger_id {
        issues.push(LedgerHeadCurrentnessIssue::HeadLedgerMismatch);
    }
    if head.acceptance_revision != currentness.acceptance_revision {
        issues.push(LedgerHeadCurrentnessIssue::HeadRevisionMismatch);
    }
    if head.acceptance_digest != currentness.acceptance_digest {
        issues.push(LedgerHeadCurrentnessIssue::HeadAcceptanceDigestMismatch);
    }
    if currentness.asserted_current_at_ms != use_at_ms {
        issues.push(LedgerHeadCurrentnessIssue::CurrentnessNotAtUseTime);
    }
    if currentness.challenge_nonce_blake3_hex != expected_challenge_nonce_blake3_hex {
        issues.push(LedgerHeadCurrentnessIssue::ChallengeNonceMismatch);
    }

    if !issues.is_empty() {
        return currentness_result(
            policy_digest,
            attestation_digest,
            head,
            currentness,
            use_at_ms,
            false,
            issues,
            None,
        );
    }

    if !(policy.issued_at_ms <= use_at_ms && use_at_ms < policy.expires_at_ms) {
        issues.push(LedgerHeadCurrentnessIssue::PolicyNotEffectiveAtUseTime);
    }
    let key = policy
        .trusted_keys
        .iter()
        .find(|key| key.key_id == currentness.signer_key_id);
    let mut signature_valid = false;
    if let Some(key) = key {
        if key.public_key_ed25519_hex != currentness.signer_public_key_ed25519_hex {
            issues.push(LedgerHeadCurrentnessIssue::SignerPublicKeyMismatch);
        }
        if !key.usable_at(use_at_ms) {
            issues.push(LedgerHeadCurrentnessIssue::SignerNotEffective);
        }
        if key.revoked_at(use_at_ms) {
            issues.push(LedgerHeadCurrentnessIssue::SignerRevoked);
        }
        if !key
            .allowed_scopes
            .contains(&LedgerHeadClaimScope::CurrentnessAttestation)
        {
            issues.push(LedgerHeadCurrentnessIssue::ScopeNotAuthorized);
        }
        let bytes = currentness
            .canonical_unsigned_bytes()
            .expect("validated currentness has canonical bytes");
        match verify_signature(
            &currentness.signer_public_key_ed25519_hex,
            &currentness.signature_ed25519_hex,
            &bytes,
        ) {
            Ok(valid) => {
                signature_valid = valid;
                if !valid {
                    issues.push(LedgerHeadCurrentnessIssue::SignatureInvalid);
                }
            }
            Err(()) => issues.push(LedgerHeadCurrentnessIssue::InvalidPublicKey),
        }
    } else {
        issues.push(LedgerHeadCurrentnessIssue::SignerUnknown);
    }

    if !issues.is_empty() {
        return currentness_result(
            policy_digest,
            attestation_digest,
            head,
            currentness,
            use_at_ms,
            signature_valid,
            issues,
            None,
        );
    }

    let current = CurrentAcceptanceLedgerHead {
        ledger_id: head.ledger_id.clone(),
        acceptance_revision: head.acceptance_revision,
        acceptance_digest: head.acceptance_digest.clone(),
        head_statement_digest: head.statement_digest.clone(),
        currentness_attestation_digest: attestation_digest
            .clone()
            .expect("validated currentness has digest"),
        authority_policy_digest: policy_digest.clone().expect("validated policy has digest"),
        use_at_ms,
    };
    currentness_result(
        policy_digest,
        attestation_digest,
        head,
        currentness,
        use_at_ms,
        true,
        Vec::new(),
        Some(current),
    )
}

#[allow(clippy::too_many_arguments)]
fn currentness_result(
    policy_digest: Option<String>,
    attestation_digest: Option<String>,
    head: &TrackedAcceptanceLedgerHead,
    currentness: &AcceptanceLedgerHeadCurrentnessAttestation,
    use_at_ms: u64,
    signature_valid: bool,
    issues: Vec<LedgerHeadCurrentnessIssue>,
    current: Option<CurrentAcceptanceLedgerHead>,
) -> LedgerHeadCurrentnessAssessment {
    let disposition = if current.is_some() {
        LedgerHeadCurrentnessDisposition::Current
    } else if issues.iter().any(|issue| {
        matches!(
            issue,
            LedgerHeadCurrentnessIssue::InvalidPolicy
                | LedgerHeadCurrentnessIssue::InvalidAttestation
                | LedgerHeadCurrentnessIssue::InvalidExpectedChallenge
                | LedgerHeadCurrentnessIssue::InvalidPublicKey
                | LedgerHeadCurrentnessIssue::SignatureInvalid
        )
    }) {
        LedgerHeadCurrentnessDisposition::Invalid
    } else {
        LedgerHeadCurrentnessDisposition::Blocked
    };

    LedgerHeadCurrentnessAssessment {
        report: LedgerHeadCurrentnessReport {
            disposition,
            ledger_id: head.ledger_id.clone(),
            acceptance_revision: head.acceptance_revision,
            acceptance_digest: head.acceptance_digest.clone(),
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

fn verify_signature(public_hex: &str, signature_hex: &str, message: &[u8]) -> Result<bool, ()> {
    let public = hex::decode(public_hex).map_err(|_| ())?;
    let public: [u8; 32] = public.try_into().map_err(|_| ())?;
    let verifying_key = VerifyingKey::from_bytes(&public).map_err(|_| ())?;
    let signature = hex::decode(signature_hex).map_err(|_| ())?;
    let signature: [u8; 64] = signature.try_into().map_err(|_| ())?;
    let signature = Signature::from_bytes(&signature);
    Ok(verifying_key.verify(message, &signature).is_ok())
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
            && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            && !digest.bytes().any(|byte| byte.is_ascii_uppercase())
    })
}

fn lower_hex_exact(value: &str, len: usize) -> bool {
    value.len() == len
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
        && !value.bytes().any(|byte| byte.is_ascii_uppercase())
}

fn valid_refs(values: &[String]) -> bool {
    !values.is_empty()
        && values.len() <= MAX_EVIDENCE_REFS
        && values.iter().all(|value| canonical_text(value))
        && values.iter().collect::<BTreeSet<_>>().len() == values.len()
}

fn unique<T: Ord + Clone>(values: &[T]) -> bool {
    values.iter().cloned().collect::<BTreeSet<_>>().len() == values.len()
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn push_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn push_optional_u64(hasher: &mut blake3::Hasher, value: Option<u64>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            push_u64(hasher, value);
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn push_sorted_refs(hasher: &mut blake3::Hasher, refs: &[String]) {
    let mut refs = refs.to_vec();
    refs.sort();
    hasher.update(&(refs.len() as u64).to_le_bytes());
    for reference in refs {
        push_field(hasher, &reference);
    }
}

fn push_vec_field(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn push_vec_optional_string(bytes: &mut Vec<u8>, value: Option<&str>) {
    match value {
        Some(value) => {
            bytes.push(1);
            push_vec_field(bytes, value);
        }
        None => bytes.push(0),
    }
}

fn push_vec_sorted_refs(bytes: &mut Vec<u8>, refs: &[String]) {
    let mut refs = refs.to_vec();
    refs.sort();
    bytes.extend_from_slice(&(refs.len() as u64).to_be_bytes());
    for reference in refs {
        push_vec_field(bytes, &reference);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    const HEAD_SEED: [u8; 32] = [1; 32];
    const CURRENT_SEED: [u8; 32] = [2; 32];

    fn digest(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn challenge(label: &str) -> String {
        blake3::hash(label.as_bytes()).to_hex().to_string()
    }

    fn public_hex(seed: [u8; 32]) -> String {
        hex::encode(SigningKey::from_bytes(&seed).verifying_key().to_bytes())
    }

    fn policy() -> LedgerHeadAuthorityPolicy {
        LedgerHeadAuthorityPolicy {
            schema_version: LEDGER_HEAD_AUTHORITY_POLICY_SCHEMA_V1.into(),
            policy_id: "policy:ledger-head:1".into(),
            sequence: 1,
            issued_at_ms: 1_000,
            expires_at_ms: 100_000,
            trusted_keys: vec![
                LedgerHeadAuthorityKey {
                    key_id: "key:head".into(),
                    public_key_ed25519_hex: public_hex(HEAD_SEED),
                    valid_from_ms: 1_000,
                    valid_until_ms: None,
                    revoked_at_ms: None,
                    allowed_scopes: vec![LedgerHeadClaimScope::HeadStatement],
                    evidence_refs: vec!["review:key:head".into()],
                },
                LedgerHeadAuthorityKey {
                    key_id: "key:current".into(),
                    public_key_ed25519_hex: public_hex(CURRENT_SEED),
                    valid_from_ms: 1_000,
                    valid_until_ms: None,
                    revoked_at_ms: None,
                    allowed_scopes: vec![LedgerHeadClaimScope::CurrentnessAttestation],
                    evidence_refs: vec!["review:key:current".into()],
                },
            ],
            evidence_refs: vec!["review:ledger-head-policy".into()],
        }
    }

    fn signed_head(
        policy: &LedgerHeadAuthorityPolicy,
        revision: u64,
        acceptance_digest: String,
        previous: Option<String>,
        issued_at_ms: u64,
    ) -> AcceptanceLedgerHeadStatement {
        let mut statement = AcceptanceLedgerHeadStatement {
            schema_version: LEDGER_HEAD_STATEMENT_SCHEMA_V1.into(),
            ledger_id: "ledger:attestation:prod-1".into(),
            acceptance_revision: revision,
            acceptance_digest,
            previous_head_statement_digest: previous,
            issued_at_ms,
            authority_policy_digest: policy.canonical_digest().unwrap(),
            signer_key_id: "key:head".into(),
            signer_public_key_ed25519_hex: public_hex(HEAD_SEED),
            evidence_refs: vec![format!("ledger:revision:{revision}")],
            signature_ed25519_hex: "00".repeat(64),
        };
        let bytes = statement.canonical_unsigned_bytes().unwrap();
        let signature = SigningKey::from_bytes(&HEAD_SEED).sign(&bytes);
        statement.signature_ed25519_hex = hex::encode(signature.to_bytes());
        statement
    }

    fn signed_currentness(
        policy: &LedgerHeadAuthorityPolicy,
        head: &TrackedAcceptanceLedgerHead,
        nonce: String,
        use_at_ms: u64,
    ) -> AcceptanceLedgerHeadCurrentnessAttestation {
        let mut attestation = AcceptanceLedgerHeadCurrentnessAttestation {
            schema_version: LEDGER_HEAD_CURRENTNESS_SCHEMA_V1.into(),
            head_statement_digest: head.statement_digest().into(),
            ledger_id: head.ledger_id().into(),
            acceptance_revision: head.acceptance_revision(),
            acceptance_digest: head.acceptance_digest().into(),
            authority_policy_digest: policy.canonical_digest().unwrap(),
            asserted_current_at_ms: use_at_ms,
            challenge_nonce_blake3_hex: nonce,
            signer_key_id: "key:current".into(),
            signer_public_key_ed25519_hex: public_hex(CURRENT_SEED),
            evidence_refs: vec!["response:ledger-currentness".into()],
            signature_ed25519_hex: "00".repeat(64),
        };
        let bytes = attestation.canonical_unsigned_bytes().unwrap();
        let signature = SigningKey::from_bytes(&CURRENT_SEED).sign(&bytes);
        attestation.signature_ed25519_hex = hex::encode(signature.to_bytes());
        attestation
    }

    fn verified_head(
        policy: &LedgerHeadAuthorityPolicy,
        statement: &AcceptanceLedgerHeadStatement,
    ) -> VerifiedAcceptanceLedgerHeadStatement {
        verify_acceptance_ledger_head_statement(policy, statement)
            .into_verified()
            .unwrap()
    }

    #[test]
    fn signed_genesis_tracks_and_currentness_mints_capability() {
        let policy = policy();
        let statement = signed_head(&policy, 1, digest("acceptance:1"), None, 2_000);
        let verified = verified_head(&policy, &statement);
        let mut tracker = AcceptanceLedgerHeadTracker::default();
        let tracked = tracker.accept(&verified).unwrap();
        let nonce = challenge("nonce:1");
        let currentness = signed_currentness(&policy, &tracked, nonce.clone(), 3_000);

        let assessed = verify_current_acceptance_ledger_head(
            &tracker,
            &tracked,
            &policy,
            &currentness,
            &nonce,
            3_000,
        );

        assert_eq!(
            assessed.report.disposition,
            LedgerHeadCurrentnessDisposition::Current
        );
        let current = assessed.current().unwrap();
        assert_eq!(current.acceptance_revision(), 1);
        assert_eq!(current.acceptance_digest(), digest("acceptance:1"));
        assert!(!current.grants_physical_authority());
    }

    #[test]
    fn fresh_tracker_cannot_bootstrap_at_non_genesis_revision() {
        let policy = policy();
        let statement = signed_head(&policy, 2, digest("acceptance:2"), Some(digest("head:1")), 2_000);
        let verified = verified_head(&policy, &statement);
        let mut tracker = AcceptanceLedgerHeadTracker::default();

        assert!(matches!(
            tracker.accept(&verified),
            Err(AcceptanceLedgerHeadTrackingError::FirstHeadMustBeGenesis {
                proposed_revision: 2
            })
        ));
    }

    #[test]
    fn next_revision_requires_exact_previous_signed_head() {
        let policy = policy();
        let first_statement = signed_head(&policy, 1, digest("acceptance:1"), None, 2_000);
        let first = verified_head(&policy, &first_statement);
        let mut tracker = AcceptanceLedgerHeadTracker::default();
        let tracked_first = tracker.accept(&first).unwrap();

        let bad_statement = signed_head(
            &policy,
            2,
            digest("acceptance:2"),
            Some(digest("wrong-head")),
            2_100,
        );
        let bad = verified_head(&policy, &bad_statement);
        assert!(matches!(
            tracker.accept(&bad),
            Err(AcceptanceLedgerHeadTrackingError::PreviousHeadDigestMismatch)
        ));

        let second_statement = signed_head(
            &policy,
            2,
            digest("acceptance:2"),
            Some(tracked_first.statement_digest().into()),
            2_100,
        );
        let second = verified_head(&policy, &second_statement);
        let tracked_second = tracker.accept(&second).unwrap();
        assert_eq!(tracked_second.acceptance_revision(), 2);
    }

    #[test]
    fn old_tracked_head_is_not_current_after_tracker_advances() {
        let policy = policy();
        let first_statement = signed_head(&policy, 1, digest("acceptance:1"), None, 2_000);
        let first = verified_head(&policy, &first_statement);
        let mut tracker = AcceptanceLedgerHeadTracker::default();
        let tracked_first = tracker.accept(&first).unwrap();
        let second_statement = signed_head(
            &policy,
            2,
            digest("acceptance:2"),
            Some(tracked_first.statement_digest().into()),
            2_100,
        );
        let second = verified_head(&policy, &second_statement);
        tracker.accept(&second).unwrap();

        let nonce = challenge("nonce:old");
        let currentness = signed_currentness(&policy, &tracked_first, nonce.clone(), 3_000);
        let assessed = verify_current_acceptance_ledger_head(
            &tracker,
            &tracked_first,
            &policy,
            &currentness,
            &nonce,
            3_000,
        );

        assert_eq!(
            assessed.report.disposition,
            LedgerHeadCurrentnessDisposition::Blocked
        );
        assert!(assessed
            .report
            .issues
            .contains(&LedgerHeadCurrentnessIssue::HeadNotLatestInTracker));
    }

    #[test]
    fn same_revision_different_statement_is_collision() {
        let policy = policy();
        let first_statement = signed_head(&policy, 1, digest("acceptance:1"), None, 2_000);
        let first = verified_head(&policy, &first_statement);
        let mut tracker = AcceptanceLedgerHeadTracker::default();
        tracker.accept(&first).unwrap();

        let collision_statement = signed_head(&policy, 1, digest("acceptance:other"), None, 2_000);
        let collision = verified_head(&policy, &collision_statement);
        assert!(matches!(
            tracker.accept(&collision),
            Err(AcceptanceLedgerHeadTrackingError::RevisionCollision { revision: 1 })
        ));
    }

    #[test]
    fn revision_gap_is_rejected() {
        let policy = policy();
        let first_statement = signed_head(&policy, 1, digest("acceptance:1"), None, 2_000);
        let first = verified_head(&policy, &first_statement);
        let mut tracker = AcceptanceLedgerHeadTracker::default();
        let tracked_first = tracker.accept(&first).unwrap();
        let third_statement = signed_head(
            &policy,
            3,
            digest("acceptance:3"),
            Some(tracked_first.statement_digest().into()),
            2_100,
        );
        let third = verified_head(&policy, &third_statement);
        assert!(matches!(
            tracker.accept(&third),
            Err(AcceptanceLedgerHeadTrackingError::RevisionGap {
                expected: 2,
                proposed: 3
            })
        ));
    }

    #[test]
    fn wrong_currentness_challenge_is_blocked() {
        let policy = policy();
        let statement = signed_head(&policy, 1, digest("acceptance:1"), None, 2_000);
        let verified = verified_head(&policy, &statement);
        let mut tracker = AcceptanceLedgerHeadTracker::default();
        let tracked = tracker.accept(&verified).unwrap();
        let signed_nonce = challenge("nonce:signed");
        let expected_nonce = challenge("nonce:expected");
        let currentness = signed_currentness(&policy, &tracked, signed_nonce, 3_000);

        let assessed = verify_current_acceptance_ledger_head(
            &tracker,
            &tracked,
            &policy,
            &currentness,
            &expected_nonce,
            3_000,
        );

        assert_eq!(
            assessed.report.disposition,
            LedgerHeadCurrentnessDisposition::Blocked
        );
        assert!(assessed
            .report
            .issues
            .contains(&LedgerHeadCurrentnessIssue::ChallengeNonceMismatch));
    }

    #[test]
    fn head_signer_cannot_substitute_for_currentness_scope() {
        let policy = policy();
        let statement = signed_head(&policy, 1, digest("acceptance:1"), None, 2_000);
        let verified = verified_head(&policy, &statement);
        let mut tracker = AcceptanceLedgerHeadTracker::default();
        let tracked = tracker.accept(&verified).unwrap();
        let nonce = challenge("nonce:scope");
        let mut currentness = AcceptanceLedgerHeadCurrentnessAttestation {
            schema_version: LEDGER_HEAD_CURRENTNESS_SCHEMA_V1.into(),
            head_statement_digest: tracked.statement_digest().into(),
            ledger_id: tracked.ledger_id().into(),
            acceptance_revision: tracked.acceptance_revision(),
            acceptance_digest: tracked.acceptance_digest().into(),
            authority_policy_digest: policy.canonical_digest().unwrap(),
            asserted_current_at_ms: 3_000,
            challenge_nonce_blake3_hex: nonce.clone(),
            signer_key_id: "key:head".into(),
            signer_public_key_ed25519_hex: public_hex(HEAD_SEED),
            evidence_refs: vec!["response:wrong-scope".into()],
            signature_ed25519_hex: "00".repeat(64),
        };
        let bytes = currentness.canonical_unsigned_bytes().unwrap();
        let signature = SigningKey::from_bytes(&HEAD_SEED).sign(&bytes);
        currentness.signature_ed25519_hex = hex::encode(signature.to_bytes());

        let assessed = verify_current_acceptance_ledger_head(
            &tracker,
            &tracked,
            &policy,
            &currentness,
            &nonce,
            3_000,
        );

        assert_eq!(
            assessed.report.disposition,
            LedgerHeadCurrentnessDisposition::Blocked
        );
        assert!(assessed
            .report
            .issues
            .contains(&LedgerHeadCurrentnessIssue::ScopeNotAuthorized));
    }
}
