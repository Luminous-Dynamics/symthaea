// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Signed, append-only current-head assurance for attestation acceptance ledgers.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use symthaea_assurance_tpm2_attestation_possession::{
    AttestationAcceptanceLedger, AttestationAcceptanceRecord,
};

pub const ACCEPTANCE_HEAD_AUTHORITY_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.acceptance-ledger-head-authority-policy.v1";
pub const ACCEPTANCE_HEAD_STATEMENT_SCHEMA_V1: &str =
    "symthaea.assurance.acceptance-ledger-head-statement.v1";
pub const ACCEPTANCE_HEAD_CURRENTNESS_SCHEMA_V1: &str =
    "symthaea.assurance.acceptance-ledger-head-currentness.v1";
pub const MAX_TRUSTED_KEYS: usize = 1_024;
pub const MAX_EVIDENCE_REFS: usize = 128;
pub const MAX_TEXT_BYTES: usize = 512;
pub const MAX_POLICY_LEDGER_RECORDS: u64 = 1_000_000;

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.acceptance-ledger-head-authority-policy.digest.v1\0";
const LEDGER_STATE_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.acceptance-ledger-state.digest.v1\0";
const HEAD_MESSAGE_DOMAIN: &[u8] =
    b"symthaea.assurance.acceptance-ledger-head-statement.message.v1\0";
const HEAD_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.acceptance-ledger-head-statement.digest.v1\0";
const CURRENTNESS_MESSAGE_DOMAIN: &[u8] =
    b"symthaea.assurance.acceptance-ledger-head-currentness.message.v1\0";
const CURRENTNESS_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.acceptance-ledger-head-currentness.digest.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AcceptanceLedgerHeadClaimScope {
    HeadStatement,
    CurrentnessAttestation,
}

impl AcceptanceLedgerHeadClaimScope {
    fn code(self) -> &'static str {
        match self {
            Self::HeadStatement => "head-statement",
            Self::CurrentnessAttestation => "currentness-attestation",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcceptanceLedgerHeadAuthorityKey {
    pub key_id: String,
    pub public_key_ed25519_hex: String,
    pub valid_from_ms: u64,
    pub valid_until_ms: Option<u64>,
    pub revoked_at_ms: Option<u64>,
    pub allowed_scopes: Vec<AcceptanceLedgerHeadClaimScope>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcceptanceLedgerHeadAuthorityPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub ledger_id: String,
    pub sequence: u64,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub max_ledger_records: u64,
    pub trusted_keys: Vec<AcceptanceLedgerHeadAuthorityKey>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptanceLedgerSnapshot {
    revision: u64,
    head_digest: String,
    state_digest: String,
}

impl AcceptanceLedgerSnapshot {
    pub const fn revision(&self) -> u64 {
        self.revision
    }
    pub fn head_digest(&self) -> &str {
        &self.head_digest
    }
    pub fn state_digest(&self) -> &str {
        &self.state_digest
    }
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcceptanceLedgerHeadStatement {
    pub schema_version: String,
    pub ledger_id: String,
    pub ledger_revision: u64,
    pub ledger_head_digest: String,
    pub ledger_state_digest: String,
    pub authority_sequence: u64,
    pub previous_head_statement_digest: Option<String>,
    pub issued_at_ms: u64,
    pub authority_policy_digest: String,
    pub signer_key_id: String,
    pub signer_public_key_ed25519_hex: String,
    pub evidence_refs: Vec<String>,
    pub signature_ed25519_hex: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcceptanceLedgerHeadCurrentnessAttestation {
    pub schema_version: String,
    pub head_statement_digest: String,
    pub ledger_id: String,
    pub ledger_revision: u64,
    pub ledger_head_digest: String,
    pub ledger_state_digest: String,
    pub authority_sequence: u64,
    pub authority_policy_digest: String,
    pub asserted_current_at_ms: u64,
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
    LedgerIdMismatch,
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
    pub ledger_revision: u64,
    pub ledger_head_digest: String,
    pub ledger_state_digest: String,
    pub authority_sequence: u64,
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
    ledger_revision: u64,
    ledger_head_digest: String,
    ledger_state_digest: String,
    authority_sequence: u64,
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
    pub const fn ledger_revision(&self) -> u64 {
        self.ledger_revision
    }
    pub fn ledger_head_digest(&self) -> &str {
        &self.ledger_head_digest
    }
    pub fn ledger_state_digest(&self) -> &str {
        &self.ledger_state_digest
    }
    pub const fn authority_sequence(&self) -> u64 {
        self.authority_sequence
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct TrackedHeadState {
    authority_sequence: u64,
    statement_digest: String,
    issued_at_ms: u64,
    ledger_revision: u64,
    ledger_head_digest: String,
    ledger_state_digest: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AcceptanceLedgerHeadTracker {
    heads: BTreeMap<String, TrackedHeadState>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcceptanceLedgerHeadTrackingError {
    PolicyDigestMismatch,
    InvalidLedgerState { reason: String },
    LedgerRevisionMismatch { statement: u64, observed: u64 },
    LedgerHeadDigestMismatch,
    LedgerStateDigestMismatch,
    FirstAuthorityStatementMustBeSequenceOne { proposed: u64 },
    FirstAuthorityStatementHasPreviousDigest,
    AuthoritySequenceRollback { latest: u64, proposed: u64 },
    AuthoritySequenceCollision { sequence: u64 },
    AuthoritySequenceGap { expected: u64, proposed: u64 },
    PreviousHeadStatementDigestMismatch,
    IssuedAtRegressed { latest: u64, proposed: u64 },
    LedgerRevisionDidNotAdvance { latest: u64, proposed: u64 },
    PreviouslyObservedHeadMissing,
    PreviouslyObservedStatePrefixMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackedAcceptanceLedgerHead {
    ledger_id: String,
    ledger_revision: u64,
    ledger_head_digest: String,
    ledger_state_digest: String,
    authority_sequence: u64,
    statement_digest: String,
    policy_digest: String,
    issued_at_ms: u64,
}

impl TrackedAcceptanceLedgerHead {
    pub fn ledger_id(&self) -> &str {
        &self.ledger_id
    }
    pub const fn ledger_revision(&self) -> u64 {
        self.ledger_revision
    }
    pub fn ledger_head_digest(&self) -> &str {
        &self.ledger_head_digest
    }
    pub fn ledger_state_digest(&self) -> &str {
        &self.ledger_state_digest
    }
    pub const fn authority_sequence(&self) -> u64 {
        self.authority_sequence
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurrentnessDisposition {
    Invalid,
    Blocked,
    Current,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CurrentnessIssue {
    InvalidPolicy,
    InvalidAttestation,
    PolicyDigestMismatch,
    HeadPolicyDigestMismatch,
    HeadStatementDigestMismatch,
    LedgerIdMismatch,
    LedgerRevisionMismatch,
    LedgerHeadDigestMismatch,
    LedgerStateDigestMismatch,
    AuthoritySequenceMismatch,
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
pub struct CurrentnessReport {
    pub disposition: CurrentnessDisposition,
    pub ledger_id: String,
    pub ledger_revision: u64,
    pub ledger_head_digest: String,
    pub ledger_state_digest: String,
    pub authority_sequence: u64,
    pub head_statement_digest: String,
    pub currentness_attestation_digest: Option<String>,
    pub policy_digest: Option<String>,
    pub use_at_ms: u64,
    pub signer_key_id: String,
    pub signature_valid: bool,
    pub issues: Vec<CurrentnessIssue>,
}

impl CurrentnessReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentAcceptanceLedgerHead {
    ledger_id: String,
    ledger_revision: u64,
    ledger_head_digest: String,
    ledger_state_digest: String,
    authority_sequence: u64,
    head_statement_digest: String,
    currentness_attestation_digest: String,
    authority_policy_digest: String,
    use_at_ms: u64,
}

impl CurrentAcceptanceLedgerHead {
    pub fn ledger_id(&self) -> &str {
        &self.ledger_id
    }
    pub const fn ledger_revision(&self) -> u64 {
        self.ledger_revision
    }
    pub fn ledger_head_digest(&self) -> &str {
        &self.ledger_head_digest
    }
    pub fn ledger_state_digest(&self) -> &str {
        &self.ledger_state_digest
    }
    pub const fn authority_sequence(&self) -> u64 {
        self.authority_sequence
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
    pub const fn establishes_authority_asserted_currentness_at_use_time(&self) -> bool {
        true
    }
    pub const fn establishes_trusted_time_provenance(&self) -> bool {
        false
    }
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentnessAssessment {
    pub report: CurrentnessReport,
    current: Option<CurrentAcceptanceLedgerHead>,
}

impl CurrentnessAssessment {
    pub fn current(&self) -> Option<&CurrentAcceptanceLedgerHead> {
        self.current.as_ref()
    }
    pub fn into_current(self) -> Option<CurrentAcceptanceLedgerHead> {
        self.current
    }
}

impl AcceptanceLedgerHeadAuthorityKey {
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

impl AcceptanceLedgerHeadAuthorityPolicy {
    pub fn validate(&self) -> bool {
        if self.schema_version != ACCEPTANCE_HEAD_AUTHORITY_POLICY_SCHEMA_V1
            || !canonical_text(&self.policy_id)
            || !canonical_text(&self.ledger_id)
            || self.sequence == 0
            || self.issued_at_ms >= self.expires_at_ms
            || self.max_ledger_records == 0
            || self.max_ledger_records > MAX_POLICY_LEDGER_RECORDS
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
        push_field(&mut hasher, &self.ledger_id);
        push_u64(&mut hasher, self.sequence);
        push_u64(&mut hasher, self.issued_at_ms);
        push_u64(&mut hasher, self.expires_at_ms);
        push_u64(&mut hasher, self.max_ledger_records);

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
            for scope in scopes {
                push_field(&mut hasher, scope.code());
            }
            push_sorted_refs(&mut hasher, &key.evidence_refs);
        }
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

impl AcceptanceLedgerHeadStatement {
    pub fn validate(&self) -> bool {
        self.schema_version == ACCEPTANCE_HEAD_STATEMENT_SCHEMA_V1
            && canonical_text(&self.ledger_id)
            && self.ledger_revision > 0
            && digest_text(&self.ledger_head_digest)
            && digest_text(&self.ledger_state_digest)
            && self.authority_sequence > 0
            && if self.authority_sequence == 1 {
                self.previous_head_statement_digest.is_none()
            } else {
                self.previous_head_statement_digest
                    .as_deref()
                    .is_some_and(digest_text)
            }
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
        bytes.extend_from_slice(&self.ledger_revision.to_be_bytes());
        push_vec_field(&mut bytes, &self.ledger_head_digest);
        push_vec_field(&mut bytes, &self.ledger_state_digest);
        bytes.extend_from_slice(&self.authority_sequence.to_be_bytes());
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

impl AcceptanceLedgerHeadCurrentnessAttestation {
    pub fn validate(&self) -> bool {
        self.schema_version == ACCEPTANCE_HEAD_CURRENTNESS_SCHEMA_V1
            && digest_text(&self.head_statement_digest)
            && canonical_text(&self.ledger_id)
            && self.ledger_revision > 0
            && digest_text(&self.ledger_head_digest)
            && digest_text(&self.ledger_state_digest)
            && self.authority_sequence > 0
            && digest_text(&self.authority_policy_digest)
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
        bytes.extend_from_slice(&self.ledger_revision.to_be_bytes());
        push_vec_field(&mut bytes, &self.ledger_head_digest);
        push_vec_field(&mut bytes, &self.ledger_state_digest);
        bytes.extend_from_slice(&self.authority_sequence.to_be_bytes());
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

pub fn snapshot_acceptance_ledger(
    ledger: &AttestationAcceptanceLedger,
    max_records: u64,
) -> Result<AcceptanceLedgerSnapshot, String> {
    validate_ledger_structure(ledger, max_records)?;
    let revision = ledger.records.len() as u64;
    let head_digest = ledger
        .records
        .last()
        .expect("validated non-empty ledger")
        .acceptance_digest();
    let state_digest = ledger_state_digest_for_prefix(ledger, revision)?;
    Ok(AcceptanceLedgerSnapshot {
        revision,
        head_digest,
        state_digest,
    })
}

pub fn verify_acceptance_ledger_head_statement(
    policy: &AcceptanceLedgerHeadAuthorityPolicy,
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
    if statement.ledger_id != policy.ledger_id {
        issues.push(HeadStatementIssue::LedgerIdMismatch);
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
        if !key
            .allowed_scopes
            .contains(&AcceptanceLedgerHeadClaimScope::HeadStatement)
        {
            issues.push(HeadStatementIssue::ScopeNotAuthorized);
        }
        let bytes = statement
            .canonical_unsigned_bytes()
            .expect("validated statement");
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
        ledger_revision: statement.ledger_revision,
        ledger_head_digest: statement.ledger_head_digest.clone(),
        ledger_state_digest: statement.ledger_state_digest.clone(),
        authority_sequence: statement.authority_sequence,
        previous_head_statement_digest: statement.previous_head_statement_digest.clone(),
        issued_at_ms: statement.issued_at_ms,
        statement_digest: statement_digest.clone().expect("validated statement"),
        policy_digest: policy_digest.clone().expect("validated policy"),
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

impl AcceptanceLedgerHeadTracker {
    pub fn observe(
        &mut self,
        head: &VerifiedAcceptanceLedgerHeadStatement,
        ledger: &AttestationAcceptanceLedger,
        policy: &AcceptanceLedgerHeadAuthorityPolicy,
    ) -> Result<TrackedAcceptanceLedgerHead, AcceptanceLedgerHeadTrackingError> {
        if policy.canonical_digest().as_deref() != Some(head.policy_digest()) {
            return Err(AcceptanceLedgerHeadTrackingError::PolicyDigestMismatch);
        }
        let snapshot = snapshot_acceptance_ledger(ledger, policy.max_ledger_records)
            .map_err(|reason| AcceptanceLedgerHeadTrackingError::InvalidLedgerState { reason })?;
        if snapshot.revision != head.ledger_revision {
            return Err(AcceptanceLedgerHeadTrackingError::LedgerRevisionMismatch {
                statement: head.ledger_revision,
                observed: snapshot.revision,
            });
        }
        if snapshot.head_digest != head.ledger_head_digest {
            return Err(AcceptanceLedgerHeadTrackingError::LedgerHeadDigestMismatch);
        }
        if snapshot.state_digest != head.ledger_state_digest {
            return Err(AcceptanceLedgerHeadTrackingError::LedgerStateDigestMismatch);
        }

        match self.heads.get(&head.ledger_id) {
            None => {
                if head.authority_sequence != 1 {
                    return Err(
                        AcceptanceLedgerHeadTrackingError::FirstAuthorityStatementMustBeSequenceOne {
                            proposed: head.authority_sequence,
                        },
                    );
                }
                if head.previous_head_statement_digest.is_some() {
                    return Err(
                        AcceptanceLedgerHeadTrackingError::FirstAuthorityStatementHasPreviousDigest,
                    );
                }
            }
            Some(latest) => {
                if head.authority_sequence < latest.authority_sequence {
                    return Err(AcceptanceLedgerHeadTrackingError::AuthoritySequenceRollback {
                        latest: latest.authority_sequence,
                        proposed: head.authority_sequence,
                    });
                }
                if head.authority_sequence == latest.authority_sequence {
                    if head.statement_digest == latest.statement_digest {
                        return Ok(tracked(head));
                    }
                    return Err(AcceptanceLedgerHeadTrackingError::AuthoritySequenceCollision {
                        sequence: head.authority_sequence,
                    });
                }
                let expected = latest.authority_sequence.saturating_add(1);
                if head.authority_sequence != expected {
                    return Err(AcceptanceLedgerHeadTrackingError::AuthoritySequenceGap {
                        expected,
                        proposed: head.authority_sequence,
                    });
                }
                if head.previous_head_statement_digest.as_deref()
                    != Some(latest.statement_digest.as_str())
                {
                    return Err(
                        AcceptanceLedgerHeadTrackingError::PreviousHeadStatementDigestMismatch,
                    );
                }
                if head.issued_at_ms < latest.issued_at_ms {
                    return Err(AcceptanceLedgerHeadTrackingError::IssuedAtRegressed {
                        latest: latest.issued_at_ms,
                        proposed: head.issued_at_ms,
                    });
                }
                if head.ledger_revision <= latest.ledger_revision {
                    return Err(
                        AcceptanceLedgerHeadTrackingError::LedgerRevisionDidNotAdvance {
                            latest: latest.ledger_revision,
                            proposed: head.ledger_revision,
                        },
                    );
                }
                let previous_record = ledger
                    .records
                    .get((latest.ledger_revision - 1) as usize)
                    .ok_or(AcceptanceLedgerHeadTrackingError::PreviouslyObservedHeadMissing)?;
                if previous_record.acceptance_digest() != latest.ledger_head_digest {
                    return Err(
                        AcceptanceLedgerHeadTrackingError::PreviouslyObservedHeadMissing,
                    );
                }
                let prefix_digest = ledger_state_digest_for_prefix(ledger, latest.ledger_revision)
                    .map_err(|_| {
                        AcceptanceLedgerHeadTrackingError::PreviouslyObservedStatePrefixMismatch
                    })?;
                if prefix_digest != latest.ledger_state_digest {
                    return Err(
                        AcceptanceLedgerHeadTrackingError::PreviouslyObservedStatePrefixMismatch,
                    );
                }
            }
        }

        self.heads.insert(
            head.ledger_id.clone(),
            TrackedHeadState {
                authority_sequence: head.authority_sequence,
                statement_digest: head.statement_digest.clone(),
                issued_at_ms: head.issued_at_ms,
                ledger_revision: head.ledger_revision,
                ledger_head_digest: head.ledger_head_digest.clone(),
                ledger_state_digest: head.ledger_state_digest.clone(),
            },
        );
        Ok(tracked(head))
    }

    pub fn latest_authority_sequence(&self, ledger_id: &str) -> Option<u64> {
        self.heads.get(ledger_id).map(|state| state.authority_sequence)
    }

    pub fn latest_statement_digest(&self, ledger_id: &str) -> Option<&str> {
        self.heads
            .get(ledger_id)
            .map(|state| state.statement_digest.as_str())
    }
}

pub fn verify_current_acceptance_ledger_head(
    head: &TrackedAcceptanceLedgerHead,
    policy: &AcceptanceLedgerHeadAuthorityPolicy,
    currentness: &AcceptanceLedgerHeadCurrentnessAttestation,
    expected_challenge_nonce_blake3_hex: &str,
    use_at_ms: u64,
) -> CurrentnessAssessment {
    let policy_digest = policy.canonical_digest();
    let attestation_digest = currentness.canonical_digest();
    let mut issues = Vec::new();

    if !policy.validate() {
        issues.push(CurrentnessIssue::InvalidPolicy);
    }
    if !currentness.validate() {
        issues.push(CurrentnessIssue::InvalidAttestation);
    }
    if policy_digest.as_deref() != Some(currentness.authority_policy_digest.as_str()) {
        issues.push(CurrentnessIssue::PolicyDigestMismatch);
    }
    if head.policy_digest != currentness.authority_policy_digest {
        issues.push(CurrentnessIssue::HeadPolicyDigestMismatch);
    }
    if head.statement_digest != currentness.head_statement_digest {
        issues.push(CurrentnessIssue::HeadStatementDigestMismatch);
    }
    if head.ledger_id != currentness.ledger_id {
        issues.push(CurrentnessIssue::LedgerIdMismatch);
    }
    if head.ledger_revision != currentness.ledger_revision {
        issues.push(CurrentnessIssue::LedgerRevisionMismatch);
    }
    if head.ledger_head_digest != currentness.ledger_head_digest {
        issues.push(CurrentnessIssue::LedgerHeadDigestMismatch);
    }
    if head.ledger_state_digest != currentness.ledger_state_digest {
        issues.push(CurrentnessIssue::LedgerStateDigestMismatch);
    }
    if head.authority_sequence != currentness.authority_sequence {
        issues.push(CurrentnessIssue::AuthoritySequenceMismatch);
    }
    if currentness.asserted_current_at_ms != use_at_ms {
        issues.push(CurrentnessIssue::CurrentnessNotAtUseTime);
    }
    if currentness.challenge_nonce_blake3_hex != expected_challenge_nonce_blake3_hex {
        issues.push(CurrentnessIssue::ChallengeNonceMismatch);
    }
    if !issues.is_empty() {
        return current_result(
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
        issues.push(CurrentnessIssue::PolicyNotEffectiveAtUseTime);
    }

    let key = policy
        .trusted_keys
        .iter()
        .find(|key| key.key_id == currentness.signer_key_id);
    let mut signature_valid = false;
    if let Some(key) = key {
        if key.public_key_ed25519_hex != currentness.signer_public_key_ed25519_hex {
            issues.push(CurrentnessIssue::SignerPublicKeyMismatch);
        }
        if !key.usable_at(use_at_ms) {
            issues.push(CurrentnessIssue::SignerNotEffective);
        }
        if key.revoked_at(use_at_ms) {
            issues.push(CurrentnessIssue::SignerRevoked);
        }
        if !key
            .allowed_scopes
            .contains(&AcceptanceLedgerHeadClaimScope::CurrentnessAttestation)
        {
            issues.push(CurrentnessIssue::ScopeNotAuthorized);
        }
        let bytes = currentness
            .canonical_unsigned_bytes()
            .expect("validated attestation");
        match verify_signature(
            &currentness.signer_public_key_ed25519_hex,
            &currentness.signature_ed25519_hex,
            &bytes,
        ) {
            Ok(valid) => {
                signature_valid = valid;
                if !valid {
                    issues.push(CurrentnessIssue::SignatureInvalid);
                }
            }
            Err(()) => issues.push(CurrentnessIssue::InvalidPublicKey),
        }
    } else {
        issues.push(CurrentnessIssue::SignerUnknown);
    }

    if !issues.is_empty() {
        return current_result(
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
        ledger_revision: head.ledger_revision,
        ledger_head_digest: head.ledger_head_digest.clone(),
        ledger_state_digest: head.ledger_state_digest.clone(),
        authority_sequence: head.authority_sequence,
        head_statement_digest: head.statement_digest.clone(),
        currentness_attestation_digest: attestation_digest
            .clone()
            .expect("validated attestation"),
        authority_policy_digest: policy_digest.clone().expect("validated policy"),
        use_at_ms,
    };
    current_result(
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

fn validate_ledger_structure(
    ledger: &AttestationAcceptanceLedger,
    max_records: u64,
) -> Result<(), String> {
    if ledger.records.is_empty() {
        return Err("empty-ledger".into());
    }
    if max_records == 0 || max_records > MAX_POLICY_LEDGER_RECORDS {
        return Err("invalid-max-records".into());
    }
    if ledger.records.len() as u64 > max_records {
        return Err("record-count-exceeds-policy".into());
    }

    let mut challenge_ids = BTreeSet::new();
    let mut nonce_digests = BTreeSet::new();
    let mut quote_digests = BTreeSet::new();
    let mut receipt_ids = BTreeSet::new();
    let mut previous_digest: Option<String> = None;
    let mut previous_accepted_at: Option<u64> = None;

    for (index, record) in ledger.records.iter().enumerate() {
        let expected_revision = index as u64 + 1;
        if record.revision != expected_revision {
            return Err(format!("revision:{expected_revision}:{}", record.revision));
        }
        if !canonical_text(&record.challenge_id)
            || !canonical_text(&record.verification_receipt_id)
        {
            return Err(format!("invalid-text:{expected_revision}"));
        }
        for digest in [
            record.nonce_digest.as_str(),
            record.possession_record_digest.as_str(),
            record.quote_artifact_digest.as_str(),
        ] {
            if !digest_text(digest) {
                return Err(format!("invalid-digest:{expected_revision}"));
            }
        }
        match (&previous_digest, &record.predecessor_acceptance_digest) {
            (None, None) => {}
            (Some(expected), Some(observed)) if expected == observed => {}
            _ => return Err(format!("predecessor:{expected_revision}")),
        }
        if record
            .predecessor_acceptance_digest
            .as_deref()
            .is_some_and(|digest| !digest_text(digest))
        {
            return Err(format!("invalid-predecessor-digest:{expected_revision}"));
        }
        if previous_accepted_at.is_some_and(|time| record.accepted_at_ms < time) {
            return Err(format!("time-regression:{expected_revision}"));
        }
        if !challenge_ids.insert(record.challenge_id.clone()) {
            return Err(format!("duplicate-challenge:{expected_revision}"));
        }
        if !nonce_digests.insert(record.nonce_digest.clone()) {
            return Err(format!("duplicate-nonce:{expected_revision}"));
        }
        if !quote_digests.insert(record.quote_artifact_digest.clone()) {
            return Err(format!("duplicate-quote:{expected_revision}"));
        }
        if !receipt_ids.insert(record.verification_receipt_id.clone()) {
            return Err(format!("duplicate-receipt:{expected_revision}"));
        }
        previous_digest = Some(record.acceptance_digest());
        previous_accepted_at = Some(record.accepted_at_ms);
    }
    Ok(())
}

fn ledger_state_digest_for_prefix(
    ledger: &AttestationAcceptanceLedger,
    revision: u64,
) -> Result<String, String> {
    if revision == 0 || revision > ledger.records.len() as u64 {
        return Err("invalid-prefix-revision".into());
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(LEDGER_STATE_DIGEST_DOMAIN);
    push_u64(&mut hasher, revision);
    for record in ledger.records.iter().take(revision as usize) {
        push_field(&mut hasher, &record.acceptance_digest());
    }
    Ok(format!("blake3:{}", hasher.finalize().to_hex()))
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
            ledger_revision: statement.ledger_revision,
            ledger_head_digest: statement.ledger_head_digest.clone(),
            ledger_state_digest: statement.ledger_state_digest.clone(),
            authority_sequence: statement.authority_sequence,
            statement_digest,
            policy_digest,
            signer_key_id: statement.signer_key_id.clone(),
            signature_valid,
            issues,
        },
        verified,
    }
}

fn tracked(head: &VerifiedAcceptanceLedgerHeadStatement) -> TrackedAcceptanceLedgerHead {
    TrackedAcceptanceLedgerHead {
        ledger_id: head.ledger_id.clone(),
        ledger_revision: head.ledger_revision,
        ledger_head_digest: head.ledger_head_digest.clone(),
        ledger_state_digest: head.ledger_state_digest.clone(),
        authority_sequence: head.authority_sequence,
        statement_digest: head.statement_digest.clone(),
        policy_digest: head.policy_digest.clone(),
        issued_at_ms: head.issued_at_ms,
    }
}

#[allow(clippy::too_many_arguments)]
fn current_result(
    policy_digest: Option<String>,
    attestation_digest: Option<String>,
    head: &TrackedAcceptanceLedgerHead,
    currentness: &AcceptanceLedgerHeadCurrentnessAttestation,
    use_at_ms: u64,
    signature_valid: bool,
    issues: Vec<CurrentnessIssue>,
    current: Option<CurrentAcceptanceLedgerHead>,
) -> CurrentnessAssessment {
    let disposition = if current.is_some() {
        CurrentnessDisposition::Current
    } else if issues.iter().any(|issue| {
        matches!(
            issue,
            CurrentnessIssue::InvalidPolicy
                | CurrentnessIssue::InvalidAttestation
                | CurrentnessIssue::PolicyDigestMismatch
                | CurrentnessIssue::HeadPolicyDigestMismatch
                | CurrentnessIssue::HeadStatementDigestMismatch
                | CurrentnessIssue::LedgerIdMismatch
                | CurrentnessIssue::LedgerRevisionMismatch
                | CurrentnessIssue::LedgerHeadDigestMismatch
                | CurrentnessIssue::LedgerStateDigestMismatch
                | CurrentnessIssue::AuthoritySequenceMismatch
                | CurrentnessIssue::InvalidPublicKey
                | CurrentnessIssue::SignatureInvalid
        )
    }) {
        CurrentnessDisposition::Invalid
    } else {
        CurrentnessDisposition::Blocked
    };
    CurrentnessAssessment {
        report: CurrentnessReport {
            disposition,
            ledger_id: head.ledger_id.clone(),
            ledger_revision: head.ledger_revision,
            ledger_head_digest: head.ledger_head_digest.clone(),
            ledger_state_digest: head.ledger_state_digest.clone(),
            authority_sequence: head.authority_sequence,
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
    !value.is_empty()
        && value.len() <= MAX_TEXT_BYTES
        && value.trim() == value
        && !value.chars().any(char::is_control)
}

fn digest_text(value: &str) -> bool {
    value
        .strip_prefix("blake3:")
        .is_some_and(|digest| lower_hex_exact(digest, 64))
}

fn lower_hex_exact(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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

fn push_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_be_bytes());
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

fn push_sorted_refs(hasher: &mut blake3::Hasher, values: &[String]) {
    let mut values = values.to_vec();
    values.sort();
    hasher.update(&(values.len() as u64).to_be_bytes());
    for value in values {
        push_field(hasher, &value);
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

fn push_vec_sorted_refs(bytes: &mut Vec<u8>, values: &[String]) {
    let mut values = values.to_vec();
    values.sort();
    bytes.extend_from_slice(&(values.len() as u64).to_be_bytes());
    for value in values {
        push_vec_field(bytes, &value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn d(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn challenge(label: &str) -> String {
        blake3::hash(label.as_bytes()).to_hex().to_string()
    }

    fn record(
        revision: u64,
        label: &str,
        predecessor: Option<String>,
    ) -> AttestationAcceptanceRecord {
        AttestationAcceptanceRecord {
            revision,
            challenge_id: format!("challenge:{label}"),
            nonce_digest: d(&format!("nonce:{label}")),
            possession_record_digest: d(&format!("possession:{label}")),
            quote_artifact_digest: d(&format!("quote:{label}")),
            verification_receipt_id: format!("receipt:{label}"),
            accepted_at_ms: 1_000 + revision,
            predecessor_acceptance_digest: predecessor,
        }
    }

    fn ledger(labels: &[&str]) -> AttestationAcceptanceLedger {
        let mut records = Vec::new();
        let mut predecessor = None;
        for (index, label) in labels.iter().enumerate() {
            let current = record(index as u64 + 1, label, predecessor.clone());
            predecessor = Some(current.acceptance_digest());
            records.push(current);
        }
        AttestationAcceptanceLedger { records }
    }

    fn authority() -> (SigningKey, AcceptanceLedgerHeadAuthorityPolicy) {
        let signing = SigningKey::from_bytes(&[31u8; 32]);
        let key = AcceptanceLedgerHeadAuthorityKey {
            key_id: "acceptance-head-authority:1".into(),
            public_key_ed25519_hex: hex::encode(signing.verifying_key().as_bytes()),
            valid_from_ms: 1,
            valid_until_ms: Some(100_000),
            revoked_at_ms: None,
            allowed_scopes: vec![
                AcceptanceLedgerHeadClaimScope::HeadStatement,
                AcceptanceLedgerHeadClaimScope::CurrentnessAttestation,
            ],
            evidence_refs: vec!["review:key".into()],
        };
        let policy = AcceptanceLedgerHeadAuthorityPolicy {
            schema_version: ACCEPTANCE_HEAD_AUTHORITY_POLICY_SCHEMA_V1.into(),
            policy_id: "policy:acceptance-head:1".into(),
            ledger_id: "ledger:attestation-acceptance:prod".into(),
            sequence: 1,
            issued_at_ms: 1,
            expires_at_ms: 100_000,
            max_ledger_records: 1_000,
            trusted_keys: vec![key],
            evidence_refs: vec!["review:policy".into()],
        };
        (signing, policy)
    }

    fn signed_head(
        signing: &SigningKey,
        policy: &AcceptanceLedgerHeadAuthorityPolicy,
        ledger: &AttestationAcceptanceLedger,
        authority_sequence: u64,
        previous: Option<String>,
    ) -> AcceptanceLedgerHeadStatement {
        let snapshot = snapshot_acceptance_ledger(ledger, policy.max_ledger_records).unwrap();
        let mut statement = AcceptanceLedgerHeadStatement {
            schema_version: ACCEPTANCE_HEAD_STATEMENT_SCHEMA_V1.into(),
            ledger_id: policy.ledger_id.clone(),
            ledger_revision: snapshot.revision(),
            ledger_head_digest: snapshot.head_digest().into(),
            ledger_state_digest: snapshot.state_digest().into(),
            authority_sequence,
            previous_head_statement_digest: previous,
            issued_at_ms: 2_000 + authority_sequence,
            authority_policy_digest: policy.canonical_digest().unwrap(),
            signer_key_id: "acceptance-head-authority:1".into(),
            signer_public_key_ed25519_hex: hex::encode(signing.verifying_key().as_bytes()),
            evidence_refs: vec![format!("head:{authority_sequence}")],
            signature_ed25519_hex: "00".repeat(64),
        };
        let bytes = statement.canonical_unsigned_bytes().unwrap();
        statement.signature_ed25519_hex = hex::encode(signing.sign(&bytes).to_bytes());
        statement
    }

    fn signed_currentness(
        signing: &SigningKey,
        policy: &AcceptanceLedgerHeadAuthorityPolicy,
        head: &TrackedAcceptanceLedgerHead,
        nonce: &str,
        use_at_ms: u64,
    ) -> AcceptanceLedgerHeadCurrentnessAttestation {
        let mut attestation = AcceptanceLedgerHeadCurrentnessAttestation {
            schema_version: ACCEPTANCE_HEAD_CURRENTNESS_SCHEMA_V1.into(),
            head_statement_digest: head.statement_digest().into(),
            ledger_id: head.ledger_id().into(),
            ledger_revision: head.ledger_revision(),
            ledger_head_digest: head.ledger_head_digest().into(),
            ledger_state_digest: head.ledger_state_digest().into(),
            authority_sequence: head.authority_sequence(),
            authority_policy_digest: policy.canonical_digest().unwrap(),
            asserted_current_at_ms: use_at_ms,
            challenge_nonce_blake3_hex: nonce.into(),
            signer_key_id: "acceptance-head-authority:1".into(),
            signer_public_key_ed25519_hex: hex::encode(signing.verifying_key().as_bytes()),
            evidence_refs: vec!["currentness:1".into()],
            signature_ed25519_hex: "00".repeat(64),
        };
        let bytes = attestation.canonical_unsigned_bytes().unwrap();
        attestation.signature_ed25519_hex = hex::encode(signing.sign(&bytes).to_bytes());
        attestation
    }

    #[test]
    fn signed_head_verifies_and_tracks_exact_ledger() {
        let (signing, policy) = authority();
        let ledger = ledger(&["a", "b"]);
        let statement = signed_head(&signing, &policy, &ledger, 1, None);
        let verified = verify_acceptance_ledger_head_statement(&policy, &statement)
            .into_verified()
            .unwrap();
        let mut tracker = AcceptanceLedgerHeadTracker::default();
        let tracked = tracker.observe(&verified, &ledger, &policy).unwrap();
        assert_eq!(tracked.ledger_revision(), 2);
        assert_eq!(
            tracker.latest_statement_digest(&policy.ledger_id),
            Some(tracked.statement_digest())
        );
    }

    #[test]
    fn fresh_tracker_rejects_non_genesis_authority_sequence() {
        let (signing, policy) = authority();
        let ledger = ledger(&["a"]);
        let statement = signed_head(&signing, &policy, &ledger, 7, Some(d("prior")));
        let verified = verify_acceptance_ledger_head_statement(&policy, &statement)
            .into_verified()
            .unwrap();
        let mut tracker = AcceptanceLedgerHeadTracker::default();
        assert!(matches!(
            tracker.observe(&verified, &ledger, &policy),
            Err(
                AcceptanceLedgerHeadTrackingError::FirstAuthorityStatementMustBeSequenceOne {
                    proposed: 7
                }
            )
        ));
    }

    #[test]
    fn tracker_rejects_fork_that_drops_previously_observed_head() {
        let (signing, policy) = authority();
        let first_ledger = ledger(&["a", "b"]);
        let first_statement = signed_head(&signing, &policy, &first_ledger, 1, None);
        let first = verify_acceptance_ledger_head_statement(&policy, &first_statement)
            .into_verified()
            .unwrap();
        let mut tracker = AcceptanceLedgerHeadTracker::default();
        let tracked_first = tracker.observe(&first, &first_ledger, &policy).unwrap();

        let fork = ledger(&["x", "y", "z"]);
        let second_statement = signed_head(
            &signing,
            &policy,
            &fork,
            2,
            Some(tracked_first.statement_digest().into()),
        );
        let second = verify_acceptance_ledger_head_statement(&policy, &second_statement)
            .into_verified()
            .unwrap();
        assert_eq!(
            tracker.observe(&second, &fork, &policy),
            Err(AcceptanceLedgerHeadTrackingError::PreviouslyObservedHeadMissing)
        );
    }

    #[test]
    fn contiguous_extension_tracks() {
        let (signing, policy) = authority();
        let first_ledger = ledger(&["a", "b"]);
        let first_statement = signed_head(&signing, &policy, &first_ledger, 1, None);
        let first = verify_acceptance_ledger_head_statement(&policy, &first_statement)
            .into_verified()
            .unwrap();
        let mut tracker = AcceptanceLedgerHeadTracker::default();
        let tracked_first = tracker.observe(&first, &first_ledger, &policy).unwrap();

        let second_ledger = ledger(&["a", "b", "c"]);
        let second_statement = signed_head(
            &signing,
            &policy,
            &second_ledger,
            2,
            Some(tracked_first.statement_digest().into()),
        );
        let second = verify_acceptance_ledger_head_statement(&policy, &second_statement)
            .into_verified()
            .unwrap();
        let tracked_second = tracker.observe(&second, &second_ledger, &policy).unwrap();
        assert_eq!(tracked_second.ledger_revision(), 3);
        assert_eq!(tracked_second.authority_sequence(), 2);
    }

    #[test]
    fn challenge_bound_exact_use_currentness_mints_capability() {
        let (signing, policy) = authority();
        let ledger = ledger(&["a", "b"]);
        let statement = signed_head(&signing, &policy, &ledger, 1, None);
        let verified = verify_acceptance_ledger_head_statement(&policy, &statement)
            .into_verified()
            .unwrap();
        let mut tracker = AcceptanceLedgerHeadTracker::default();
        let tracked = tracker.observe(&verified, &ledger, &policy).unwrap();
        let nonce = challenge("request:current-head");
        let currentness = signed_currentness(&signing, &policy, &tracked, &nonce, 5_000);
        let assessment = verify_current_acceptance_ledger_head(
            &tracked,
            &policy,
            &currentness,
            &nonce,
            5_000,
        );
        assert_eq!(assessment.report.disposition, CurrentnessDisposition::Current);
        let current = assessment.into_current().unwrap();
        assert_eq!(current.ledger_revision(), 2);
        assert!(current.establishes_authority_asserted_currentness_at_use_time());
        assert!(!current.establishes_trusted_time_provenance());
        assert!(!current.grants_physical_authority());
    }

    #[test]
    fn stale_challenge_cannot_be_reused() {
        let (signing, policy) = authority();
        let ledger = ledger(&["a"]);
        let statement = signed_head(&signing, &policy, &ledger, 1, None);
        let verified = verify_acceptance_ledger_head_statement(&policy, &statement)
            .into_verified()
            .unwrap();
        let mut tracker = AcceptanceLedgerHeadTracker::default();
        let tracked = tracker.observe(&verified, &ledger, &policy).unwrap();
        let nonce_a = challenge("request:a");
        let currentness = signed_currentness(&signing, &policy, &tracked, &nonce_a, 5_000);
        let assessment = verify_current_acceptance_ledger_head(
            &tracked,
            &policy,
            &currentness,
            &challenge("request:b"),
            5_000,
        );
        assert_eq!(assessment.report.disposition, CurrentnessDisposition::Blocked);
        assert!(assessment
            .report
            .issues
            .contains(&CurrentnessIssue::ChallengeNonceMismatch));
    }
}
