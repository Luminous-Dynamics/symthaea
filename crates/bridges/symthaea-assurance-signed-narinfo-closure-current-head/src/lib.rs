// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Authority-current head assurance for one exact signed-narinfo-backed Nix
//! runtime-closure profile.
//!
//! This crate does not re-verify Nix narinfo signatures or the closure graph.
//! That theorem is owned by `symthaea-assurance-signed-narinfo-runtime-closure`.
//! Here the exact opaque qualification produced by that crate is the subject of
//! a signed authority head, a strict observed head lineage, and one fresh
//! challenge-bound currentness assertion.

#![deny(unsafe_code)]

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use symthaea_assurance_signed_narinfo_runtime_closure::
    SignedNarInfoBackedNixRuntimeClosure;

pub const SIGNED_NARINFO_HEAD_AUTHORITY_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.signed-narinfo-closure-head-authority-policy.v1";
pub const SIGNED_NARINFO_HEAD_STATEMENT_SCHEMA_V1: &str =
    "symthaea.assurance.signed-narinfo-closure-head-statement.v1";
pub const SIGNED_NARINFO_HEAD_CURRENTNESS_SCHEMA_V1: &str =
    "symthaea.assurance.signed-narinfo-closure-head-currentness.v1";
pub const SIGNED_NARINFO_HEAD_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.signed-narinfo-closure-head-report.v1";
pub const SIGNED_NARINFO_CURRENTNESS_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.signed-narinfo-closure-currentness-report.v1";

const POLICY_DOMAIN: &[u8] =
    b"symthaea.assurance.signed-narinfo-closure-head-authority-policy.digest.v1\0";
const HEAD_MESSAGE_DOMAIN: &[u8] =
    b"symthaea.assurance.signed-narinfo-closure-head-statement.message.v1\0";
const HEAD_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.signed-narinfo-closure-head-statement.digest.v1\0";
const CURRENTNESS_MESSAGE_DOMAIN: &[u8] =
    b"symthaea.assurance.signed-narinfo-closure-head-currentness.message.v1\0";
const CURRENTNESS_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.signed-narinfo-closure-head-currentness.digest.v1\0";
const CURRENT_CAPABILITY_DOMAIN: &[u8] =
    b"symthaea.assurance.current-signed-narinfo-runtime-closure.digest.v1\0";
const NIX_BASE32: &[u8] = b"0123456789abcdfghijklmnpqrsvwxyz";
const MAX_TEXT_BYTES: usize = 4096;
const MAX_EVIDENCE_REFS: usize = 128;
const MAX_TRUSTED_KEYS: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SignedNarInfoHeadClaimScope {
    HeadStatement,
    CurrentnessAttestation,
}

impl SignedNarInfoHeadClaimScope {
    fn code(self) -> &'static str {
        match self {
            Self::HeadStatement => "head-statement",
            Self::CurrentnessAttestation => "currentness-attestation",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedNarInfoHeadAuthorityKey {
    pub key_id: String,
    pub public_key_ed25519_base64: String,
    pub valid_from_ms: u64,
    pub valid_until_ms: Option<u64>,
    pub revoked_at_ms: Option<u64>,
    pub allowed_scopes: Vec<SignedNarInfoHeadClaimScope>,
    pub evidence_refs: Vec<String>,
}

impl SignedNarInfoHeadAuthorityKey {
    fn validate(&self) -> bool {
        canonical_text(&self.key_id)
            && decode_exact::<32>(&self.public_key_ed25519_base64).is_some()
            && self
                .valid_until_ms
                .map(|until| until > self.valid_from_ms)
                .unwrap_or(true)
            && self
                .revoked_at_ms
                .map(|revoked| revoked >= self.valid_from_ms)
                .unwrap_or(true)
            && !self.allowed_scopes.is_empty()
            && unique(self.allowed_scopes.iter().copied())
            && valid_refs(&self.evidence_refs)
    }

    fn usable_at(&self, at_ms: u64, scope: SignedNarInfoHeadClaimScope) -> bool {
        at_ms >= self.valid_from_ms
            && self
                .valid_until_ms
                .map(|until| at_ms < until)
                .unwrap_or(true)
            && self.revoked_at_ms.map(|at| at_ms < at).unwrap_or(true)
            && self.allowed_scopes.contains(&scope)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedNarInfoHeadAuthorityPolicy {
    pub schema_version: String,
    pub policy_id: String,
    /// Stable reviewed lineage for this exact closure profile/root.
    pub subject_id: String,
    pub expected_signed_closure_policy_digest: String,
    pub expected_root_store_path: String,
    pub policy_sequence: u64,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub trusted_keys: Vec<SignedNarInfoHeadAuthorityKey>,
    pub evidence_refs: Vec<String>,
}

impl SignedNarInfoHeadAuthorityPolicy {
    pub fn validate(&self) -> bool {
        if self.schema_version != SIGNED_NARINFO_HEAD_AUTHORITY_POLICY_SCHEMA_V1
            || !canonical_text(&self.policy_id)
            || !canonical_text(&self.subject_id)
            || !valid_blake3(&self.expected_signed_closure_policy_digest)
            || !valid_store_path(&self.expected_root_store_path)
            || self.policy_sequence == 0
            || self.issued_at_ms >= self.expires_at_ms
            || self.trusted_keys.is_empty()
            || self.trusted_keys.len() > MAX_TRUSTED_KEYS
            || !valid_refs(&self.evidence_refs)
        {
            return false;
        }

        let mut ids = BTreeSet::new();
        let keys_valid = self
            .trusted_keys
            .iter()
            .all(|key| key.validate() && ids.insert(key.key_id.as_str()));
        let has_head_signer = self.trusted_keys.iter().any(|key| {
            key.allowed_scopes
                .contains(&SignedNarInfoHeadClaimScope::HeadStatement)
        });
        let has_currentness_signer = self.trusted_keys.iter().any(|key| {
            key.allowed_scopes
                .contains(&SignedNarInfoHeadClaimScope::CurrentnessAttestation)
        });
        keys_valid && has_head_signer && has_currentness_signer
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut h = blake3::Hasher::new();
        h.update(POLICY_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.subject_id.as_str(),
            self.expected_signed_closure_policy_digest.as_str(),
            self.expected_root_store_path.as_str(),
        ] {
            push_field(&mut h, value);
        }
        push_u64(&mut h, self.policy_sequence);
        push_u64(&mut h, self.issued_at_ms);
        push_u64(&mut h, self.expires_at_ms);
        let mut keys = self.trusted_keys.iter().collect::<Vec<_>>();
        keys.sort_by(|left, right| left.key_id.cmp(&right.key_id));
        push_u64(&mut h, keys.len() as u64);
        for key in keys {
            push_field(&mut h, &key.key_id);
            push_field(&mut h, &key.public_key_ed25519_base64);
            push_u64(&mut h, key.valid_from_ms);
            push_optional_u64(&mut h, key.valid_until_ms);
            push_optional_u64(&mut h, key.revoked_at_ms);
            let mut scopes = key.allowed_scopes.clone();
            scopes.sort();
            push_u64(&mut h, scopes.len() as u64);
            for scope in scopes {
                push_field(&mut h, scope.code());
            }
            push_sorted_strings(&mut h, &key.evidence_refs);
        }
        push_sorted_strings(&mut h, &self.evidence_refs);
        Some(blake3_text(h.finalize()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedNarInfoClosureHeadStatement {
    pub schema_version: String,
    pub subject_id: String,
    pub signed_closure_policy_digest: String,
    pub signed_closure_qualification_digest: String,
    pub capsule_digest: String,
    pub signed_graph_digest: String,
    pub local_closure_digest: String,
    pub root_store_path: String,
    pub authority_sequence: u64,
    pub previous_head_statement_digest: Option<String>,
    pub issued_at_ms: u64,
    pub authority_policy_digest: String,
    pub signer_key_id: String,
    pub signer_public_key_ed25519_base64: String,
    pub evidence_refs: Vec<String>,
    pub signature_ed25519_base64: String,
}

impl SignedNarInfoClosureHeadStatement {
    pub fn validate(&self) -> bool {
        self.schema_version == SIGNED_NARINFO_HEAD_STATEMENT_SCHEMA_V1
            && canonical_text(&self.subject_id)
            && valid_blake3(&self.signed_closure_policy_digest)
            && valid_blake3(&self.signed_closure_qualification_digest)
            && valid_blake3(&self.capsule_digest)
            && valid_blake3(&self.signed_graph_digest)
            && valid_blake3(&self.local_closure_digest)
            && valid_store_path(&self.root_store_path)
            && self.authority_sequence > 0
            && if self.authority_sequence == 1 {
                self.previous_head_statement_digest.is_none()
            } else {
                self.previous_head_statement_digest
                    .as_deref()
                    .is_some_and(valid_blake3)
            }
            && valid_blake3(&self.authority_policy_digest)
            && canonical_text(&self.signer_key_id)
            && decode_exact::<32>(&self.signer_public_key_ed25519_base64).is_some()
            && valid_refs(&self.evidence_refs)
            && decode_exact::<64>(&self.signature_ed25519_base64).is_some()
    }

    pub fn canonical_unsigned_bytes(&self) -> Option<Vec<u8>> {
        if !self.validate() {
            return None;
        }
        let mut out = Vec::new();
        out.extend_from_slice(HEAD_MESSAGE_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.subject_id.as_str(),
            self.signed_closure_policy_digest.as_str(),
            self.signed_closure_qualification_digest.as_str(),
            self.capsule_digest.as_str(),
            self.signed_graph_digest.as_str(),
            self.local_closure_digest.as_str(),
            self.root_store_path.as_str(),
        ] {
            push_vec_field(&mut out, value);
        }
        out.extend_from_slice(&self.authority_sequence.to_be_bytes());
        push_vec_optional_string(&mut out, self.previous_head_statement_digest.as_deref());
        out.extend_from_slice(&self.issued_at_ms.to_be_bytes());
        push_vec_field(&mut out, &self.authority_policy_digest);
        push_vec_field(&mut out, &self.signer_key_id);
        push_vec_field(&mut out, &self.signer_public_key_ed25519_base64);
        push_vec_sorted_strings(&mut out, &self.evidence_refs);
        Some(out)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        let unsigned = self.canonical_unsigned_bytes()?;
        let mut h = blake3::Hasher::new();
        h.update(HEAD_DIGEST_DOMAIN);
        h.update(&(unsigned.len() as u64).to_be_bytes());
        h.update(&unsigned);
        push_field(&mut h, &self.signature_ed25519_base64);
        Some(blake3_text(h.finalize()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedNarInfoClosureHeadCurrentnessAttestation {
    pub schema_version: String,
    pub head_statement_digest: String,
    pub subject_id: String,
    pub signed_closure_policy_digest: String,
    pub signed_closure_qualification_digest: String,
    pub capsule_digest: String,
    pub signed_graph_digest: String,
    pub local_closure_digest: String,
    pub root_store_path: String,
    pub authority_sequence: u64,
    pub authority_policy_digest: String,
    pub asserted_current_at_ms: u64,
    pub challenge_nonce_blake3_hex: String,
    pub signer_key_id: String,
    pub signer_public_key_ed25519_base64: String,
    pub evidence_refs: Vec<String>,
    pub signature_ed25519_base64: String,
}

impl SignedNarInfoClosureHeadCurrentnessAttestation {
    pub fn validate(&self) -> bool {
        self.schema_version == SIGNED_NARINFO_HEAD_CURRENTNESS_SCHEMA_V1
            && valid_blake3(&self.head_statement_digest)
            && canonical_text(&self.subject_id)
            && valid_blake3(&self.signed_closure_policy_digest)
            && valid_blake3(&self.signed_closure_qualification_digest)
            && valid_blake3(&self.capsule_digest)
            && valid_blake3(&self.signed_graph_digest)
            && valid_blake3(&self.local_closure_digest)
            && valid_store_path(&self.root_store_path)
            && self.authority_sequence > 0
            && valid_blake3(&self.authority_policy_digest)
            && valid_challenge(&self.challenge_nonce_blake3_hex)
            && canonical_text(&self.signer_key_id)
            && decode_exact::<32>(&self.signer_public_key_ed25519_base64).is_some()
            && valid_refs(&self.evidence_refs)
            && decode_exact::<64>(&self.signature_ed25519_base64).is_some()
    }

    pub fn canonical_unsigned_bytes(&self) -> Option<Vec<u8>> {
        if !self.validate() {
            return None;
        }
        let mut out = Vec::new();
        out.extend_from_slice(CURRENTNESS_MESSAGE_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.head_statement_digest.as_str(),
            self.subject_id.as_str(),
            self.signed_closure_policy_digest.as_str(),
            self.signed_closure_qualification_digest.as_str(),
            self.capsule_digest.as_str(),
            self.signed_graph_digest.as_str(),
            self.local_closure_digest.as_str(),
            self.root_store_path.as_str(),
        ] {
            push_vec_field(&mut out, value);
        }
        out.extend_from_slice(&self.authority_sequence.to_be_bytes());
        push_vec_field(&mut out, &self.authority_policy_digest);
        out.extend_from_slice(&self.asserted_current_at_ms.to_be_bytes());
        push_vec_field(&mut out, &self.challenge_nonce_blake3_hex);
        push_vec_field(&mut out, &self.signer_key_id);
        push_vec_field(&mut out, &self.signer_public_key_ed25519_base64);
        push_vec_sorted_strings(&mut out, &self.evidence_refs);
        Some(out)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        let unsigned = self.canonical_unsigned_bytes()?;
        let mut h = blake3::Hasher::new();
        h.update(CURRENTNESS_DIGEST_DOMAIN);
        h.update(&(unsigned.len() as u64).to_be_bytes());
        h.update(&unsigned);
        push_field(&mut h, &self.signature_ed25519_base64);
        Some(blake3_text(h.finalize()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignedNarInfoHeadIssue {
    InvalidPolicy,
    InvalidStatement,
    PolicyDigestMismatch,
    SubjectMismatch,
    SignedClosurePolicyMismatch,
    SignedClosureQualificationMismatch,
    CapsuleMismatch,
    SignedGraphMismatch,
    LocalClosureMismatch,
    RootStorePathMismatch,
    PolicyNotEffectiveAtStatementTime,
    SignerUnknown,
    SignerPublicKeyMismatch,
    SignerNotEffectiveOrAuthorized,
    SignatureInvalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedNarInfoHeadReport {
    pub schema_version: String,
    pub statement_digest: Option<String>,
    pub policy_digest: Option<String>,
    pub subject_id: String,
    pub authority_sequence: u64,
    pub signer_key_id: String,
    pub signature_valid: bool,
    pub issues: Vec<SignedNarInfoHeadIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedSignedNarInfoClosureHeadStatement {
    subject_id: String,
    signed_closure_policy_digest: String,
    signed_closure_qualification_digest: String,
    capsule_digest: String,
    signed_graph_digest: String,
    local_closure_digest: String,
    root_store_path: String,
    authority_sequence: u64,
    previous_head_statement_digest: Option<String>,
    issued_at_ms: u64,
    statement_digest: String,
    authority_policy_digest: String,
}

impl VerifiedSignedNarInfoClosureHeadStatement {
    pub fn subject_id(&self) -> &str {
        &self.subject_id
    }
    pub fn signed_closure_policy_digest(&self) -> &str {
        &self.signed_closure_policy_digest
    }
    pub fn signed_closure_qualification_digest(&self) -> &str {
        &self.signed_closure_qualification_digest
    }
    pub fn capsule_digest(&self) -> &str {
        &self.capsule_digest
    }
    pub fn signed_graph_digest(&self) -> &str {
        &self.signed_graph_digest
    }
    pub fn local_closure_digest(&self) -> &str {
        &self.local_closure_digest
    }
    pub fn root_store_path(&self) -> &str {
        &self.root_store_path
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
    pub fn authority_policy_digest(&self) -> &str {
        &self.authority_policy_digest
    }
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

pub fn verify_signed_narinfo_closure_head_statement(
    policy: &SignedNarInfoHeadAuthorityPolicy,
    closure: &SignedNarInfoBackedNixRuntimeClosure,
    statement: &SignedNarInfoClosureHeadStatement,
) -> Result<VerifiedSignedNarInfoClosureHeadStatement, SignedNarInfoHeadReport> {
    let policy_digest = policy.canonical_digest();
    let statement_digest = statement.canonical_digest();
    let mut report = SignedNarInfoHeadReport {
        schema_version: SIGNED_NARINFO_HEAD_REPORT_SCHEMA_V1.into(),
        statement_digest: statement_digest.clone(),
        policy_digest: policy_digest.clone(),
        subject_id: statement.subject_id.clone(),
        authority_sequence: statement.authority_sequence,
        signer_key_id: statement.signer_key_id.clone(),
        signature_valid: false,
        issues: Vec::new(),
    };

    if !policy.validate() {
        report.issues.push(SignedNarInfoHeadIssue::InvalidPolicy);
    }
    if !statement.validate() {
        report.issues.push(SignedNarInfoHeadIssue::InvalidStatement);
    }
    if policy_digest.as_deref() != Some(statement.authority_policy_digest.as_str()) {
        report.issues.push(SignedNarInfoHeadIssue::PolicyDigestMismatch);
    }
    if statement.subject_id != policy.subject_id {
        report.issues.push(SignedNarInfoHeadIssue::SubjectMismatch);
    }
    if closure.policy_digest() != policy.expected_signed_closure_policy_digest
        || statement.signed_closure_policy_digest != policy.expected_signed_closure_policy_digest
    {
        report
            .issues
            .push(SignedNarInfoHeadIssue::SignedClosurePolicyMismatch);
    }
    if statement.signed_closure_qualification_digest != closure.qualification_digest() {
        report
            .issues
            .push(SignedNarInfoHeadIssue::SignedClosureQualificationMismatch);
    }
    if statement.capsule_digest != closure.capsule_digest() {
        report.issues.push(SignedNarInfoHeadIssue::CapsuleMismatch);
    }
    if statement.signed_graph_digest != closure.signed_graph_digest() {
        report
            .issues
            .push(SignedNarInfoHeadIssue::SignedGraphMismatch);
    }
    if statement.local_closure_digest != closure.local_closure_digest() {
        report
            .issues
            .push(SignedNarInfoHeadIssue::LocalClosureMismatch);
    }
    if statement.root_store_path != policy.expected_root_store_path
        || closure.root_store_path() != policy.expected_root_store_path
    {
        report
            .issues
            .push(SignedNarInfoHeadIssue::RootStorePathMismatch);
    }
    if !report.issues.is_empty() {
        return Err(report);
    }

    if !(policy.issued_at_ms <= statement.issued_at_ms
        && statement.issued_at_ms < policy.expires_at_ms)
    {
        report
            .issues
            .push(SignedNarInfoHeadIssue::PolicyNotEffectiveAtStatementTime);
    }
    let Some(key) = policy
        .trusted_keys
        .iter()
        .find(|key| key.key_id == statement.signer_key_id)
    else {
        report.issues.push(SignedNarInfoHeadIssue::SignerUnknown);
        return Err(report);
    };
    if key.public_key_ed25519_base64 != statement.signer_public_key_ed25519_base64 {
        report
            .issues
            .push(SignedNarInfoHeadIssue::SignerPublicKeyMismatch);
    }
    if !key.usable_at(
        statement.issued_at_ms,
        SignedNarInfoHeadClaimScope::HeadStatement,
    ) {
        report
            .issues
            .push(SignedNarInfoHeadIssue::SignerNotEffectiveOrAuthorized);
    }
    if report.issues.is_empty() {
        let message = statement
            .canonical_unsigned_bytes()
            .expect("validated statement has canonical bytes");
        report.signature_valid = verify_signature(
            &statement.signer_public_key_ed25519_base64,
            &statement.signature_ed25519_base64,
            &message,
        );
        if !report.signature_valid {
            report.issues.push(SignedNarInfoHeadIssue::SignatureInvalid);
        }
    }
    if !report.issues.is_empty() {
        return Err(report);
    }

    Ok(VerifiedSignedNarInfoClosureHeadStatement {
        subject_id: statement.subject_id.clone(),
        signed_closure_policy_digest: statement.signed_closure_policy_digest.clone(),
        signed_closure_qualification_digest: statement.signed_closure_qualification_digest.clone(),
        capsule_digest: statement.capsule_digest.clone(),
        signed_graph_digest: statement.signed_graph_digest.clone(),
        local_closure_digest: statement.local_closure_digest.clone(),
        root_store_path: statement.root_store_path.clone(),
        authority_sequence: statement.authority_sequence,
        previous_head_statement_digest: statement.previous_head_statement_digest.clone(),
        issued_at_ms: statement.issued_at_ms,
        statement_digest: statement_digest.expect("validated statement has digest"),
        authority_policy_digest: policy_digest.expect("validated policy has digest"),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TrackedState {
    authority_sequence: u64,
    statement_digest: String,
    authority_policy_digest: String,
    issued_at_ms: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SignedNarInfoClosureHeadTracker {
    heads: BTreeMap<String, TrackedState>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignedNarInfoHeadTrackingError {
    FirstHeadMustBeSequenceOne { proposed: u64 },
    FirstHeadHasPredecessor,
    AuthorityPolicyChanged { expected: String, proposed: String },
    SequenceRollback { latest: u64, proposed: u64 },
    SequenceCollision { sequence: u64 },
    SequenceGap { expected: u64, proposed: u64 },
    PreviousHeadDigestMismatch,
    IssuedAtRegression { latest: u64, proposed: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrackedSignedNarInfoClosureHead {
    subject_id: String,
    signed_closure_policy_digest: String,
    signed_closure_qualification_digest: String,
    capsule_digest: String,
    signed_graph_digest: String,
    local_closure_digest: String,
    root_store_path: String,
    authority_sequence: u64,
    statement_digest: String,
    authority_policy_digest: String,
    issued_at_ms: u64,
}

impl TrackedSignedNarInfoClosureHead {
    pub fn subject_id(&self) -> &str {
        &self.subject_id
    }
    pub fn signed_closure_policy_digest(&self) -> &str {
        &self.signed_closure_policy_digest
    }
    pub fn signed_closure_qualification_digest(&self) -> &str {
        &self.signed_closure_qualification_digest
    }
    pub fn capsule_digest(&self) -> &str {
        &self.capsule_digest
    }
    pub fn signed_graph_digest(&self) -> &str {
        &self.signed_graph_digest
    }
    pub fn local_closure_digest(&self) -> &str {
        &self.local_closure_digest
    }
    pub fn root_store_path(&self) -> &str {
        &self.root_store_path
    }
    pub const fn authority_sequence(&self) -> u64 {
        self.authority_sequence
    }
    pub fn statement_digest(&self) -> &str {
        &self.statement_digest
    }
    pub fn authority_policy_digest(&self) -> &str {
        &self.authority_policy_digest
    }
    pub const fn issued_at_ms(&self) -> u64 {
        self.issued_at_ms
    }
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

impl SignedNarInfoClosureHeadTracker {
    pub fn observe(
        &mut self,
        head: &VerifiedSignedNarInfoClosureHeadStatement,
    ) -> Result<TrackedSignedNarInfoClosureHead, SignedNarInfoHeadTrackingError> {
        match self.heads.get(head.subject_id()) {
            None => {
                if head.authority_sequence() != 1 {
                    return Err(SignedNarInfoHeadTrackingError::FirstHeadMustBeSequenceOne {
                        proposed: head.authority_sequence(),
                    });
                }
                if head.previous_head_statement_digest().is_some() {
                    return Err(SignedNarInfoHeadTrackingError::FirstHeadHasPredecessor);
                }
            }
            Some(latest) => {
                if head.authority_policy_digest() != latest.authority_policy_digest {
                    return Err(SignedNarInfoHeadTrackingError::AuthorityPolicyChanged {
                        expected: latest.authority_policy_digest.clone(),
                        proposed: head.authority_policy_digest().to_string(),
                    });
                }
                if head.authority_sequence() < latest.authority_sequence {
                    return Err(SignedNarInfoHeadTrackingError::SequenceRollback {
                        latest: latest.authority_sequence,
                        proposed: head.authority_sequence(),
                    });
                }
                if head.authority_sequence() == latest.authority_sequence {
                    if head.statement_digest() == latest.statement_digest {
                        return Ok(tracked(head));
                    }
                    return Err(SignedNarInfoHeadTrackingError::SequenceCollision {
                        sequence: head.authority_sequence(),
                    });
                }
                let expected = latest
                    .authority_sequence
                    .checked_add(1)
                    .ok_or(SignedNarInfoHeadTrackingError::SequenceGap {
                        expected: u64::MAX,
                        proposed: head.authority_sequence(),
                    })?;
                if head.authority_sequence() != expected {
                    return Err(SignedNarInfoHeadTrackingError::SequenceGap {
                        expected,
                        proposed: head.authority_sequence(),
                    });
                }
                if head.previous_head_statement_digest() != Some(latest.statement_digest.as_str()) {
                    return Err(SignedNarInfoHeadTrackingError::PreviousHeadDigestMismatch);
                }
                if head.issued_at_ms() < latest.issued_at_ms {
                    return Err(SignedNarInfoHeadTrackingError::IssuedAtRegression {
                        latest: latest.issued_at_ms,
                        proposed: head.issued_at_ms(),
                    });
                }
            }
        }
        self.heads.insert(
            head.subject_id().to_string(),
            TrackedState {
                authority_sequence: head.authority_sequence(),
                statement_digest: head.statement_digest().to_string(),
                authority_policy_digest: head.authority_policy_digest().to_string(),
                issued_at_ms: head.issued_at_ms(),
            },
        );
        Ok(tracked(head))
    }

    pub fn is_latest(&self, head: &TrackedSignedNarInfoClosureHead) -> bool {
        self.heads.get(head.subject_id()).is_some_and(|latest| {
            latest.authority_sequence == head.authority_sequence
                && latest.statement_digest == head.statement_digest
                && latest.authority_policy_digest == head.authority_policy_digest
        })
    }

    pub fn latest_sequence(&self, subject_id: &str) -> Option<u64> {
        self.heads
            .get(subject_id)
            .map(|head| head.authority_sequence)
    }

    pub fn latest_statement_digest(&self, subject_id: &str) -> Option<&str> {
        self.heads
            .get(subject_id)
            .map(|head| head.statement_digest.as_str())
    }

    pub fn latest_authority_policy_digest(&self, subject_id: &str) -> Option<&str> {
        self.heads
            .get(subject_id)
            .map(|head| head.authority_policy_digest.as_str())
    }
}

fn tracked(head: &VerifiedSignedNarInfoClosureHeadStatement) -> TrackedSignedNarInfoClosureHead {
    TrackedSignedNarInfoClosureHead {
        subject_id: head.subject_id.clone(),
        signed_closure_policy_digest: head.signed_closure_policy_digest.clone(),
        signed_closure_qualification_digest: head.signed_closure_qualification_digest.clone(),
        capsule_digest: head.capsule_digest.clone(),
        signed_graph_digest: head.signed_graph_digest.clone(),
        local_closure_digest: head.local_closure_digest.clone(),
        root_store_path: head.root_store_path.clone(),
        authority_sequence: head.authority_sequence,
        statement_digest: head.statement_digest.clone(),
        authority_policy_digest: head.authority_policy_digest.clone(),
        issued_at_ms: head.issued_at_ms,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignedNarInfoCurrentnessIssue {
    InvalidPolicy,
    InvalidAttestation,
    TrackedHeadNotLatest,
    PolicyDigestMismatch,
    SubjectMismatch,
    HeadStatementDigestMismatch,
    SignedClosurePolicyMismatch,
    SignedClosureQualificationMismatch,
    CapsuleMismatch,
    SignedGraphMismatch,
    LocalClosureMismatch,
    RootStorePathMismatch,
    AuthoritySequenceMismatch,
    HeadIssuedAfterUseTime,
    CurrentnessNotAtUseTime,
    ChallengeMismatch,
    PolicyNotEffectiveAtUseTime,
    SignerUnknown,
    SignerPublicKeyMismatch,
    SignerNotEffectiveOrAuthorized,
    SignatureInvalid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedNarInfoCurrentnessReport {
    pub schema_version: String,
    pub subject_id: String,
    pub authority_sequence: u64,
    pub head_statement_digest: String,
    pub currentness_attestation_digest: Option<String>,
    pub authority_policy_digest: Option<String>,
    pub use_at_ms: u64,
    pub challenge_nonce_blake3_hex: String,
    pub signer_key_id: String,
    pub signature_valid: bool,
    pub issues: Vec<SignedNarInfoCurrentnessIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentSignedNarInfoRuntimeClosure {
    currentness_digest: String,
    signed_closure_policy_digest: String,
    signed_closure_qualification_digest: String,
    capsule_digest: String,
    signed_graph_digest: String,
    local_closure_digest: String,
    root_store_path: String,
    subject_id: String,
    authority_sequence: u64,
    head_statement_digest: String,
    currentness_attestation_digest: String,
    authority_policy_digest: String,
    challenge_nonce_blake3_hex: String,
    use_at_ms: u64,
}

impl CurrentSignedNarInfoRuntimeClosure {
    pub fn currentness_digest(&self) -> &str {
        &self.currentness_digest
    }
    pub fn signed_closure_policy_digest(&self) -> &str {
        &self.signed_closure_policy_digest
    }
    pub fn signed_closure_qualification_digest(&self) -> &str {
        &self.signed_closure_qualification_digest
    }
    pub fn capsule_digest(&self) -> &str {
        &self.capsule_digest
    }
    pub fn signed_graph_digest(&self) -> &str {
        &self.signed_graph_digest
    }
    pub fn local_closure_digest(&self) -> &str {
        &self.local_closure_digest
    }
    pub fn root_store_path(&self) -> &str {
        &self.root_store_path
    }
    pub fn subject_id(&self) -> &str {
        &self.subject_id
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
    pub fn challenge_nonce_blake3_hex(&self) -> &str {
        &self.challenge_nonce_blake3_hex
    }
    pub const fn use_at_ms(&self) -> u64 {
        self.use_at_ms
    }

    pub const fn signed_graph_current_under_reviewed_authority_at_use(&self) -> bool {
        true
    }
    pub const fn stale_observed_predecessor_rejected_after_tracker_advance(&self) -> bool {
        true
    }
    pub const fn authority_policy_immutable_within_tracked_lineage(&self) -> bool {
        true
    }
    pub const fn currentness_challenge_bound(&self) -> bool {
        true
    }
    pub const fn currentness_challenge_nonzero(&self) -> bool {
        true
    }
    pub const fn head_not_issued_after_use_time(&self) -> bool {
        true
    }
    pub const fn tracker_persistence_rollback_resistance_established(&self) -> bool {
        false
    }
    pub const fn authority_non_compromise_established(&self) -> bool {
        false
    }
    pub const fn global_cache_latestness_established(&self) -> bool {
        false
    }
    pub const fn nix_database_global_currentness_established(&self) -> bool {
        false
    }
    pub const fn trusted_time_established(&self) -> bool {
        false
    }
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

pub fn verify_current_signed_narinfo_runtime_closure(
    tracker: &SignedNarInfoClosureHeadTracker,
    tracked: &TrackedSignedNarInfoClosureHead,
    policy: &SignedNarInfoHeadAuthorityPolicy,
    attestation: &SignedNarInfoClosureHeadCurrentnessAttestation,
    expected_challenge_nonce_blake3_hex: &str,
    use_at_ms: u64,
) -> Result<CurrentSignedNarInfoRuntimeClosure, SignedNarInfoCurrentnessReport> {
    let policy_digest = policy.canonical_digest();
    let attestation_digest = attestation.canonical_digest();
    let mut report = SignedNarInfoCurrentnessReport {
        schema_version: SIGNED_NARINFO_CURRENTNESS_REPORT_SCHEMA_V1.into(),
        subject_id: tracked.subject_id.clone(),
        authority_sequence: tracked.authority_sequence,
        head_statement_digest: tracked.statement_digest.clone(),
        currentness_attestation_digest: attestation_digest.clone(),
        authority_policy_digest: policy_digest.clone(),
        use_at_ms,
        challenge_nonce_blake3_hex: expected_challenge_nonce_blake3_hex.into(),
        signer_key_id: attestation.signer_key_id.clone(),
        signature_valid: false,
        issues: Vec::new(),
    };

    if !policy.validate() {
        report
            .issues
            .push(SignedNarInfoCurrentnessIssue::InvalidPolicy);
    }
    if !attestation.validate() {
        report
            .issues
            .push(SignedNarInfoCurrentnessIssue::InvalidAttestation);
    }
    if !tracker.is_latest(tracked) {
        report
            .issues
            .push(SignedNarInfoCurrentnessIssue::TrackedHeadNotLatest);
    }
    if policy_digest.as_deref() != Some(attestation.authority_policy_digest.as_str())
        || tracked.authority_policy_digest != attestation.authority_policy_digest
    {
        report
            .issues
            .push(SignedNarInfoCurrentnessIssue::PolicyDigestMismatch);
    }
    if tracked.subject_id != policy.subject_id || tracked.subject_id != attestation.subject_id {
        report
            .issues
            .push(SignedNarInfoCurrentnessIssue::SubjectMismatch);
    }
    if tracked.statement_digest != attestation.head_statement_digest {
        report
            .issues
            .push(SignedNarInfoCurrentnessIssue::HeadStatementDigestMismatch);
    }
    if tracked.signed_closure_policy_digest != policy.expected_signed_closure_policy_digest
        || tracked.signed_closure_policy_digest != attestation.signed_closure_policy_digest
    {
        report
            .issues
            .push(SignedNarInfoCurrentnessIssue::SignedClosurePolicyMismatch);
    }
    if tracked.signed_closure_qualification_digest
        != attestation.signed_closure_qualification_digest
    {
        report
            .issues
            .push(SignedNarInfoCurrentnessIssue::SignedClosureQualificationMismatch);
    }
    if tracked.capsule_digest != attestation.capsule_digest {
        report
            .issues
            .push(SignedNarInfoCurrentnessIssue::CapsuleMismatch);
    }
    if tracked.signed_graph_digest != attestation.signed_graph_digest {
        report
            .issues
            .push(SignedNarInfoCurrentnessIssue::SignedGraphMismatch);
    }
    if tracked.local_closure_digest != attestation.local_closure_digest {
        report
            .issues
            .push(SignedNarInfoCurrentnessIssue::LocalClosureMismatch);
    }
    if tracked.root_store_path != policy.expected_root_store_path
        || tracked.root_store_path != attestation.root_store_path
    {
        report
            .issues
            .push(SignedNarInfoCurrentnessIssue::RootStorePathMismatch);
    }
    if tracked.authority_sequence != attestation.authority_sequence {
        report
            .issues
            .push(SignedNarInfoCurrentnessIssue::AuthoritySequenceMismatch);
    }
    if tracked.issued_at_ms > use_at_ms {
        report
            .issues
            .push(SignedNarInfoCurrentnessIssue::HeadIssuedAfterUseTime);
    }
    if attestation.asserted_current_at_ms != use_at_ms {
        report
            .issues
            .push(SignedNarInfoCurrentnessIssue::CurrentnessNotAtUseTime);
    }
    if attestation.challenge_nonce_blake3_hex != expected_challenge_nonce_blake3_hex
        || !valid_challenge(expected_challenge_nonce_blake3_hex)
    {
        report
            .issues
            .push(SignedNarInfoCurrentnessIssue::ChallengeMismatch);
    }
    if !report.issues.is_empty() {
        return Err(report);
    }

    if !(policy.issued_at_ms <= use_at_ms && use_at_ms < policy.expires_at_ms) {
        report
            .issues
            .push(SignedNarInfoCurrentnessIssue::PolicyNotEffectiveAtUseTime);
    }
    let Some(key) = policy
        .trusted_keys
        .iter()
        .find(|key| key.key_id == attestation.signer_key_id)
    else {
        report
            .issues
            .push(SignedNarInfoCurrentnessIssue::SignerUnknown);
        return Err(report);
    };
    if key.public_key_ed25519_base64 != attestation.signer_public_key_ed25519_base64 {
        report
            .issues
            .push(SignedNarInfoCurrentnessIssue::SignerPublicKeyMismatch);
    }
    if !key.usable_at(
        use_at_ms,
        SignedNarInfoHeadClaimScope::CurrentnessAttestation,
    ) {
        report
            .issues
            .push(SignedNarInfoCurrentnessIssue::SignerNotEffectiveOrAuthorized);
    }
    if report.issues.is_empty() {
        let message = attestation
            .canonical_unsigned_bytes()
            .expect("validated attestation has canonical bytes");
        report.signature_valid = verify_signature(
            &attestation.signer_public_key_ed25519_base64,
            &attestation.signature_ed25519_base64,
            &message,
        );
        if !report.signature_valid {
            report
                .issues
                .push(SignedNarInfoCurrentnessIssue::SignatureInvalid);
        }
    }
    if !report.issues.is_empty() {
        return Err(report);
    }

    let policy_digest = policy_digest.expect("validated policy has digest");
    let attestation_digest = attestation_digest.expect("validated attestation has digest");
    let currentness_digest = current_capability_digest(
        &policy_digest,
        &tracked.statement_digest,
        &attestation_digest,
        &tracked.signed_closure_qualification_digest,
        &tracked.capsule_digest,
        &tracked.signed_graph_digest,
        &tracked.local_closure_digest,
        &tracked.root_store_path,
        tracked.authority_sequence,
        expected_challenge_nonce_blake3_hex,
        use_at_ms,
    );
    Ok(CurrentSignedNarInfoRuntimeClosure {
        currentness_digest,
        signed_closure_policy_digest: tracked.signed_closure_policy_digest.clone(),
        signed_closure_qualification_digest: tracked.signed_closure_qualification_digest.clone(),
        capsule_digest: tracked.capsule_digest.clone(),
        signed_graph_digest: tracked.signed_graph_digest.clone(),
        local_closure_digest: tracked.local_closure_digest.clone(),
        root_store_path: tracked.root_store_path.clone(),
        subject_id: tracked.subject_id.clone(),
        authority_sequence: tracked.authority_sequence,
        head_statement_digest: tracked.statement_digest.clone(),
        currentness_attestation_digest: attestation_digest,
        authority_policy_digest: policy_digest,
        challenge_nonce_blake3_hex: expected_challenge_nonce_blake3_hex.into(),
        use_at_ms,
    })
}

#[allow(clippy::too_many_arguments)]
fn current_capability_digest(
    policy_digest: &str,
    head_statement_digest: &str,
    currentness_attestation_digest: &str,
    signed_closure_qualification_digest: &str,
    capsule_digest: &str,
    signed_graph_digest: &str,
    local_closure_digest: &str,
    root_store_path: &str,
    authority_sequence: u64,
    challenge_nonce_blake3_hex: &str,
    use_at_ms: u64,
) -> String {
    let mut h = blake3::Hasher::new();
    h.update(CURRENT_CAPABILITY_DOMAIN);
    for value in [
        policy_digest,
        head_statement_digest,
        currentness_attestation_digest,
        signed_closure_qualification_digest,
        capsule_digest,
        signed_graph_digest,
        local_closure_digest,
        root_store_path,
        challenge_nonce_blake3_hex,
    ] {
        push_field(&mut h, value);
    }
    push_u64(&mut h, authority_sequence);
    push_u64(&mut h, use_at_ms);
    blake3_text(h.finalize())
}

fn verify_signature(public_key_base64: &str, signature_base64: &str, message: &[u8]) -> bool {
    let Some(public_key) = decode_exact::<32>(public_key_base64) else {
        return false;
    };
    let Some(signature) = decode_exact::<64>(signature_base64) else {
        return false;
    };
    let Ok(key) = VerifyingKey::from_bytes(&public_key) else {
        return false;
    };
    key.verify(message, &Signature::from_bytes(&signature)).is_ok()
}

fn decode_exact<const N: usize>(value: &str) -> Option<[u8; N]> {
    if value.is_empty() || value.chars().any(char::is_whitespace) {
        return None;
    }
    BASE64.decode(value.as_bytes()).ok()?.try_into().ok()
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_TEXT_BYTES
        && !value.chars().any(char::is_control)
}

fn valid_blake3(value: &str) -> bool {
    value
        .strip_prefix("blake3:")
        .is_some_and(|digest| lower_hex_exact(digest, 64))
}

fn lower_hex_exact(value: &str, expected: usize) -> bool {
    value.len() == expected
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_challenge(value: &str) -> bool {
    lower_hex_exact(value, 64) && value.bytes().any(|byte| byte != b'0')
}

fn valid_store_path(value: &str) -> bool {
    let Some(component) = value.strip_prefix("/nix/store/") else {
        return false;
    };
    if component.contains('/') || component.len() <= 33 || component.as_bytes()[32] != b'-' {
        return false;
    }
    component.as_bytes()[..32]
        .iter()
        .all(|byte| NIX_BASE32.contains(byte))
        && component[33..].bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'-' | b'.' | b'_' | b'?' | b'=')
        })
}

fn valid_refs(values: &[String]) -> bool {
    !values.is_empty()
        && values.len() <= MAX_EVIDENCE_REFS
        && values.iter().all(|value| canonical_text(value))
        && unique(values.iter().map(String::as_str))
}

fn unique<T: Ord>(values: impl Iterator<Item = T>) -> bool {
    let mut seen = BTreeSet::new();
    values.into_iter().all(|value| seen.insert(value))
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

fn push_sorted_strings(hasher: &mut blake3::Hasher, values: &[String]) {
    let mut values = values.to_vec();
    values.sort();
    push_u64(hasher, values.len() as u64);
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

fn push_vec_sorted_strings(bytes: &mut Vec<u8>, values: &[String]) {
    let mut values = values.to_vec();
    values.sort();
    bytes.extend_from_slice(&(values.len() as u64).to_be_bytes());
    for value in values {
        push_vec_field(bytes, &value);
    }
}

fn blake3_text(hash: blake3::Hash) -> String {
    format!("blake3:{}", hash.to_hex())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    const ROOT: &str = "/nix/store/00000000000000000000000000000000-runtime";

    fn d(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn nonce(label: &str) -> String {
        blake3::hash(label.as_bytes()).to_hex().to_string()
    }

    fn authority() -> (SigningKey, SignedNarInfoHeadAuthorityPolicy) {
        let signing = SigningKey::from_bytes(&[19u8; 32]);
        let key = SignedNarInfoHeadAuthorityKey {
            key_id: "narinfo-head:1".into(),
            public_key_ed25519_base64: BASE64.encode(signing.verifying_key().to_bytes()),
            valid_from_ms: 1,
            valid_until_ms: Some(100_000),
            revoked_at_ms: None,
            allowed_scopes: vec![
                SignedNarInfoHeadClaimScope::HeadStatement,
                SignedNarInfoHeadClaimScope::CurrentnessAttestation,
            ],
            evidence_refs: vec!["review:key".into()],
        };
        let policy = SignedNarInfoHeadAuthorityPolicy {
            schema_version: SIGNED_NARINFO_HEAD_AUTHORITY_POLICY_SCHEMA_V1.into(),
            policy_id: "policy:narinfo-head:1".into(),
            subject_id: "runtime-closure:prod".into(),
            expected_signed_closure_policy_digest: d("signed-closure-policy"),
            expected_root_store_path: ROOT.into(),
            policy_sequence: 1,
            issued_at_ms: 1,
            expires_at_ms: 100_000,
            trusted_keys: vec![key],
            evidence_refs: vec!["review:policy".into()],
        };
        (signing, policy)
    }

    fn verified_head(
        policy: &SignedNarInfoHeadAuthorityPolicy,
        sequence: u64,
        previous: Option<String>,
        suffix: &str,
    ) -> VerifiedSignedNarInfoClosureHeadStatement {
        VerifiedSignedNarInfoClosureHeadStatement {
            subject_id: policy.subject_id.clone(),
            signed_closure_policy_digest: policy.expected_signed_closure_policy_digest.clone(),
            signed_closure_qualification_digest: d(&format!("qualification:{suffix}")),
            capsule_digest: d(&format!("capsule:{suffix}")),
            signed_graph_digest: d(&format!("graph:{suffix}")),
            local_closure_digest: d(&format!("local:{suffix}")),
            root_store_path: ROOT.into(),
            authority_sequence: sequence,
            previous_head_statement_digest: previous,
            issued_at_ms: 1_000 + sequence,
            statement_digest: d(&format!("head:{sequence}:{suffix}")),
            authority_policy_digest: policy.canonical_digest().unwrap(),
        }
    }

    fn signed_currentness(
        signing: &SigningKey,
        policy: &SignedNarInfoHeadAuthorityPolicy,
        tracked: &TrackedSignedNarInfoClosureHead,
        challenge: &str,
        use_at_ms: u64,
    ) -> SignedNarInfoClosureHeadCurrentnessAttestation {
        let mut value = SignedNarInfoClosureHeadCurrentnessAttestation {
            schema_version: SIGNED_NARINFO_HEAD_CURRENTNESS_SCHEMA_V1.into(),
            head_statement_digest: tracked.statement_digest.clone(),
            subject_id: tracked.subject_id.clone(),
            signed_closure_policy_digest: tracked.signed_closure_policy_digest.clone(),
            signed_closure_qualification_digest: tracked.signed_closure_qualification_digest.clone(),
            capsule_digest: tracked.capsule_digest.clone(),
            signed_graph_digest: tracked.signed_graph_digest.clone(),
            local_closure_digest: tracked.local_closure_digest.clone(),
            root_store_path: tracked.root_store_path.clone(),
            authority_sequence: tracked.authority_sequence,
            authority_policy_digest: policy.canonical_digest().unwrap(),
            asserted_current_at_ms: use_at_ms,
            challenge_nonce_blake3_hex: challenge.into(),
            signer_key_id: "narinfo-head:1".into(),
            signer_public_key_ed25519_base64: BASE64.encode(signing.verifying_key().to_bytes()),
            evidence_refs: vec!["currentness:test".into()],
            signature_ed25519_base64: BASE64.encode([0u8; 64]),
        };
        let bytes = value.canonical_unsigned_bytes().unwrap();
        value.signature_ed25519_base64 = BASE64.encode(signing.sign(&bytes).to_bytes());
        value
    }

    #[test]
    fn policy_reference_order_is_nonsemantic() {
        let (_, mut left) = authority();
        left.evidence_refs = vec!["review:a".into(), "review:b".into()];
        let mut right = left.clone();
        right.evidence_refs.reverse();
        assert_eq!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn policy_requires_both_head_and_currentness_scopes() {
        let (_, mut policy) = authority();
        policy.trusted_keys[0].allowed_scopes = vec![SignedNarInfoHeadClaimScope::HeadStatement];
        assert!(!policy.validate());
    }

    #[test]
    fn store_path_validation_matches_nix_hash_alphabet() {
        assert!(valid_store_path(ROOT));
        assert!(!valid_store_path(
            "/nix/store/eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee-runtime"
        ));
    }

    #[test]
    fn fresh_tracker_requires_sequence_one() {
        let (_, policy) = authority();
        let head = verified_head(&policy, 2, Some(d("old")), "two");
        let mut tracker = SignedNarInfoClosureHeadTracker::default();
        assert!(matches!(
            tracker.observe(&head),
            Err(SignedNarInfoHeadTrackingError::FirstHeadMustBeSequenceOne { proposed: 2 })
        ));
    }

    #[test]
    fn successor_makes_predecessor_non_current() {
        let (_, policy) = authority();
        let first = verified_head(&policy, 1, None, "one");
        let mut tracker = SignedNarInfoClosureHeadTracker::default();
        let tracked_first = tracker.observe(&first).unwrap();
        assert!(tracker.is_latest(&tracked_first));
        let second = verified_head(
            &policy,
            2,
            Some(tracked_first.statement_digest().into()),
            "two",
        );
        let tracked_second = tracker.observe(&second).unwrap();
        assert!(!tracker.is_latest(&tracked_first));
        assert!(tracker.is_latest(&tracked_second));
    }

    #[test]
    fn tracker_rejects_silent_authority_policy_rotation() {
        let (_, policy) = authority();
        let first = verified_head(&policy, 1, None, "one");
        let mut tracker = SignedNarInfoClosureHeadTracker::default();
        let tracked_first = tracker.observe(&first).unwrap();

        let mut rotated = policy.clone();
        rotated.policy_sequence = 2;
        rotated.evidence_refs = vec!["review:rotated-policy".into()];
        let second = verified_head(
            &rotated,
            2,
            Some(tracked_first.statement_digest().into()),
            "two",
        );
        assert!(matches!(
            tracker.observe(&second),
            Err(SignedNarInfoHeadTrackingError::AuthorityPolicyChanged { .. })
        ));
    }

    #[test]
    fn sibling_at_same_sequence_is_collision() {
        let (_, policy) = authority();
        let first = verified_head(&policy, 1, None, "one");
        let mut tracker = SignedNarInfoClosureHeadTracker::default();
        tracker.observe(&first).unwrap();
        let sibling = verified_head(&policy, 1, None, "sibling");
        assert!(matches!(
            tracker.observe(&sibling),
            Err(SignedNarInfoHeadTrackingError::SequenceCollision { sequence: 1 })
        ));
    }

    #[test]
    fn challenge_bound_currentness_rejects_stale_tracked_head() {
        let (signing, policy) = authority();
        let first = verified_head(&policy, 1, None, "one");
        let mut tracker = SignedNarInfoClosureHeadTracker::default();
        let tracked_first = tracker.observe(&first).unwrap();
        let second = verified_head(
            &policy,
            2,
            Some(tracked_first.statement_digest().into()),
            "two",
        );
        tracker.observe(&second).unwrap();
        let challenge = nonce("request");
        let attestation = signed_currentness(&signing, &policy, &tracked_first, &challenge, 5_000);
        let report = verify_current_signed_narinfo_runtime_closure(
            &tracker,
            &tracked_first,
            &policy,
            &attestation,
            &challenge,
            5_000,
        )
        .unwrap_err();
        assert!(report
            .issues
            .contains(&SignedNarInfoCurrentnessIssue::TrackedHeadNotLatest));
    }

    #[test]
    fn currentness_rejects_use_before_head_issue_time() {
        let (signing, policy) = authority();
        let head = verified_head(&policy, 1, None, "one");
        let mut tracker = SignedNarInfoClosureHeadTracker::default();
        let tracked = tracker.observe(&head).unwrap();
        let challenge = nonce("chronology");
        let attestation = signed_currentness(&signing, &policy, &tracked, &challenge, 500);
        let report = verify_current_signed_narinfo_runtime_closure(
            &tracker,
            &tracked,
            &policy,
            &attestation,
            &challenge,
            500,
        )
        .unwrap_err();
        assert!(report
            .issues
            .contains(&SignedNarInfoCurrentnessIssue::HeadIssuedAfterUseTime));
    }

    #[test]
    fn zero_challenge_is_reserved_and_invalid() {
        let (signing, policy) = authority();
        let head = verified_head(&policy, 1, None, "one");
        let mut tracker = SignedNarInfoClosureHeadTracker::default();
        let tracked = tracker.observe(&head).unwrap();
        let mut attestation =
            signed_currentness(&signing, &policy, &tracked, &nonce("valid"), 5_000);
        attestation.challenge_nonce_blake3_hex = "0".repeat(64);
        assert!(!attestation.validate());
    }

    #[test]
    fn wrong_challenge_is_blocked_before_signature_promotion() {
        let (signing, policy) = authority();
        let head = verified_head(&policy, 1, None, "one");
        let mut tracker = SignedNarInfoClosureHeadTracker::default();
        let tracked = tracker.observe(&head).unwrap();
        let challenge = nonce("a");
        let attestation = signed_currentness(&signing, &policy, &tracked, &challenge, 5_000);
        let report = verify_current_signed_narinfo_runtime_closure(
            &tracker,
            &tracked,
            &policy,
            &attestation,
            &nonce("b"),
            5_000,
        )
        .unwrap_err();
        assert!(report
            .issues
            .contains(&SignedNarInfoCurrentnessIssue::ChallengeMismatch));
    }
}
