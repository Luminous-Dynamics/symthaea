// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scoped cryptographic provenance for verifier-independence evidence.
//!
//! Signature validity is deliberately separated from issuer trust, scope
//! authorization, substantive correctness, and verifier independence.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use symthaea_evidence_verifier_diversity::VerifierFaultDomainProfile;

pub const PROFILE_RECORD_SCHEMA_V1: &str = "symthaea.assurance.verifier-profile-record.v1";
pub const PROFILE_PROVENANCE_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.verifier-profile-provenance-policy.v1";
pub const PROFILE_ATTESTATION_SCHEMA_V1: &str =
    "symthaea.assurance.verifier-profile-attestation.v1";
pub const MAX_TEXT_BYTES: usize = 512;
pub const MAX_EVIDENCE_REFS: usize = 128;
pub const MAX_TRUSTED_KEYS: usize = 1_024;

const RAW_PROFILE_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.verifier-fault-domain-profile.digest.v1\0";
const PROFILE_RECORD_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.verifier-profile-record.digest.v1\0";
const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.verifier-profile-provenance-policy.digest.v1\0";
const ATTESTATION_MESSAGE_DOMAIN: &[u8] =
    b"symthaea.assurance.verifier-profile-attestation.message.v1\0";
const ATTESTATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.verifier-profile-attestation.digest.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ProfileClaimScope {
    ProfileIdentity,
    FaultDomainAssignment,
    GraphBinding,
    RelationCompletenessBinding,
}

impl ProfileClaimScope {
    fn code(self) -> &'static str {
        match self {
            Self::ProfileIdentity => "profile-identity",
            Self::FaultDomainAssignment => "fault-domain-assignment",
            Self::GraphBinding => "graph-binding",
            Self::RelationCompletenessBinding => "relation-completeness-binding",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionedVerifierProfileRecord {
    pub schema_version: String,
    pub profile_id: String,
    pub revision: u64,
    pub profile: VerifierFaultDomainProfile,
    pub effective_from_ms: u64,
    pub effective_until_ms: Option<u64>,
    pub graph_digest: String,
    pub relation_completeness_digest: String,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileIssuerKeyRecord {
    pub key_id: String,
    pub public_key_ed25519_hex: String,
    pub valid_from_ms: u64,
    pub valid_until_ms: Option<u64>,
    /// A revocation after an attestation was issued does not by itself define
    /// whether historical reliance remains acceptable. This field only answers
    /// whether the key was revoked at the attestation issue time; lifecycle is
    /// a separate follow-on theorem.
    pub revoked_at_ms: Option<u64>,
    pub allowed_scopes: Vec<ProfileClaimScope>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileProvenancePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub sequence: u64,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub required_scopes: Vec<ProfileClaimScope>,
    pub trusted_keys: Vec<ProfileIssuerKeyRecord>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileAttestationEnvelope {
    pub schema_version: String,
    pub record_digest: String,
    pub policy_digest: String,
    pub issuer_key_id: String,
    pub issuer_public_key_ed25519_hex: String,
    pub issued_at_ms: u64,
    pub expires_at_ms: Option<u64>,
    pub scopes: Vec<ProfileClaimScope>,
    pub nonce_blake3_hex: String,
    pub signature_ed25519_hex: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProvenanceDisposition {
    Accepted,
    SignatureValidIssuerUntrusted,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProvenanceIssue {
    InvalidRecord,
    InvalidPolicy,
    InvalidAttestation,
    RecordDigestMismatch,
    PolicyDigestMismatch,
    InvalidIssuerPublicKey,
    SignatureInvalid,
    IssuerKeyUnknown,
    IssuerPublicKeyMismatch,
    PolicyNotEffectiveAtIssueTime,
    IssuerKeyNotEffectiveAtIssueTime,
    IssuerKeyRevokedAtIssueTime,
    RequiredScopeMissing(ProfileClaimScope),
    ScopeNotAuthorized(ProfileClaimScope),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileProvenanceReport {
    pub disposition: ProvenanceDisposition,
    pub profile_id: String,
    pub profile_revision: u64,
    pub raw_profile_digest: Option<String>,
    pub record_digest: Option<String>,
    pub graph_digest: String,
    pub relation_completeness_digest: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub attestation_digest: Option<String>,
    pub issuer_key_id: String,
    pub signature_valid: bool,
    pub issuer_trusted_for_claim: bool,
    pub scope_authorized: bool,
    pub issues: Vec<ProvenanceIssue>,
}

impl ProfileProvenanceReport {
    pub const fn substantive_correctness_established(&self) -> bool {
        false
    }

    pub const fn verifier_independence_established(&self) -> bool {
        false
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthenticatedProfileProvenance {
    record_digest: String,
    raw_profile_digest: String,
    graph_digest: String,
    relation_completeness_digest: String,
    policy_digest: String,
    attestation_digest: String,
    issuer_key_id: String,
    effective_from_ms: u64,
    effective_until_ms: Option<u64>,
    attestation_issued_at_ms: u64,
    attestation_expires_at_ms: Option<u64>,
}

impl AuthenticatedProfileProvenance {
    pub fn record_digest(&self) -> &str {
        &self.record_digest
    }

    pub fn raw_profile_digest(&self) -> &str {
        &self.raw_profile_digest
    }

    pub fn graph_digest(&self) -> &str {
        &self.graph_digest
    }

    pub fn relation_completeness_digest(&self) -> &str {
        &self.relation_completeness_digest
    }

    pub fn policy_digest(&self) -> &str {
        &self.policy_digest
    }

    pub fn attestation_digest(&self) -> &str {
        &self.attestation_digest
    }

    pub fn issuer_key_id(&self) -> &str {
        &self.issuer_key_id
    }

    pub fn record_applies_at(&self, at_ms: u64) -> bool {
        at_ms >= self.effective_from_ms
            && self
                .effective_until_ms
                .map(|until| at_ms < until)
                .unwrap_or(true)
    }

    /// Strong anti-post-hoc predicate for a downstream verifier event. The
    /// record must apply at the event and the trusted attestation must already
    /// exist and not yet have expired at that event.
    pub fn prospectively_established_at(&self, at_ms: u64) -> bool {
        self.record_applies_at(at_ms)
            && self.attestation_issued_at_ms <= at_ms
            && self
                .attestation_expires_at_ms
                .map(|expires| at_ms < expires)
                .unwrap_or(true)
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileProvenanceVerification {
    pub report: ProfileProvenanceReport,
    authenticated: Option<AuthenticatedProfileProvenance>,
}

impl ProfileProvenanceVerification {
    pub fn authenticated(&self) -> Option<&AuthenticatedProfileProvenance> {
        self.authenticated.as_ref()
    }

    pub fn into_authenticated(self) -> Option<AuthenticatedProfileProvenance> {
        self.authenticated
    }
}

impl VersionedVerifierProfileRecord {
    pub fn validate(&self) -> bool {
        self.schema_version == PROFILE_RECORD_SCHEMA_V1
            && canonical_text(&self.profile_id)
            && self.revision > 0
            && canonical_profile(&self.profile)
            && self
                .effective_until_ms
                .map(|until| until > self.effective_from_ms)
                .unwrap_or(true)
            && digest_text(&self.graph_digest)
            && digest_text(&self.relation_completeness_digest)
            && valid_refs(&self.evidence_refs)
    }

    pub fn raw_profile_digest(&self) -> Option<String> {
        if !canonical_profile(&self.profile) {
            return None;
        }
        Some(digest_raw_profile(&self.profile))
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let raw_profile_digest = self.raw_profile_digest()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(PROFILE_RECORD_DIGEST_DOMAIN);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.profile_id);
        push_u64(&mut hasher, self.revision);
        push_field(&mut hasher, &raw_profile_digest);
        push_u64(&mut hasher, self.effective_from_ms);
        push_optional_u64(&mut hasher, self.effective_until_ms);
        push_field(&mut hasher, &self.graph_digest);
        push_field(&mut hasher, &self.relation_completeness_digest);
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

impl ProfileIssuerKeyRecord {
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

    fn usable_at_issue_time(&self, issued_at_ms: u64) -> bool {
        issued_at_ms >= self.valid_from_ms
            && self
                .valid_until_ms
                .map(|until| issued_at_ms < until)
                .unwrap_or(true)
    }

    fn revoked_at_issue_time(&self, issued_at_ms: u64) -> bool {
        self.revoked_at_ms
            .map(|revoked| issued_at_ms >= revoked)
            .unwrap_or(false)
    }
}

impl ProfileProvenancePolicy {
    pub fn validate(&self) -> bool {
        if self.schema_version != PROFILE_PROVENANCE_POLICY_SCHEMA_V1
            || !canonical_text(&self.policy_id)
            || self.sequence == 0
            || self.issued_at_ms >= self.expires_at_ms
            || self.required_scopes.is_empty()
            || !unique(&self.required_scopes)
            || self.trusted_keys.is_empty()
            || self.trusted_keys.len() > MAX_TRUSTED_KEYS
            || !valid_refs(&self.evidence_refs)
        {
            return false;
        }
        let mut key_ids = BTreeSet::new();
        self.trusted_keys.iter().all(|key| {
            key.validate() && key_ids.insert(key.key_id.as_str())
        })
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

        let mut scopes = self.required_scopes.clone();
        scopes.sort();
        for scope in scopes {
            push_field(&mut hasher, scope.code());
        }

        let mut keys = self.trusted_keys.iter().collect::<Vec<_>>();
        keys.sort_by(|left, right| left.key_id.cmp(&right.key_id));
        for key in keys {
            push_field(&mut hasher, &key.key_id);
            push_field(&mut hasher, &key.public_key_ed25519_hex);
            push_u64(&mut hasher, key.valid_from_ms);
            push_optional_u64(&mut hasher, key.valid_until_ms);
            push_optional_u64(&mut hasher, key.revoked_at_ms);
            let mut allowed = key.allowed_scopes.clone();
            allowed.sort();
            for scope in allowed {
                push_field(&mut hasher, scope.code());
            }
            push_sorted_refs(&mut hasher, &key.evidence_refs);
        }
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

impl ProfileAttestationEnvelope {
    pub fn validate(&self) -> bool {
        self.schema_version == PROFILE_ATTESTATION_SCHEMA_V1
            && digest_text(&self.record_digest)
            && digest_text(&self.policy_digest)
            && canonical_text(&self.issuer_key_id)
            && lower_hex_exact(&self.issuer_public_key_ed25519_hex, 64)
            && self
                .expires_at_ms
                .map(|expires| expires > self.issued_at_ms)
                .unwrap_or(true)
            && !self.scopes.is_empty()
            && unique(&self.scopes)
            && lower_hex_exact(&self.nonce_blake3_hex, 64)
            && lower_hex_exact(&self.signature_ed25519_hex, 128)
    }

    pub fn canonical_unsigned_bytes(&self) -> Option<Vec<u8>> {
        if !self.validate() {
            return None;
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(ATTESTATION_MESSAGE_DOMAIN);
        push_vec_field(&mut bytes, &self.schema_version);
        push_vec_field(&mut bytes, &self.record_digest);
        push_vec_field(&mut bytes, &self.policy_digest);
        push_vec_field(&mut bytes, &self.issuer_key_id);
        push_vec_field(&mut bytes, &self.issuer_public_key_ed25519_hex);
        bytes.extend_from_slice(&self.issued_at_ms.to_be_bytes());
        push_vec_optional_u64(&mut bytes, self.expires_at_ms);
        let mut scopes = self.scopes.clone();
        scopes.sort();
        for scope in scopes {
            push_vec_field(&mut bytes, scope.code());
        }
        push_vec_field(&mut bytes, &self.nonce_blake3_hex);
        Some(bytes)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        let unsigned = self.canonical_unsigned_bytes()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(ATTESTATION_DIGEST_DOMAIN);
        hasher.update(&(unsigned.len() as u64).to_be_bytes());
        hasher.update(&unsigned);
        push_field(&mut hasher, &self.signature_ed25519_hex);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

pub fn verify_profile_provenance(
    record: &VersionedVerifierProfileRecord,
    policy: &ProfileProvenancePolicy,
    attestation: &ProfileAttestationEnvelope,
) -> ProfileProvenanceVerification {
    let raw_profile_digest = record.raw_profile_digest();
    let record_digest = record.canonical_digest();
    let policy_digest = policy.canonical_digest();
    let attestation_digest = attestation.canonical_digest();
    let mut issues = Vec::new();

    if !record.validate() {
        issues.push(ProvenanceIssue::InvalidRecord);
    }
    if !policy.validate() {
        issues.push(ProvenanceIssue::InvalidPolicy);
    }
    if !attestation.validate() {
        issues.push(ProvenanceIssue::InvalidAttestation);
    }
    if record_digest.as_deref() != Some(attestation.record_digest.as_str()) {
        issues.push(ProvenanceIssue::RecordDigestMismatch);
    }
    if policy_digest.as_deref() != Some(attestation.policy_digest.as_str()) {
        issues.push(ProvenanceIssue::PolicyDigestMismatch);
    }

    if !issues.is_empty() {
        return rejected(
            record,
            policy,
            attestation,
            raw_profile_digest,
            record_digest,
            policy_digest,
            attestation_digest,
            false,
            false,
            false,
            issues,
        );
    }

    let unsigned = attestation.canonical_unsigned_bytes().expect("validated attestation");
    let signature_valid = match verify_signature(attestation, &unsigned) {
        Ok(valid) => valid,
        Err(()) => {
            issues.push(ProvenanceIssue::InvalidIssuerPublicKey);
            false
        }
    };
    if !signature_valid {
        if !issues.contains(&ProvenanceIssue::InvalidIssuerPublicKey) {
            issues.push(ProvenanceIssue::SignatureInvalid);
        }
        return rejected(
            record,
            policy,
            attestation,
            raw_profile_digest,
            record_digest,
            policy_digest,
            attestation_digest,
            false,
            false,
            false,
            issues,
        );
    }

    let Some(key) = policy
        .trusted_keys
        .iter()
        .find(|key| key.key_id == attestation.issuer_key_id)
    else {
        issues.push(ProvenanceIssue::IssuerKeyUnknown);
        return untrusted(
            record,
            policy,
            attestation,
            raw_profile_digest,
            record_digest,
            policy_digest,
            attestation_digest,
            issues,
        );
    };

    if key.public_key_ed25519_hex != attestation.issuer_public_key_ed25519_hex {
        issues.push(ProvenanceIssue::IssuerPublicKeyMismatch);
    }
    if !(policy.issued_at_ms <= attestation.issued_at_ms
        && attestation.issued_at_ms < policy.expires_at_ms)
    {
        issues.push(ProvenanceIssue::PolicyNotEffectiveAtIssueTime);
    }
    if !key.usable_at_issue_time(attestation.issued_at_ms) {
        issues.push(ProvenanceIssue::IssuerKeyNotEffectiveAtIssueTime);
    }
    if key.revoked_at_issue_time(attestation.issued_at_ms) {
        issues.push(ProvenanceIssue::IssuerKeyRevokedAtIssueTime);
    }

    let claimed = attestation.scopes.iter().copied().collect::<BTreeSet<_>>();
    let allowed = key.allowed_scopes.iter().copied().collect::<BTreeSet<_>>();
    for required in &policy.required_scopes {
        if !claimed.contains(required) {
            issues.push(ProvenanceIssue::RequiredScopeMissing(*required));
        }
    }
    for scope in &attestation.scopes {
        if !allowed.contains(scope) {
            issues.push(ProvenanceIssue::ScopeNotAuthorized(*scope));
        }
    }

    if !issues.is_empty() {
        return untrusted(
            record,
            policy,
            attestation,
            raw_profile_digest,
            record_digest,
            policy_digest,
            attestation_digest,
            issues,
        );
    }

    let raw_profile_digest = raw_profile_digest.expect("validated record");
    let record_digest = record_digest.expect("validated record");
    let policy_digest = policy_digest.expect("validated policy");
    let attestation_digest = attestation_digest.expect("validated attestation");
    let authenticated = AuthenticatedProfileProvenance {
        record_digest: record_digest.clone(),
        raw_profile_digest: raw_profile_digest.clone(),
        graph_digest: record.graph_digest.clone(),
        relation_completeness_digest: record.relation_completeness_digest.clone(),
        policy_digest: policy_digest.clone(),
        attestation_digest: attestation_digest.clone(),
        issuer_key_id: attestation.issuer_key_id.clone(),
        effective_from_ms: record.effective_from_ms,
        effective_until_ms: record.effective_until_ms,
        attestation_issued_at_ms: attestation.issued_at_ms,
        attestation_expires_at_ms: attestation.expires_at_ms,
    };
    ProfileProvenanceVerification {
        report: report(
            record,
            policy,
            attestation,
            Some(raw_profile_digest),
            Some(record_digest),
            Some(policy_digest),
            Some(attestation_digest),
            ProvenanceDisposition::Accepted,
            true,
            true,
            true,
            Vec::new(),
        ),
        authenticated: Some(authenticated),
    }
}

fn untrusted(
    record: &VersionedVerifierProfileRecord,
    policy: &ProfileProvenancePolicy,
    attestation: &ProfileAttestationEnvelope,
    raw_profile_digest: Option<String>,
    record_digest: Option<String>,
    policy_digest: Option<String>,
    attestation_digest: Option<String>,
    issues: Vec<ProvenanceIssue>,
) -> ProfileProvenanceVerification {
    let scope_authorized = !issues.iter().any(|issue| {
        matches!(
            issue,
            ProvenanceIssue::RequiredScopeMissing(_) | ProvenanceIssue::ScopeNotAuthorized(_)
        )
    });
    ProfileProvenanceVerification {
        report: report(
            record,
            policy,
            attestation,
            raw_profile_digest,
            record_digest,
            policy_digest,
            attestation_digest,
            ProvenanceDisposition::SignatureValidIssuerUntrusted,
            true,
            false,
            scope_authorized,
            issues,
        ),
        authenticated: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn rejected(
    record: &VersionedVerifierProfileRecord,
    policy: &ProfileProvenancePolicy,
    attestation: &ProfileAttestationEnvelope,
    raw_profile_digest: Option<String>,
    record_digest: Option<String>,
    policy_digest: Option<String>,
    attestation_digest: Option<String>,
    signature_valid: bool,
    issuer_trusted_for_claim: bool,
    scope_authorized: bool,
    issues: Vec<ProvenanceIssue>,
) -> ProfileProvenanceVerification {
    ProfileProvenanceVerification {
        report: report(
            record,
            policy,
            attestation,
            raw_profile_digest,
            record_digest,
            policy_digest,
            attestation_digest,
            ProvenanceDisposition::Rejected,
            signature_valid,
            issuer_trusted_for_claim,
            scope_authorized,
            issues,
        ),
        authenticated: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn report(
    record: &VersionedVerifierProfileRecord,
    policy: &ProfileProvenancePolicy,
    attestation: &ProfileAttestationEnvelope,
    raw_profile_digest: Option<String>,
    record_digest: Option<String>,
    policy_digest: Option<String>,
    attestation_digest: Option<String>,
    disposition: ProvenanceDisposition,
    signature_valid: bool,
    issuer_trusted_for_claim: bool,
    scope_authorized: bool,
    issues: Vec<ProvenanceIssue>,
) -> ProfileProvenanceReport {
    ProfileProvenanceReport {
        disposition,
        profile_id: record.profile_id.clone(),
        profile_revision: record.revision,
        raw_profile_digest,
        record_digest,
        graph_digest: record.graph_digest.clone(),
        relation_completeness_digest: record.relation_completeness_digest.clone(),
        policy_id: policy.policy_id.clone(),
        policy_digest,
        attestation_digest,
        issuer_key_id: attestation.issuer_key_id.clone(),
        signature_valid,
        issuer_trusted_for_claim,
        scope_authorized,
        issues,
    }
}

fn verify_signature(attestation: &ProfileAttestationEnvelope, message: &[u8]) -> Result<bool, ()> {
    let public = hex::decode(&attestation.issuer_public_key_ed25519_hex).map_err(|_| ())?;
    let signature = hex::decode(&attestation.signature_ed25519_hex).map_err(|_| ())?;
    let public: [u8; 32] = public.try_into().map_err(|_| ())?;
    let signature: [u8; 64] = signature.try_into().map_err(|_| ())?;
    let key = VerifyingKey::from_bytes(&public).map_err(|_| ())?;
    let signature = Signature::from_bytes(&signature);
    Ok(key.verify(message, &signature).is_ok())
}

fn canonical_profile(profile: &VerifierFaultDomainProfile) -> bool {
    profile.validate()
        && canonical_text(&profile.verifier_ref)
        && canonical_text(&profile.organization_domain)
        && canonical_text(&profile.review_process_domain)
        && canonical_text(&profile.toolchain_domain)
        && canonical_text(&profile.evidence_source_domain)
        && valid_refs(&profile.evidence_refs)
}

fn digest_raw_profile(profile: &VerifierFaultDomainProfile) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RAW_PROFILE_DIGEST_DOMAIN);
    push_field(&mut hasher, &profile.verifier_ref);
    push_field(&mut hasher, &profile.organization_domain);
    push_field(&mut hasher, &profile.review_process_domain);
    push_field(&mut hasher, &profile.toolchain_domain);
    push_field(&mut hasher, &profile.evidence_source_domain);
    push_sorted_refs(&mut hasher, &profile.evidence_refs);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn canonical_text(value: &str) -> bool {
    let trimmed = value.trim();
    !trimmed.is_empty() && trimmed == value && value.len() <= MAX_TEXT_BYTES
}

fn digest_text(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("blake3:") else {
        return false;
    };
    lower_hex_exact(hex, 64)
}

fn lower_hex_exact(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_refs(refs: &[String]) -> bool {
    !refs.is_empty()
        && refs.len() <= MAX_EVIDENCE_REFS
        && refs.iter().all(|value| canonical_text(value))
        && refs.iter().collect::<BTreeSet<_>>().len() == refs.len()
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
        None => hasher.update(&[0]),
    }
}

fn push_sorted_refs(hasher: &mut blake3::Hasher, refs: &[String]) {
    let mut refs = refs.iter().collect::<Vec<_>>();
    refs.sort();
    for value in refs {
        push_field(hasher, value);
    }
}

fn push_vec_field(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn push_vec_optional_u64(bytes: &mut Vec<u8>, value: Option<u64>) {
    match value {
        Some(value) => {
            bytes.push(1);
            bytes.extend_from_slice(&value.to_be_bytes());
        }
        None => bytes.push(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn profile() -> VerifierFaultDomainProfile {
        VerifierFaultDomainProfile {
            verifier_ref: "verifier:a".into(),
            organization_domain: "org:a".into(),
            review_process_domain: "process:a".into(),
            toolchain_domain: "tool:a".into(),
            evidence_source_domain: "source:a".into(),
            evidence_refs: vec!["evidence:profile:a".into()],
        }
    }

    fn digest(byte: char) -> String {
        format!("blake3:{}", byte.to_string().repeat(64))
    }

    fn record() -> VersionedVerifierProfileRecord {
        VersionedVerifierProfileRecord {
            schema_version: PROFILE_RECORD_SCHEMA_V1.into(),
            profile_id: "profile:a".into(),
            revision: 7,
            profile: profile(),
            effective_from_ms: 1_000,
            effective_until_ms: Some(10_000),
            graph_digest: digest('a'),
            relation_completeness_digest: digest('b'),
            evidence_refs: vec!["review:profile:a:r7".into()],
        }
    }

    fn policy(signing_key: &SigningKey) -> ProfileProvenancePolicy {
        ProfileProvenancePolicy {
            schema_version: PROFILE_PROVENANCE_POLICY_SCHEMA_V1.into(),
            policy_id: "profile-trust:v1".into(),
            sequence: 3,
            issued_at_ms: 500,
            expires_at_ms: 20_000,
            required_scopes: vec![
                ProfileClaimScope::ProfileIdentity,
                ProfileClaimScope::FaultDomainAssignment,
                ProfileClaimScope::GraphBinding,
                ProfileClaimScope::RelationCompletenessBinding,
            ],
            trusted_keys: vec![ProfileIssuerKeyRecord {
                key_id: "issuer:a".into(),
                public_key_ed25519_hex: hex::encode(signing_key.verifying_key().as_bytes()),
                valid_from_ms: 400,
                valid_until_ms: Some(15_000),
                revoked_at_ms: None,
                allowed_scopes: vec![
                    ProfileClaimScope::ProfileIdentity,
                    ProfileClaimScope::FaultDomainAssignment,
                    ProfileClaimScope::GraphBinding,
                    ProfileClaimScope::RelationCompletenessBinding,
                ],
                evidence_refs: vec!["trust:key:issuer:a".into()],
            }],
            evidence_refs: vec!["review:profile-trust:v1".into()],
        }
    }

    fn signed_attestation(
        record: &VersionedVerifierProfileRecord,
        policy: &ProfileProvenancePolicy,
        signing_key: &SigningKey,
        issued_at_ms: u64,
    ) -> ProfileAttestationEnvelope {
        let mut envelope = ProfileAttestationEnvelope {
            schema_version: PROFILE_ATTESTATION_SCHEMA_V1.into(),
            record_digest: record.canonical_digest().unwrap(),
            policy_digest: policy.canonical_digest().unwrap(),
            issuer_key_id: "issuer:a".into(),
            issuer_public_key_ed25519_hex: hex::encode(signing_key.verifying_key().as_bytes()),
            issued_at_ms,
            expires_at_ms: Some(12_000),
            scopes: policy.required_scopes.clone(),
            nonce_blake3_hex: "c".repeat(64),
            signature_ed25519_hex: "0".repeat(128),
        };
        let message = envelope.canonical_unsigned_bytes().unwrap();
        envelope.signature_ed25519_hex = hex::encode(signing_key.sign(&message).to_bytes());
        envelope
    }

    #[test]
    fn trusted_scoped_signature_mints_opaque_authenticated_provenance() {
        let signing_key = SigningKey::from_bytes(&[7u8; 32]);
        let record = record();
        let policy = policy(&signing_key);
        let attestation = signed_attestation(&record, &policy, &signing_key, 900);
        let verification = verify_profile_provenance(&record, &policy, &attestation);
        assert_eq!(verification.report.disposition, ProvenanceDisposition::Accepted);
        assert!(verification.report.signature_valid);
        assert!(verification.report.issuer_trusted_for_claim);
        let authenticated = verification.authenticated().unwrap();
        assert!(authenticated.prospectively_established_at(1_500));
        assert!(!authenticated.grants_physical_authority());
        assert!(!verification.report.substantive_correctness_established());
        assert!(!verification.report.verifier_independence_established());
    }

    #[test]
    fn valid_signature_from_unknown_issuer_is_not_trust() {
        let signing_key = SigningKey::from_bytes(&[7u8; 32]);
        let unknown_key = SigningKey::from_bytes(&[8u8; 32]);
        let record = record();
        let policy = policy(&signing_key);
        let mut attestation = signed_attestation(&record, &policy, &unknown_key, 900);
        attestation.issuer_key_id = "issuer:unknown".into();
        let message = attestation.canonical_unsigned_bytes().unwrap();
        attestation.signature_ed25519_hex = hex::encode(unknown_key.sign(&message).to_bytes());
        let verification = verify_profile_provenance(&record, &policy, &attestation);
        assert_eq!(
            verification.report.disposition,
            ProvenanceDisposition::SignatureValidIssuerUntrusted
        );
        assert!(verification.report.signature_valid);
        assert!(verification.authenticated().is_none());
    }

    #[test]
    fn unauthorized_claim_scope_is_not_trust() {
        let signing_key = SigningKey::from_bytes(&[7u8; 32]);
        let record = record();
        let mut policy = policy(&signing_key);
        policy.trusted_keys[0]
            .allowed_scopes
            .retain(|scope| *scope != ProfileClaimScope::RelationCompletenessBinding);
        let attestation = signed_attestation(&record, &policy, &signing_key, 900);
        let verification = verify_profile_provenance(&record, &policy, &attestation);
        assert_eq!(
            verification.report.disposition,
            ProvenanceDisposition::SignatureValidIssuerUntrusted
        );
        assert!(!verification.report.scope_authorized);
    }

    #[test]
    fn record_rebinding_is_rejected_even_with_otherwise_valid_signature() {
        let signing_key = SigningKey::from_bytes(&[7u8; 32]);
        let record = record();
        let policy = policy(&signing_key);
        let mut attestation = signed_attestation(&record, &policy, &signing_key, 900);
        attestation.record_digest = digest('d');
        let message = attestation.canonical_unsigned_bytes().unwrap();
        attestation.signature_ed25519_hex = hex::encode(signing_key.sign(&message).to_bytes());
        let verification = verify_profile_provenance(&record, &policy, &attestation);
        assert_eq!(verification.report.disposition, ProvenanceDisposition::Rejected);
        assert!(verification
            .report
            .issues
            .contains(&ProvenanceIssue::RecordDigestMismatch));
    }

    #[test]
    fn policy_created_after_attestation_is_untrusted_not_authenticity_failure() {
        let signing_key = SigningKey::from_bytes(&[7u8; 32]);
        let record = record();
        let mut policy = policy(&signing_key);
        policy.issued_at_ms = 1_100;
        let attestation = signed_attestation(&record, &policy, &signing_key, 900);
        let verification = verify_profile_provenance(&record, &policy, &attestation);
        assert_eq!(
            verification.report.disposition,
            ProvenanceDisposition::SignatureValidIssuerUntrusted
        );
        assert!(verification.report.signature_valid);
        assert!(verification
            .report
            .issues
            .contains(&ProvenanceIssue::PolicyNotEffectiveAtIssueTime));
    }

    #[test]
    fn key_revoked_before_issue_is_untrusted() {
        let signing_key = SigningKey::from_bytes(&[7u8; 32]);
        let record = record();
        let mut policy = policy(&signing_key);
        policy.trusted_keys[0].revoked_at_ms = Some(800);
        let attestation = signed_attestation(&record, &policy, &signing_key, 900);
        let verification = verify_profile_provenance(&record, &policy, &attestation);
        assert_eq!(
            verification.report.disposition,
            ProvenanceDisposition::SignatureValidIssuerUntrusted
        );
        assert!(verification
            .report
            .issues
            .contains(&ProvenanceIssue::IssuerKeyRevokedAtIssueTime));
    }

    #[test]
    fn graph_or_completeness_change_changes_record_identity() {
        let mut left = record();
        let mut right = left.clone();
        right.graph_digest = digest('d');
        assert_ne!(left.canonical_digest(), right.canonical_digest());
        left.graph_digest = right.graph_digest.clone();
        right.relation_completeness_digest = digest('e');
        assert_ne!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn post_hoc_attestation_does_not_establish_earlier_verification() {
        let signing_key = SigningKey::from_bytes(&[7u8; 32]);
        let record = record();
        let policy = policy(&signing_key);
        let attestation = signed_attestation(&record, &policy, &signing_key, 2_000);
        let authenticated = verify_profile_provenance(&record, &policy, &attestation)
            .into_authenticated()
            .unwrap();
        assert!(authenticated.record_applies_at(1_500));
        assert!(!authenticated.prospectively_established_at(1_500));
        assert!(authenticated.prospectively_established_at(2_500));
    }

    #[test]
    fn policy_digest_is_key_order_independent() {
        let key_a = SigningKey::from_bytes(&[7u8; 32]);
        let key_b = SigningKey::from_bytes(&[8u8; 32]);
        let mut left = policy(&key_a);
        let mut second = left.trusted_keys[0].clone();
        second.key_id = "issuer:b".into();
        second.public_key_ed25519_hex = hex::encode(key_b.verifying_key().as_bytes());
        left.trusted_keys.push(second.clone());
        let mut right = left.clone();
        right.trusted_keys.reverse();
        assert_eq!(left.canonical_digest(), right.canonical_digest());
    }
}
