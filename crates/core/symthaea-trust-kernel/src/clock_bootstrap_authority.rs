// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! External-authority admission for the first trusted clock bootstrap interval.
//!
//! V2 binds the exact initial clock-evaluation-policy identity into the claim
//! authenticated by the external provider. Serializable claim/evidence records
//! remain audit material only; the verified capability is private and one-way.
//! No candidate clock window or candidate time participates in this boundary.

use crate::digest::{Sha256Digest, domain_hash};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub const CLOCK_BOOTSTRAP_CLAIM_SCHEMA: &str = "symthaea.trust.clock-bootstrap-claim.v2";
pub const CLOCK_BOOTSTRAP_AUTHORITY_EVIDENCE_SCHEMA: &str =
    "symthaea.trust.clock-bootstrap-authority-evidence.v2";
pub const VERIFIED_CLOCK_BOOTSTRAP_AUTHORITY_SCHEMA: &str =
    "symthaea.trust.verified-clock-bootstrap-authority.v2";

const CLOCK_BOOTSTRAP_CLAIM_DOMAIN: &[u8] = b"symthaea.trust.clock-bootstrap-claim.v2\0";
const CLOCK_BOOTSTRAP_AUTHORITY_EVIDENCE_DOMAIN: &[u8] =
    b"symthaea.trust.clock-bootstrap-authority-evidence.v2\0";
const VERIFIED_CLOCK_BOOTSTRAP_AUTHORITY_DOMAIN: &[u8] =
    b"symthaea.trust.verified-clock-bootstrap-authority.v2\0";
const CLOCK_BOOTSTRAP_PURPOSE: &str = "ClockBootstrap";
const MAX_BOOTSTRAP_PROVIDER_ID_BYTES: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockBootstrapClaimV2 {
    schema: String,
    purpose: String,
    trust_snapshot_digest: Sha256Digest,
    clock_evaluation_policy_id: Sha256Digest,
    trusted_lower_unix_ms: u64,
    trusted_upper_unix_ms: u64,
    #[serde(rename = "id")]
    id: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClockBootstrapAuthorityEvidenceV2 {
    schema: String,
    claim_id: Sha256Digest,
    provider_id: String,
    authority_policy_digest: Sha256Digest,
    external_evidence_digest: Sha256Digest,
    #[serde(rename = "id")]
    id: Sha256Digest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VerifiedClockBootstrapAuthorityIdV2(Sha256Digest);

#[derive(Debug, Clone)]
#[must_use]
pub struct VerifiedClockBootstrapAuthorityV2 {
    id: VerifiedClockBootstrapAuthorityIdV2,
    claim_id: Sha256Digest,
    authority_evidence_id: Sha256Digest,
    provider_id: String,
    authority_policy_digest: Sha256Digest,
    clock_evaluation_policy_id: Sha256Digest,
}

pub trait ClockBootstrapAuthorityVerifier {
    fn provider_id(&self) -> &str;
    fn authority_policy_digest(&self) -> Sha256Digest;
    fn verify_clock_bootstrap_authority(
        &self,
        canonical_claim_bytes: &[u8],
        external_evidence_digest: Sha256Digest,
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClockBootstrapAuthorityError {
    InvalidClaimWindow,
    InvalidProviderId,
    ClaimIdentityMismatch,
    EvidenceIdentityMismatch,
    EvidenceClaimMismatch,
    ProviderMismatch,
    AuthorityPolicyMismatch,
    ExternalAuthorityRejected,
    ExternalVerifierError(String),
    Encoding(String),
}

impl ClockBootstrapClaimV2 {
    pub fn new(
        trust_snapshot_digest: Sha256Digest,
        clock_evaluation_policy_id: Sha256Digest,
        trusted_lower_unix_ms: u64,
        trusted_upper_unix_ms: u64,
    ) -> Result<Self, ClockBootstrapAuthorityError> {
        if trusted_lower_unix_ms > trusted_upper_unix_ms {
            return Err(ClockBootstrapAuthorityError::InvalidClaimWindow);
        }
        let mut value = Self {
            schema: CLOCK_BOOTSTRAP_CLAIM_SCHEMA.to_string(),
            purpose: CLOCK_BOOTSTRAP_PURPOSE.to_string(),
            trust_snapshot_digest,
            clock_evaluation_policy_id,
            trusted_lower_unix_ms,
            trusted_upper_unix_ms,
            id: Sha256Digest([0; 32]),
        };
        value.id = compute_claim_id(&value)?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), ClockBootstrapAuthorityError> {
        if self.schema != CLOCK_BOOTSTRAP_CLAIM_SCHEMA
            || self.purpose != CLOCK_BOOTSTRAP_PURPOSE
            || self.trusted_lower_unix_ms > self.trusted_upper_unix_ms
        {
            return Err(ClockBootstrapAuthorityError::InvalidClaimWindow);
        }
        if compute_claim_id(self)? != self.id {
            return Err(ClockBootstrapAuthorityError::ClaimIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> Sha256Digest { self.id }
    pub fn trust_snapshot_digest(&self) -> Sha256Digest { self.trust_snapshot_digest }
    pub fn clock_evaluation_policy_id(&self) -> Sha256Digest { self.clock_evaluation_policy_id }
    pub fn trusted_lower_unix_ms(&self) -> u64 { self.trusted_lower_unix_ms }
    pub fn trusted_upper_unix_ms(&self) -> u64 { self.trusted_upper_unix_ms }
}

impl ClockBootstrapAuthorityEvidenceV2 {
    pub fn new(
        claim: &ClockBootstrapClaimV2,
        provider_id: impl Into<String>,
        authority_policy_digest: Sha256Digest,
        external_evidence_digest: Sha256Digest,
    ) -> Result<Self, ClockBootstrapAuthorityError> {
        claim.validate()?;
        let provider_id = provider_id.into();
        validate_provider_id(&provider_id)?;
        let mut value = Self {
            schema: CLOCK_BOOTSTRAP_AUTHORITY_EVIDENCE_SCHEMA.to_string(),
            claim_id: claim.id(),
            provider_id,
            authority_policy_digest,
            external_evidence_digest,
            id: Sha256Digest([0; 32]),
        };
        value.id = compute_evidence_id(&value)?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), ClockBootstrapAuthorityError> {
        if self.schema != CLOCK_BOOTSTRAP_AUTHORITY_EVIDENCE_SCHEMA {
            return Err(ClockBootstrapAuthorityError::EvidenceIdentityMismatch);
        }
        validate_provider_id(&self.provider_id)?;
        if compute_evidence_id(self)? != self.id {
            return Err(ClockBootstrapAuthorityError::EvidenceIdentityMismatch);
        }
        Ok(())
    }

    pub fn id(&self) -> Sha256Digest { self.id }
    pub fn claim_id(&self) -> Sha256Digest { self.claim_id }
    pub fn provider_id(&self) -> &str { &self.provider_id }
    pub fn authority_policy_digest(&self) -> Sha256Digest { self.authority_policy_digest }
    pub fn external_evidence_digest(&self) -> Sha256Digest { self.external_evidence_digest }
}

impl VerifiedClockBootstrapAuthorityIdV2 {
    pub fn as_digest(self) -> Sha256Digest { self.0 }
    pub fn to_hex(self) -> String { self.0.to_hex() }
}

impl VerifiedClockBootstrapAuthorityV2 {
    pub fn id(&self) -> VerifiedClockBootstrapAuthorityIdV2 { self.id }
    pub fn claim_id(&self) -> Sha256Digest { self.claim_id }
    pub fn authority_evidence_id(&self) -> Sha256Digest { self.authority_evidence_id }
    pub fn provider_id(&self) -> &str { &self.provider_id }
    pub fn authority_policy_digest(&self) -> Sha256Digest { self.authority_policy_digest }
    pub fn clock_evaluation_policy_id(&self) -> Sha256Digest { self.clock_evaluation_policy_id }
}

pub fn verify_clock_bootstrap_authority(
    claim: &ClockBootstrapClaimV2,
    evidence: &ClockBootstrapAuthorityEvidenceV2,
    verifier: &dyn ClockBootstrapAuthorityVerifier,
) -> Result<VerifiedClockBootstrapAuthorityV2, ClockBootstrapAuthorityError> {
    claim.validate()?;
    evidence.validate()?;
    if evidence.claim_id() != claim.id() {
        return Err(ClockBootstrapAuthorityError::EvidenceClaimMismatch);
    }

    let configured_provider = verifier.provider_id();
    validate_provider_id(configured_provider)?;
    if evidence.provider_id() != configured_provider {
        return Err(ClockBootstrapAuthorityError::ProviderMismatch);
    }
    let configured_policy = verifier.authority_policy_digest();
    if evidence.authority_policy_digest() != configured_policy {
        return Err(ClockBootstrapAuthorityError::AuthorityPolicyMismatch);
    }

    let canonical_claim_bytes = canonical_claim_preimage_bytes(claim)?;
    match verifier.verify_clock_bootstrap_authority(
        &canonical_claim_bytes,
        evidence.external_evidence_digest(),
    ) {
        Ok(true) => {}
        Ok(false) => return Err(ClockBootstrapAuthorityError::ExternalAuthorityRejected),
        Err(reason) => return Err(ClockBootstrapAuthorityError::ExternalVerifierError(reason)),
    }

    let id = VerifiedClockBootstrapAuthorityIdV2(compute_verified_authority_id(
        claim.id(),
        evidence.id(),
        evidence.provider_id(),
        evidence.authority_policy_digest(),
    )?);
    Ok(VerifiedClockBootstrapAuthorityV2 {
        id,
        claim_id: claim.id(),
        authority_evidence_id: evidence.id(),
        provider_id: evidence.provider_id().to_string(),
        authority_policy_digest: evidence.authority_policy_digest(),
        clock_evaluation_policy_id: claim.clock_evaluation_policy_id(),
    })
}

fn compute_claim_id(claim: &ClockBootstrapClaimV2) -> Result<Sha256Digest, ClockBootstrapAuthorityError> {
    Ok(domain_hash(CLOCK_BOOTSTRAP_CLAIM_DOMAIN, &canonical_claim_preimage_bytes(claim)?))
}

fn canonical_claim_preimage_bytes(claim: &ClockBootstrapClaimV2) -> Result<Vec<u8>, ClockBootstrapAuthorityError> {
    canonical_json_bytes([
        ("clock_evaluation_policy_id", Value::String(claim.clock_evaluation_policy_id.to_hex())),
        ("purpose", Value::String(claim.purpose.clone())),
        ("schema", Value::String(claim.schema.clone())),
        ("trust_snapshot_digest", Value::String(claim.trust_snapshot_digest.to_hex())),
        ("trusted_lower_unix_ms", Value::from(claim.trusted_lower_unix_ms)),
        ("trusted_upper_unix_ms", Value::from(claim.trusted_upper_unix_ms)),
    ])
}

fn compute_evidence_id(evidence: &ClockBootstrapAuthorityEvidenceV2) -> Result<Sha256Digest, ClockBootstrapAuthorityError> {
    let bytes = canonical_json_bytes([
        ("authority_policy_digest", Value::String(evidence.authority_policy_digest.to_hex())),
        ("claim_id", Value::String(evidence.claim_id.to_hex())),
        ("external_evidence_digest", Value::String(evidence.external_evidence_digest.to_hex())),
        ("provider_id", Value::String(evidence.provider_id.clone())),
        ("schema", Value::String(evidence.schema.clone())),
    ])?;
    Ok(domain_hash(CLOCK_BOOTSTRAP_AUTHORITY_EVIDENCE_DOMAIN, &bytes))
}

fn compute_verified_authority_id(
    claim_id: Sha256Digest,
    authority_evidence_id: Sha256Digest,
    provider_id: &str,
    authority_policy_digest: Sha256Digest,
) -> Result<Sha256Digest, ClockBootstrapAuthorityError> {
    let bytes = canonical_json_bytes([
        ("authority_evidence_id", Value::String(authority_evidence_id.to_hex())),
        ("authority_policy_digest", Value::String(authority_policy_digest.to_hex())),
        ("claim_id", Value::String(claim_id.to_hex())),
        ("provider_id", Value::String(provider_id.to_string())),
        ("schema", Value::String(VERIFIED_CLOCK_BOOTSTRAP_AUTHORITY_SCHEMA.to_string())),
    ])?;
    Ok(domain_hash(VERIFIED_CLOCK_BOOTSTRAP_AUTHORITY_DOMAIN, &bytes))
}

fn canonical_json_bytes<const N: usize>(fields: [(&str, Value); N]) -> Result<Vec<u8>, ClockBootstrapAuthorityError> {
    let map = fields.into_iter().map(|(key, value)| (key.to_string(), value)).collect::<BTreeMap<_, _>>();
    serde_json::to_vec(&map).map_err(|error| ClockBootstrapAuthorityError::Encoding(error.to_string()))
}

fn validate_provider_id(provider_id: &str) -> Result<(), ClockBootstrapAuthorityError> {
    if provider_id.is_empty()
        || provider_id != provider_id.trim()
        || provider_id.len() > MAX_BOOTSTRAP_PROVIDER_ID_BYTES
        || provider_id.chars().any(char::is_control)
    {
        return Err(ClockBootstrapAuthorityError::InvalidProviderId);
    }
    Ok(())
}
