// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Crate entry and context binding for verified welfare-intervention authority.

#![deny(unsafe_code)]

#[path = "lib.rs"]
#[allow(unused_imports)]
mod implementation;

pub use implementation::*;

use serde::{Deserialize, Serialize};
use symthaea_core::intervention_interlock::InterventionRequest;
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::crypto_digest::{Sha256, Sha256Digest};
use symthaea_fabrication_kernel::trust::{TrustSnapshot, digest_trust_snapshot};
use thiserror::Error;

pub const WELFARE_AUTHORITY_POLICY_MANIFEST_SCHEMA: &str =
    "symthaea.welfare.authority-policy-manifest.v1";
const MAX_POLICY_ID_BYTES: usize = 256;
const MAX_CONTEXT_NONCE_BYTES: usize = 64;

/// Serializable signer enrollment used to make welfare-authority policy identity explicit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WelfareAuthorityPolicyBindingRecord {
    pub algorithm: SignatureAlgorithm,
    pub key_id: String,
    pub role: WelfareAuthorityRole,
}

impl From<WelfareAuthoritySignerBinding> for WelfareAuthorityPolicyBindingRecord {
    fn from(value: WelfareAuthoritySignerBinding) -> Self {
        Self {
            algorithm: value.algorithm,
            key_id: value.key_id,
            role: value.role,
        }
    }
}

/// Canonical policy manifest whose digest is signed indirectly through the authority ID.
///
/// This closes a retrospective-policy-widening gap: a signature created when a key was not
/// enrolled must not become valid later merely because operator policy changed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WelfareAuthorityPolicyManifest {
    pub schema_version: String,
    pub policy_id: String,
    pub bindings: Vec<WelfareAuthorityPolicyBindingRecord>,
    pub maximum_authorization_duration_s: u64,
    pub maximum_statement_age_s: u64,
    pub maximum_signatures: usize,
}

impl WelfareAuthorityPolicyManifest {
    pub fn new(
        policy_id: impl Into<String>,
        bindings: impl IntoIterator<Item = WelfareAuthoritySignerBinding>,
    ) -> Result<Self, WelfareAuthorityContextError> {
        let manifest = Self {
            schema_version: WELFARE_AUTHORITY_POLICY_MANIFEST_SCHEMA.into(),
            policy_id: policy_id.into(),
            bindings: bindings.into_iter().map(Into::into).collect(),
            maximum_authorization_duration_s: 15 * 60,
            maximum_statement_age_s: 5 * 60,
            maximum_signatures: MAX_AUTHORITY_SIGNATURES,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), WelfareAuthorityContextError> {
        if self.schema_version != WELFARE_AUTHORITY_POLICY_MANIFEST_SCHEMA {
            return Err(WelfareAuthorityContextError::UnsupportedPolicyManifestSchema);
        }
        validate_context_id("policy_id", &self.policy_id, MAX_POLICY_ID_BYTES)?;
        if self.bindings.is_empty()
            || self.maximum_authorization_duration_s == 0
            || self.maximum_statement_age_s == 0
            || self.maximum_signatures == 0
            || self.maximum_signatures > MAX_AUTHORITY_SIGNATURES
        {
            return Err(WelfareAuthorityContextError::InvalidPolicyManifest);
        }
        self.build_policy().map(|_| ())
    }

    pub fn build_policy(&self) -> Result<WelfareAuthorityPolicy, WelfareAuthorityContextError> {
        if self.schema_version != WELFARE_AUTHORITY_POLICY_MANIFEST_SCHEMA {
            return Err(WelfareAuthorityContextError::UnsupportedPolicyManifestSchema);
        }
        validate_context_id("policy_id", &self.policy_id, MAX_POLICY_ID_BYTES)?;
        let bindings = self.bindings.iter().cloned().map(|binding| {
            WelfareAuthoritySignerBinding::new(binding.algorithm, binding.key_id, binding.role)
        });
        let mut policy = WelfareAuthorityPolicy::new(bindings)
            .map_err(WelfareAuthorityContextError::Authority)?;
        policy.maximum_authorization_duration_s = self.maximum_authorization_duration_s;
        policy.maximum_statement_age_s = self.maximum_statement_age_s;
        policy.maximum_signatures = self.maximum_signatures;
        policy
            .validate()
            .map_err(WelfareAuthorityContextError::Authority)?;
        Ok(policy)
    }
}

/// Digest the canonical, order-independent welfare-authority policy manifest.
pub fn digest_welfare_authority_policy_manifest(
    manifest: &WelfareAuthorityPolicyManifest,
) -> Result<Sha256Digest, WelfareAuthorityContextError> {
    manifest.validate()?;
    let mut canonical = manifest.clone();
    canonical.bindings.sort_by(|left, right| {
        (&left.algorithm, left.key_id.as_str(), left.role).cmp(&(
            &right.algorithm,
            right.key_id.as_str(),
            right.role,
        ))
    });
    let encoded = serde_json::to_vec(&canonical)
        .map_err(|error| WelfareAuthorityContextError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.welfare.authority-policy-manifest-digest.v1\0");
    hasher.update(&encoded);
    Ok(hasher.finalize())
}

/// Derive an authority ID that commits the statement signatures to both the signer-policy
/// manifest and the exact lifecycle-aware trust snapshot used at issuance.
pub fn context_bound_authority_id(
    nonce: &str,
    manifest: &WelfareAuthorityPolicyManifest,
    trust_snapshot: &TrustSnapshot,
) -> Result<String, WelfareAuthorityContextError> {
    validate_context_id("authority_nonce", nonce, MAX_CONTEXT_NONCE_BYTES)?;
    let policy_digest = digest_welfare_authority_policy_manifest(manifest)?;
    let trust_digest = digest_trust_snapshot(trust_snapshot)
        .map_err(|error| WelfareAuthorityContextError::TrustSnapshot(format!("{error:?}")))?;
    let id = format!(
        "ctx-v1:{nonce}:{}:{}",
        hex_context_digest(policy_digest),
        hex_context_digest(trust_digest)
    );
    if id.len() > MAX_AUTHORITY_ID_BYTES {
        return Err(WelfareAuthorityContextError::ContextAuthorityIdTooLong {
            actual: id.len(),
            maximum: MAX_AUTHORITY_ID_BYTES,
        });
    }
    Ok(id)
}

/// Construct an authority statement on the exact policy + trust context that signers saw.
#[allow(clippy::too_many_arguments)]
pub fn build_context_bound_authority_statement(
    nonce: &str,
    manifest: &WelfareAuthorityPolicyManifest,
    trust_snapshot: &TrustSnapshot,
    authority_epoch: u64,
    sequence: u64,
    issued_at_unix_s: u64,
    expires_at_unix_s: u64,
    request: &InterventionRequest,
) -> Result<WelfareInterventionAuthorityStatement, WelfareAuthorityContextError> {
    manifest.validate()?;
    if !trust_snapshot.is_fresh_at(issued_at_unix_s) {
        return Err(WelfareAuthorityContextError::TrustSnapshotNotFreshAtIssue);
    }
    let authority_id = context_bound_authority_id(nonce, manifest, trust_snapshot)?;
    WelfareInterventionAuthorityStatement::for_request(
        authority_id,
        authority_epoch,
        sequence,
        issued_at_unix_s,
        expires_at_unix_s,
        request,
    )
    .map_err(WelfareAuthorityContextError::Authority)
}

/// Verify a signed authority only under the exact policy/trust context committed at issuance.
///
/// Callers should prefer this over bare `verify_welfare_authority` for runtime intervention
/// authority. Trust-snapshot rollover or policy mutation requires a freshly issued signature.
pub fn verify_context_bound_welfare_authority(
    nonce: &str,
    signed: &SignedWelfareInterventionAuthority,
    manifest: &WelfareAuthorityPolicyManifest,
    trust_snapshot: &TrustSnapshot,
    unix_s: u64,
    verifier: &dyn WelfareAuthoritySignatureVerifier,
) -> Result<VerifiedWelfareInterventionAuthority, WelfareAuthorityContextError> {
    let expected_id = context_bound_authority_id(nonce, manifest, trust_snapshot)?;
    if signed.statement.authority_id != expected_id {
        return Err(WelfareAuthorityContextError::VerificationContextMismatch);
    }
    let policy = manifest.build_policy()?;
    verify_welfare_authority(signed, trust_snapshot, &policy, unix_s, verifier)
        .map_err(WelfareAuthorityContextError::Authority)
}

fn validate_context_id(
    field: &'static str,
    value: &str,
    maximum: usize,
) -> Result<(), WelfareAuthorityContextError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > maximum
        || value.chars().any(char::is_control)
    {
        return Err(WelfareAuthorityContextError::InvalidContextIdentifier {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn hex_context_digest(digest: Sha256Digest) -> String {
    let mut out = String::with_capacity(64);
    for byte in digest.0 {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum WelfareAuthorityContextError {
    #[error("unsupported welfare-authority policy-manifest schema")]
    UnsupportedPolicyManifestSchema,
    #[error("invalid welfare-authority policy manifest")]
    InvalidPolicyManifest,
    #[error("invalid context identifier in {field}: {value:?}")]
    InvalidContextIdentifier { field: &'static str, value: String },
    #[error("context-bound authority id too long: {actual} > {maximum}")]
    ContextAuthorityIdTooLong { actual: usize, maximum: usize },
    #[error("trust snapshot is not fresh at authority issuance time")]
    TrustSnapshotNotFreshAtIssue,
    #[error("trust snapshot invalid: {0}")]
    TrustSnapshot(String),
    #[error("signed authority was issued for a different policy/trust context")]
    VerificationContextMismatch,
    #[error("welfare-authority operation failed: {0}")]
    Authority(#[source] WelfareAuthorityError),
    #[error("encoding failed: {0}")]
    Encoding(String),
}
