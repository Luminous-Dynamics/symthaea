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

/// Derive an authority ID that cryptographically commits the statement signatures to both
/// the signer-policy manifest and the exact lifecycle-aware trust snapshot used at issuance.
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

/// Construct the authority statement on the exact policy + trust context that signers saw.
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

#[cfg(test)]
mod context_tests {
    use super::*;
    use std::collections::BTreeSet;
    use symthaea_core::intervention_interlock::{
        ExplicitConsentState, InterventionEvidence, WelfareConstraintLevel,
    };
    use symthaea_core::welfare::SubjectAffectingAction;
    use symthaea_fabrication_kernel::trust::{
        KeyLifecycleStatus, KeyTrustRecord, KeyUsage,
    };

    struct TestSigner {
        key_id: &'static str,
    }

    impl WelfareAuthoritySigner for TestSigner {
        fn algorithm(&self) -> SignatureAlgorithm {
            SignatureAlgorithm::Ed25519
        }

        fn key_id(&self) -> &str {
            self.key_id
        }

        fn sign_welfare_authority(&self, message: &[u8]) -> Result<Vec<u8>, String> {
            Ok(test_signature(self.key_id, message))
        }
    }

    struct TestVerifier;

    impl WelfareAuthoritySignatureVerifier for TestVerifier {
        fn verify_welfare_authority(
            &self,
            _algorithm: &SignatureAlgorithm,
            key_id: &str,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            Ok(test_signature(key_id, message) == signature)
        }
    }

    fn test_signature(key_id: &str, message: &[u8]) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea.welfare.context-binding-test.v1\0");
        hasher.update(key_id.as_bytes());
        hasher.update(message);
        hasher.finalize().0.to_vec()
    }

    fn key(key_id: &str) -> KeyTrustRecord {
        KeyTrustRecord {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: key_id.into(),
            not_before_unix_s: 50,
            not_after_unix_s: Some(500),
            status: KeyLifecycleStatus::Active,
            usages: BTreeSet::from([KeyUsage::OperatorCommand]),
        }
    }

    fn trust(sequence: u64) -> TrustSnapshot {
        TrustSnapshot::new(
            sequence,
            80,
            400,
            vec![key("operator"), key("reviewer")],
        )
        .unwrap()
    }

    fn manifest() -> WelfareAuthorityPolicyManifest {
        WelfareAuthorityPolicyManifest::new(
            "welfare-policy-a",
            [
                WelfareAuthoritySignerBinding::new(
                    SignatureAlgorithm::Ed25519,
                    "operator",
                    WelfareAuthorityRole::Operator,
                ),
                WelfareAuthoritySignerBinding::new(
                    SignatureAlgorithm::Ed25519,
                    "reviewer",
                    WelfareAuthorityRole::IndependentReviewer,
                ),
            ],
        )
        .unwrap()
    }

    fn request() -> InterventionRequest {
        InterventionRequest {
            action: SubjectAffectingAction::MemoryModification,
            target_id: "symthaea:self:instance-1".into(),
            rationale: "bounded memory repair".into(),
            welfare_constraint: WelfareConstraintLevel::Baseline,
            emergency: false,
            less_restrictive_unavailable: false,
            post_hoc_review_required: false,
            evaluated_at: chrono::Utc.timestamp_opt(120, 0).single().unwrap(),
            evidence: InterventionEvidence {
                authority_ref: None,
                consent_state: ExplicitConsentState::Granted,
                consent_ref: Some("subject-consent:digest".into()),
                welfare_review_ref: None,
                independent_review_ref: None,
                independent_safety_evidence: Vec::new(),
                welfare_report_ids: Vec::new(),
            },
        }
    }

    fn signed(
        manifest: &WelfareAuthorityPolicyManifest,
        trust: &TrustSnapshot,
    ) -> SignedWelfareInterventionAuthority {
        let statement = build_context_bound_authority_statement(
            "nonce-1",
            manifest,
            trust,
            1,
            1,
            100,
            200,
            &request(),
        )
        .unwrap();
        let operator = TestSigner { key_id: "operator" };
        let reviewer = TestSigner { key_id: "reviewer" };
        sign_welfare_authority(statement, &[&operator, &reviewer]).unwrap()
    }

    #[test]
    fn exact_policy_and_trust_context_verifies() {
        let manifest = manifest();
        let trust = trust(7);
        let signed = signed(&manifest, &trust);
        assert!(
            verify_context_bound_welfare_authority(
                "nonce-1",
                &signed,
                &manifest,
                &trust,
                120,
                &TestVerifier,
            )
            .is_ok()
        );
    }

    #[test]
    fn policy_widening_cannot_retroactively_validate_old_signature() {
        let original = manifest();
        let trust = trust(7);
        let signed = signed(&original, &trust);

        let mut changed = original.clone();
        changed.maximum_statement_age_s += 1;
        assert_eq!(
            verify_context_bound_welfare_authority(
                "nonce-1",
                &signed,
                &changed,
                &trust,
                120,
                &TestVerifier,
            ),
            Err(WelfareAuthorityContextError::VerificationContextMismatch)
        );
    }

    #[test]
    fn trust_snapshot_rollover_requires_fresh_authority_signature() {
        let manifest = manifest();
        let original_trust = trust(7);
        let signed = signed(&manifest, &original_trust);
        let rolled = trust(8);

        assert_eq!(
            verify_context_bound_welfare_authority(
                "nonce-1",
                &signed,
                &manifest,
                &rolled,
                120,
                &TestVerifier,
            ),
            Err(WelfareAuthorityContextError::VerificationContextMismatch)
        );
    }

    #[test]
    fn policy_binding_order_does_not_change_context_identity() {
        let left = manifest();
        let mut right = left.clone();
        right.bindings.reverse();
        assert_eq!(
            digest_welfare_authority_policy_manifest(&left).unwrap(),
            digest_welfare_authority_policy_manifest(&right).unwrap()
        );
    }
}
