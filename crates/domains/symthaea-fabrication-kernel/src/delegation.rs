// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bounded, manifest-scoped delegation of release roles.
//!
//! Delegation does not create a new trusted key and does not authorize arbitrary
//! artifacts. A currently trusted role holder may sign a short-lived grant that
//! delegates one role to one already lifecycle-verified manifest signer for one
//! exact manifest digest.

use crate::attestation::{SignatureAlgorithm, VerifiedAttestation};
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::release::{
    ReleaseAuthority, ReleaseAuthorization, ReleaseEvaluationReport, ReleasePolicy, SignerRole,
    authorize_release_with_role_overrides,
};
use crate::trust::{KeyEligibility, KeyUsage, TrustSnapshot, digest_trust_snapshot};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const DELEGATION_GRANT_SCHEMA: &str = "symthaea.fabrication.delegation-grant.v1";
pub const MAX_DELEGATION_GRANTS: usize = 64;
pub const MAX_DELEGATION_KEY_ID_BYTES: usize = 256;
pub const MAX_DELEGATION_NONCE_BYTES: usize = 256;
pub const MAX_DELEGATION_SIGNATURE_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DelegationGrantBody {
    pub schema_version: String,
    pub delegator_algorithm: SignatureAlgorithm,
    pub delegator_key_id: String,
    pub delegate_algorithm: SignatureAlgorithm,
    pub delegate_key_id: String,
    pub role: SignerRole,
    pub manifest_digest: Sha256Digest,
    pub not_before_unix_s: u64,
    pub not_after_unix_s: u64,
    pub nonce: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedDelegationGrant {
    pub body: DelegationGrantBody,
    pub signature: Vec<u8>,
}

pub trait DelegationSigner {
    fn algorithm(&self) -> SignatureAlgorithm;
    fn key_id(&self) -> &str;
    fn sign_delegation(&self, message: &[u8]) -> Result<Vec<u8>, String>;
}

pub trait DelegationVerifier {
    fn verify_delegation(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DelegationBuildError {
    InvalidWindow,
    InvalidAlgorithm,
    InvalidKeyId,
    KeyIdTooLong,
    InvalidNonce,
    NonceTooLong,
    IdentityMismatch,
    Encoding(String),
    Signing(String),
    EmptySignature,
    SignatureTooLarge,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DelegationViolation {
    TooManyGrants { actual: usize, maximum: usize },
    UnsupportedSchema { index: usize },
    InvalidWindow { index: usize },
    GrantNotYetValid { index: usize },
    GrantExpired { index: usize },
    InvalidKeyId { index: usize },
    InvalidNonce { index: usize },
    DuplicateGrant { index: usize },
    ManifestMismatch { index: usize },
    DelegateNotManifestSigner { index: usize },
    DelegatorNotPolicyRoleHolder { index: usize },
    TrustSnapshotInvalid,
    TrustSnapshotStale,
    DelegatorUnknown { index: usize },
    DelegatorNotYetValid { index: usize },
    DelegatorExpired { index: usize },
    DelegatorRetired { index: usize },
    DelegatorRevoked { index: usize },
    DelegatorUsageNotAllowed { index: usize },
    EmptySignature { index: usize },
    SignatureTooLarge { index: usize },
    InvalidSignature { index: usize },
    VerificationProviderError { index: usize, reason: String },
    Encoding { index: usize },
    ReleasePolicy(ReleaseEvaluationReport),
}

#[derive(Debug, Clone)]
pub struct DelegatedReleaseAuthorization {
    release: ReleaseAuthorization,
    delegation_digest: Sha256Digest,
    grants: Vec<DelegationGrantBody>,
}

impl DelegatedReleaseAuthorization {
    pub fn release(&self) -> &ReleaseAuthorization {
        &self.release
    }
    pub fn delegation_digest(&self) -> Sha256Digest {
        self.delegation_digest
    }
    pub fn grants(&self) -> &[DelegationGrantBody] {
        &self.grants
    }
}

impl ReleaseAuthority for DelegatedReleaseAuthorization {
    fn manifest_digest(&self) -> Sha256Digest {
        self.release.manifest_digest()
    }
    fn policy_digest(&self) -> Sha256Digest {
        self.release.policy_digest()
    }
    fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.release.trust_snapshot_digest()
    }
    fn evaluation_time_unix_s(&self) -> u64 {
        self.release.evaluation_time_unix_s()
    }
    fn delegation_digest(&self) -> Option<Sha256Digest> {
        Some(self.delegation_digest)
    }
}

pub fn sign_delegation_grant(
    delegate_algorithm: SignatureAlgorithm,
    delegate_key_id: impl Into<String>,
    role: SignerRole,
    manifest_digest: Sha256Digest,
    not_before_unix_s: u64,
    not_after_unix_s: u64,
    nonce: impl Into<String>,
    signer: &dyn DelegationSigner,
) -> Result<SignedDelegationGrant, DelegationBuildError> {
    let body = DelegationGrantBody {
        schema_version: DELEGATION_GRANT_SCHEMA.into(),
        delegator_algorithm: signer.algorithm(),
        delegator_key_id: signer.key_id().to_string(),
        delegate_algorithm,
        delegate_key_id: delegate_key_id.into(),
        role,
        manifest_digest,
        not_before_unix_s,
        not_after_unix_s,
        nonce: nonce.into(),
    };
    validate_body(&body)?;
    if body.delegator_algorithm != signer.algorithm() || body.delegator_key_id != signer.key_id() {
        return Err(DelegationBuildError::IdentityMismatch);
    }
    let message = delegation_message(&body)?;
    let signature = signer
        .sign_delegation(&message)
        .map_err(DelegationBuildError::Signing)?;
    if signature.is_empty() {
        return Err(DelegationBuildError::EmptySignature);
    }
    if signature.len() > MAX_DELEGATION_SIGNATURE_BYTES {
        return Err(DelegationBuildError::SignatureTooLarge);
    }
    Ok(SignedDelegationGrant { body, signature })
}

pub fn authorize_release_with_delegations(
    attestation: &VerifiedAttestation,
    policy: &ReleasePolicy,
    grants: &[SignedDelegationGrant],
    verifier: &dyn DelegationVerifier,
    snapshot: &TrustSnapshot,
    evaluation_time_unix_s: u64,
) -> Result<DelegatedReleaseAuthorization, Vec<DelegationViolation>> {
    let mut violations = Vec::new();
    if grants.len() > MAX_DELEGATION_GRANTS {
        violations.push(DelegationViolation::TooManyGrants {
            actual: grants.len(),
            maximum: MAX_DELEGATION_GRANTS,
        });
    }
    if snapshot.validate().is_err() || digest_trust_snapshot(snapshot).is_err() {
        violations.push(DelegationViolation::TrustSnapshotInvalid);
    } else if !snapshot.is_fresh_at(evaluation_time_unix_s) {
        violations.push(DelegationViolation::TrustSnapshotStale);
    }
    if attestation.trust_snapshot_digest() != digest_trust_snapshot(snapshot).ok()
        || attestation.evaluation_time_unix_s() != Some(evaluation_time_unix_s)
    {
        violations.push(DelegationViolation::TrustSnapshotInvalid);
    }

    let valid_signers: BTreeSet<_> = attestation.valid_signers().iter().cloned().collect();
    let role_holders: BTreeMap<_, _> = policy
        .bindings
        .iter()
        .map(|binding| {
            (
                (binding.algorithm.clone(), binding.key_id.clone()),
                &binding.roles,
            )
        })
        .collect();
    let mut seen = BTreeSet::new();
    let mut overrides: BTreeMap<(SignatureAlgorithm, String), BTreeSet<SignerRole>> =
        BTreeMap::new();
    let mut accepted_bodies = Vec::new();

    for (index, grant) in grants.iter().enumerate().take(MAX_DELEGATION_GRANTS) {
        let body = &grant.body;
        if body.schema_version != DELEGATION_GRANT_SCHEMA {
            violations.push(DelegationViolation::UnsupportedSchema { index });
        }
        if validate_body(body).is_err() {
            if body.not_before_unix_s >= body.not_after_unix_s {
                violations.push(DelegationViolation::InvalidWindow { index });
            } else if !canonical_id(&body.delegator_key_id) || !canonical_id(&body.delegate_key_id)
            {
                violations.push(DelegationViolation::InvalidKeyId { index });
            } else {
                violations.push(DelegationViolation::InvalidNonce { index });
            }
            continue;
        }
        if evaluation_time_unix_s < body.not_before_unix_s {
            violations.push(DelegationViolation::GrantNotYetValid { index });
        }
        if evaluation_time_unix_s >= body.not_after_unix_s {
            violations.push(DelegationViolation::GrantExpired { index });
        }
        let identity = (
            body.delegator_algorithm.clone(),
            body.delegator_key_id.clone(),
            body.delegate_algorithm.clone(),
            body.delegate_key_id.clone(),
            body.role.clone(),
            body.manifest_digest,
            body.nonce.clone(),
        );
        if !seen.insert(identity) {
            violations.push(DelegationViolation::DuplicateGrant { index });
        }
        if body.manifest_digest != attestation.manifest_digest() {
            violations.push(DelegationViolation::ManifestMismatch { index });
        }
        let delegate = (
            body.delegate_algorithm.clone(),
            body.delegate_key_id.clone(),
        );
        if !valid_signers.contains(&delegate) {
            violations.push(DelegationViolation::DelegateNotManifestSigner { index });
        }
        let delegator = (
            body.delegator_algorithm.clone(),
            body.delegator_key_id.clone(),
        );
        if !role_holders
            .get(&delegator)
            .is_some_and(|roles| roles.contains(&body.role))
        {
            violations.push(DelegationViolation::DelegatorNotPolicyRoleHolder { index });
        }
        match snapshot.key_eligibility(
            &body.delegator_algorithm,
            &body.delegator_key_id,
            KeyUsage::FabricationManifest,
            evaluation_time_unix_s,
        ) {
            KeyEligibility::Eligible => {}
            KeyEligibility::Unknown => {
                violations.push(DelegationViolation::DelegatorUnknown { index })
            }
            KeyEligibility::NotYetValid => {
                violations.push(DelegationViolation::DelegatorNotYetValid { index })
            }
            KeyEligibility::Expired => {
                violations.push(DelegationViolation::DelegatorExpired { index })
            }
            KeyEligibility::Retired => {
                violations.push(DelegationViolation::DelegatorRetired { index })
            }
            KeyEligibility::Revoked => {
                violations.push(DelegationViolation::DelegatorRevoked { index })
            }
            KeyEligibility::UsageNotAllowed => {
                violations.push(DelegationViolation::DelegatorUsageNotAllowed { index })
            }
        }
        if grant.signature.is_empty() {
            violations.push(DelegationViolation::EmptySignature { index });
        } else if grant.signature.len() > MAX_DELEGATION_SIGNATURE_BYTES {
            violations.push(DelegationViolation::SignatureTooLarge { index });
        } else {
            match delegation_message(body) {
                Ok(message) => match verifier.verify_delegation(
                    &body.delegator_algorithm,
                    &body.delegator_key_id,
                    &message,
                    &grant.signature,
                ) {
                    Ok(true) => {}
                    Ok(false) => violations.push(DelegationViolation::InvalidSignature { index }),
                    Err(reason) => violations
                        .push(DelegationViolation::VerificationProviderError { index, reason }),
                },
                Err(_) => violations.push(DelegationViolation::Encoding { index }),
            }
        }

        if !violations
            .iter()
            .any(|violation| violation_index(violation) == Some(index))
        {
            overrides
                .entry(delegate)
                .or_default()
                .insert(body.role.clone());
            accepted_bodies.push(body.clone());
        }
    }

    if !violations.is_empty() {
        return Err(violations);
    }
    let release = authorize_release_with_role_overrides(attestation, policy, &overrides)
        .map_err(|report| vec![DelegationViolation::ReleasePolicy(report)])?;
    let delegation_digest = digest_delegation_grants(&accepted_bodies)
        .map_err(|_| vec![DelegationViolation::Encoding { index: 0 }])?;
    Ok(DelegatedReleaseAuthorization {
        release,
        delegation_digest,
        grants: accepted_bodies,
    })
}

pub fn digest_delegation_grants(
    grants: &[DelegationGrantBody],
) -> Result<Sha256Digest, DelegationBuildError> {
    let mut canonical = grants.to_vec();
    for body in &canonical {
        validate_body(body)?;
    }
    canonical.sort_by(|left, right| {
        (
            &left.delegator_algorithm,
            left.delegator_key_id.as_str(),
            &left.delegate_algorithm,
            left.delegate_key_id.as_str(),
            &left.role,
            left.manifest_digest.to_hex(),
            left.nonce.as_str(),
        )
            .cmp(&(
                &right.delegator_algorithm,
                right.delegator_key_id.as_str(),
                &right.delegate_algorithm,
                right.delegate_key_id.as_str(),
                &right.role,
                right.manifest_digest.to_hex(),
                right.nonce.as_str(),
            ))
    });
    let bytes = serde_json::to_vec(&canonical)
        .map_err(|error| DelegationBuildError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.delegation-set-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

fn validate_body(body: &DelegationGrantBody) -> Result<(), DelegationBuildError> {
    if body.schema_version != DELEGATION_GRANT_SCHEMA {
        return Err(DelegationBuildError::Encoding("unsupported schema".into()));
    }
    if !body.delegator_algorithm.is_canonical() || !body.delegate_algorithm.is_canonical() {
        return Err(DelegationBuildError::InvalidAlgorithm);
    }
    if body.not_before_unix_s >= body.not_after_unix_s {
        return Err(DelegationBuildError::InvalidWindow);
    }
    for key_id in [&body.delegator_key_id, &body.delegate_key_id] {
        if !canonical_id(key_id) {
            return Err(DelegationBuildError::InvalidKeyId);
        }
        if key_id.len() > MAX_DELEGATION_KEY_ID_BYTES {
            return Err(DelegationBuildError::KeyIdTooLong);
        }
    }
    if body.nonce.trim().is_empty() || body.nonce != body.nonce.trim() {
        return Err(DelegationBuildError::InvalidNonce);
    }
    if body.nonce.len() > MAX_DELEGATION_NONCE_BYTES {
        return Err(DelegationBuildError::NonceTooLong);
    }
    Ok(())
}

fn canonical_id(value: &str) -> bool {
    !value.trim().is_empty() && value == value.trim() && value.len() <= MAX_DELEGATION_KEY_ID_BYTES
}

fn delegation_message(body: &DelegationGrantBody) -> Result<Vec<u8>, DelegationBuildError> {
    validate_body(body)?;
    let canonical = serde_json::to_vec(body)
        .map_err(|error| DelegationBuildError::Encoding(error.to_string()))?;
    let mut message = b"symthaea.fabrication.delegation-signature.v1\0".to_vec();
    message.extend_from_slice(&canonical);
    Ok(message)
}

fn violation_index(violation: &DelegationViolation) -> Option<usize> {
    match violation {
        DelegationViolation::UnsupportedSchema { index }
        | DelegationViolation::InvalidWindow { index }
        | DelegationViolation::GrantNotYetValid { index }
        | DelegationViolation::GrantExpired { index }
        | DelegationViolation::InvalidKeyId { index }
        | DelegationViolation::InvalidNonce { index }
        | DelegationViolation::DuplicateGrant { index }
        | DelegationViolation::ManifestMismatch { index }
        | DelegationViolation::DelegateNotManifestSigner { index }
        | DelegationViolation::DelegatorNotPolicyRoleHolder { index }
        | DelegationViolation::DelegatorUnknown { index }
        | DelegationViolation::DelegatorNotYetValid { index }
        | DelegationViolation::DelegatorExpired { index }
        | DelegationViolation::DelegatorRetired { index }
        | DelegationViolation::DelegatorRevoked { index }
        | DelegationViolation::DelegatorUsageNotAllowed { index }
        | DelegationViolation::EmptySignature { index }
        | DelegationViolation::SignatureTooLarge { index }
        | DelegationViolation::InvalidSignature { index }
        | DelegationViolation::VerificationProviderError { index, .. }
        | DelegationViolation::Encoding { index } => Some(*index),
        DelegationViolation::TooManyGrants { .. }
        | DelegationViolation::TrustSnapshotInvalid
        | DelegationViolation::TrustSnapshotStale
        | DelegationViolation::ReleasePolicy(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attestation::{
        AttestationPolicy, AttestationTrustContext, ManifestSignatureVerifier, ManifestSigner,
        attest_fabrication_manifest, verify_attestation_authority_with_trust,
    };
    use crate::crypto_digest::sha256;
    use crate::provenance::{FabricationManifest, StableFingerprint};
    use crate::release::{ReleaseQuorumRequirement, ReleaseSignerBinding};
    use crate::trust::{KeyLifecycleStatus, KeyTrustRecord};

    struct Provider(&'static str);
    impl ManifestSigner for Provider {
        fn algorithm(&self) -> SignatureAlgorithm {
            SignatureAlgorithm::Ed25519
        }
        fn key_id(&self) -> &str {
            self.0
        }
        fn sign(&self, message: &[u8]) -> Result<Vec<u8>, String> {
            let mut bytes = self.0.as_bytes().to_vec();
            bytes.extend_from_slice(message);
            Ok(sha256(&bytes).0.to_vec())
        }
    }
    impl DelegationSigner for Provider {
        fn algorithm(&self) -> SignatureAlgorithm {
            SignatureAlgorithm::Ed25519
        }
        fn key_id(&self) -> &str {
            self.0
        }
        fn sign_delegation(&self, message: &[u8]) -> Result<Vec<u8>, String> {
            let mut bytes = self.0.as_bytes().to_vec();
            bytes.extend_from_slice(message);
            Ok(sha256(&bytes).0.to_vec())
        }
    }
    struct Verifier;
    impl ManifestSignatureVerifier for Verifier {
        fn verify(
            &self,
            _: &SignatureAlgorithm,
            key_id: &str,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            let mut bytes = key_id.as_bytes().to_vec();
            bytes.extend_from_slice(message);
            Ok(signature == sha256(&bytes).0.as_slice())
        }
    }
    impl DelegationVerifier for Verifier {
        fn verify_delegation(
            &self,
            _: &SignatureAlgorithm,
            key_id: &str,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            let mut bytes = key_id.as_bytes().to_vec();
            bytes.extend_from_slice(message);
            Ok(signature == sha256(&bytes).0.as_slice())
        }
    }

    fn manifest() -> FabricationManifest {
        let f = StableFingerprint([1, 2, 3, 4]);
        FabricationManifest {
            schema_version: "symthaea.fabrication.manifest.v1".into(),
            geometry: f,
            process_policy: f,
            process_evidence: f,
            minimum_feature_policy: f,
            minimum_feature_evidence: f,
            slice_config: f,
            slice_layers: f,
            toolpath_config: f,
            machine_profile: f,
            gcode_program: f,
            pipeline: f,
            layer_count: 1,
            command_count: 1,
            total_extrusion_mm: 1.0,
        }
    }

    #[test]
    fn exact_manifest_delegation_satisfies_missing_role() {
        let delegate = Provider("delegate");
        let attested = attest_fabrication_manifest(manifest(), &[&delegate]).unwrap();
        let records = vec!["owner", "delegate"]
            .into_iter()
            .map(|key_id| KeyTrustRecord {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: key_id.into(),
                not_before_unix_s: 1,
                not_after_unix_s: None,
                status: KeyLifecycleStatus::Active,
                usages: BTreeSet::from([KeyUsage::FabricationManifest]),
            })
            .collect();
        let snapshot = TrustSnapshot::new(1, 1, 1000, records).unwrap();
        let verified = verify_attestation_authority_with_trust(
            attested,
            &AttestationPolicy::default(),
            &Verifier,
            AttestationTrustContext {
                evaluation_time_unix_s: 100,
                snapshot: &snapshot,
            },
        )
        .unwrap();
        let policy = ReleasePolicy::new(
            1,
            4,
            false,
            vec![ReleaseQuorumRequirement {
                role: SignerRole::SafetyAuthority,
                minimum_distinct_signers: 1,
            }],
            vec![ReleaseSignerBinding {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "owner".into(),
                roles: BTreeSet::from([SignerRole::SafetyAuthority]),
            }],
        )
        .unwrap();
        let owner = Provider("owner");
        let grant = sign_delegation_grant(
            SignatureAlgorithm::Ed25519,
            "delegate",
            SignerRole::SafetyAuthority,
            verified.manifest_digest(),
            50,
            200,
            "grant-1",
            &owner,
        )
        .unwrap();
        let authorization = authorize_release_with_delegations(
            &verified,
            &policy,
            &[grant],
            &Verifier,
            &snapshot,
            100,
        )
        .unwrap();
        assert_eq!(
            authorization.release().role_counts()[&SignerRole::SafetyAuthority],
            1
        );
    }

    #[test]
    fn delegation_cannot_cross_manifest_boundary() {
        let owner = Provider("owner");
        let grant = sign_delegation_grant(
            SignatureAlgorithm::Ed25519,
            "delegate",
            SignerRole::SafetyAuthority,
            sha256(b"other"),
            50,
            200,
            "grant-1",
            &owner,
        )
        .unwrap();
        assert_ne!(grant.body.manifest_digest, sha256(b"manifest"));
    }
}
