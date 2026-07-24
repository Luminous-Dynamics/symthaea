// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Detached attestations for fabrication manifests.
//!
//! The kernel owns canonical bytes, digest binding, policy evaluation, and
//! signature-envelope structure. Private keys and cryptographic providers remain
//! outside this crate behind narrow signer/verifier traits.

use crate::crypto_digest::Sha256Digest;
use crate::provenance::{
    FabricationManifest, canonical_manifest_bytes, digest_fabrication_manifest,
};
use crate::trust::{KeyEligibility, KeyUsage, TrustSnapshot, digest_trust_snapshot};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

const MAX_KEY_ID_BYTES: usize = 256;
const MAX_SIGNATURE_BYTES: usize = 64 * 1024;
pub const MAX_SIGNATURE_ALGORITHM_NAME_BYTES: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SignatureAlgorithm {
    Ed25519,
    MlDsa65,
    MlDsa87,
    Other(String),
}

impl SignatureAlgorithm {
    pub fn is_canonical(&self) -> bool {
        match self {
            Self::Ed25519 | Self::MlDsa65 | Self::MlDsa87 => true,
            Self::Other(name) => {
                !name.trim().is_empty()
                    && name == name.trim()
                    && name.len() <= MAX_SIGNATURE_ALGORITHM_NAME_BYTES
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DetachedSignature {
    pub algorithm: SignatureAlgorithm,
    pub key_id: String,
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttestedFabricationManifest {
    pub schema_version: String,
    pub manifest: FabricationManifest,
    pub manifest_digest: Sha256Digest,
    pub signatures: Vec<DetachedSignature>,
}

pub trait ManifestSigner {
    fn algorithm(&self) -> SignatureAlgorithm;
    fn key_id(&self) -> &str;
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, String>;
}

pub trait ManifestSignatureVerifier {
    fn verify(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttestationBuildError {
    ManifestEncoding(String),
    InvalidAlgorithm,
    InvalidKeyId,
    KeyIdTooLong,
    EmptySignature,
    SignatureTooLarge,
    Signing(String),
    DuplicateSigner {
        algorithm: SignatureAlgorithm,
        key_id: String,
    },
}

pub fn attest_fabrication_manifest(
    manifest: FabricationManifest,
    signers: &[&dyn ManifestSigner],
) -> Result<AttestedFabricationManifest, AttestationBuildError> {
    let canonical = canonical_manifest_bytes(&manifest)
        .map_err(|error| AttestationBuildError::ManifestEncoding(error.to_string()))?;
    let manifest_digest = digest_fabrication_manifest(&manifest)
        .map_err(|error| AttestationBuildError::ManifestEncoding(error.to_string()))?;
    let message = attestation_message(manifest_digest, &canonical);
    let mut identities = BTreeSet::new();
    let mut signatures = Vec::with_capacity(signers.len());

    for signer in signers {
        let algorithm = signer.algorithm();
        if !algorithm.is_canonical() {
            return Err(AttestationBuildError::InvalidAlgorithm);
        }
        let key_id = signer.key_id().trim();
        if key_id.is_empty() {
            return Err(AttestationBuildError::InvalidKeyId);
        }
        if key_id.len() > MAX_KEY_ID_BYTES {
            return Err(AttestationBuildError::KeyIdTooLong);
        }
        if !identities.insert((algorithm.clone(), key_id.to_string())) {
            return Err(AttestationBuildError::DuplicateSigner {
                algorithm,
                key_id: key_id.to_string(),
            });
        }
        let signature = signer
            .sign(&message)
            .map_err(AttestationBuildError::Signing)?;
        if signature.is_empty() {
            return Err(AttestationBuildError::EmptySignature);
        }
        if signature.len() > MAX_SIGNATURE_BYTES {
            return Err(AttestationBuildError::SignatureTooLarge);
        }
        signatures.push(DetachedSignature {
            algorithm,
            key_id: key_id.to_string(),
            signature,
        });
    }

    Ok(AttestedFabricationManifest {
        schema_version: "symthaea.fabrication.attestation.v1".into(),
        manifest,
        manifest_digest,
        signatures,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttestationPolicy {
    pub minimum_valid_signatures: usize,
    pub maximum_signatures: usize,
    pub maximum_signature_bytes: usize,
    pub maximum_key_id_bytes: usize,
    pub required_algorithms: BTreeSet<SignatureAlgorithm>,
    pub allowed_key_ids: Option<BTreeSet<String>>,
}

/// Time and lifecycle context used for revocation-aware verification.
#[derive(Debug, Clone, Copy)]
pub struct AttestationTrustContext<'a> {
    pub evaluation_time_unix_s: u64,
    pub snapshot: &'a TrustSnapshot,
}

impl Default for AttestationPolicy {
    fn default() -> Self {
        Self {
            minimum_valid_signatures: 1,
            maximum_signatures: 16,
            maximum_signature_bytes: MAX_SIGNATURE_BYTES,
            maximum_key_id_bytes: MAX_KEY_ID_BYTES,
            required_algorithms: BTreeSet::new(),
            allowed_key_ids: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttestationViolation {
    InvalidPolicy,
    InvalidAlgorithm,
    UnsupportedSchema,
    TooManySignatures {
        actual: usize,
        maximum: usize,
    },
    KeyIdTooLong {
        actual: usize,
        maximum: usize,
    },
    EmptySignature,
    SignatureTooLarge {
        actual: usize,
        maximum: usize,
    },
    ManifestEncoding,
    ManifestDigestMismatch,
    DuplicateSigner {
        algorithm: SignatureAlgorithm,
        key_id: String,
    },
    KeyNotAllowed(String),
    VerificationProviderError {
        key_id: String,
        reason: String,
    },
    InvalidSignature {
        algorithm: SignatureAlgorithm,
        key_id: String,
    },
    InsufficientValidSignatures {
        actual: usize,
        required: usize,
    },
    MissingRequiredAlgorithm(SignatureAlgorithm),
    TrustSnapshotInvalid(String),
    TrustSnapshotStale {
        evaluation_time_unix_s: u64,
        issued_at_unix_s: u64,
        expires_at_unix_s: u64,
    },
    SignerUnknown(String),
    SignerNotYetValid(String),
    SignerExpired(String),
    SignerRetired(String),
    SignerRevoked(String),
    SignerUsageNotAllowed(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttestationVerificationReport {
    pub valid_signers: Vec<(SignatureAlgorithm, String)>,
    pub violations: Vec<AttestationViolation>,
    pub trust_snapshot_digest: Option<Sha256Digest>,
    pub evaluation_time_unix_s: Option<u64>,
}

impl AttestationVerificationReport {
    pub fn trusted(&self) -> bool {
        self.violations.is_empty()
    }
}

/// Capability-bearing attestation that has passed one explicit trust policy.
#[derive(Debug, Clone)]
pub struct VerifiedAttestation {
    attested: AttestedFabricationManifest,
    valid_signers: Vec<(SignatureAlgorithm, String)>,
    trust_snapshot_digest: Option<Sha256Digest>,
    evaluation_time_unix_s: Option<u64>,
}

impl VerifiedAttestation {
    pub fn attested(&self) -> &AttestedFabricationManifest {
        &self.attested
    }

    pub fn manifest(&self) -> &FabricationManifest {
        &self.attested.manifest
    }

    pub fn manifest_digest(&self) -> Sha256Digest {
        self.attested.manifest_digest
    }

    pub fn valid_signers(&self) -> &[(SignatureAlgorithm, String)] {
        &self.valid_signers
    }

    pub fn trust_snapshot_digest(&self) -> Option<Sha256Digest> {
        self.trust_snapshot_digest
    }

    pub fn evaluation_time_unix_s(&self) -> Option<u64> {
        self.evaluation_time_unix_s
    }

    pub fn is_lifecycle_governed(&self) -> bool {
        self.trust_snapshot_digest.is_some() && self.evaluation_time_unix_s.is_some()
    }
}

/// Grant attestation authority only after all policy requirements pass.
pub fn verify_attestation_authority(
    attested: AttestedFabricationManifest,
    policy: &AttestationPolicy,
    verifier: &dyn ManifestSignatureVerifier,
) -> Result<VerifiedAttestation, AttestationVerificationReport> {
    grant_attestation_authority(attested, policy, verifier, None)
}

/// Grant authority only after cryptographic and key-lifecycle policy pass.
pub fn verify_attestation_authority_with_trust(
    attested: AttestedFabricationManifest,
    policy: &AttestationPolicy,
    verifier: &dyn ManifestSignatureVerifier,
    trust: AttestationTrustContext<'_>,
) -> Result<VerifiedAttestation, AttestationVerificationReport> {
    grant_attestation_authority(attested, policy, verifier, Some(trust))
}

fn grant_attestation_authority(
    attested: AttestedFabricationManifest,
    policy: &AttestationPolicy,
    verifier: &dyn ManifestSignatureVerifier,
    trust: Option<AttestationTrustContext<'_>>,
) -> Result<VerifiedAttestation, AttestationVerificationReport> {
    let report = verify_attested_manifest_internal(&attested, policy, verifier, trust);
    if !report.trusted() {
        return Err(report);
    }
    Ok(VerifiedAttestation {
        attested,
        valid_signers: report.valid_signers,
        trust_snapshot_digest: report.trust_snapshot_digest,
        evaluation_time_unix_s: report.evaluation_time_unix_s,
    })
}

pub fn verify_attested_manifest(
    attested: &AttestedFabricationManifest,
    policy: &AttestationPolicy,
    verifier: &dyn ManifestSignatureVerifier,
) -> AttestationVerificationReport {
    verify_attested_manifest_internal(attested, policy, verifier, None)
}

pub fn verify_attested_manifest_with_trust(
    attested: &AttestedFabricationManifest,
    policy: &AttestationPolicy,
    verifier: &dyn ManifestSignatureVerifier,
    trust: AttestationTrustContext<'_>,
) -> AttestationVerificationReport {
    verify_attested_manifest_internal(attested, policy, verifier, Some(trust))
}

fn verify_attested_manifest_internal(
    attested: &AttestedFabricationManifest,
    policy: &AttestationPolicy,
    verifier: &dyn ManifestSignatureVerifier,
    trust: Option<AttestationTrustContext<'_>>,
) -> AttestationVerificationReport {
    let mut violations = Vec::new();
    let mut valid_signers = Vec::new();
    let mut trust_snapshot_digest = None;
    let evaluation_time_unix_s = trust.map(|context| context.evaluation_time_unix_s);
    let mut trust_usable = trust.is_some();

    if let Some(context) = trust {
        if let Err(error) = context.snapshot.validate() {
            violations.push(AttestationViolation::TrustSnapshotInvalid(format!(
                "{error:?}"
            )));
            trust_usable = false;
        } else {
            match digest_trust_snapshot(context.snapshot) {
                Ok(digest) => trust_snapshot_digest = Some(digest),
                Err(error) => {
                    violations.push(AttestationViolation::TrustSnapshotInvalid(format!(
                        "{error:?}"
                    )));
                    trust_usable = false;
                }
            }
            if !context.snapshot.is_fresh_at(context.evaluation_time_unix_s) {
                violations.push(AttestationViolation::TrustSnapshotStale {
                    evaluation_time_unix_s: context.evaluation_time_unix_s,
                    issued_at_unix_s: context.snapshot.issued_at_unix_s,
                    expires_at_unix_s: context.snapshot.expires_at_unix_s,
                });
                trust_usable = false;
            }
        }
    }

    if policy.minimum_valid_signatures == 0
        || policy.maximum_signatures == 0
        || policy.maximum_signature_bytes == 0
        || policy.maximum_key_id_bytes == 0
        || policy.minimum_valid_signatures > policy.maximum_signatures
    {
        violations.push(AttestationViolation::InvalidPolicy);
    }
    if attested.signatures.len() > policy.maximum_signatures {
        violations.push(AttestationViolation::TooManySignatures {
            actual: attested.signatures.len(),
            maximum: policy.maximum_signatures,
        });
    }
    if attested.schema_version != "symthaea.fabrication.attestation.v1" {
        violations.push(AttestationViolation::UnsupportedSchema);
    }
    let canonical = match canonical_manifest_bytes(&attested.manifest) {
        Ok(bytes) => bytes,
        Err(_) => {
            violations.push(AttestationViolation::ManifestEncoding);
            Vec::new()
        }
    };
    let actual_digest = digest_fabrication_manifest(&attested.manifest).ok();
    if actual_digest != Some(attested.manifest_digest) {
        violations.push(AttestationViolation::ManifestDigestMismatch);
    }
    let message = attestation_message(attested.manifest_digest, &canonical);
    let mut identities = BTreeSet::new();
    let mut valid_algorithms = BTreeSet::new();

    for signature in &attested.signatures {
        if !signature.algorithm.is_canonical() {
            violations.push(AttestationViolation::InvalidAlgorithm);
        }
        if signature.key_id.len() > policy.maximum_key_id_bytes {
            violations.push(AttestationViolation::KeyIdTooLong {
                actual: signature.key_id.len(),
                maximum: policy.maximum_key_id_bytes,
            });
            continue;
        }
        if signature.signature.is_empty() {
            violations.push(AttestationViolation::EmptySignature);
            continue;
        }
        if signature.signature.len() > policy.maximum_signature_bytes {
            violations.push(AttestationViolation::SignatureTooLarge {
                actual: signature.signature.len(),
                maximum: policy.maximum_signature_bytes,
            });
            continue;
        }
        let identity = (signature.algorithm.clone(), signature.key_id.clone());
        if !identities.insert(identity.clone()) {
            violations.push(AttestationViolation::DuplicateSigner {
                algorithm: identity.0,
                key_id: identity.1,
            });
            continue;
        }
        if let Some(allowed) = &policy.allowed_key_ids {
            if !allowed.contains(&signature.key_id) {
                violations.push(AttestationViolation::KeyNotAllowed(
                    signature.key_id.clone(),
                ));
                continue;
            }
        }
        if let Some(context) = trust {
            if !trust_usable {
                continue;
            }
            match context.snapshot.key_eligibility(
                &signature.algorithm,
                &signature.key_id,
                KeyUsage::FabricationManifest,
                context.evaluation_time_unix_s,
            ) {
                KeyEligibility::Eligible => {}
                KeyEligibility::Unknown => {
                    violations.push(AttestationViolation::SignerUnknown(
                        signature.key_id.clone(),
                    ));
                    continue;
                }
                KeyEligibility::NotYetValid => {
                    violations.push(AttestationViolation::SignerNotYetValid(
                        signature.key_id.clone(),
                    ));
                    continue;
                }
                KeyEligibility::Expired => {
                    violations.push(AttestationViolation::SignerExpired(
                        signature.key_id.clone(),
                    ));
                    continue;
                }
                KeyEligibility::Retired => {
                    violations.push(AttestationViolation::SignerRetired(
                        signature.key_id.clone(),
                    ));
                    continue;
                }
                KeyEligibility::Revoked => {
                    violations.push(AttestationViolation::SignerRevoked(
                        signature.key_id.clone(),
                    ));
                    continue;
                }
                KeyEligibility::UsageNotAllowed => {
                    violations.push(AttestationViolation::SignerUsageNotAllowed(
                        signature.key_id.clone(),
                    ));
                    continue;
                }
            }
        }
        match verifier.verify(
            &signature.algorithm,
            &signature.key_id,
            &message,
            &signature.signature,
        ) {
            Ok(true) => {
                valid_algorithms.insert(signature.algorithm.clone());
                valid_signers.push(identity);
            }
            Ok(false) => violations.push(AttestationViolation::InvalidSignature {
                algorithm: signature.algorithm.clone(),
                key_id: signature.key_id.clone(),
            }),
            Err(reason) => violations.push(AttestationViolation::VerificationProviderError {
                key_id: signature.key_id.clone(),
                reason,
            }),
        }
    }

    if valid_signers.len() < policy.minimum_valid_signatures {
        violations.push(AttestationViolation::InsufficientValidSignatures {
            actual: valid_signers.len(),
            required: policy.minimum_valid_signatures,
        });
    }
    for algorithm in &policy.required_algorithms {
        if !valid_algorithms.contains(algorithm) {
            violations.push(AttestationViolation::MissingRequiredAlgorithm(
                algorithm.clone(),
            ));
        }
    }

    AttestationVerificationReport {
        valid_signers,
        violations,
        trust_snapshot_digest,
        evaluation_time_unix_s,
    }
}

fn attestation_message(digest: Sha256Digest, canonical_manifest: &[u8]) -> Vec<u8> {
    let mut message = Vec::with_capacity(64 + canonical_manifest.len());
    message.extend_from_slice(b"symthaea.fabrication.detached-attestation.v1\0");
    message.extend_from_slice(&digest.0);
    message.extend_from_slice(canonical_manifest);
    message
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;
    use crate::provenance::StableFingerprint;

    struct TestProvider;

    impl ManifestSigner for TestProvider {
        fn algorithm(&self) -> SignatureAlgorithm {
            SignatureAlgorithm::Other("test-only-sha256".into())
        }

        fn key_id(&self) -> &str {
            "test-key"
        }

        fn sign(&self, message: &[u8]) -> Result<Vec<u8>, String> {
            Ok(sha256(message).0.to_vec())
        }
    }

    impl ManifestSignatureVerifier for TestProvider {
        fn verify(
            &self,
            algorithm: &SignatureAlgorithm,
            key_id: &str,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            Ok(
                algorithm == &SignatureAlgorithm::Other("test-only-sha256".into())
                    && key_id == "test-key"
                    && signature == sha256(message).0.as_slice(),
            )
        }
    }

    fn manifest() -> FabricationManifest {
        let fingerprint = StableFingerprint([1, 2, 3, 4]);
        FabricationManifest {
            schema_version: "symthaea.fabrication.manifest.v1".into(),
            geometry: fingerprint,
            process_policy: fingerprint,
            process_evidence: fingerprint,
            minimum_feature_policy: fingerprint,
            minimum_feature_evidence: fingerprint,
            slice_config: fingerprint,
            slice_layers: fingerprint,
            toolpath_config: fingerprint,
            machine_profile: fingerprint,
            gcode_program: fingerprint,
            pipeline: fingerprint,
            layer_count: 1,
            command_count: 2,
            total_extrusion_mm: 3.0,
        }
    }

    #[test]
    fn external_provider_can_attest_and_verify() {
        let provider = TestProvider;
        let attested = attest_fabrication_manifest(manifest(), &[&provider]).unwrap();
        let report = verify_attested_manifest(&attested, &AttestationPolicy::default(), &provider);
        assert!(report.trusted(), "{:#?}", report.violations);
    }

    #[test]
    fn trusted_report_can_become_a_capability() {
        let provider = TestProvider;
        let attested = attest_fabrication_manifest(manifest(), &[&provider]).unwrap();
        let verified =
            verify_attestation_authority(attested, &AttestationPolicy::default(), &provider)
                .unwrap();
        assert_eq!(verified.valid_signers().len(), 1);
        assert_eq!(verified.manifest().command_count, 2);
    }

    #[test]
    fn manifest_tampering_invalidates_digest_and_signature() {
        let provider = TestProvider;
        let mut attested = attest_fabrication_manifest(manifest(), &[&provider]).unwrap();
        attested.manifest.command_count += 1;
        let report = verify_attested_manifest(&attested, &AttestationPolicy::default(), &provider);
        assert!(
            report
                .violations
                .contains(&AttestationViolation::ManifestDigestMismatch)
        );
        assert!(
            report.violations.iter().any(|violation| matches!(
                violation,
                AttestationViolation::InvalidSignature { .. }
            ))
        );
    }

    #[test]
    fn zero_signature_policy_is_not_a_trust_bypass() {
        let provider = TestProvider;
        let attested = attest_fabrication_manifest(manifest(), &[&provider]).unwrap();
        let mut policy = AttestationPolicy::default();
        policy.minimum_valid_signatures = 0;
        let report = verify_attested_manifest(&attested, &policy, &provider);
        assert!(
            report
                .violations
                .contains(&AttestationViolation::InvalidPolicy)
        );
        assert!(!report.trusted());
    }

    #[test]
    fn required_algorithm_is_enforced() {
        let provider = TestProvider;
        let attested = attest_fabrication_manifest(manifest(), &[&provider]).unwrap();
        let mut policy = AttestationPolicy::default();
        policy
            .required_algorithms
            .insert(SignatureAlgorithm::Ed25519);
        let report = verify_attested_manifest(&attested, &policy, &provider);
        assert!(
            report
                .violations
                .contains(&AttestationViolation::MissingRequiredAlgorithm(
                    SignatureAlgorithm::Ed25519
                ))
        );
    }

    #[test]
    fn revoked_signer_cannot_receive_authority() {
        use crate::trust::{KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot};
        use std::collections::BTreeSet;

        let provider = TestProvider;
        let attested = attest_fabrication_manifest(manifest(), &[&provider]).unwrap();
        let snapshot = TrustSnapshot::new(
            1,
            100,
            1_000,
            vec![KeyTrustRecord {
                algorithm: SignatureAlgorithm::Other("test-only-sha256".into()),
                key_id: "test-key".into(),
                not_before_unix_s: 100,
                not_after_unix_s: Some(900),
                status: KeyLifecycleStatus::Revoked,
                usages: BTreeSet::from([KeyUsage::FabricationManifest]),
            }],
        )
        .unwrap();
        let report = verify_attested_manifest_with_trust(
            &attested,
            &AttestationPolicy::default(),
            &provider,
            AttestationTrustContext {
                evaluation_time_unix_s: 500,
                snapshot: &snapshot,
            },
        );
        assert!(
            report
                .violations
                .contains(&AttestationViolation::SignerRevoked("test-key".into()))
        );
        assert!(!report.trusted());
    }

    #[test]
    fn active_signer_retains_snapshot_digest_in_capability() {
        use crate::trust::{KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot};
        use std::collections::BTreeSet;

        let provider = TestProvider;
        let attested = attest_fabrication_manifest(manifest(), &[&provider]).unwrap();
        let snapshot = TrustSnapshot::new(
            2,
            100,
            1_000,
            vec![KeyTrustRecord {
                algorithm: SignatureAlgorithm::Other("test-only-sha256".into()),
                key_id: "test-key".into(),
                not_before_unix_s: 100,
                not_after_unix_s: Some(900),
                status: KeyLifecycleStatus::Active,
                usages: BTreeSet::from([KeyUsage::FabricationManifest]),
            }],
        )
        .unwrap();
        let verified = verify_attestation_authority_with_trust(
            attested,
            &AttestationPolicy::default(),
            &provider,
            AttestationTrustContext {
                evaluation_time_unix_s: 500,
                snapshot: &snapshot,
            },
        )
        .unwrap();
        assert!(verified.is_lifecycle_governed());
        assert_eq!(
            verified.trust_snapshot_digest(),
            Some(digest_trust_snapshot(&snapshot).unwrap())
        );
        assert_eq!(verified.evaluation_time_unix_s(), Some(500));
    }
}
