// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Detached attestations over already content-addressed subjects and payloads.
//!
//! The trust core owns message framing, digest binding, substitution checks,
//! signature-policy evaluation, and lifecycle-aware authority minting. Private
//! keys and cryptographic implementations stay behind narrow traits.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::identity::{FramedDigest, Sha256Digest, TrustUsage};
use crate::trust::{KeyEligibility, TrustSnapshot, TrustSnapshotError};

pub const ATTESTATION_SCHEMA: &str = "symthaea.detached-attestation.v1";
const ATTESTATION_MESSAGE_DOMAIN: &str = "symthaea.detached-attestation.message.v1";
const ATTESTATION_IDENTITY_DOMAIN: &str = "symthaea.detached-attestation.identity.v1";
pub const MAX_KEY_ID_BYTES: usize = 256;
pub const MAX_SIGNATURE_BYTES: usize = 64 * 1024;
pub const MAX_SIGNATURES: usize = 64;
pub const MAX_ALGORITHM_NAME_BYTES: usize = 128;

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
                !name.is_empty()
                    && name == name.trim()
                    && name.len() <= MAX_ALGORITHM_NAME_BYTES
                    && name.bytes().all(|byte| {
                        byte.is_ascii_alphanumeric()
                            || matches!(byte, b'.' | b'_' | b':' | b'/' | b'-')
                    })
            }
        }
    }
}

pub(crate) fn digest_signature_algorithm(
    digest: &mut FramedDigest,
    value: &SignatureAlgorithm,
) {
    match value {
        SignatureAlgorithm::Ed25519 => digest.text("builtin:ed25519"),
        SignatureAlgorithm::MlDsa65 => digest.text("builtin:ml-dsa-65"),
        SignatureAlgorithm::MlDsa87 => digest.text("builtin:ml-dsa-87"),
        SignatureAlgorithm::Other(name) => {
            digest.text("other");
            digest.text(name);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DetachedSignature {
    pub algorithm: SignatureAlgorithm,
    pub key_id: String,
    pub signature: Vec<u8>,
}

/// Signed envelope around already content-addressed authority inputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttestationEnvelope {
    pub schema_version: String,
    pub purpose: TrustUsage,
    pub subject_sha256: Sha256Digest,
    pub payload_sha256: Sha256Digest,
    pub context_sha256: Option<Sha256Digest>,
    pub signatures: Vec<DetachedSignature>,
}

/// Exact values the verifier expected before any signature may grant authority.
#[derive(Debug, Clone, Copy)]
pub struct AttestationExpectation<'a> {
    pub purpose: &'a TrustUsage,
    pub subject_sha256: &'a Sha256Digest,
    pub payload_sha256: &'a Sha256Digest,
    pub context_sha256: Option<&'a Sha256Digest>,
}

pub trait AttestationSigner {
    fn algorithm(&self) -> SignatureAlgorithm;
    fn key_id(&self) -> &str;
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, String>;
}

pub trait AttestationSignatureVerifier {
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
    TooManySigners { actual: usize, maximum: usize },
    InvalidAlgorithm,
    InvalidKeyId,
    KeyIdTooLong,
    EmptySignature,
    SignatureTooLarge,
    Signing { key_id: String, reason: String },
    DuplicateSigner { algorithm: SignatureAlgorithm, key_id: String },
}

pub fn attest_digests(
    purpose: TrustUsage,
    subject_sha256: Sha256Digest,
    payload_sha256: Sha256Digest,
    context_sha256: Option<Sha256Digest>,
    signers: &[&dyn AttestationSigner],
) -> Result<AttestationEnvelope, AttestationBuildError> {
    if signers.len() > MAX_SIGNATURES {
        return Err(AttestationBuildError::TooManySigners {
            actual: signers.len(),
            maximum: MAX_SIGNATURES,
        });
    }
    let message = attestation_message(
        &purpose,
        &subject_sha256,
        &payload_sha256,
        context_sha256.as_ref(),
    );
    let mut identities = BTreeSet::new();
    let mut signatures = Vec::with_capacity(signers.len());

    for signer in signers {
        let algorithm = signer.algorithm();
        if !algorithm.is_canonical() {
            return Err(AttestationBuildError::InvalidAlgorithm);
        }
        let key_id = signer.key_id();
        if key_id.is_empty() || key_id != key_id.trim() {
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
            .map_err(|reason| AttestationBuildError::Signing {
                key_id: key_id.to_string(),
                reason,
            })?;
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

    Ok(AttestationEnvelope {
        schema_version: ATTESTATION_SCHEMA.into(),
        purpose,
        subject_sha256,
        payload_sha256,
        context_sha256,
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

impl AttestationPolicy {
    fn is_valid(&self) -> bool {
        if self.minimum_valid_signatures == 0
            || self.maximum_signatures == 0
            || self.maximum_signatures > MAX_SIGNATURES
            || self.maximum_signature_bytes == 0
            || self.maximum_signature_bytes > MAX_SIGNATURE_BYTES
            || self.maximum_key_id_bytes == 0
            || self.maximum_key_id_bytes > MAX_KEY_ID_BYTES
            || self.minimum_valid_signatures > self.maximum_signatures
            || self.required_algorithms.iter().any(|item| !item.is_canonical())
        {
            return false;
        }
        self.allowed_key_ids.as_ref().is_none_or(|ids| {
            ids.iter().all(|key_id| {
                !key_id.is_empty()
                    && key_id == key_id.trim()
                    && key_id.len() <= self.maximum_key_id_bytes
            })
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AttestationTrustContext<'a> {
    pub evaluation_time_unix_s: u64,
    pub snapshot: &'a TrustSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttestationViolation {
    UnsupportedSchema,
    PurposeMismatch,
    SubjectMismatch,
    PayloadMismatch,
    ContextMismatch,
    InvalidPolicy,
    TooManySignatures { actual: usize, maximum: usize },
    InvalidAlgorithm,
    InvalidKeyId,
    KeyIdTooLong { actual: usize, maximum: usize },
    EmptySignature,
    SignatureTooLarge { actual: usize, maximum: usize },
    DuplicateSigner { algorithm: SignatureAlgorithm, key_id: String },
    KeyNotAllowed { key_id: String },
    VerificationProviderError { key_id: String, reason: String },
    InvalidSignature { algorithm: SignatureAlgorithm, key_id: String },
    InsufficientValidSignatures { actual: usize, required: usize },
    MissingRequiredAlgorithm { algorithm: SignatureAlgorithm },
    TrustSnapshotInvalid(TrustSnapshotError),
    TrustSnapshotStale,
    SignerUnknown { key_id: String },
    SignerNotYetValid { key_id: String },
    SignerExpired { key_id: String },
    SignerRetired { key_id: String },
    SignerRevoked { key_id: String },
    SignerUsageNotAllowed { key_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttestationVerificationReport {
    pub valid_signers: Vec<(SignatureAlgorithm, String)>,
    pub violations: Vec<AttestationViolation>,
    pub attestation_sha256: Sha256Digest,
    pub trust_snapshot_sha256: Option<Sha256Digest>,
    pub evaluation_time_unix_s: Option<u64>,
}

impl AttestationVerificationReport {
    /// All supplied cryptographic/policy/expectation checks passed. This remains
    /// non-authorizing when no lifecycle trust snapshot was supplied.
    pub fn verification_passed(&self) -> bool {
        self.violations.is_empty()
    }

    /// Authority-capable trust additionally requires a bound lifecycle snapshot
    /// and evaluation time. Diagnostic verification can never return true here.
    pub fn trusted(&self) -> bool {
        self.verification_passed()
            && self.trust_snapshot_sha256.is_some()
            && self.evaluation_time_unix_s.is_some()
    }
}

/// Capability-bearing attestation. It is intentionally not deserializable;
/// rehydration must rerun signature, expectation, and trust-lifecycle checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VerifiedAttestation {
    envelope: AttestationEnvelope,
    attestation_sha256: Sha256Digest,
    valid_signers: Vec<(SignatureAlgorithm, String)>,
    trust_snapshot_sha256: Sha256Digest,
    evaluation_time_unix_s: u64,
}

impl VerifiedAttestation {
    pub fn envelope(&self) -> &AttestationEnvelope {
        &self.envelope
    }

    pub fn attestation_sha256(&self) -> &Sha256Digest {
        &self.attestation_sha256
    }

    pub fn valid_signers(&self) -> &[(SignatureAlgorithm, String)] {
        &self.valid_signers
    }

    pub fn trust_snapshot_sha256(&self) -> &Sha256Digest {
        &self.trust_snapshot_sha256
    }

    pub fn evaluation_time_unix_s(&self) -> u64 {
        self.evaluation_time_unix_s
    }
}

/// Diagnostic verification. This never mints authority and may be used without
/// a trust snapshot to inspect cryptographic/policy structure.
pub fn verify_attestation(
    envelope: &AttestationEnvelope,
    expectation: AttestationExpectation<'_>,
    policy: &AttestationPolicy,
    verifier: &dyn AttestationSignatureVerifier,
) -> AttestationVerificationReport {
    verify_internal(envelope, expectation, policy, verifier, None)
}

/// Mint authority only when exact expectation, signature policy, cryptographic
/// verification, trust freshness, key lifecycle, and purpose authorization all
/// pass together.
pub fn verify_attestation_authority(
    envelope: AttestationEnvelope,
    expectation: AttestationExpectation<'_>,
    policy: &AttestationPolicy,
    verifier: &dyn AttestationSignatureVerifier,
    trust: AttestationTrustContext<'_>,
) -> Result<VerifiedAttestation, AttestationVerificationReport> {
    let report = verify_internal(&envelope, expectation, policy, verifier, Some(trust));
    if !report.trusted() {
        return Err(report);
    }
    Ok(VerifiedAttestation {
        envelope,
        attestation_sha256: report.attestation_sha256,
        valid_signers: report.valid_signers,
        trust_snapshot_sha256: report
            .trust_snapshot_sha256
            .expect("trusted lifecycle verification must bind a trust snapshot"),
        evaluation_time_unix_s: report
            .evaluation_time_unix_s
            .expect("trusted lifecycle verification must bind evaluation time"),
    })
}

fn verify_internal(
    envelope: &AttestationEnvelope,
    expectation: AttestationExpectation<'_>,
    policy: &AttestationPolicy,
    verifier: &dyn AttestationSignatureVerifier,
    trust: Option<AttestationTrustContext<'_>>,
) -> AttestationVerificationReport {
    let mut violations = Vec::new();
    let mut valid_signers = Vec::new();
    let attestation_sha256 = attestation_digest(envelope);
    let mut trust_snapshot_sha256 = None;
    let evaluation_time_unix_s = trust.map(|context| context.evaluation_time_unix_s);
    let mut trust_usable = false;

    if envelope.schema_version != ATTESTATION_SCHEMA {
        violations.push(AttestationViolation::UnsupportedSchema);
    }
    if &envelope.purpose != expectation.purpose {
        violations.push(AttestationViolation::PurposeMismatch);
    }
    if &envelope.subject_sha256 != expectation.subject_sha256 {
        violations.push(AttestationViolation::SubjectMismatch);
    }
    if &envelope.payload_sha256 != expectation.payload_sha256 {
        violations.push(AttestationViolation::PayloadMismatch);
    }
    if envelope.context_sha256.as_ref() != expectation.context_sha256 {
        violations.push(AttestationViolation::ContextMismatch);
    }

    if !policy.is_valid() {
        violations.push(AttestationViolation::InvalidPolicy);
    }
    if envelope.signatures.len() > policy.maximum_signatures {
        violations.push(AttestationViolation::TooManySignatures {
            actual: envelope.signatures.len(),
            maximum: policy.maximum_signatures,
        });
    }

    if let Some(context) = trust {
        match context.snapshot.digest() {
            Ok(digest) => {
                trust_snapshot_sha256 = Some(digest);
                if context.snapshot.is_fresh_at(context.evaluation_time_unix_s) {
                    trust_usable = true;
                } else {
                    violations.push(AttestationViolation::TrustSnapshotStale);
                }
            }
            Err(error) => violations.push(AttestationViolation::TrustSnapshotInvalid(error)),
        }
    }

    let message = attestation_message(
        &envelope.purpose,
        &envelope.subject_sha256,
        &envelope.payload_sha256,
        envelope.context_sha256.as_ref(),
    );
    let mut identities = BTreeSet::new();
    let mut valid_algorithms = BTreeSet::new();

    for signature in &envelope.signatures {
        if !signature.algorithm.is_canonical() {
            violations.push(AttestationViolation::InvalidAlgorithm);
            continue;
        }
        if signature.key_id.is_empty() || signature.key_id != signature.key_id.trim() {
            violations.push(AttestationViolation::InvalidKeyId);
            continue;
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
                violations.push(AttestationViolation::KeyNotAllowed {
                    key_id: signature.key_id.clone(),
                });
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
                &envelope.purpose,
                context.evaluation_time_unix_s,
            ) {
                KeyEligibility::Eligible => {}
                KeyEligibility::Unknown => {
                    violations.push(AttestationViolation::SignerUnknown {
                        key_id: signature.key_id.clone(),
                    });
                    continue;
                }
                KeyEligibility::NotYetValid => {
                    violations.push(AttestationViolation::SignerNotYetValid {
                        key_id: signature.key_id.clone(),
                    });
                    continue;
                }
                KeyEligibility::Expired => {
                    violations.push(AttestationViolation::SignerExpired {
                        key_id: signature.key_id.clone(),
                    });
                    continue;
                }
                KeyEligibility::Retired => {
                    violations.push(AttestationViolation::SignerRetired {
                        key_id: signature.key_id.clone(),
                    });
                    continue;
                }
                KeyEligibility::Revoked => {
                    violations.push(AttestationViolation::SignerRevoked {
                        key_id: signature.key_id.clone(),
                    });
                    continue;
                }
                KeyEligibility::UsageNotAllowed => {
                    violations.push(AttestationViolation::SignerUsageNotAllowed {
                        key_id: signature.key_id.clone(),
                    });
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
                valid_signers.push((signature.algorithm.clone(), signature.key_id.clone()));
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
    for required in &policy.required_algorithms {
        if !valid_algorithms.contains(required) {
            violations.push(AttestationViolation::MissingRequiredAlgorithm {
                algorithm: required.clone(),
            });
        }
    }

    valid_signers.sort();
    violations.sort_by_key(violation_sort_key);

    AttestationVerificationReport {
        valid_signers,
        violations,
        attestation_sha256,
        trust_snapshot_sha256,
        evaluation_time_unix_s,
    }
}

pub fn attestation_digest(envelope: &AttestationEnvelope) -> Sha256Digest {
    let mut digest = FramedDigest::new(ATTESTATION_IDENTITY_DOMAIN);
    digest.text(&envelope.schema_version);
    digest.text(envelope.purpose.as_str());
    digest.text(envelope.subject_sha256.as_str());
    digest.text(envelope.payload_sha256.as_str());
    digest.optional_sha(envelope.context_sha256.as_ref());

    let mut signatures = envelope.signatures.clone();
    signatures.sort();
    for signature in signatures {
        digest.text("signature");
        digest_signature_algorithm(&mut digest, &signature.algorithm);
        digest.text(&signature.key_id);
        digest.text(Sha256Digest::of_bytes(&signature.signature).as_str());
    }
    digest.digest()
}

fn attestation_message(
    purpose: &TrustUsage,
    subject_sha256: &Sha256Digest,
    payload_sha256: &Sha256Digest,
    context_sha256: Option<&Sha256Digest>,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    append_frame(&mut bytes, ATTESTATION_MESSAGE_DOMAIN);
    append_frame(&mut bytes, ATTESTATION_SCHEMA);
    append_frame(&mut bytes, purpose.as_str());
    append_frame(&mut bytes, subject_sha256.as_str());
    append_frame(&mut bytes, payload_sha256.as_str());
    match context_sha256 {
        Some(value) => {
            append_frame(&mut bytes, "some-context");
            append_frame(&mut bytes, value.as_str());
        }
        None => append_frame(&mut bytes, "no-context"),
    }
    bytes
}

fn append_frame(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn violation_sort_key(value: &AttestationViolation) -> String {
    format!("{value:?}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trust::{KeyLifecycleStatus, KeyTrustRecord, TrustSnapshot};

    struct EchoSigner {
        key_id: &'static str,
    }

    impl AttestationSigner for EchoSigner {
        fn algorithm(&self) -> SignatureAlgorithm {
            SignatureAlgorithm::Ed25519
        }

        fn key_id(&self) -> &str {
            self.key_id
        }

        fn sign(&self, message: &[u8]) -> Result<Vec<u8>, String> {
            Ok(Sha256Digest::of_bytes(message).as_str().as_bytes().to_vec())
        }
    }

    struct EchoVerifier;

    impl AttestationSignatureVerifier for EchoVerifier {
        fn verify(
            &self,
            _algorithm: &SignatureAlgorithm,
            _key_id: &str,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            Ok(signature == Sha256Digest::of_bytes(message).as_str().as_bytes())
        }
    }

    fn usage() -> TrustUsage {
        TrustUsage::parse("science.qualification").unwrap()
    }

    fn snapshot(status: KeyLifecycleStatus) -> TrustSnapshot {
        TrustSnapshot::new(
            1,
            100,
            1_000,
            vec![KeyTrustRecord {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "reviewer".into(),
                not_before_unix_s: 100,
                not_after_unix_s: Some(900),
                status,
                usages: BTreeSet::from([usage()]),
            }],
        )
        .unwrap()
    }

    fn envelope() -> AttestationEnvelope {
        let signer = EchoSigner { key_id: "reviewer" };
        attest_digests(
            usage(),
            Sha256Digest::of_bytes(b"claim"),
            Sha256Digest::of_bytes(b"review-ready-capability"),
            Some(Sha256Digest::of_bytes(b"qualification-profile")),
            &[&signer],
        )
        .unwrap()
    }

    #[test]
    fn lifecycle_authority_mints_private_capability() {
        let envelope = envelope();
        let purpose = usage();
        let subject = envelope.subject_sha256.clone();
        let payload = envelope.payload_sha256.clone();
        let context = envelope.context_sha256.clone();
        let trust_snapshot = snapshot(KeyLifecycleStatus::Active);
        let verified = verify_attestation_authority(
            envelope,
            AttestationExpectation {
                purpose: &purpose,
                subject_sha256: &subject,
                payload_sha256: &payload,
                context_sha256: context.as_ref(),
            },
            &AttestationPolicy::default(),
            &EchoVerifier,
            AttestationTrustContext {
                evaluation_time_unix_s: 500,
                snapshot: &trust_snapshot,
            },
        )
        .unwrap();
        assert_eq!(verified.valid_signers().len(), 1);
    }

    #[test]
    fn diagnostic_verification_is_not_trusted_authority() {
        let envelope = envelope();
        let purpose = usage();
        let subject = envelope.subject_sha256.clone();
        let payload = envelope.payload_sha256.clone();
        let context = envelope.context_sha256.clone();
        let report = verify_attestation(
            &envelope,
            AttestationExpectation {
                purpose: &purpose,
                subject_sha256: &subject,
                payload_sha256: &payload,
                context_sha256: context.as_ref(),
            },
            &AttestationPolicy::default(),
            &EchoVerifier,
        );
        assert!(report.verification_passed());
        assert!(!report.trusted());
    }

    #[test]
    fn revoked_signer_cannot_mint_authority() {
        let envelope = envelope();
        let purpose = usage();
        let subject = envelope.subject_sha256.clone();
        let payload = envelope.payload_sha256.clone();
        let context = envelope.context_sha256.clone();
        let trust_snapshot = snapshot(KeyLifecycleStatus::Revoked);
        let report = verify_attestation_authority(
            envelope,
            AttestationExpectation {
                purpose: &purpose,
                subject_sha256: &subject,
                payload_sha256: &payload,
                context_sha256: context.as_ref(),
            },
            &AttestationPolicy::default(),
            &EchoVerifier,
            AttestationTrustContext {
                evaluation_time_unix_s: 500,
                snapshot: &trust_snapshot,
            },
        )
        .unwrap_err();
        assert!(report
            .violations
            .iter()
            .any(|item| matches!(item, AttestationViolation::SignerRevoked { .. })));
    }

    #[test]
    fn payload_substitution_is_rejected_even_with_valid_signature() {
        let envelope = envelope();
        let purpose = usage();
        let subject = envelope.subject_sha256.clone();
        let wrong_payload = Sha256Digest::of_bytes(b"other-payload");
        let context = envelope.context_sha256.clone();
        let trust_snapshot = snapshot(KeyLifecycleStatus::Active);
        let report = verify_attestation_authority(
            envelope,
            AttestationExpectation {
                purpose: &purpose,
                subject_sha256: &subject,
                payload_sha256: &wrong_payload,
                context_sha256: context.as_ref(),
            },
            &AttestationPolicy::default(),
            &EchoVerifier,
            AttestationTrustContext {
                evaluation_time_unix_s: 500,
                snapshot: &trust_snapshot,
            },
        )
        .unwrap_err();
        assert!(report.violations.contains(&AttestationViolation::PayloadMismatch));
    }

    #[test]
    fn stale_snapshot_cannot_mint_authority() {
        let envelope = envelope();
        let purpose = usage();
        let subject = envelope.subject_sha256.clone();
        let payload = envelope.payload_sha256.clone();
        let context = envelope.context_sha256.clone();
        let trust_snapshot = snapshot(KeyLifecycleStatus::Active);
        let report = verify_attestation_authority(
            envelope,
            AttestationExpectation {
                purpose: &purpose,
                subject_sha256: &subject,
                payload_sha256: &payload,
                context_sha256: context.as_ref(),
            },
            &AttestationPolicy::default(),
            &EchoVerifier,
            AttestationTrustContext {
                evaluation_time_unix_s: 1_000,
                snapshot: &trust_snapshot,
            },
        )
        .unwrap_err();
        assert!(report.violations.contains(&AttestationViolation::TrustSnapshotStale));
    }

    #[test]
    fn custom_algorithm_cannot_alias_builtin_identity() {
        let builtin = AttestationEnvelope {
            schema_version: ATTESTATION_SCHEMA.into(),
            purpose: usage(),
            subject_sha256: Sha256Digest::of_bytes(b"subject"),
            payload_sha256: Sha256Digest::of_bytes(b"payload"),
            context_sha256: None,
            signatures: vec![DetachedSignature {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "same".into(),
                signature: vec![1],
            }],
        };
        let custom = AttestationEnvelope {
            signatures: vec![DetachedSignature {
                algorithm: SignatureAlgorithm::Other("ed25519".into()),
                key_id: "same".into(),
                signature: vec![1],
            }],
            ..builtin.clone()
        };
        assert_ne!(attestation_digest(&builtin), attestation_digest(&custom));
    }

    #[test]
    fn malformed_policy_fails_closed() {
        let envelope = envelope();
        let purpose = usage();
        let subject = envelope.subject_sha256.clone();
        let payload = envelope.payload_sha256.clone();
        let context = envelope.context_sha256.clone();
        let policy = AttestationPolicy {
            required_algorithms: BTreeSet::from([SignatureAlgorithm::Other(" bad ".into())]),
            ..AttestationPolicy::default()
        };
        let report = verify_attestation(
            &envelope,
            AttestationExpectation {
                purpose: &purpose,
                subject_sha256: &subject,
                payload_sha256: &payload,
                context_sha256: context.as_ref(),
            },
            &policy,
            &EchoVerifier,
        );
        assert!(report.violations.contains(&AttestationViolation::InvalidPolicy));
        assert!(!report.trusted());
    }
}
