// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Detached attestations over already content-addressed subjects and payloads.
//!
//! Authority verification binds five things together: the signed envelope, the
//! exact signature policy, the current tracker-accepted trust snapshot, the
//! evaluation time, and the exact verification-key material committed by that
//! snapshot. Private keys and cryptographic implementations stay behind narrow
//! traits.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::identity::{FramedDigest, Sha256Digest, TrustUsage};
use crate::trust::{
    KeyEligibility, TrustSnapshot, TrustSnapshotCurrentnessError, TrustSnapshotError,
    TrustSnapshotTracker,
};

pub const ATTESTATION_SCHEMA: &str = "symthaea.detached-attestation.v1";
const ATTESTATION_MESSAGE_DOMAIN: &str = "symthaea.detached-attestation.message.v1";
const ATTESTATION_IDENTITY_DOMAIN: &str = "symthaea.detached-attestation.identity.v1";
const ATTESTATION_POLICY_DOMAIN: &str = "symthaea.detached-attestation-policy.identity.v1";
const VERIFIED_AUTHORITY_DOMAIN: &str = "symthaea.verified-attestation-authority.identity.v1";
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttestationEnvelope {
    pub schema_version: String,
    pub purpose: TrustUsage,
    pub subject_sha256: Sha256Digest,
    pub payload_sha256: Sha256Digest,
    pub context_sha256: Option<Sha256Digest>,
    pub signatures: Vec<DetachedSignature>,
}

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

/// Cryptographic verification provider.
///
/// When `expected_verification_key_sha256` is `Some`, implementations MUST
/// resolve the exact public/verification key bytes, hash their canonical form,
/// fail if the digest differs, and only then verify the signature. Authority
/// verification always supplies `Some`; diagnostic verification may supply
/// `None` and can never mint `VerifiedAttestation`.
pub trait AttestationSignatureVerifier {
    fn verify(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        expected_verification_key_sha256: Option<&Sha256Digest>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttestationPolicyError {
    InvalidPolicy,
}

impl AttestationPolicy {
    pub fn validate(&self) -> Result<(), AttestationPolicyError> {
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
            return Err(AttestationPolicyError::InvalidPolicy);
        }
        if self.allowed_key_ids.as_ref().is_some_and(|ids| {
            ids.iter().any(|key_id| {
                key_id.is_empty()
                    || key_id != key_id.trim()
                    || key_id.len() > self.maximum_key_id_bytes
            })
        }) {
            return Err(AttestationPolicyError::InvalidPolicy);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Sha256Digest, AttestationPolicyError> {
        self.validate()?;
        let mut digest = FramedDigest::new(ATTESTATION_POLICY_DOMAIN);
        digest.text(&self.minimum_valid_signatures.to_string());
        digest.text(&self.maximum_signatures.to_string());
        digest.text(&self.maximum_signature_bytes.to_string());
        digest.text(&self.maximum_key_id_bytes.to_string());
        for algorithm in &self.required_algorithms {
            digest.text("required-algorithm");
            digest_signature_algorithm(&mut digest, algorithm);
        }
        match &self.allowed_key_ids {
            None => digest.text("allow-any-key-id"),
            Some(ids) => {
                digest.text("restricted-key-ids");
                for key_id in ids {
                    digest.text(key_id);
                }
            }
        }
        Ok(digest.digest())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AttestationTrustContext<'a> {
    pub evaluation_time_unix_s: u64,
    pub snapshot: &'a TrustSnapshot,
    pub tracker: &'a TrustSnapshotTracker,
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
    TrustSnapshotNotCurrent(TrustSnapshotCurrentnessError),
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
    pub policy_sha256: Option<Sha256Digest>,
    pub trust_snapshot_sha256: Option<Sha256Digest>,
    pub evaluation_time_unix_s: Option<u64>,
}

impl AttestationVerificationReport {
    pub fn verification_passed(&self) -> bool {
        self.violations.is_empty()
    }

    pub fn trusted(&self) -> bool {
        self.verification_passed()
            && self.policy_sha256.is_some()
            && self.trust_snapshot_sha256.is_some()
            && self.evaluation_time_unix_s.is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VerifiedAttestation {
    envelope: AttestationEnvelope,
    attestation_sha256: Sha256Digest,
    policy_sha256: Sha256Digest,
    trust_snapshot_sha256: Sha256Digest,
    evaluation_time_unix_s: u64,
    authority_sha256: Sha256Digest,
    valid_signers: Vec<(SignatureAlgorithm, String)>,
}

impl VerifiedAttestation {
    pub fn envelope(&self) -> &AttestationEnvelope {
        &self.envelope
    }
    pub fn attestation_sha256(&self) -> &Sha256Digest {
        &self.attestation_sha256
    }
    pub fn policy_sha256(&self) -> &Sha256Digest {
        &self.policy_sha256
    }
    pub fn trust_snapshot_sha256(&self) -> &Sha256Digest {
        &self.trust_snapshot_sha256
    }
    pub fn evaluation_time_unix_s(&self) -> u64 {
        self.evaluation_time_unix_s
    }
    pub fn authority_sha256(&self) -> &Sha256Digest {
        &self.authority_sha256
    }
    pub fn valid_signers(&self) -> &[(SignatureAlgorithm, String)] {
        &self.valid_signers
    }
}

pub fn verify_attestation(
    envelope: &AttestationEnvelope,
    expectation: AttestationExpectation<'_>,
    policy: &AttestationPolicy,
    verifier: &dyn AttestationSignatureVerifier,
) -> AttestationVerificationReport {
    verify_internal(envelope, expectation, policy, verifier, None)
}

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
    let policy_sha256 = report
        .policy_sha256
        .clone()
        .expect("trusted verification must bind an exact attestation policy");
    let trust_snapshot_sha256 = report
        .trust_snapshot_sha256
        .clone()
        .expect("trusted verification must bind a current trust snapshot");
    let evaluation_time_unix_s = report
        .evaluation_time_unix_s
        .expect("trusted verification must bind evaluation time");
    let authority_sha256 = verified_authority_digest(
        &report.attestation_sha256,
        &policy_sha256,
        &trust_snapshot_sha256,
        evaluation_time_unix_s,
    );
    Ok(VerifiedAttestation {
        envelope,
        attestation_sha256: report.attestation_sha256,
        policy_sha256,
        trust_snapshot_sha256,
        evaluation_time_unix_s,
        authority_sha256,
        valid_signers: report.valid_signers,
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
    let policy_sha256 = match policy.digest() {
        Ok(digest) => Some(digest),
        Err(AttestationPolicyError::InvalidPolicy) => {
            violations.push(AttestationViolation::InvalidPolicy);
            None
        }
    };
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
    if envelope.signatures.len() > policy.maximum_signatures {
        violations.push(AttestationViolation::TooManySignatures {
            actual: envelope.signatures.len(),
            maximum: policy.maximum_signatures,
        });
    }

    if let Some(context) = trust {
        match context.tracker.require_current(context.snapshot) {
            Ok(digest) => {
                trust_snapshot_sha256 = Some(digest);
                if context.snapshot.is_fresh_at(context.evaluation_time_unix_s) {
                    trust_usable = true;
                } else {
                    violations.push(AttestationViolation::TrustSnapshotStale);
                }
            }
            Err(TrustSnapshotCurrentnessError::InvalidSnapshot(error)) => {
                violations.push(AttestationViolation::TrustSnapshotInvalid(error));
            }
            Err(error) => violations.push(AttestationViolation::TrustSnapshotNotCurrent(error)),
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

        let expected_key_digest = if let Some(context) = trust {
            if !trust_usable {
                continue;
            }
            match context.snapshot.key_eligibility(
                &signature.algorithm,
                &signature.key_id,
                &envelope.purpose,
                context.evaluation_time_unix_s,
            ) {
                KeyEligibility::Eligible => context
                    .snapshot
                    .key_record(&signature.algorithm, &signature.key_id)
                    .map(|record| &record.verification_key_sha256),
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
        } else {
            None
        };

        match verifier.verify(
            &signature.algorithm,
            &signature.key_id,
            expected_key_digest,
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
        policy_sha256,
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

fn verified_authority_digest(
    attestation_sha256: &Sha256Digest,
    policy_sha256: &Sha256Digest,
    trust_snapshot_sha256: &Sha256Digest,
    evaluation_time_unix_s: u64,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(VERIFIED_AUTHORITY_DOMAIN);
    digest.text(attestation_sha256.as_str());
    digest.text(policy_sha256.as_str());
    digest.text(trust_snapshot_sha256.as_str());
    digest.text(&evaluation_time_unix_s.to_string());
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
    use crate::trust::{KeyLifecycleStatus, KeyTrustRecord};

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
            key_id: &str,
            expected_verification_key_sha256: Option<&Sha256Digest>,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            let resolved = Sha256Digest::of_bytes(
                format!("verification-key:{key_id}").as_bytes(),
            );
            if let Some(expected) = expected_verification_key_sha256 {
                if expected != &resolved {
                    return Err("verification-key digest mismatch".into());
                }
            }
            Ok(signature == Sha256Digest::of_bytes(message).as_str().as_bytes())
        }
    }

    fn usage() -> TrustUsage {
        TrustUsage::parse("science.qualification").unwrap()
    }

    fn key_material(key_id: &str) -> Sha256Digest {
        Sha256Digest::of_bytes(format!("verification-key:{key_id}").as_bytes())
    }

    fn snapshot_with_sequence(status: KeyLifecycleStatus, sequence: u64) -> TrustSnapshot {
        TrustSnapshot::new(
            sequence,
            100 + sequence,
            1_000,
            vec![KeyTrustRecord {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "reviewer".into(),
                verification_key_sha256: key_material("reviewer"),
                not_before_unix_s: 100,
                not_after_unix_s: Some(900),
                status,
                usages: BTreeSet::from([usage()]),
            }],
        )
        .unwrap()
    }

    fn snapshot(status: KeyLifecycleStatus) -> TrustSnapshot {
        snapshot_with_sequence(status, 1)
    }

    fn current_tracker(snapshot: &TrustSnapshot) -> TrustSnapshotTracker {
        let mut tracker = TrustSnapshotTracker::default();
        tracker.accept(snapshot).unwrap();
        tracker
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

    fn expectation<'a>(
        purpose: &'a TrustUsage,
        envelope: &'a AttestationEnvelope,
    ) -> AttestationExpectation<'a> {
        AttestationExpectation {
            purpose,
            subject_sha256: &envelope.subject_sha256,
            payload_sha256: &envelope.payload_sha256,
            context_sha256: envelope.context_sha256.as_ref(),
        }
    }

    #[test]
    fn lifecycle_authority_mints_private_capability() {
        let envelope = envelope();
        let purpose = usage();
        let trust_snapshot = snapshot(KeyLifecycleStatus::Active);
        let tracker = current_tracker(&trust_snapshot);
        let verified = verify_attestation_authority(
            envelope.clone(),
            expectation(&purpose, &envelope),
            &AttestationPolicy::default(),
            &EchoVerifier,
            AttestationTrustContext {
                evaluation_time_unix_s: 500,
                snapshot: &trust_snapshot,
                tracker: &tracker,
            },
        )
        .unwrap();
        assert_eq!(verified.valid_signers().len(), 1);
    }

    #[test]
    fn diagnostic_verification_is_not_trusted_authority() {
        let envelope = envelope();
        let purpose = usage();
        let report = verify_attestation(
            &envelope,
            expectation(&purpose, &envelope),
            &AttestationPolicy::default(),
            &EchoVerifier,
        );
        assert!(report.verification_passed());
        assert!(!report.trusted());
    }

    #[test]
    fn verification_key_substitution_cannot_mint_authority() {
        let envelope = envelope();
        let purpose = usage();
        let mut trust_snapshot = snapshot(KeyLifecycleStatus::Active);
        trust_snapshot.keys[0].verification_key_sha256 =
            Sha256Digest::of_bytes(b"different-key-material");
        let tracker = current_tracker(&trust_snapshot);
        let report = verify_attestation_authority(
            envelope.clone(),
            expectation(&purpose, &envelope),
            &AttestationPolicy::default(),
            &EchoVerifier,
            AttestationTrustContext {
                evaluation_time_unix_s: 500,
                snapshot: &trust_snapshot,
                tracker: &tracker,
            },
        )
        .unwrap_err();
        assert!(report.violations.iter().any(|item| matches!(
            item,
            AttestationViolation::VerificationProviderError { reason, .. }
                if reason.contains("verification-key digest mismatch")
        )));
    }

    #[test]
    fn revoked_signer_cannot_mint_authority() {
        let envelope = envelope();
        let purpose = usage();
        let trust_snapshot = snapshot(KeyLifecycleStatus::Revoked);
        let tracker = current_tracker(&trust_snapshot);
        let report = verify_attestation_authority(
            envelope.clone(),
            expectation(&purpose, &envelope),
            &AttestationPolicy::default(),
            &EchoVerifier,
            AttestationTrustContext {
                evaluation_time_unix_s: 500,
                snapshot: &trust_snapshot,
                tracker: &tracker,
            },
        )
        .unwrap_err();
        assert!(report.violations.iter().any(|item| matches!(
            item,
            AttestationViolation::SignerRevoked { .. }
        )));
    }

    #[test]
    fn older_fresh_snapshot_cannot_mint_after_tracker_advance() {
        let envelope = envelope();
        let purpose = usage();
        let first = snapshot_with_sequence(KeyLifecycleStatus::Active, 1);
        let second = snapshot_with_sequence(KeyLifecycleStatus::Revoked, 2);
        let mut tracker = TrustSnapshotTracker::default();
        tracker.accept(&first).unwrap();
        tracker.accept(&second).unwrap();
        let report = verify_attestation_authority(
            envelope.clone(),
            expectation(&purpose, &envelope),
            &AttestationPolicy::default(),
            &EchoVerifier,
            AttestationTrustContext {
                evaluation_time_unix_s: 500,
                snapshot: &first,
                tracker: &tracker,
            },
        )
        .unwrap_err();
        assert!(report.violations.iter().any(|item| matches!(
            item,
            AttestationViolation::TrustSnapshotNotCurrent(
                TrustSnapshotCurrentnessError::SequenceMismatch {
                    accepted: 2,
                    presented: 1,
                }
            )
        )));
    }

    #[test]
    fn policy_changes_authority_identity_even_for_same_envelope() {
        let envelope = envelope();
        let purpose = usage();
        let trust_snapshot = snapshot(KeyLifecycleStatus::Active);
        let tracker = current_tracker(&trust_snapshot);
        let default_policy = AttestationPolicy::default();
        let other_policy = AttestationPolicy {
            maximum_signatures: 8,
            ..AttestationPolicy::default()
        };
        let left = verify_attestation_authority(
            envelope.clone(),
            expectation(&purpose, &envelope),
            &default_policy,
            &EchoVerifier,
            AttestationTrustContext {
                evaluation_time_unix_s: 500,
                snapshot: &trust_snapshot,
                tracker: &tracker,
            },
        )
        .unwrap();
        let right = verify_attestation_authority(
            envelope.clone(),
            expectation(&purpose, &envelope),
            &other_policy,
            &EchoVerifier,
            AttestationTrustContext {
                evaluation_time_unix_s: 500,
                snapshot: &trust_snapshot,
                tracker: &tracker,
            },
        )
        .unwrap();
        assert_eq!(left.attestation_sha256(), right.attestation_sha256());
        assert_ne!(left.policy_sha256(), right.policy_sha256());
        assert_ne!(left.authority_sha256(), right.authority_sha256());
    }
}
