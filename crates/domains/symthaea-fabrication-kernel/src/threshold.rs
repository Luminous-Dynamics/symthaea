// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generic lifecycle-governed multi-signature threshold ceremonies.
//!
//! This module does not claim an aggregated threshold-signature primitive. It
//! verifies an explicit quorum of independent detached signatures over one
//! canonical payload and preserves every signer identity as audit evidence.

use crate::attestation::{DetachedSignature, SignatureAlgorithm};
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::signer_compromise_tracker::SignerCompromiseTracker;
use crate::trust::{KeyEligibility, KeyUsage, TrustSnapshot, digest_trust_snapshot};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const THRESHOLD_APPROVAL_SCHEMA: &str = "symthaea.fabrication.threshold-approval.v1";
pub const SIGNED_THRESHOLD_APPROVAL_SCHEMA: &str =
    "symthaea.fabrication.signed-threshold-approval.v1";
pub const MAX_THRESHOLD_APPROVALS: usize = 64;
pub const MAX_THRESHOLD_SIGNATURE_BYTES: usize = 64 * 1024;
pub const MAX_THRESHOLD_PURPOSE_BYTES: usize = 256;
pub const MAX_THRESHOLD_KEY_ID_BYTES: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThresholdApproval {
    pub schema_version: String,
    pub purpose: String,
    pub payload_digest: Sha256Digest,
    pub issued_at_unix_s: u64,
    pub expires_at_unix_s: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedThresholdApproval {
    pub schema_version: String,
    pub approval: ThresholdApproval,
    pub approval_digest: Sha256Digest,
    pub signature: DetachedSignature,
}

pub trait ThresholdApprovalSigner {
    fn algorithm(&self) -> SignatureAlgorithm;
    fn key_id(&self) -> &str;
    fn sign_threshold_approval(&self, message: &[u8]) -> Result<Vec<u8>, String>;
}

pub trait ThresholdApprovalVerifier {
    fn verify_threshold_approval(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThresholdCeremonyError {
    UnsupportedSchema,
    InvalidPurpose,
    InvalidWindow,
    InvalidAlgorithm,
    InvalidKeyId,
    EmptySignature,
    SignatureTooLarge { actual: usize, maximum: usize },
    Signing(String),
    Encoding(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThresholdCeremonyPolicy {
    pub minimum_distinct_signers: usize,
    pub maximum_approvals: usize,
    pub require_algorithm_diversity: bool,
    pub required_algorithms: BTreeSet<SignatureAlgorithm>,
    pub allowed_key_ids: Option<BTreeSet<String>>,
    pub key_usage: KeyUsage,
}

impl Default for ThresholdCeremonyPolicy {
    fn default() -> Self {
        Self {
            minimum_distinct_signers: 2,
            maximum_approvals: MAX_THRESHOLD_APPROVALS,
            require_algorithm_diversity: true,
            required_algorithms: BTreeSet::new(),
            allowed_key_ids: None,
            key_usage: KeyUsage::ThresholdCeremony,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThresholdCeremonyViolation {
    InvalidPolicy,
    TooManyApprovals {
        actual: usize,
        maximum: usize,
    },
    UnsupportedSchema,
    InvalidApproval(ThresholdCeremonyError),
    PurposeMismatch(String),
    PayloadMismatch(String),
    ApprovalDigestMismatch(String),
    NotYetValid(String),
    Expired(String),
    DuplicateSigner {
        algorithm: SignatureAlgorithm,
        key_id: String,
    },
    KeyNotAllowed(String),
    SignerUnknown(String),
    SignerNotYetValid(String),
    SignerExpired(String),
    SignerRetired(String),
    SignerRevoked(String),
    SignerUsageNotAllowed(String),
    InvalidSignature(String),
    VerificationProviderError {
        key_id: String,
        reason: String,
    },
    InsufficientDistinctSigners {
        actual: usize,
        required: usize,
    },
    MissingAlgorithm(SignatureAlgorithm),
    MissingAlgorithmDiversity,
    TrustSnapshotInvalid(String),
    TrustSnapshotStale,
    CompromisedSigner(String),
}

#[derive(Debug, Clone)]
pub struct VerifiedThresholdCeremony {
    payload_digest: Sha256Digest,
    ceremony_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    purpose: String,
    signers: Vec<(SignatureAlgorithm, String)>,
}

impl VerifiedThresholdCeremony {
    pub fn payload_digest(&self) -> Sha256Digest {
        self.payload_digest
    }
    pub fn ceremony_digest(&self) -> Sha256Digest {
        self.ceremony_digest
    }
    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }
    pub fn purpose(&self) -> &str {
        &self.purpose
    }
    pub fn signers(&self) -> &[(SignatureAlgorithm, String)] {
        &self.signers
    }
}

impl ThresholdApproval {
    pub fn validate(&self) -> Result<(), ThresholdCeremonyError> {
        if self.schema_version != THRESHOLD_APPROVAL_SCHEMA {
            return Err(ThresholdCeremonyError::UnsupportedSchema);
        }
        validate_purpose(&self.purpose)?;
        if self.issued_at_unix_s >= self.expires_at_unix_s {
            return Err(ThresholdCeremonyError::InvalidWindow);
        }
        Ok(())
    }
}

pub fn canonical_threshold_approval_bytes(
    approval: &ThresholdApproval,
) -> Result<Vec<u8>, ThresholdCeremonyError> {
    approval.validate()?;
    serde_json::to_vec(approval)
        .map_err(|error| ThresholdCeremonyError::Encoding(error.to_string()))
}

pub fn digest_threshold_approval(
    approval: &ThresholdApproval,
) -> Result<Sha256Digest, ThresholdCeremonyError> {
    let bytes = canonical_threshold_approval_bytes(approval)?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.threshold-approval-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn sign_threshold_approval(
    purpose: impl Into<String>,
    payload_digest: Sha256Digest,
    issued_at_unix_s: u64,
    expires_at_unix_s: u64,
    signer: &dyn ThresholdApprovalSigner,
) -> Result<SignedThresholdApproval, ThresholdCeremonyError> {
    let approval = ThresholdApproval {
        schema_version: THRESHOLD_APPROVAL_SCHEMA.into(),
        purpose: purpose.into(),
        payload_digest,
        issued_at_unix_s,
        expires_at_unix_s,
    };
    approval.validate()?;
    let algorithm = signer.algorithm();
    if !algorithm.is_canonical() {
        return Err(ThresholdCeremonyError::InvalidAlgorithm);
    }
    let key_id = signer.key_id();
    if key_id.trim().is_empty()
        || key_id != key_id.trim()
        || key_id.len() > MAX_THRESHOLD_KEY_ID_BYTES
    {
        return Err(ThresholdCeremonyError::InvalidKeyId);
    }
    let approval_digest = digest_threshold_approval(&approval)?;
    let message = threshold_signature_message(approval_digest);
    let signature = signer
        .sign_threshold_approval(&message)
        .map_err(ThresholdCeremonyError::Signing)?;
    if signature.is_empty() {
        return Err(ThresholdCeremonyError::EmptySignature);
    }
    if signature.len() > MAX_THRESHOLD_SIGNATURE_BYTES {
        return Err(ThresholdCeremonyError::SignatureTooLarge {
            actual: signature.len(),
            maximum: MAX_THRESHOLD_SIGNATURE_BYTES,
        });
    }
    Ok(SignedThresholdApproval {
        schema_version: SIGNED_THRESHOLD_APPROVAL_SCHEMA.into(),
        approval,
        approval_digest,
        signature: DetachedSignature {
            algorithm,
            key_id: key_id.to_string(),
            signature,
        },
    })
}

pub fn verify_threshold_ceremony(
    purpose: &str,
    payload_digest: Sha256Digest,
    approvals: &[SignedThresholdApproval],
    policy: &ThresholdCeremonyPolicy,
    trust_snapshot: &TrustSnapshot,
    now_unix_s: u64,
    verifier: &dyn ThresholdApprovalVerifier,
) -> Result<VerifiedThresholdCeremony, Vec<ThresholdCeremonyViolation>> {
    let mut violations = Vec::new();
    if validate_policy(policy).is_err() {
        violations.push(ThresholdCeremonyViolation::InvalidPolicy);
    }
    if validate_purpose(purpose).is_err() {
        violations.push(ThresholdCeremonyViolation::InvalidApproval(
            ThresholdCeremonyError::InvalidPurpose,
        ));
    }
    if approvals.len() > policy.maximum_approvals {
        violations.push(ThresholdCeremonyViolation::TooManyApprovals {
            actual: approvals.len(),
            maximum: policy.maximum_approvals,
        });
    }
    if let Err(error) = trust_snapshot.validate() {
        violations.push(ThresholdCeremonyViolation::TrustSnapshotInvalid(format!(
            "{error:?}"
        )));
    }
    if !trust_snapshot.is_fresh_at(now_unix_s) {
        violations.push(ThresholdCeremonyViolation::TrustSnapshotStale);
    }

    let mut identities = BTreeSet::new();
    let mut algorithms = BTreeSet::new();
    let mut valid = Vec::new();
    let mut approved_digests = Vec::new();
    for signed in approvals.iter().take(policy.maximum_approvals) {
        if signed.schema_version != SIGNED_THRESHOLD_APPROVAL_SCHEMA {
            violations.push(ThresholdCeremonyViolation::UnsupportedSchema);
            continue;
        }
        if let Err(error) = signed.approval.validate() {
            violations.push(ThresholdCeremonyViolation::InvalidApproval(error));
            continue;
        }
        let key_id = signed.signature.key_id.clone();
        let algorithm = signed.signature.algorithm.clone();
        if !algorithm.is_canonical() {
            violations.push(ThresholdCeremonyViolation::InvalidApproval(
                ThresholdCeremonyError::InvalidAlgorithm,
            ));
            continue;
        }
        if key_id.trim().is_empty()
            || key_id != key_id.trim()
            || key_id.len() > MAX_THRESHOLD_KEY_ID_BYTES
        {
            violations.push(ThresholdCeremonyViolation::InvalidApproval(
                ThresholdCeremonyError::InvalidKeyId,
            ));
            continue;
        }
        if signed.signature.signature.is_empty() {
            violations.push(ThresholdCeremonyViolation::InvalidApproval(
                ThresholdCeremonyError::EmptySignature,
            ));
            continue;
        }
        if signed.signature.signature.len() > MAX_THRESHOLD_SIGNATURE_BYTES {
            violations.push(ThresholdCeremonyViolation::InvalidApproval(
                ThresholdCeremonyError::SignatureTooLarge {
                    actual: signed.signature.signature.len(),
                    maximum: MAX_THRESHOLD_SIGNATURE_BYTES,
                },
            ));
            continue;
        }
        if signed.approval.purpose != purpose {
            violations.push(ThresholdCeremonyViolation::PurposeMismatch(key_id));
            continue;
        }
        if signed.approval.payload_digest != payload_digest {
            violations.push(ThresholdCeremonyViolation::PayloadMismatch(key_id));
            continue;
        }
        let expected_digest = match digest_threshold_approval(&signed.approval) {
            Ok(digest) => digest,
            Err(error) => {
                violations.push(ThresholdCeremonyViolation::InvalidApproval(error));
                continue;
            }
        };
        if expected_digest != signed.approval_digest {
            violations.push(ThresholdCeremonyViolation::ApprovalDigestMismatch(key_id));
            continue;
        }
        if now_unix_s < signed.approval.issued_at_unix_s {
            violations.push(ThresholdCeremonyViolation::NotYetValid(key_id));
            continue;
        }
        if now_unix_s >= signed.approval.expires_at_unix_s {
            violations.push(ThresholdCeremonyViolation::Expired(key_id));
            continue;
        }
        if !identities.insert((algorithm.clone(), key_id.clone())) {
            violations.push(ThresholdCeremonyViolation::DuplicateSigner { algorithm, key_id });
            continue;
        }
        if policy
            .allowed_key_ids
            .as_ref()
            .is_some_and(|allowed| !allowed.contains(&key_id))
        {
            violations.push(ThresholdCeremonyViolation::KeyNotAllowed(key_id));
            continue;
        }
        match trust_snapshot.key_eligibility(&algorithm, &key_id, policy.key_usage, now_unix_s) {
            KeyEligibility::Eligible => {}
            KeyEligibility::Unknown => {
                violations.push(ThresholdCeremonyViolation::SignerUnknown(key_id));
                continue;
            }
            KeyEligibility::NotYetValid => {
                violations.push(ThresholdCeremonyViolation::SignerNotYetValid(key_id));
                continue;
            }
            KeyEligibility::Expired => {
                violations.push(ThresholdCeremonyViolation::SignerExpired(key_id));
                continue;
            }
            KeyEligibility::Retired => {
                violations.push(ThresholdCeremonyViolation::SignerRetired(key_id));
                continue;
            }
            KeyEligibility::Revoked => {
                violations.push(ThresholdCeremonyViolation::SignerRevoked(key_id));
                continue;
            }
            KeyEligibility::UsageNotAllowed => {
                violations.push(ThresholdCeremonyViolation::SignerUsageNotAllowed(key_id));
                continue;
            }
        }
        let message = threshold_signature_message(signed.approval_digest);
        match verifier.verify_threshold_approval(
            &algorithm,
            &key_id,
            &message,
            &signed.signature.signature,
        ) {
            Ok(true) => {
                algorithms.insert(algorithm.clone());
                valid.push((algorithm, key_id));
                approved_digests.push(signed.approval_digest);
            }
            Ok(false) => violations.push(ThresholdCeremonyViolation::InvalidSignature(key_id)),
            Err(reason) => violations
                .push(ThresholdCeremonyViolation::VerificationProviderError { key_id, reason }),
        }
    }

    if valid.len() < policy.minimum_distinct_signers {
        violations.push(ThresholdCeremonyViolation::InsufficientDistinctSigners {
            actual: valid.len(),
            required: policy.minimum_distinct_signers,
        });
    }
    for required in &policy.required_algorithms {
        if !algorithms.contains(required) {
            violations.push(ThresholdCeremonyViolation::MissingAlgorithm(
                required.clone(),
            ));
        }
    }
    if policy.require_algorithm_diversity && algorithms.len() < 2 {
        violations.push(ThresholdCeremonyViolation::MissingAlgorithmDiversity);
    }
    if !violations.is_empty() {
        return Err(violations);
    }
    valid.sort();
    approved_digests.sort();
    let trust_snapshot_digest = digest_trust_snapshot(trust_snapshot).map_err(|error| {
        vec![ThresholdCeremonyViolation::TrustSnapshotInvalid(format!(
            "{error:?}"
        ))]
    })?;
    let ceremony_digest = digest_verified_ceremony(
        purpose,
        payload_digest,
        trust_snapshot_digest,
        &valid,
        &approved_digests,
    );
    Ok(VerifiedThresholdCeremony {
        payload_digest,
        ceremony_digest,
        trust_snapshot_digest,
        purpose: purpose.to_string(),
        signers: valid,
    })
}

pub fn verify_threshold_ceremony_with_containment(
    purpose: &str,
    payload_digest: Sha256Digest,
    approvals: &[SignedThresholdApproval],
    policy: &ThresholdCeremonyPolicy,
    trust_snapshot: &TrustSnapshot,
    compromise_tracker: &SignerCompromiseTracker,
    now_unix_s: u64,
    verifier: &dyn ThresholdApprovalVerifier,
) -> Result<VerifiedThresholdCeremony, Vec<ThresholdCeremonyViolation>> {
    let ceremony = verify_threshold_ceremony(
        purpose,
        payload_digest,
        approvals,
        policy,
        trust_snapshot,
        now_unix_s,
        verifier,
    )?;
    let compromised: Vec<_> = ceremony
        .signers()
        .iter()
        .filter_map(|(algorithm, key_id)| {
            compromise_tracker
                .is_compromised_at(algorithm, key_id, policy.key_usage, now_unix_s)
                .then_some(ThresholdCeremonyViolation::CompromisedSigner(
                    key_id.clone(),
                ))
        })
        .collect();
    if compromised.is_empty() {
        Ok(ceremony)
    } else {
        Err(compromised)
    }
}

fn validate_policy(policy: &ThresholdCeremonyPolicy) -> Result<(), ()> {
    if policy.minimum_distinct_signers == 0
        || policy.maximum_approvals == 0
        || policy.minimum_distinct_signers > policy.maximum_approvals
        || policy.maximum_approvals > MAX_THRESHOLD_APPROVALS
        || policy
            .required_algorithms
            .iter()
            .any(|algorithm| !algorithm.is_canonical())
        || policy
            .allowed_key_ids
            .as_ref()
            .is_some_and(|ids| ids.iter().any(|id| id.trim().is_empty() || id != id.trim()))
    {
        return Err(());
    }
    Ok(())
}

fn validate_purpose(value: &str) -> Result<(), ThresholdCeremonyError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_THRESHOLD_PURPOSE_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(ThresholdCeremonyError::InvalidPurpose);
    }
    Ok(())
}

fn threshold_signature_message(digest: Sha256Digest) -> Vec<u8> {
    let mut message = b"symthaea.fabrication.threshold-approval-signature.v1\0".to_vec();
    message.extend_from_slice(&digest.0);
    message
}

fn digest_verified_ceremony(
    purpose: &str,
    payload_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    signers: &[(SignatureAlgorithm, String)],
    approval_digests: &[Sha256Digest],
) -> Sha256Digest {
    let bytes = serde_json::to_vec(&(
        purpose,
        payload_digest,
        trust_snapshot_digest,
        signers,
        approval_digests,
    ))
    .expect("verified threshold ceremony is serializable");
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.verified-threshold-ceremony.v1\0");
    hasher.update(&bytes);
    hasher.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;
    use crate::trust::{KeyLifecycleStatus, KeyTrustRecord};

    struct Provider {
        algorithm: SignatureAlgorithm,
        key_id: &'static str,
    }
    impl ThresholdApprovalSigner for Provider {
        fn algorithm(&self) -> SignatureAlgorithm {
            self.algorithm.clone()
        }
        fn key_id(&self) -> &str {
            self.key_id
        }
        fn sign_threshold_approval(&self, message: &[u8]) -> Result<Vec<u8>, String> {
            Ok(sha256(message).0.to_vec())
        }
    }
    impl ThresholdApprovalVerifier for Provider {
        fn verify_threshold_approval(
            &self,
            _: &SignatureAlgorithm,
            _: &str,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            Ok(signature == sha256(message).0.as_slice())
        }
    }

    fn trust() -> TrustSnapshot {
        TrustSnapshot::new(
            1,
            100,
            1_000,
            vec![
                KeyTrustRecord {
                    algorithm: SignatureAlgorithm::Ed25519,
                    key_id: "a".into(),
                    not_before_unix_s: 100,
                    not_after_unix_s: None,
                    status: KeyLifecycleStatus::Active,
                    usages: BTreeSet::from([KeyUsage::ThresholdCeremony]),
                },
                KeyTrustRecord {
                    algorithm: SignatureAlgorithm::MlDsa65,
                    key_id: "b".into(),
                    not_before_unix_s: 100,
                    not_after_unix_s: None,
                    status: KeyLifecycleStatus::Active,
                    usages: BTreeSet::from([KeyUsage::ThresholdCeremony]),
                },
            ],
        )
        .unwrap()
    }

    #[test]
    fn diverse_quorum_verifies_exact_payload() {
        let payload = sha256(b"release");
        let a = Provider {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: "a",
        };
        let b = Provider {
            algorithm: SignatureAlgorithm::MlDsa65,
            key_id: "b",
        };
        let approvals = vec![
            sign_threshold_approval("release-promotion", payload, 200, 400, &a).unwrap(),
            sign_threshold_approval("release-promotion", payload, 200, 400, &b).unwrap(),
        ];
        let verified = verify_threshold_ceremony(
            "release-promotion",
            payload,
            &approvals,
            &ThresholdCeremonyPolicy::default(),
            &trust(),
            250,
            &a,
        )
        .unwrap();
        assert_eq!(verified.payload_digest(), payload);
        assert_eq!(verified.signers().len(), 2);
    }

    #[test]
    fn payload_substitution_is_rejected() {
        let a = Provider {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: "a",
        };
        let signed =
            sign_threshold_approval("release-promotion", sha256(b"one"), 200, 400, &a).unwrap();
        let violations = verify_threshold_ceremony(
            "release-promotion",
            sha256(b"two"),
            &[signed],
            &ThresholdCeremonyPolicy {
                minimum_distinct_signers: 1,
                require_algorithm_diversity: false,
                ..ThresholdCeremonyPolicy::default()
            },
            &trust(),
            250,
            &a,
        )
        .unwrap_err();
        assert!(
            violations
                .iter()
                .any(|v| matches!(v, ThresholdCeremonyViolation::PayloadMismatch(_)))
        );
    }
}
