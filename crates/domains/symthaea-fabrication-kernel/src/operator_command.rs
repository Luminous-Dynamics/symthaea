// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Signed, bounded operator commands for one exact governed print execution.
//!
//! Commands are data, not authority, until their signatures and signer lifecycle
//! are checked against a fresh trust snapshot. The resulting capability retains
//! the exact command digest and trust evidence used during verification.

use crate::attestation::{DetachedSignature, SignatureAlgorithm};
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::trust::{KeyEligibility, KeyUsage, TrustSnapshot, digest_trust_snapshot};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const OPERATOR_COMMAND_SCHEMA: &str = "symthaea.fabrication.operator-command.v1";
pub const SIGNED_OPERATOR_COMMAND_SCHEMA: &str = "symthaea.fabrication.signed-operator-command.v1";
pub const MAX_OPERATOR_COMMAND_SIGNATURES: usize = 16;
pub const MAX_OPERATOR_COMMAND_SIGNATURE_BYTES: usize = 64 * 1024;
pub const MAX_OPERATOR_REASON_BYTES: usize = 2048;
pub const MAX_OPERATOR_IDENTIFIER_BYTES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OperatorCommandKind {
    Pause,
    Resume,
    Cancel,
    EmergencyStop,
}

impl OperatorCommandKind {
    pub fn severity(self) -> u8 {
        match self {
            Self::Resume => 0,
            Self::Pause => 1,
            Self::Cancel => 2,
            Self::EmergencyStop => 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorCommand {
    pub schema_version: String,
    pub manifest_digest: Sha256Digest,
    pub machine_id: String,
    pub session_digest: Sha256Digest,
    pub printer_job_id: String,
    pub command_sequence: u64,
    pub issued_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
    pub kind: OperatorCommandKind,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedOperatorCommand {
    pub schema_version: String,
    pub command: OperatorCommand,
    pub command_digest: Sha256Digest,
    pub signatures: Vec<DetachedSignature>,
}

pub trait OperatorCommandSigner {
    fn algorithm(&self) -> SignatureAlgorithm;
    fn key_id(&self) -> &str;
    fn sign_operator_command(&self, message: &[u8]) -> Result<Vec<u8>, String>;
}

pub trait OperatorCommandVerifier {
    fn verify_operator_command(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperatorCommandError {
    UnsupportedSchema,
    UnsupportedSignedSchema,
    InvalidWindow,
    CommandSequenceZero,
    EmptyIdentifier(&'static str),
    NonCanonicalIdentifier(&'static str),
    IdentifierTooLong {
        field: &'static str,
        actual: usize,
        maximum: usize,
    },
    EmptyReason,
    ReasonTooLong {
        actual: usize,
        maximum: usize,
    },
    InvalidAlgorithm,
    EmptySignature,
    SignatureTooLarge {
        actual: usize,
        maximum: usize,
    },
    TooManySignatures {
        actual: usize,
        maximum: usize,
    },
    DuplicateSigner {
        algorithm: SignatureAlgorithm,
        key_id: String,
    },
    Signing {
        key_id: String,
        reason: String,
    },
    Encoding(String),
}

impl OperatorCommand {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        manifest_digest: Sha256Digest,
        machine_id: impl Into<String>,
        session_digest: Sha256Digest,
        printer_job_id: impl Into<String>,
        command_sequence: u64,
        issued_at_unix_ms: u64,
        expires_at_unix_ms: u64,
        kind: OperatorCommandKind,
        reason: impl Into<String>,
    ) -> Result<Self, OperatorCommandError> {
        let command = Self {
            schema_version: OPERATOR_COMMAND_SCHEMA.into(),
            manifest_digest,
            machine_id: machine_id.into(),
            session_digest,
            printer_job_id: printer_job_id.into(),
            command_sequence,
            issued_at_unix_ms,
            expires_at_unix_ms,
            kind,
            reason: reason.into(),
        };
        command.validate()?;
        Ok(command)
    }

    pub fn validate(&self) -> Result<(), OperatorCommandError> {
        if self.schema_version != OPERATOR_COMMAND_SCHEMA {
            return Err(OperatorCommandError::UnsupportedSchema);
        }
        if self.command_sequence == 0 {
            return Err(OperatorCommandError::CommandSequenceZero);
        }
        if self.issued_at_unix_ms >= self.expires_at_unix_ms {
            return Err(OperatorCommandError::InvalidWindow);
        }
        validate_identifier("machine_id", &self.machine_id)?;
        validate_identifier("printer_job_id", &self.printer_job_id)?;
        if self.reason.trim().is_empty() {
            return Err(OperatorCommandError::EmptyReason);
        }
        if self.reason != self.reason.trim() {
            return Err(OperatorCommandError::NonCanonicalIdentifier("reason"));
        }
        if self.reason.len() > MAX_OPERATOR_REASON_BYTES {
            return Err(OperatorCommandError::ReasonTooLong {
                actual: self.reason.len(),
                maximum: MAX_OPERATOR_REASON_BYTES,
            });
        }
        Ok(())
    }

    pub fn is_fresh_at(&self, now_unix_ms: u64) -> bool {
        now_unix_ms >= self.issued_at_unix_ms && now_unix_ms < self.expires_at_unix_ms
    }
}

pub fn canonical_operator_command_bytes(
    command: &OperatorCommand,
) -> Result<Vec<u8>, OperatorCommandError> {
    command.validate()?;
    serde_json::to_vec(command).map_err(|error| OperatorCommandError::Encoding(error.to_string()))
}

pub fn digest_operator_command(
    command: &OperatorCommand,
) -> Result<Sha256Digest, OperatorCommandError> {
    let bytes = canonical_operator_command_bytes(command)?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.operator-command-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn sign_operator_command(
    command: OperatorCommand,
    signers: &[&dyn OperatorCommandSigner],
) -> Result<SignedOperatorCommand, OperatorCommandError> {
    command.validate()?;
    if signers.len() > MAX_OPERATOR_COMMAND_SIGNATURES {
        return Err(OperatorCommandError::TooManySignatures {
            actual: signers.len(),
            maximum: MAX_OPERATOR_COMMAND_SIGNATURES,
        });
    }
    let command_digest = digest_operator_command(&command)?;
    let message = operator_command_signature_message(command_digest);
    let mut identities = BTreeSet::new();
    let mut signatures = Vec::with_capacity(signers.len());
    for signer in signers {
        let algorithm = signer.algorithm();
        if !algorithm.is_canonical() {
            return Err(OperatorCommandError::InvalidAlgorithm);
        }
        let key_id = signer.key_id();
        validate_identifier("key_id", key_id)?;
        if !identities.insert((algorithm.clone(), key_id.to_string())) {
            return Err(OperatorCommandError::DuplicateSigner {
                algorithm,
                key_id: key_id.to_string(),
            });
        }
        let signature = signer.sign_operator_command(&message).map_err(|reason| {
            OperatorCommandError::Signing {
                key_id: key_id.to_string(),
                reason,
            }
        })?;
        if signature.is_empty() {
            return Err(OperatorCommandError::EmptySignature);
        }
        if signature.len() > MAX_OPERATOR_COMMAND_SIGNATURE_BYTES {
            return Err(OperatorCommandError::SignatureTooLarge {
                actual: signature.len(),
                maximum: MAX_OPERATOR_COMMAND_SIGNATURE_BYTES,
            });
        }
        signatures.push(DetachedSignature {
            algorithm,
            key_id: key_id.to_string(),
            signature,
        });
    }
    Ok(SignedOperatorCommand {
        schema_version: SIGNED_OPERATOR_COMMAND_SCHEMA.into(),
        command,
        command_digest,
        signatures,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorCommandPolicy {
    pub minimum_valid_signatures: usize,
    pub maximum_signatures: usize,
    pub require_algorithm_diversity: bool,
    pub allowed_key_ids: Option<BTreeSet<String>>,
}

impl Default for OperatorCommandPolicy {
    fn default() -> Self {
        Self {
            minimum_valid_signatures: 1,
            maximum_signatures: MAX_OPERATOR_COMMAND_SIGNATURES,
            require_algorithm_diversity: false,
            allowed_key_ids: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperatorCommandViolation {
    InvalidPolicy,
    UnsupportedSchema,
    InvalidCommand(OperatorCommandError),
    DigestMismatch,
    NotYetValid,
    Expired,
    ManifestMismatch,
    MachineMismatch,
    SessionMismatch,
    PrinterJobMismatch,
    TooManySignatures {
        actual: usize,
        maximum: usize,
    },
    DuplicateSigner {
        algorithm: SignatureAlgorithm,
        key_id: String,
    },
    SignatureTooLarge {
        key_id: String,
        actual: usize,
        maximum: usize,
    },
    KeyNotAllowed(String),
    SignerUnknown(String),
    SignerNotYetValid(String),
    SignerExpired(String),
    SignerRetired(String),
    SignerRevoked(String),
    SignerUsageNotAllowed(String),
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
    MissingAlgorithmDiversity,
    TrustSnapshotInvalid(String),
    TrustSnapshotStale,
}

#[derive(Debug, Clone, Copy)]
pub struct OperatorCommandExpectation<'a> {
    pub manifest_digest: Sha256Digest,
    pub machine_id: &'a str,
    pub session_digest: Sha256Digest,
    pub printer_job_id: &'a str,
    pub now_unix_ms: u64,
    pub trust_snapshot: &'a TrustSnapshot,
}

#[derive(Debug, Clone)]
pub struct VerifiedOperatorCommand {
    signed: SignedOperatorCommand,
    valid_signers: Vec<(SignatureAlgorithm, String)>,
    trust_snapshot_digest: Sha256Digest,
}

impl VerifiedOperatorCommand {
    pub fn command(&self) -> &OperatorCommand {
        &self.signed.command
    }
    pub fn command_digest(&self) -> Sha256Digest {
        self.signed.command_digest
    }
    pub fn valid_signers(&self) -> &[(SignatureAlgorithm, String)] {
        &self.valid_signers
    }
    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }
}

pub fn verify_operator_command(
    signed: SignedOperatorCommand,
    policy: &OperatorCommandPolicy,
    expectation: OperatorCommandExpectation<'_>,
    verifier: &dyn OperatorCommandVerifier,
) -> Result<VerifiedOperatorCommand, Vec<OperatorCommandViolation>> {
    let mut violations = Vec::new();
    if policy.minimum_valid_signatures == 0
        || policy.maximum_signatures == 0
        || policy.minimum_valid_signatures > policy.maximum_signatures
    {
        violations.push(OperatorCommandViolation::InvalidPolicy);
    }
    if signed.schema_version != SIGNED_OPERATOR_COMMAND_SCHEMA {
        violations.push(OperatorCommandViolation::UnsupportedSchema);
    }
    if let Err(error) = signed.command.validate() {
        violations.push(OperatorCommandViolation::InvalidCommand(error));
    }
    match digest_operator_command(&signed.command) {
        Ok(digest) if digest != signed.command_digest => {
            violations.push(OperatorCommandViolation::DigestMismatch)
        }
        Err(error) => violations.push(OperatorCommandViolation::InvalidCommand(error)),
        Ok(_) => {}
    }
    if expectation.now_unix_ms < signed.command.issued_at_unix_ms {
        violations.push(OperatorCommandViolation::NotYetValid);
    }
    if expectation.now_unix_ms >= signed.command.expires_at_unix_ms {
        violations.push(OperatorCommandViolation::Expired);
    }
    if signed.command.manifest_digest != expectation.manifest_digest {
        violations.push(OperatorCommandViolation::ManifestMismatch);
    }
    if signed.command.machine_id != expectation.machine_id {
        violations.push(OperatorCommandViolation::MachineMismatch);
    }
    if signed.command.session_digest != expectation.session_digest {
        violations.push(OperatorCommandViolation::SessionMismatch);
    }
    if signed.command.printer_job_id != expectation.printer_job_id {
        violations.push(OperatorCommandViolation::PrinterJobMismatch);
    }
    if signed.signatures.len() > policy.maximum_signatures {
        violations.push(OperatorCommandViolation::TooManySignatures {
            actual: signed.signatures.len(),
            maximum: policy.maximum_signatures,
        });
    }
    if expectation.trust_snapshot.validate().is_err() {
        violations.push(OperatorCommandViolation::TrustSnapshotInvalid(
            "snapshot validation failed".into(),
        ));
    }
    let now_unix_s = expectation.now_unix_ms / 1_000;
    if !expectation.trust_snapshot.is_fresh_at(now_unix_s) {
        violations.push(OperatorCommandViolation::TrustSnapshotStale);
    }

    let message = operator_command_signature_message(signed.command_digest);
    let mut seen = BTreeSet::new();
    let mut valid_signers = Vec::new();
    for signature in &signed.signatures {
        if signature.signature.len() > MAX_OPERATOR_COMMAND_SIGNATURE_BYTES {
            violations.push(OperatorCommandViolation::SignatureTooLarge {
                key_id: signature.key_id.clone(),
                actual: signature.signature.len(),
                maximum: MAX_OPERATOR_COMMAND_SIGNATURE_BYTES,
            });
            continue;
        }
        let identity = (signature.algorithm.clone(), signature.key_id.clone());
        if !seen.insert(identity.clone()) {
            violations.push(OperatorCommandViolation::DuplicateSigner {
                algorithm: identity.0,
                key_id: identity.1,
            });
            continue;
        }
        if policy
            .allowed_key_ids
            .as_ref()
            .is_some_and(|allowed| !allowed.contains(&signature.key_id))
        {
            violations.push(OperatorCommandViolation::KeyNotAllowed(
                signature.key_id.clone(),
            ));
            continue;
        }
        match expectation.trust_snapshot.key_eligibility(
            &signature.algorithm,
            &signature.key_id,
            KeyUsage::OperatorCommand,
            now_unix_s,
        ) {
            KeyEligibility::Eligible => {}
            KeyEligibility::Unknown => {
                violations.push(OperatorCommandViolation::SignerUnknown(
                    signature.key_id.clone(),
                ));
                continue;
            }
            KeyEligibility::NotYetValid => {
                violations.push(OperatorCommandViolation::SignerNotYetValid(
                    signature.key_id.clone(),
                ));
                continue;
            }
            KeyEligibility::Expired => {
                violations.push(OperatorCommandViolation::SignerExpired(
                    signature.key_id.clone(),
                ));
                continue;
            }
            KeyEligibility::Retired => {
                violations.push(OperatorCommandViolation::SignerRetired(
                    signature.key_id.clone(),
                ));
                continue;
            }
            KeyEligibility::Revoked => {
                violations.push(OperatorCommandViolation::SignerRevoked(
                    signature.key_id.clone(),
                ));
                continue;
            }
            KeyEligibility::UsageNotAllowed => {
                violations.push(OperatorCommandViolation::SignerUsageNotAllowed(
                    signature.key_id.clone(),
                ));
                continue;
            }
        }
        match verifier.verify_operator_command(
            &signature.algorithm,
            &signature.key_id,
            &message,
            &signature.signature,
        ) {
            Ok(true) => valid_signers.push(identity),
            Ok(false) => violations.push(OperatorCommandViolation::InvalidSignature {
                algorithm: signature.algorithm.clone(),
                key_id: signature.key_id.clone(),
            }),
            Err(reason) => violations.push(OperatorCommandViolation::VerificationProviderError {
                key_id: signature.key_id.clone(),
                reason,
            }),
        }
    }
    if valid_signers.len() < policy.minimum_valid_signatures {
        violations.push(OperatorCommandViolation::InsufficientValidSignatures {
            actual: valid_signers.len(),
            required: policy.minimum_valid_signatures,
        });
    }
    if policy.require_algorithm_diversity
        && valid_signers
            .iter()
            .map(|(algorithm, _)| algorithm)
            .collect::<BTreeSet<_>>()
            .len()
            < 2
    {
        violations.push(OperatorCommandViolation::MissingAlgorithmDiversity);
    }
    let trust_snapshot_digest =
        digest_trust_snapshot(expectation.trust_snapshot).map_err(|error| {
            vec![OperatorCommandViolation::TrustSnapshotInvalid(format!(
                "{error:?}"
            ))]
        })?;
    if !violations.is_empty() {
        return Err(violations);
    }
    Ok(VerifiedOperatorCommand {
        signed,
        valid_signers,
        trust_snapshot_digest,
    })
}

fn operator_command_signature_message(command_digest: Sha256Digest) -> Vec<u8> {
    let mut message = b"symthaea.fabrication.operator-command-signature.v1\0".to_vec();
    message.extend_from_slice(&command_digest.0);
    message
}

fn validate_identifier(field: &'static str, value: &str) -> Result<(), OperatorCommandError> {
    if value.trim().is_empty() {
        return Err(OperatorCommandError::EmptyIdentifier(field));
    }
    if value != value.trim() {
        return Err(OperatorCommandError::NonCanonicalIdentifier(field));
    }
    if value.len() > MAX_OPERATOR_IDENTIFIER_BYTES {
        return Err(OperatorCommandError::IdentifierTooLong {
            field,
            actual: value.len(),
            maximum: MAX_OPERATOR_IDENTIFIER_BYTES,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;
    use crate::trust::{KeyLifecycleStatus, KeyTrustRecord};

    struct Provider;

    impl OperatorCommandSigner for Provider {
        fn algorithm(&self) -> SignatureAlgorithm {
            SignatureAlgorithm::Ed25519
        }
        fn key_id(&self) -> &str {
            "operator-1"
        }
        fn sign_operator_command(&self, message: &[u8]) -> Result<Vec<u8>, String> {
            Ok(sha256(message).0.to_vec())
        }
    }

    impl OperatorCommandVerifier for Provider {
        fn verify_operator_command(
            &self,
            _algorithm: &SignatureAlgorithm,
            _key_id: &str,
            message: &[u8],
            signature: &[u8],
        ) -> Result<bool, String> {
            Ok(signature == sha256(message).0.as_slice())
        }
    }

    fn snapshot() -> TrustSnapshot {
        TrustSnapshot::new(
            1,
            100,
            1_000,
            vec![KeyTrustRecord {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "operator-1".into(),
                not_before_unix_s: 100,
                not_after_unix_s: None,
                status: KeyLifecycleStatus::Active,
                usages: BTreeSet::from([KeyUsage::OperatorCommand]),
            }],
        )
        .unwrap()
    }

    fn command() -> OperatorCommand {
        OperatorCommand::new(
            sha256(b"manifest"),
            "machine-1",
            sha256(b"session"),
            "job-1",
            1,
            500_000,
            510_000,
            OperatorCommandKind::Pause,
            "operator requested inspection",
        )
        .unwrap()
    }

    #[test]
    fn exact_command_context_verifies() {
        let signed = sign_operator_command(command(), &[&Provider]).unwrap();
        let snapshot = snapshot();
        let verified = verify_operator_command(
            signed,
            &OperatorCommandPolicy::default(),
            OperatorCommandExpectation {
                manifest_digest: sha256(b"manifest"),
                machine_id: "machine-1",
                session_digest: sha256(b"session"),
                printer_job_id: "job-1",
                now_unix_ms: 501_000,
                trust_snapshot: &snapshot,
            },
            &Provider,
        )
        .unwrap();
        assert_eq!(verified.command().kind, OperatorCommandKind::Pause);
    }

    #[test]
    fn cross_job_substitution_is_rejected() {
        let signed = sign_operator_command(command(), &[&Provider]).unwrap();
        let snapshot = snapshot();
        let violations = verify_operator_command(
            signed,
            &OperatorCommandPolicy::default(),
            OperatorCommandExpectation {
                manifest_digest: sha256(b"manifest"),
                machine_id: "machine-1",
                session_digest: sha256(b"session"),
                printer_job_id: "job-2",
                now_unix_ms: 501_000,
                trust_snapshot: &snapshot,
            },
            &Provider,
        )
        .unwrap_err();
        assert!(violations.contains(&OperatorCommandViolation::PrinterJobMismatch));
    }
}
