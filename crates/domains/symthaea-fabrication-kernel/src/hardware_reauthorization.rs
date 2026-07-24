// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Machine-signed hardware reauthorization after a software upgrade.

use crate::attestation::{DetachedSignature, SignatureAlgorithm};
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::trust::{KeyEligibility, KeyUsage, TrustSnapshot, digest_trust_snapshot};
use crate::upgrade_handoff::AuthorizedUpgradeHandoff;
use crate::upgrade_probation::AuthorizedUpgradeProbationClearance;
use serde::{Deserialize, Serialize};

pub const HARDWARE_REAUTHORIZATION_SCHEMA: &str =
    "symthaea.fabrication.hardware-reauthorization.v1";
pub const MAX_HARDWARE_ID_BYTES: usize = 256;
pub const MAX_HARDWARE_SIGNATURE_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareReauthorizationStatement {
    pub schema_version: String,
    pub reauthorization_sequence: u64,
    pub handoff_digest: Sha256Digest,
    pub successor_source_tree_digest: Sha256Digest,
    pub successor_executable_digest: Sha256Digest,
    pub machine_id: String,
    pub hardware_identity_digest: Sha256Digest,
    pub machine_profile_digest: Sha256Digest,
    pub firmware_digest: Sha256Digest,
    pub calibration_digest: Sha256Digest,
    pub capability_digest: Sha256Digest,
    pub issued_at_unix_s: u64,
    pub expires_at_unix_s: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedHardwareReauthorization {
    pub statement: HardwareReauthorizationStatement,
    pub statement_digest: Sha256Digest,
    pub signature: DetachedSignature,
}

pub trait HardwareReauthorizationSigner {
    fn algorithm(&self) -> SignatureAlgorithm;
    fn key_id(&self) -> &str;
    fn sign_hardware_reauthorization(&self, message: &[u8]) -> Result<Vec<u8>, String>;
}

pub trait HardwareReauthorizationVerifier {
    fn verify_hardware_reauthorization(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HardwareReauthorizationPolicy {
    pub maximum_authorization_duration_s: u64,
    pub maximum_statement_age_s: u64,
}

impl Default for HardwareReauthorizationPolicy {
    fn default() -> Self {
        Self {
            maximum_authorization_duration_s: 24 * 60 * 60,
            maximum_statement_age_s: 5 * 60,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HardwareReauthorizationError {
    UnsupportedSchema,
    InvalidPolicy,
    SequenceZero,
    InvalidMachineId,
    ZeroDigest(&'static str),
    InvalidWindow,
    HandoffMismatch,
    SuccessorMismatch,
    ProbationMismatch,
    ProbationExpired,
    StatementTooOld,
    InvalidAlgorithm,
    InvalidKeyId,
    EmptySignature,
    SignatureTooLarge,
    StatementDigestMismatch,
    TrustSnapshotInvalid(String),
    TrustSnapshotStale,
    SignerIneligible(KeyEligibility),
    InvalidSignature,
    VerificationProviderError(String),
    Signing(String),
    Encoding(String),
}

#[derive(Debug, Clone)]
pub struct VerifiedHardwareReauthorization {
    statement: HardwareReauthorizationStatement,
    statement_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    signer: (SignatureAlgorithm, String),
}

impl VerifiedHardwareReauthorization {
    pub fn statement(&self) -> &HardwareReauthorizationStatement {
        &self.statement
    }
    pub fn statement_digest(&self) -> Sha256Digest {
        self.statement_digest
    }
    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }
    pub fn signer(&self) -> &(SignatureAlgorithm, String) {
        &self.signer
    }

    pub fn permits_machine(&self, machine_id: &str, unix_s: u64) -> bool {
        self.statement.machine_id == machine_id
            && unix_s >= self.statement.issued_at_unix_s
            && unix_s < self.statement.expires_at_unix_s
    }
}

impl HardwareReauthorizationStatement {
    pub fn validate(&self) -> Result<(), HardwareReauthorizationError> {
        if self.schema_version != HARDWARE_REAUTHORIZATION_SCHEMA {
            return Err(HardwareReauthorizationError::UnsupportedSchema);
        }
        if self.reauthorization_sequence == 0 {
            return Err(HardwareReauthorizationError::SequenceZero);
        }
        validate_id(&self.machine_id)?;
        for (name, digest) in [
            ("handoff_digest", self.handoff_digest),
            (
                "successor_source_tree_digest",
                self.successor_source_tree_digest,
            ),
            (
                "successor_executable_digest",
                self.successor_executable_digest,
            ),
            ("hardware_identity_digest", self.hardware_identity_digest),
            ("machine_profile_digest", self.machine_profile_digest),
            ("firmware_digest", self.firmware_digest),
            ("calibration_digest", self.calibration_digest),
            ("capability_digest", self.capability_digest),
        ] {
            if digest.0 == [0; 32] {
                return Err(HardwareReauthorizationError::ZeroDigest(name));
            }
        }
        if self.issued_at_unix_s >= self.expires_at_unix_s {
            return Err(HardwareReauthorizationError::InvalidWindow);
        }
        Ok(())
    }
}

pub fn digest_hardware_reauthorization_statement(
    statement: &HardwareReauthorizationStatement,
) -> Result<Sha256Digest, HardwareReauthorizationError> {
    statement.validate()?;
    let bytes = serde_json::to_vec(statement)
        .map_err(|error| HardwareReauthorizationError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.hardware-reauthorization-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn sign_hardware_reauthorization(
    statement: HardwareReauthorizationStatement,
    signer: &dyn HardwareReauthorizationSigner,
) -> Result<SignedHardwareReauthorization, HardwareReauthorizationError> {
    let statement_digest = digest_hardware_reauthorization_statement(&statement)?;
    let algorithm = signer.algorithm();
    if !algorithm.is_canonical() {
        return Err(HardwareReauthorizationError::InvalidAlgorithm);
    }
    validate_id(signer.key_id()).map_err(|_| HardwareReauthorizationError::InvalidKeyId)?;
    let signature = signer
        .sign_hardware_reauthorization(&signature_message(statement_digest))
        .map_err(HardwareReauthorizationError::Signing)?;
    if signature.is_empty() {
        return Err(HardwareReauthorizationError::EmptySignature);
    }
    if signature.len() > MAX_HARDWARE_SIGNATURE_BYTES {
        return Err(HardwareReauthorizationError::SignatureTooLarge);
    }
    Ok(SignedHardwareReauthorization {
        statement,
        statement_digest,
        signature: DetachedSignature {
            algorithm,
            key_id: signer.key_id().to_string(),
            signature,
        },
    })
}

#[allow(clippy::too_many_arguments)]
pub fn verify_hardware_reauthorization(
    signed: &SignedHardwareReauthorization,
    handoff: &AuthorizedUpgradeHandoff,
    probation: &AuthorizedUpgradeProbationClearance,
    trust_snapshot: &TrustSnapshot,
    now_unix_s: u64,
    policy: &HardwareReauthorizationPolicy,
    verifier: &dyn HardwareReauthorizationVerifier,
) -> Result<VerifiedHardwareReauthorization, HardwareReauthorizationError> {
    if policy.maximum_authorization_duration_s == 0 || policy.maximum_statement_age_s == 0 {
        return Err(HardwareReauthorizationError::InvalidPolicy);
    }
    signed.statement.validate()?;
    if signed.statement.handoff_digest != handoff.plan_digest {
        return Err(HardwareReauthorizationError::HandoffMismatch);
    }
    if signed.statement.successor_source_tree_digest != handoff.plan.successor.source_tree_digest
        || signed.statement.successor_executable_digest != handoff.plan.successor.executable_digest
    {
        return Err(HardwareReauthorizationError::SuccessorMismatch);
    }
    if probation.evidence().handoff_digest != handoff.plan_digest {
        return Err(HardwareReauthorizationError::ProbationMismatch);
    }
    let now_ms = now_unix_s.saturating_mul(1_000);
    if !probation.permits_finalization(handoff.plan_digest, now_ms) {
        return Err(HardwareReauthorizationError::ProbationExpired);
    }
    if signed
        .statement
        .expires_at_unix_s
        .saturating_sub(signed.statement.issued_at_unix_s)
        > policy.maximum_authorization_duration_s
        || now_unix_s < signed.statement.issued_at_unix_s
        || now_unix_s >= signed.statement.expires_at_unix_s
    {
        return Err(HardwareReauthorizationError::InvalidWindow);
    }
    if now_unix_s.saturating_sub(signed.statement.issued_at_unix_s) > policy.maximum_statement_age_s
    {
        return Err(HardwareReauthorizationError::StatementTooOld);
    }
    if !signed.signature.algorithm.is_canonical() {
        return Err(HardwareReauthorizationError::InvalidAlgorithm);
    }
    if signed.signature.key_id.trim().is_empty()
        || signed.signature.key_id != signed.signature.key_id.trim()
        || signed.signature.key_id.len() > MAX_HARDWARE_ID_BYTES
        || signed.signature.key_id.chars().any(char::is_control)
    {
        return Err(HardwareReauthorizationError::InvalidKeyId);
    }
    let expected = digest_hardware_reauthorization_statement(&signed.statement)?;
    if expected != signed.statement_digest {
        return Err(HardwareReauthorizationError::StatementDigestMismatch);
    }
    trust_snapshot.validate().map_err(|error| {
        HardwareReauthorizationError::TrustSnapshotInvalid(format!("{error:?}"))
    })?;
    if !trust_snapshot.is_fresh_at(now_unix_s) {
        return Err(HardwareReauthorizationError::TrustSnapshotStale);
    }
    let eligibility = trust_snapshot.key_eligibility(
        &signed.signature.algorithm,
        &signed.signature.key_id,
        KeyUsage::HardwareReauthorization,
        now_unix_s,
    );
    if eligibility != KeyEligibility::Eligible {
        return Err(HardwareReauthorizationError::SignerIneligible(eligibility));
    }
    if signed.signature.signature.is_empty() {
        return Err(HardwareReauthorizationError::EmptySignature);
    }
    if signed.signature.signature.len() > MAX_HARDWARE_SIGNATURE_BYTES {
        return Err(HardwareReauthorizationError::SignatureTooLarge);
    }
    match verifier.verify_hardware_reauthorization(
        &signed.signature.algorithm,
        &signed.signature.key_id,
        &signature_message(expected),
        &signed.signature.signature,
    ) {
        Ok(true) => {}
        Ok(false) => return Err(HardwareReauthorizationError::InvalidSignature),
        Err(error) => {
            return Err(HardwareReauthorizationError::VerificationProviderError(
                error,
            ));
        }
    }
    let trust_snapshot_digest = digest_trust_snapshot(trust_snapshot).map_err(|error| {
        HardwareReauthorizationError::TrustSnapshotInvalid(format!("{error:?}"))
    })?;
    Ok(VerifiedHardwareReauthorization {
        statement: signed.statement.clone(),
        statement_digest: expected,
        trust_snapshot_digest,
        signer: (
            signed.signature.algorithm.clone(),
            signed.signature.key_id.clone(),
        ),
    })
}

fn signature_message(digest: Sha256Digest) -> Vec<u8> {
    let mut message = b"symthaea.fabrication.hardware-reauthorization-signature.v1\0".to_vec();
    message.extend_from_slice(&digest.0);
    message
}

fn validate_id(value: &str) -> Result<(), HardwareReauthorizationError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > MAX_HARDWARE_ID_BYTES
        || value.chars().any(char::is_control)
    {
        return Err(HardwareReauthorizationError::InvalidMachineId);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_digest::sha256;

    fn statement() -> HardwareReauthorizationStatement {
        HardwareReauthorizationStatement {
            schema_version: HARDWARE_REAUTHORIZATION_SCHEMA.into(),
            reauthorization_sequence: 1,
            handoff_digest: sha256(b"handoff"),
            successor_source_tree_digest: sha256(b"source"),
            successor_executable_digest: sha256(b"exe"),
            machine_id: "machine-a".into(),
            hardware_identity_digest: sha256(b"hardware"),
            machine_profile_digest: sha256(b"profile"),
            firmware_digest: sha256(b"firmware"),
            calibration_digest: sha256(b"calibration"),
            capability_digest: sha256(b"capability"),
            issued_at_unix_s: 10,
            expires_at_unix_s: 20,
        }
    }

    #[test]
    fn firmware_changes_statement_identity() {
        let first = statement();
        let mut second = statement();
        second.firmware_digest = sha256(b"other");
        assert_ne!(
            digest_hardware_reauthorization_statement(&first).unwrap(),
            digest_hardware_reauthorization_statement(&second).unwrap()
        );
    }

    #[test]
    fn noncanonical_machine_id_is_rejected() {
        let mut value = statement();
        value.machine_id = " machine-a".into();
        assert_eq!(
            value.validate(),
            Err(HardwareReauthorizationError::InvalidMachineId)
        );
    }
}
