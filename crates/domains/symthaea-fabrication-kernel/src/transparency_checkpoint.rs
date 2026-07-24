// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Signed transparency checkpoints and persistent anti-rollback tracking.

use crate::attestation::{DetachedSignature, SignatureAlgorithm};
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::transparency::{TransparencyError, TransparencyLog};
use crate::trust::{KeyEligibility, KeyUsage, TrustSnapshot};
use serde::{Deserialize, Serialize};

pub const TRANSPARENCY_CHECKPOINT_SCHEMA: &str = "symthaea.fabrication.transparency-checkpoint.v1";
pub const SIGNED_TRANSPARENCY_CHECKPOINT_SCHEMA: &str =
    "symthaea.fabrication.signed-transparency-checkpoint.v1";
pub const MAX_TRANSPARENCY_CHECKPOINT_SIGNATURE_BYTES: usize = 64 * 1024;
pub const MAX_TRANSPARENCY_CHECKPOINT_KEY_ID_BYTES: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransparencyCheckpoint {
    pub schema_version: String,
    pub log_size: u64,
    pub root_digest: Sha256Digest,
    pub previous_checkpoint_digest: Option<Sha256Digest>,
    pub issued_at_unix_s: u64,
    pub expires_at_unix_s: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedTransparencyCheckpoint {
    pub schema_version: String,
    pub checkpoint: TransparencyCheckpoint,
    pub checkpoint_digest: Sha256Digest,
    pub signature: DetachedSignature,
}

pub trait TransparencyCheckpointSigner {
    fn algorithm(&self) -> SignatureAlgorithm;
    fn key_id(&self) -> &str;
    fn sign_transparency_checkpoint(&self, message: &[u8]) -> Result<Vec<u8>, String>;
}

pub trait TransparencyCheckpointVerifier {
    fn verify_transparency_checkpoint(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransparencyCheckpointError {
    UnsupportedSchema,
    InvalidWindow,
    Log(TransparencyError),
    LogRootMismatch,
    InvalidAlgorithm,
    InvalidKeyId,
    EmptySignature,
    SignatureTooLarge { actual: usize, maximum: usize },
    Signing(String),
    Encoding(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransparencyCheckpointViolation {
    UnsupportedSchema,
    InvalidCheckpoint(TransparencyCheckpointError),
    DigestMismatch,
    NotYetValid,
    Expired,
    TrustSnapshotInvalid(String),
    TrustSnapshotStale,
    SignerUnknown,
    SignerNotYetValid,
    SignerExpired,
    SignerRetired,
    SignerRevoked,
    SignerUsageNotAllowed,
    InvalidSignature,
    VerificationProviderError(String),
}

#[derive(Debug, Clone)]
pub struct VerifiedTransparencyCheckpoint {
    checkpoint: TransparencyCheckpoint,
    checkpoint_digest: Sha256Digest,
    signer: (SignatureAlgorithm, String),
}

impl VerifiedTransparencyCheckpoint {
    pub fn checkpoint(&self) -> &TransparencyCheckpoint {
        &self.checkpoint
    }
    pub fn checkpoint_digest(&self) -> Sha256Digest {
        self.checkpoint_digest
    }
    pub fn signer(&self) -> &(SignatureAlgorithm, String) {
        &self.signer
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransparencyCheckpointTracker {
    latest_log_size: Option<u64>,
    latest_root_digest: Option<Sha256Digest>,
    latest_checkpoint_digest: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransparencyCheckpointTrackingError {
    SizeRollback { latest: u64, proposed: u64 },
    SameSizeSubstitution,
    PreviousCheckpointMismatch,
}

pub fn sign_transparency_checkpoint(
    log: &TransparencyLog,
    previous_checkpoint_digest: Option<Sha256Digest>,
    issued_at_unix_s: u64,
    expires_at_unix_s: u64,
    signer: &dyn TransparencyCheckpointSigner,
) -> Result<SignedTransparencyCheckpoint, TransparencyCheckpointError> {
    let checkpoint = TransparencyCheckpoint {
        schema_version: TRANSPARENCY_CHECKPOINT_SCHEMA.into(),
        log_size: log.entries.len() as u64,
        root_digest: log.root().map_err(TransparencyCheckpointError::Log)?,
        previous_checkpoint_digest,
        issued_at_unix_s,
        expires_at_unix_s,
    };
    checkpoint.validate()?;
    let algorithm = signer.algorithm();
    if !algorithm.is_canonical() {
        return Err(TransparencyCheckpointError::InvalidAlgorithm);
    }
    let key_id = signer.key_id();
    if key_id.trim().is_empty()
        || key_id != key_id.trim()
        || key_id.len() > MAX_TRANSPARENCY_CHECKPOINT_KEY_ID_BYTES
    {
        return Err(TransparencyCheckpointError::InvalidKeyId);
    }
    let checkpoint_digest = digest_transparency_checkpoint(&checkpoint)?;
    let signature = signer
        .sign_transparency_checkpoint(&checkpoint_signature_message(checkpoint_digest))
        .map_err(TransparencyCheckpointError::Signing)?;
    if signature.is_empty() {
        return Err(TransparencyCheckpointError::EmptySignature);
    }
    if signature.len() > MAX_TRANSPARENCY_CHECKPOINT_SIGNATURE_BYTES {
        return Err(TransparencyCheckpointError::SignatureTooLarge {
            actual: signature.len(),
            maximum: MAX_TRANSPARENCY_CHECKPOINT_SIGNATURE_BYTES,
        });
    }
    Ok(SignedTransparencyCheckpoint {
        schema_version: SIGNED_TRANSPARENCY_CHECKPOINT_SCHEMA.into(),
        checkpoint,
        checkpoint_digest,
        signature: DetachedSignature {
            algorithm,
            key_id: key_id.to_string(),
            signature,
        },
    })
}

impl TransparencyCheckpoint {
    pub fn validate(&self) -> Result<(), TransparencyCheckpointError> {
        if self.schema_version != TRANSPARENCY_CHECKPOINT_SCHEMA {
            return Err(TransparencyCheckpointError::UnsupportedSchema);
        }
        if self.issued_at_unix_s >= self.expires_at_unix_s {
            return Err(TransparencyCheckpointError::InvalidWindow);
        }
        Ok(())
    }
}

pub fn digest_transparency_checkpoint(
    checkpoint: &TransparencyCheckpoint,
) -> Result<Sha256Digest, TransparencyCheckpointError> {
    checkpoint.validate()?;
    let bytes = serde_json::to_vec(checkpoint)
        .map_err(|error| TransparencyCheckpointError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.transparency-checkpoint-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn verify_transparency_checkpoint(
    signed: &SignedTransparencyCheckpoint,
    log: &TransparencyLog,
    trust_snapshot: &TrustSnapshot,
    now_unix_s: u64,
    verifier: &dyn TransparencyCheckpointVerifier,
) -> Result<VerifiedTransparencyCheckpoint, Vec<TransparencyCheckpointViolation>> {
    let mut violations = Vec::new();
    if signed.schema_version != SIGNED_TRANSPARENCY_CHECKPOINT_SCHEMA {
        violations.push(TransparencyCheckpointViolation::UnsupportedSchema);
    }
    if let Err(error) = signed.checkpoint.validate() {
        violations.push(TransparencyCheckpointViolation::InvalidCheckpoint(error));
    }
    if signed.checkpoint.log_size != log.entries.len() as u64
        || log.root().ok() != Some(signed.checkpoint.root_digest)
    {
        violations.push(TransparencyCheckpointViolation::InvalidCheckpoint(
            TransparencyCheckpointError::LogRootMismatch,
        ));
    }
    if digest_transparency_checkpoint(&signed.checkpoint).ok() != Some(signed.checkpoint_digest) {
        violations.push(TransparencyCheckpointViolation::DigestMismatch);
    }
    if now_unix_s < signed.checkpoint.issued_at_unix_s {
        violations.push(TransparencyCheckpointViolation::NotYetValid);
    }
    if now_unix_s >= signed.checkpoint.expires_at_unix_s {
        violations.push(TransparencyCheckpointViolation::Expired);
    }
    if let Err(error) = trust_snapshot.validate() {
        violations.push(TransparencyCheckpointViolation::TrustSnapshotInvalid(
            format!("{error:?}"),
        ));
    }
    if !trust_snapshot.is_fresh_at(now_unix_s) {
        violations.push(TransparencyCheckpointViolation::TrustSnapshotStale);
    }
    if !signed.signature.algorithm.is_canonical()
        || signed.signature.key_id.trim().is_empty()
        || signed.signature.key_id != signed.signature.key_id.trim()
        || signed.signature.key_id.len() > MAX_TRANSPARENCY_CHECKPOINT_KEY_ID_BYTES
        || signed.signature.signature.is_empty()
        || signed.signature.signature.len() > MAX_TRANSPARENCY_CHECKPOINT_SIGNATURE_BYTES
    {
        violations.push(TransparencyCheckpointViolation::InvalidCheckpoint(
            TransparencyCheckpointError::InvalidKeyId,
        ));
    }
    match trust_snapshot.key_eligibility(
        &signed.signature.algorithm,
        &signed.signature.key_id,
        KeyUsage::TransparencyLog,
        now_unix_s,
    ) {
        KeyEligibility::Eligible => {}
        KeyEligibility::Unknown => violations.push(TransparencyCheckpointViolation::SignerUnknown),
        KeyEligibility::NotYetValid => {
            violations.push(TransparencyCheckpointViolation::SignerNotYetValid)
        }
        KeyEligibility::Expired => violations.push(TransparencyCheckpointViolation::SignerExpired),
        KeyEligibility::Retired => violations.push(TransparencyCheckpointViolation::SignerRetired),
        KeyEligibility::Revoked => violations.push(TransparencyCheckpointViolation::SignerRevoked),
        KeyEligibility::UsageNotAllowed => {
            violations.push(TransparencyCheckpointViolation::SignerUsageNotAllowed)
        }
    }
    match verifier.verify_transparency_checkpoint(
        &signed.signature.algorithm,
        &signed.signature.key_id,
        &checkpoint_signature_message(signed.checkpoint_digest),
        &signed.signature.signature,
    ) {
        Ok(true) => {}
        Ok(false) => violations.push(TransparencyCheckpointViolation::InvalidSignature),
        Err(reason) => violations.push(TransparencyCheckpointViolation::VerificationProviderError(
            reason,
        )),
    }
    if !violations.is_empty() {
        return Err(violations);
    }
    Ok(VerifiedTransparencyCheckpoint {
        checkpoint: signed.checkpoint.clone(),
        checkpoint_digest: signed.checkpoint_digest,
        signer: (
            signed.signature.algorithm.clone(),
            signed.signature.key_id.clone(),
        ),
    })
}

impl TransparencyCheckpointTracker {
    pub fn accept(
        &mut self,
        verified: &VerifiedTransparencyCheckpoint,
    ) -> Result<(), TransparencyCheckpointTrackingError> {
        let checkpoint = verified.checkpoint();
        if let Some(latest_size) = self.latest_log_size {
            if checkpoint.log_size < latest_size {
                return Err(TransparencyCheckpointTrackingError::SizeRollback {
                    latest: latest_size,
                    proposed: checkpoint.log_size,
                });
            }
            if checkpoint.log_size == latest_size {
                if self.latest_checkpoint_digest == Some(verified.checkpoint_digest())
                    && self.latest_root_digest == Some(checkpoint.root_digest)
                {
                    return Ok(());
                }
                return Err(TransparencyCheckpointTrackingError::SameSizeSubstitution);
            }
            if checkpoint.previous_checkpoint_digest != self.latest_checkpoint_digest {
                return Err(TransparencyCheckpointTrackingError::PreviousCheckpointMismatch);
            }
        } else if checkpoint.previous_checkpoint_digest.is_some() {
            return Err(TransparencyCheckpointTrackingError::PreviousCheckpointMismatch);
        }
        self.latest_log_size = Some(checkpoint.log_size);
        self.latest_root_digest = Some(checkpoint.root_digest);
        self.latest_checkpoint_digest = Some(verified.checkpoint_digest());
        Ok(())
    }
}

fn checkpoint_signature_message(digest: Sha256Digest) -> Vec<u8> {
    let mut message = b"symthaea.fabrication.transparency-checkpoint-signature.v1\0".to_vec();
    message.extend_from_slice(&digest.0);
    message
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracker_rejects_unlinked_growth() {
        let mut tracker = TransparencyCheckpointTracker::default();
        tracker.latest_log_size = Some(1);
        tracker.latest_root_digest = Some(Sha256Digest([1; 32]));
        tracker.latest_checkpoint_digest = Some(Sha256Digest([2; 32]));
        let verified = VerifiedTransparencyCheckpoint {
            checkpoint: TransparencyCheckpoint {
                schema_version: TRANSPARENCY_CHECKPOINT_SCHEMA.into(),
                log_size: 2,
                root_digest: Sha256Digest([3; 32]),
                previous_checkpoint_digest: Some(Sha256Digest([9; 32])),
                issued_at_unix_s: 10,
                expires_at_unix_s: 20,
            },
            checkpoint_digest: Sha256Digest([4; 32]),
            signer: (SignatureAlgorithm::Ed25519, "log".into()),
        };
        assert_eq!(
            tracker.accept(&verified),
            Err(TransparencyCheckpointTrackingError::PreviousCheckpointMismatch)
        );
    }
}
