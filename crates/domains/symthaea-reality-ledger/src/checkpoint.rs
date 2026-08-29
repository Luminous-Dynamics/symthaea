// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Authenticated-checkpoint contract above the append-only RealityLedger.
//!
//! This module deliberately does not implement signatures. Xenia or another
//! authority/signature provider may verify attestations externally.

use serde::{Deserialize, Serialize};

use crate::digest::{DigestAlgorithm, TypedDigest};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerCheckpoint {
    pub schema_version: u32,
    pub checkpoint_id: String,
    pub record_count: u64,
    pub ledger_head: TypedDigest,
}

impl LedgerCheckpoint {
    pub fn validate(&self) -> Result<(), CheckpointError> {
        if self.schema_version == 0 {
            return Err(CheckpointError::InvalidSchemaVersion);
        }
        if self.checkpoint_id.trim().is_empty() {
            return Err(CheckpointError::MissingCheckpointId);
        }
        if self.record_count == 0 {
            return Err(CheckpointError::EmptyCheckpoint);
        }
        self.ledger_head
            .validate()
            .map_err(|error| CheckpointError::InvalidDigest(error.to_string()))?;
        Ok(())
    }

    pub fn digest(&self) -> Result<TypedDigest, CheckpointError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        feed(&mut hasher, b"symthaea.reality-ledger-checkpoint.v1");
        hasher.update(&self.schema_version.to_le_bytes());
        feed(&mut hasher, self.checkpoint_id.as_bytes());
        hasher.update(&self.record_count.to_le_bytes());
        feed(&mut hasher, self.ledger_head.domain.as_bytes());
        feed(&mut hasher, algorithm_name(&self.ledger_head.algorithm).as_bytes());
        feed(&mut hasher, self.ledger_head.value.as_bytes());
        TypedDigest::new(
            "symthaea.reality-ledger-checkpoint.v1",
            DigestAlgorithm::Blake3,
            hasher.finalize().to_hex().to_string(),
        )
        .map_err(|error| CheckpointError::InvalidDigest(error.to_string()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointAttestation {
    pub checkpoint_digest: TypedDigest,
    pub signer_id: String,
    pub authority_scope: String,
    pub signature_scheme: String,
    /// Opaque signature/attestation bytes encoded by the external verifier.
    pub signature: String,
}

impl CheckpointAttestation {
    pub fn validate_structure(&self, checkpoint: &LedgerCheckpoint) -> Result<(), CheckpointError> {
        let expected = checkpoint.digest()?;
        if !self.checkpoint_digest.same_typed_value(&expected) {
            return Err(CheckpointError::CheckpointDigestMismatch);
        }
        for value in [
            self.signer_id.as_str(),
            self.authority_scope.as_str(),
            self.signature_scheme.as_str(),
            self.signature.as_str(),
        ] {
            if value.trim().is_empty() {
                return Err(CheckpointError::MissingAttestationField);
            }
        }
        Ok(())
    }
}

fn algorithm_name(algorithm: &DigestAlgorithm) -> String {
    match algorithm {
        DigestAlgorithm::Blake3 => "blake3".into(),
        DigestAlgorithm::Sha256 => "sha256".into(),
        DigestAlgorithm::Other(name) => format!("other:{name}"),
    }
}

fn feed(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CheckpointError {
    #[error("checkpoint schema version must be non-zero")]
    InvalidSchemaVersion,
    #[error("checkpoint id may not be empty")]
    MissingCheckpointId,
    #[error("cannot attest an empty ledger")]
    EmptyCheckpoint,
    #[error("invalid typed digest: {0}")]
    InvalidDigest(String),
    #[error("attestation references a different checkpoint digest")]
    CheckpointDigestMismatch,
    #[error("attestation fields may not be empty")]
    MissingAttestationField,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_ledger_replacement_needs_a_new_external_attestation() {
        let head = TypedDigest::new(
            "symthaea.reality-ledger-head.v1",
            DigestAlgorithm::Blake3,
            "head-a",
        )
        .unwrap();
        let checkpoint = LedgerCheckpoint {
            schema_version: 1,
            checkpoint_id: "cp-1".into(),
            record_count: 10,
            ledger_head: head,
        };
        let digest = checkpoint.digest().unwrap();
        let attestation = CheckpointAttestation {
            checkpoint_digest: digest,
            signer_id: "xenia-agent".into(),
            authority_scope: "reality-ledger-checkpoint".into(),
            signature_scheme: "external-test".into(),
            signature: "opaque-signature".into(),
        };
        attestation.validate_structure(&checkpoint).unwrap();

        let mut replacement = checkpoint.clone();
        replacement.ledger_head.value = "rewritten-head".into();
        assert_eq!(
            attestation.validate_structure(&replacement),
            Err(CheckpointError::CheckpointDigestMismatch)
        );
    }
}
