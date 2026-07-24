// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent witness cosigning for transparency checkpoints.
//!
//! A log operator signature proves who issued a checkpoint. Witness cosigning
//! adds evidence that independent organizations in distinct regions observed
//! the same checkpoint before release authority relied upon it.

use crate::attestation::{DetachedSignature, SignatureAlgorithm};
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::transparency_checkpoint::VerifiedTransparencyCheckpoint;
use crate::trust::{KeyEligibility, KeyUsage, TrustSnapshot, digest_trust_snapshot};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const TRANSPARENCY_WITNESS_SCHEMA: &str = "symthaea.fabrication.transparency-witness.v1";
pub const SIGNED_TRANSPARENCY_WITNESS_SCHEMA: &str =
    "symthaea.fabrication.signed-transparency-witness.v1";
pub const MAX_TRANSPARENCY_WITNESSES: usize = 64;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransparencyWitnessStatement {
    pub schema_version: String,
    pub checkpoint_digest: Sha256Digest,
    pub checkpoint_log_size: u64,
    pub checkpoint_root_digest: Sha256Digest,
    pub witness_organization: String,
    pub witness_region: String,
    pub observed_at_unix_s: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedTransparencyWitness {
    pub schema_version: String,
    pub statement: TransparencyWitnessStatement,
    pub statement_digest: Sha256Digest,
    pub signature: DetachedSignature,
}

pub trait TransparencyWitnessSigner {
    fn algorithm(&self) -> SignatureAlgorithm;
    fn key_id(&self) -> &str;
    fn sign_transparency_witness(&self, message: &[u8]) -> Result<Vec<u8>, String>;
}

pub trait TransparencyWitnessVerifier {
    fn verify_transparency_witness(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransparencyWitnessPolicy {
    pub minimum_distinct_witnesses: usize,
    pub minimum_distinct_organizations: usize,
    pub minimum_distinct_regions: usize,
    pub maximum_observation_age_s: u64,
    pub maximum_witnesses: usize,
    pub require_algorithm_diversity: bool,
}

impl Default for TransparencyWitnessPolicy {
    fn default() -> Self {
        Self {
            minimum_distinct_witnesses: 2,
            minimum_distinct_organizations: 2,
            minimum_distinct_regions: 2,
            maximum_observation_age_s: 300,
            maximum_witnesses: MAX_TRANSPARENCY_WITNESSES,
            require_algorithm_diversity: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransparencyWitnessError {
    UnsupportedSchema,
    InvalidStatement,
    InvalidWindow,
    InvalidAlgorithm,
    InvalidKeyId,
    EmptySignature,
    SignatureTooLarge,
    Signing(String),
    Encoding(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransparencyWitnessViolation {
    InvalidPolicy,
    TooManyWitnesses,
    UnsupportedSchema,
    InvalidStatement(TransparencyWitnessError),
    CheckpointMismatch,
    DigestMismatch,
    ObservationInFuture,
    ObservationStale,
    DuplicateSigner(String),
    SignerUnknown(String),
    SignerNotYetValid(String),
    SignerExpired(String),
    SignerRetired(String),
    SignerRevoked(String),
    SignerUsageNotAllowed(String),
    InvalidSignature(String),
    VerificationProviderError { key_id: String, reason: String },
    InsufficientWitnesses { actual: usize, required: usize },
    InsufficientOrganizations { actual: usize, required: usize },
    InsufficientRegions { actual: usize, required: usize },
    MissingAlgorithmDiversity,
    TrustSnapshotInvalid,
    TrustSnapshotStale,
}

#[derive(Debug, Clone)]
pub struct VerifiedTransparencyWitnessQuorum {
    checkpoint_digest: Sha256Digest,
    witness_quorum_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    witnesses: Vec<(SignatureAlgorithm, String, String, String)>,
}

impl VerifiedTransparencyWitnessQuorum {
    pub fn checkpoint_digest(&self) -> Sha256Digest {
        self.checkpoint_digest
    }
    pub fn witness_quorum_digest(&self) -> Sha256Digest {
        self.witness_quorum_digest
    }
    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }
    pub fn witnesses(&self) -> &[(SignatureAlgorithm, String, String, String)] {
        &self.witnesses
    }
}

pub fn sign_transparency_witness(
    checkpoint: &VerifiedTransparencyCheckpoint,
    witness_organization: impl Into<String>,
    witness_region: impl Into<String>,
    observed_at_unix_s: u64,
    signer: &dyn TransparencyWitnessSigner,
) -> Result<SignedTransparencyWitness, TransparencyWitnessError> {
    let statement = TransparencyWitnessStatement {
        schema_version: TRANSPARENCY_WITNESS_SCHEMA.into(),
        checkpoint_digest: checkpoint.checkpoint_digest(),
        checkpoint_log_size: checkpoint.checkpoint().log_size,
        checkpoint_root_digest: checkpoint.checkpoint().root_digest,
        witness_organization: witness_organization.into(),
        witness_region: witness_region.into(),
        observed_at_unix_s,
    };
    validate_statement(&statement)?;
    let statement_digest = digest_transparency_witness_statement(&statement)?;
    let algorithm = signer.algorithm();
    if !algorithm.is_canonical() {
        return Err(TransparencyWitnessError::InvalidAlgorithm);
    }
    validate_identifier(signer.key_id())?;
    let signature = signer
        .sign_transparency_witness(&witness_signature_message(statement_digest))
        .map_err(TransparencyWitnessError::Signing)?;
    if signature.is_empty() {
        return Err(TransparencyWitnessError::EmptySignature);
    }
    if signature.len() > 64 * 1024 {
        return Err(TransparencyWitnessError::SignatureTooLarge);
    }
    Ok(SignedTransparencyWitness {
        schema_version: SIGNED_TRANSPARENCY_WITNESS_SCHEMA.into(),
        statement,
        statement_digest,
        signature: DetachedSignature {
            algorithm,
            key_id: signer.key_id().to_string(),
            signature,
        },
    })
}

pub fn digest_transparency_witness_statement(
    statement: &TransparencyWitnessStatement,
) -> Result<Sha256Digest, TransparencyWitnessError> {
    validate_statement(statement)?;
    let bytes = serde_json::to_vec(statement)
        .map_err(|error| TransparencyWitnessError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.transparency-witness-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn verify_transparency_witness_quorum(
    checkpoint: &VerifiedTransparencyCheckpoint,
    signed_witnesses: &[SignedTransparencyWitness],
    policy: &TransparencyWitnessPolicy,
    trust_snapshot: &TrustSnapshot,
    now_unix_s: u64,
    verifier: &dyn TransparencyWitnessVerifier,
) -> Result<VerifiedTransparencyWitnessQuorum, Vec<TransparencyWitnessViolation>> {
    let mut violations = Vec::new();
    if policy.minimum_distinct_witnesses == 0
        || policy.minimum_distinct_organizations == 0
        || policy.minimum_distinct_regions == 0
        || policy.maximum_observation_age_s == 0
        || policy.maximum_witnesses == 0
        || policy.maximum_witnesses > MAX_TRANSPARENCY_WITNESSES
    {
        violations.push(TransparencyWitnessViolation::InvalidPolicy);
    }
    if signed_witnesses.len() > policy.maximum_witnesses {
        violations.push(TransparencyWitnessViolation::TooManyWitnesses);
    }
    if trust_snapshot.validate().is_err() {
        violations.push(TransparencyWitnessViolation::TrustSnapshotInvalid);
    }
    if !trust_snapshot.is_fresh_at(now_unix_s) {
        violations.push(TransparencyWitnessViolation::TrustSnapshotStale);
    }

    let mut signers = BTreeSet::new();
    let mut organizations = BTreeSet::new();
    let mut regions = BTreeSet::new();
    let mut algorithms = BTreeSet::new();
    let mut valid = Vec::new();
    let mut valid_digests = Vec::new();
    for signed in signed_witnesses {
        if signed.schema_version != SIGNED_TRANSPARENCY_WITNESS_SCHEMA {
            violations.push(TransparencyWitnessViolation::UnsupportedSchema);
            continue;
        }
        if let Err(error) = validate_statement(&signed.statement) {
            violations.push(TransparencyWitnessViolation::InvalidStatement(error));
            continue;
        }
        if signed.statement.checkpoint_digest != checkpoint.checkpoint_digest()
            || signed.statement.checkpoint_log_size != checkpoint.checkpoint().log_size
            || signed.statement.checkpoint_root_digest != checkpoint.checkpoint().root_digest
        {
            violations.push(TransparencyWitnessViolation::CheckpointMismatch);
            continue;
        }
        if digest_transparency_witness_statement(&signed.statement).ok()
            != Some(signed.statement_digest)
        {
            violations.push(TransparencyWitnessViolation::DigestMismatch);
            continue;
        }
        if signed.statement.observed_at_unix_s > now_unix_s {
            violations.push(TransparencyWitnessViolation::ObservationInFuture);
            continue;
        }
        if now_unix_s.saturating_sub(signed.statement.observed_at_unix_s)
            > policy.maximum_observation_age_s
        {
            violations.push(TransparencyWitnessViolation::ObservationStale);
            continue;
        }
        let identity = (
            signed.signature.algorithm.clone(),
            signed.signature.key_id.clone(),
        );
        if !signers.insert(identity.clone()) {
            violations.push(TransparencyWitnessViolation::DuplicateSigner(
                signed.signature.key_id.clone(),
            ));
            continue;
        }
        match trust_snapshot.key_eligibility(
            &signed.signature.algorithm,
            &signed.signature.key_id,
            KeyUsage::TransparencyWitness,
            now_unix_s,
        ) {
            KeyEligibility::Eligible => {}
            KeyEligibility::Unknown => {
                violations.push(TransparencyWitnessViolation::SignerUnknown(
                    signed.signature.key_id.clone(),
                ));
                continue;
            }
            KeyEligibility::NotYetValid => {
                violations.push(TransparencyWitnessViolation::SignerNotYetValid(
                    signed.signature.key_id.clone(),
                ));
                continue;
            }
            KeyEligibility::Expired => {
                violations.push(TransparencyWitnessViolation::SignerExpired(
                    signed.signature.key_id.clone(),
                ));
                continue;
            }
            KeyEligibility::Retired => {
                violations.push(TransparencyWitnessViolation::SignerRetired(
                    signed.signature.key_id.clone(),
                ));
                continue;
            }
            KeyEligibility::Revoked => {
                violations.push(TransparencyWitnessViolation::SignerRevoked(
                    signed.signature.key_id.clone(),
                ));
                continue;
            }
            KeyEligibility::UsageNotAllowed => {
                violations.push(TransparencyWitnessViolation::SignerUsageNotAllowed(
                    signed.signature.key_id.clone(),
                ));
                continue;
            }
        }
        match verifier.verify_transparency_witness(
            &signed.signature.algorithm,
            &signed.signature.key_id,
            &witness_signature_message(signed.statement_digest),
            &signed.signature.signature,
        ) {
            Ok(true) => {}
            Ok(false) => {
                violations.push(TransparencyWitnessViolation::InvalidSignature(
                    signed.signature.key_id.clone(),
                ));
                continue;
            }
            Err(reason) => {
                violations.push(TransparencyWitnessViolation::VerificationProviderError {
                    key_id: signed.signature.key_id.clone(),
                    reason,
                });
                continue;
            }
        }
        algorithms.insert(signed.signature.algorithm.clone());
        organizations.insert(signed.statement.witness_organization.clone());
        regions.insert(signed.statement.witness_region.clone());
        valid_digests.push(signed.statement_digest);
        valid.push((
            signed.signature.algorithm.clone(),
            signed.signature.key_id.clone(),
            signed.statement.witness_organization.clone(),
            signed.statement.witness_region.clone(),
        ));
    }

    if valid.len() < policy.minimum_distinct_witnesses {
        violations.push(TransparencyWitnessViolation::InsufficientWitnesses {
            actual: valid.len(),
            required: policy.minimum_distinct_witnesses,
        });
    }
    if organizations.len() < policy.minimum_distinct_organizations {
        violations.push(TransparencyWitnessViolation::InsufficientOrganizations {
            actual: organizations.len(),
            required: policy.minimum_distinct_organizations,
        });
    }
    if regions.len() < policy.minimum_distinct_regions {
        violations.push(TransparencyWitnessViolation::InsufficientRegions {
            actual: regions.len(),
            required: policy.minimum_distinct_regions,
        });
    }
    if policy.require_algorithm_diversity && algorithms.len() < 2 {
        violations.push(TransparencyWitnessViolation::MissingAlgorithmDiversity);
    }
    if !violations.is_empty() {
        return Err(violations);
    }

    valid.sort();
    valid_digests.sort();
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.transparency-witness-quorum.v1\0");
    hasher.update(&checkpoint.checkpoint_digest().0);
    for digest in valid_digests {
        hasher.update(&digest.0);
    }
    Ok(VerifiedTransparencyWitnessQuorum {
        checkpoint_digest: checkpoint.checkpoint_digest(),
        witness_quorum_digest: hasher.finalize(),
        trust_snapshot_digest: digest_trust_snapshot(trust_snapshot)
            .expect("validated trust snapshot must have canonical digest"),
        witnesses: valid,
    })
}

fn validate_statement(
    statement: &TransparencyWitnessStatement,
) -> Result<(), TransparencyWitnessError> {
    if statement.schema_version != TRANSPARENCY_WITNESS_SCHEMA {
        return Err(TransparencyWitnessError::UnsupportedSchema);
    }
    validate_identifier(&statement.witness_organization)?;
    validate_identifier(&statement.witness_region)?;
    if statement.checkpoint_log_size == 0 {
        return Err(TransparencyWitnessError::InvalidStatement);
    }
    Ok(())
}

fn validate_identifier(value: &str) -> Result<(), TransparencyWitnessError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > 256
        || value.chars().any(char::is_control)
    {
        return Err(TransparencyWitnessError::InvalidKeyId);
    }
    Ok(())
}

fn witness_signature_message(digest: Sha256Digest) -> Vec<u8> {
    let mut message = b"symthaea.fabrication.transparency-witness-signature.v1\0".to_vec();
    message.extend_from_slice(&digest.0);
    message
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_witness_identity_is_rejected() {
        let statement = TransparencyWitnessStatement {
            schema_version: TRANSPARENCY_WITNESS_SCHEMA.into(),
            checkpoint_digest: Sha256Digest([1; 32]),
            checkpoint_log_size: 1,
            checkpoint_root_digest: Sha256Digest([2; 32]),
            witness_organization: " org".into(),
            witness_region: "region-a".into(),
            observed_at_unix_s: 10,
        };
        assert_eq!(
            validate_statement(&statement),
            Err(TransparencyWitnessError::InvalidKeyId)
        );
    }
}
