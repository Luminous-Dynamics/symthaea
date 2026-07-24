// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Signed build provenance for reproducible release artifact sets.
//!
//! Artifact inventories prove which bytes are promoted. Provenance adds an
//! independently signed statement about the source, builder, build environment,
//! inputs, and reproducibility comparison used to produce those bytes.

use crate::artifact_set::{ArtifactSetError, ReleaseArtifactSet, digest_release_artifact_set};
use crate::attestation::{DetachedSignature, SignatureAlgorithm};
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::trust::{KeyEligibility, KeyUsage, TrustSnapshot, digest_trust_snapshot};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const ARTIFACT_PROVENANCE_SCHEMA: &str = "symthaea.fabrication.artifact-provenance.v1";
pub const SIGNED_ARTIFACT_PROVENANCE_SCHEMA: &str =
    "symthaea.fabrication.signed-artifact-provenance.v1";
pub const MAX_PROVENANCE_INPUTS: usize = 4096;
pub const MAX_PROVENANCE_STATEMENTS: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceInput {
    pub name: String,
    pub digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactProvenanceStatement {
    pub schema_version: String,
    pub artifact_set_digest: Sha256Digest,
    pub source_tree_digest: Sha256Digest,
    pub builder_id: String,
    pub builder_region: String,
    pub build_environment_digest: Sha256Digest,
    pub dependency_lock_digest: Sha256Digest,
    pub inputs: Vec<ProvenanceInput>,
    pub reproducible_match_count: u16,
    pub built_at_unix_s: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedArtifactProvenance {
    pub schema_version: String,
    pub statement: ArtifactProvenanceStatement,
    pub statement_digest: Sha256Digest,
    pub signature: DetachedSignature,
}

pub trait ArtifactProvenanceSigner {
    fn algorithm(&self) -> SignatureAlgorithm;
    fn key_id(&self) -> &str;
    fn sign_artifact_provenance(&self, message: &[u8]) -> Result<Vec<u8>, String>;
}

pub trait ArtifactProvenanceVerifier {
    fn verify_artifact_provenance(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactProvenancePolicy {
    pub minimum_distinct_builders: usize,
    pub minimum_distinct_regions: usize,
    pub minimum_reproducible_match_count: u16,
    pub maximum_statement_age_s: u64,
    pub maximum_statements: usize,
    pub require_algorithm_diversity: bool,
}

impl Default for ArtifactProvenancePolicy {
    fn default() -> Self {
        Self {
            minimum_distinct_builders: 2,
            minimum_distinct_regions: 2,
            minimum_reproducible_match_count: 2,
            maximum_statement_age_s: 3_600,
            maximum_statements: MAX_PROVENANCE_STATEMENTS,
            require_algorithm_diversity: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactProvenanceError {
    UnsupportedSchema,
    InvalidIdentifier,
    EmptyInputs,
    TooManyInputs,
    DuplicateInput(String),
    NonCanonicalInputOrder,
    InvalidReproducibilityCount,
    InvalidAlgorithm,
    EmptySignature,
    SignatureTooLarge,
    ArtifactSet(ArtifactSetError),
    Signing(String),
    Encoding(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactProvenanceViolation {
    InvalidPolicy,
    TooManyStatements,
    UnsupportedSchema,
    InvalidStatement(ArtifactProvenanceError),
    ArtifactSetMismatch,
    StatementDigestMismatch,
    StatementInFuture,
    StatementStale,
    DuplicateSigner(String),
    DuplicateBuilder(String),
    InsufficientReproducibility { actual: u16, required: u16 },
    SignerUnknown(String),
    SignerNotYetValid(String),
    SignerExpired(String),
    SignerRetired(String),
    SignerRevoked(String),
    SignerUsageNotAllowed(String),
    InvalidSignature(String),
    VerificationProviderError { key_id: String, reason: String },
    InsufficientBuilders { actual: usize, required: usize },
    InsufficientRegions { actual: usize, required: usize },
    MissingAlgorithmDiversity,
    TrustSnapshotInvalid,
    TrustSnapshotStale,
}

#[derive(Debug, Clone)]
pub struct VerifiedArtifactProvenance {
    artifact_set_digest: Sha256Digest,
    provenance_digest: Sha256Digest,
    trust_snapshot_digest: Sha256Digest,
    builders: Vec<(String, String, SignatureAlgorithm, String)>,
}

impl VerifiedArtifactProvenance {
    pub fn artifact_set_digest(&self) -> Sha256Digest {
        self.artifact_set_digest
    }
    pub fn provenance_digest(&self) -> Sha256Digest {
        self.provenance_digest
    }
    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }
    pub fn builders(&self) -> &[(String, String, SignatureAlgorithm, String)] {
        &self.builders
    }
}

pub fn build_artifact_provenance_statement(
    artifact_set: &ReleaseArtifactSet,
    builder_id: impl Into<String>,
    builder_region: impl Into<String>,
    build_environment_digest: Sha256Digest,
    dependency_lock_digest: Sha256Digest,
    mut inputs: Vec<ProvenanceInput>,
    reproducible_match_count: u16,
    built_at_unix_s: u64,
) -> Result<ArtifactProvenanceStatement, ArtifactProvenanceError> {
    artifact_set
        .validate()
        .map_err(ArtifactProvenanceError::ArtifactSet)?;
    inputs.sort_by(|left, right| left.name.cmp(&right.name));
    let statement = ArtifactProvenanceStatement {
        schema_version: ARTIFACT_PROVENANCE_SCHEMA.into(),
        artifact_set_digest: digest_release_artifact_set(artifact_set)
            .map_err(ArtifactProvenanceError::ArtifactSet)?,
        source_tree_digest: artifact_set.source_tree_digest,
        builder_id: builder_id.into(),
        builder_region: builder_region.into(),
        build_environment_digest,
        dependency_lock_digest,
        inputs,
        reproducible_match_count,
        built_at_unix_s,
    };
    validate_statement(&statement)?;
    Ok(statement)
}

pub fn sign_artifact_provenance(
    statement: ArtifactProvenanceStatement,
    signer: &dyn ArtifactProvenanceSigner,
) -> Result<SignedArtifactProvenance, ArtifactProvenanceError> {
    validate_statement(&statement)?;
    let statement_digest = digest_artifact_provenance_statement(&statement)?;
    let algorithm = signer.algorithm();
    if !algorithm.is_canonical() {
        return Err(ArtifactProvenanceError::InvalidAlgorithm);
    }
    validate_identifier(signer.key_id())?;
    let signature = signer
        .sign_artifact_provenance(&provenance_signature_message(statement_digest))
        .map_err(ArtifactProvenanceError::Signing)?;
    if signature.is_empty() {
        return Err(ArtifactProvenanceError::EmptySignature);
    }
    if signature.len() > 64 * 1024 {
        return Err(ArtifactProvenanceError::SignatureTooLarge);
    }
    Ok(SignedArtifactProvenance {
        schema_version: SIGNED_ARTIFACT_PROVENANCE_SCHEMA.into(),
        statement,
        statement_digest,
        signature: DetachedSignature {
            algorithm,
            key_id: signer.key_id().to_string(),
            signature,
        },
    })
}

pub fn digest_artifact_provenance_statement(
    statement: &ArtifactProvenanceStatement,
) -> Result<Sha256Digest, ArtifactProvenanceError> {
    validate_statement(statement)?;
    let bytes = serde_json::to_vec(statement)
        .map_err(|error| ArtifactProvenanceError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.artifact-provenance-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn verify_artifact_provenance(
    artifact_set: &ReleaseArtifactSet,
    statements: &[SignedArtifactProvenance],
    policy: &ArtifactProvenancePolicy,
    trust_snapshot: &TrustSnapshot,
    now_unix_s: u64,
    verifier: &dyn ArtifactProvenanceVerifier,
) -> Result<VerifiedArtifactProvenance, Vec<ArtifactProvenanceViolation>> {
    let mut violations = Vec::new();
    if policy.minimum_distinct_builders == 0
        || policy.minimum_distinct_regions == 0
        || policy.minimum_reproducible_match_count == 0
        || policy.maximum_statement_age_s == 0
        || policy.maximum_statements == 0
        || policy.maximum_statements > MAX_PROVENANCE_STATEMENTS
    {
        violations.push(ArtifactProvenanceViolation::InvalidPolicy);
    }
    if statements.len() > policy.maximum_statements {
        violations.push(ArtifactProvenanceViolation::TooManyStatements);
    }
    let artifact_set_digest = match digest_release_artifact_set(artifact_set) {
        Ok(digest) => digest,
        Err(error) => {
            violations.push(ArtifactProvenanceViolation::InvalidStatement(
                ArtifactProvenanceError::ArtifactSet(error),
            ));
            Sha256Digest([0; 32])
        }
    };
    if trust_snapshot.validate().is_err() {
        violations.push(ArtifactProvenanceViolation::TrustSnapshotInvalid);
    }
    if !trust_snapshot.is_fresh_at(now_unix_s) {
        violations.push(ArtifactProvenanceViolation::TrustSnapshotStale);
    }

    let mut signer_ids = BTreeSet::new();
    let mut builders = BTreeSet::new();
    let mut regions = BTreeSet::new();
    let mut algorithms = BTreeSet::new();
    let mut valid_builders = Vec::new();
    let mut valid_digests = Vec::new();
    for signed in statements {
        if signed.schema_version != SIGNED_ARTIFACT_PROVENANCE_SCHEMA {
            violations.push(ArtifactProvenanceViolation::UnsupportedSchema);
            continue;
        }
        if let Err(error) = validate_statement(&signed.statement) {
            violations.push(ArtifactProvenanceViolation::InvalidStatement(error));
            continue;
        }
        if signed.statement.artifact_set_digest != artifact_set_digest
            || signed.statement.source_tree_digest != artifact_set.source_tree_digest
        {
            violations.push(ArtifactProvenanceViolation::ArtifactSetMismatch);
            continue;
        }
        if digest_artifact_provenance_statement(&signed.statement).ok()
            != Some(signed.statement_digest)
        {
            violations.push(ArtifactProvenanceViolation::StatementDigestMismatch);
            continue;
        }
        if signed.statement.built_at_unix_s > now_unix_s {
            violations.push(ArtifactProvenanceViolation::StatementInFuture);
            continue;
        }
        if now_unix_s.saturating_sub(signed.statement.built_at_unix_s)
            > policy.maximum_statement_age_s
        {
            violations.push(ArtifactProvenanceViolation::StatementStale);
            continue;
        }
        if signed.statement.reproducible_match_count < policy.minimum_reproducible_match_count {
            violations.push(ArtifactProvenanceViolation::InsufficientReproducibility {
                actual: signed.statement.reproducible_match_count,
                required: policy.minimum_reproducible_match_count,
            });
            continue;
        }
        let signer_identity = (
            signed.signature.algorithm.clone(),
            signed.signature.key_id.clone(),
        );
        if !signer_ids.insert(signer_identity) {
            violations.push(ArtifactProvenanceViolation::DuplicateSigner(
                signed.signature.key_id.clone(),
            ));
            continue;
        }
        if !builders.insert(signed.statement.builder_id.clone()) {
            violations.push(ArtifactProvenanceViolation::DuplicateBuilder(
                signed.statement.builder_id.clone(),
            ));
            continue;
        }
        match trust_snapshot.key_eligibility(
            &signed.signature.algorithm,
            &signed.signature.key_id,
            KeyUsage::ArtifactProvenance,
            now_unix_s,
        ) {
            KeyEligibility::Eligible => {}
            KeyEligibility::Unknown => {
                violations.push(ArtifactProvenanceViolation::SignerUnknown(
                    signed.signature.key_id.clone(),
                ));
                continue;
            }
            KeyEligibility::NotYetValid => {
                violations.push(ArtifactProvenanceViolation::SignerNotYetValid(
                    signed.signature.key_id.clone(),
                ));
                continue;
            }
            KeyEligibility::Expired => {
                violations.push(ArtifactProvenanceViolation::SignerExpired(
                    signed.signature.key_id.clone(),
                ));
                continue;
            }
            KeyEligibility::Retired => {
                violations.push(ArtifactProvenanceViolation::SignerRetired(
                    signed.signature.key_id.clone(),
                ));
                continue;
            }
            KeyEligibility::Revoked => {
                violations.push(ArtifactProvenanceViolation::SignerRevoked(
                    signed.signature.key_id.clone(),
                ));
                continue;
            }
            KeyEligibility::UsageNotAllowed => {
                violations.push(ArtifactProvenanceViolation::SignerUsageNotAllowed(
                    signed.signature.key_id.clone(),
                ));
                continue;
            }
        }
        match verifier.verify_artifact_provenance(
            &signed.signature.algorithm,
            &signed.signature.key_id,
            &provenance_signature_message(signed.statement_digest),
            &signed.signature.signature,
        ) {
            Ok(true) => {}
            Ok(false) => {
                violations.push(ArtifactProvenanceViolation::InvalidSignature(
                    signed.signature.key_id.clone(),
                ));
                continue;
            }
            Err(reason) => {
                violations.push(ArtifactProvenanceViolation::VerificationProviderError {
                    key_id: signed.signature.key_id.clone(),
                    reason,
                });
                continue;
            }
        }
        algorithms.insert(signed.signature.algorithm.clone());
        regions.insert(signed.statement.builder_region.clone());
        valid_digests.push(signed.statement_digest);
        valid_builders.push((
            signed.statement.builder_id.clone(),
            signed.statement.builder_region.clone(),
            signed.signature.algorithm.clone(),
            signed.signature.key_id.clone(),
        ));
    }

    if valid_builders.len() < policy.minimum_distinct_builders {
        violations.push(ArtifactProvenanceViolation::InsufficientBuilders {
            actual: valid_builders.len(),
            required: policy.minimum_distinct_builders,
        });
    }
    if regions.len() < policy.minimum_distinct_regions {
        violations.push(ArtifactProvenanceViolation::InsufficientRegions {
            actual: regions.len(),
            required: policy.minimum_distinct_regions,
        });
    }
    if policy.require_algorithm_diversity && algorithms.len() < 2 {
        violations.push(ArtifactProvenanceViolation::MissingAlgorithmDiversity);
    }
    if !violations.is_empty() {
        return Err(violations);
    }

    valid_builders.sort();
    valid_digests.sort();
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.artifact-provenance-quorum.v1\0");
    hasher.update(&artifact_set_digest.0);
    for digest in valid_digests {
        hasher.update(&digest.0);
    }
    Ok(VerifiedArtifactProvenance {
        artifact_set_digest,
        provenance_digest: hasher.finalize(),
        trust_snapshot_digest: digest_trust_snapshot(trust_snapshot)
            .expect("validated trust snapshot must have canonical digest"),
        builders: valid_builders,
    })
}

fn validate_statement(
    statement: &ArtifactProvenanceStatement,
) -> Result<(), ArtifactProvenanceError> {
    if statement.schema_version != ARTIFACT_PROVENANCE_SCHEMA {
        return Err(ArtifactProvenanceError::UnsupportedSchema);
    }
    validate_identifier(&statement.builder_id)?;
    validate_identifier(&statement.builder_region)?;
    if statement.inputs.is_empty() {
        return Err(ArtifactProvenanceError::EmptyInputs);
    }
    if statement.inputs.len() > MAX_PROVENANCE_INPUTS {
        return Err(ArtifactProvenanceError::TooManyInputs);
    }
    if statement.reproducible_match_count == 0 {
        return Err(ArtifactProvenanceError::InvalidReproducibilityCount);
    }
    let mut names = BTreeSet::new();
    let mut previous: Option<&str> = None;
    for input in &statement.inputs {
        validate_identifier(&input.name)?;
        if !names.insert(input.name.clone()) {
            return Err(ArtifactProvenanceError::DuplicateInput(input.name.clone()));
        }
        if previous.is_some_and(|value| value >= input.name.as_str()) {
            return Err(ArtifactProvenanceError::NonCanonicalInputOrder);
        }
        previous = Some(&input.name);
    }
    Ok(())
}

fn validate_identifier(value: &str) -> Result<(), ArtifactProvenanceError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > 256
        || value.chars().any(char::is_control)
    {
        return Err(ArtifactProvenanceError::InvalidIdentifier);
    }
    Ok(())
}

fn provenance_signature_message(digest: Sha256Digest) -> Vec<u8> {
    let mut message = b"symthaea.fabrication.artifact-provenance-signature.v1\0".to_vec();
    message.extend_from_slice(&digest.0);
    message
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statement_inputs_must_be_canonical() {
        let statement = ArtifactProvenanceStatement {
            schema_version: ARTIFACT_PROVENANCE_SCHEMA.into(),
            artifact_set_digest: Sha256Digest([1; 32]),
            source_tree_digest: Sha256Digest([2; 32]),
            builder_id: "builder-a".into(),
            builder_region: "region-a".into(),
            build_environment_digest: Sha256Digest([3; 32]),
            dependency_lock_digest: Sha256Digest([4; 32]),
            inputs: vec![
                ProvenanceInput {
                    name: "z".into(),
                    digest: Sha256Digest([5; 32]),
                },
                ProvenanceInput {
                    name: "a".into(),
                    digest: Sha256Digest([6; 32]),
                },
            ],
            reproducible_match_count: 2,
            built_at_unix_s: 10,
        };
        assert_eq!(
            validate_statement(&statement),
            Err(ArtifactProvenanceError::NonCanonicalInputOrder)
        );
    }
}
