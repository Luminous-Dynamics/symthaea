// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Principal-bound witness federation for authenticated transparency checkpoints.
//!
//! Witness statements contain checkpoint/observation semantics only. Principal,
//! organization, region, algorithm, and verification-key identity come from the
//! exact root-authorized role proof, never from witness-supplied identity labels.

use std::collections::BTreeSet;

use serde::Serialize;

use crate::attestation::digest_signature_algorithm;
use crate::{
    AuthenticatedTransparencyCheckpoint, AuthorizedTrustRoleAttestation, FramedDigest,
    RoleSignerIdentity, RootRoleQuorumProof, Sha256Digest, SignatureAlgorithm, TrustRole,
    TrustedTime,
};

const WITNESS_STATEMENT_DOMAIN: &str = "symthaea.transparency-witness-statement.identity.v1";
const WITNESS_OBSERVATION_DOMAIN: &str = "symthaea.verified-transparency-witness.identity.v1";
const WITNESS_QUORUM_DOMAIN: &str = "symthaea.transparency-witness-quorum.identity.v1";
pub const MAX_WITNESS_OBSERVATIONS: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TransparencyWitnessStatement {
    checkpoint_sha256: Sha256Digest,
    tree_head_sha256: Sha256Digest,
    tree_size: u64,
    root_sha256: Sha256Digest,
    observed_time_authority_sha256: Sha256Digest,
    observed_earliest_unix_s: u64,
    observed_latest_unix_s: u64,
    statement_sha256: Sha256Digest,
}

impl TransparencyWitnessStatement {
    pub fn new(
        checkpoint: &AuthenticatedTransparencyCheckpoint,
        observed_time: &TrustedTime,
    ) -> Self {
        let (earliest, latest) = observed_time.consensus_interval();
        let statement_sha256 = witness_statement_digest(
            checkpoint.checkpoint_sha256(),
            checkpoint.statement().tree_head_sha256(),
            checkpoint.statement().tree_size(),
            checkpoint.statement().root_sha256(),
            observed_time.authority_sha256(),
            earliest,
            latest,
        );
        Self {
            checkpoint_sha256: checkpoint.checkpoint_sha256().clone(),
            tree_head_sha256: checkpoint.statement().tree_head_sha256().clone(),
            tree_size: checkpoint.statement().tree_size(),
            root_sha256: checkpoint.statement().root_sha256().clone(),
            observed_time_authority_sha256: observed_time.authority_sha256().clone(),
            observed_earliest_unix_s: earliest,
            observed_latest_unix_s: latest,
            statement_sha256,
        }
    }

    pub fn checkpoint_sha256(&self) -> &Sha256Digest { &self.checkpoint_sha256 }
    pub fn tree_head_sha256(&self) -> &Sha256Digest { &self.tree_head_sha256 }
    pub fn tree_size(&self) -> u64 { self.tree_size }
    pub fn root_sha256(&self) -> &Sha256Digest { &self.root_sha256 }
    pub fn observed_time_authority_sha256(&self) -> &Sha256Digest {
        &self.observed_time_authority_sha256
    }
    pub fn observed_interval(&self) -> (u64, u64) {
        (self.observed_earliest_unix_s, self.observed_latest_unix_s)
    }
    pub fn statement_sha256(&self) -> &Sha256Digest { &self.statement_sha256 }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransparencyWitnessObservationError {
    CheckpointMismatch,
    ObservedTimeMismatch,
    ObservationNotDefinitelyAfterCheckpoint,
    TimeDependsOnCheckpoint,
    WrongRole,
    RootAuthorityMismatch,
    SubjectMismatch,
    PayloadMismatch,
    ContextMismatch,
    RoleProofMismatch,
    RoleProofWrongRole,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VerifiedTransparencyWitnessObservation {
    checkpoint_sha256: Sha256Digest,
    witness_role_authority_sha256: Sha256Digest,
    root_authority_sha256: Sha256Digest,
    trust_snapshot_authority_sha256: Sha256Digest,
    observed_time_authority_sha256: Sha256Digest,
    observed_earliest_unix_s: u64,
    observed_latest_unix_s: u64,
    statement_sha256: Sha256Digest,
    signers: Vec<RoleSignerIdentity>,
    observation_sha256: Sha256Digest,
}

impl VerifiedTransparencyWitnessObservation {
    pub fn checkpoint_sha256(&self) -> &Sha256Digest { &self.checkpoint_sha256 }
    pub fn witness_role_authority_sha256(&self) -> &Sha256Digest {
        &self.witness_role_authority_sha256
    }
    pub fn root_authority_sha256(&self) -> &Sha256Digest { &self.root_authority_sha256 }
    pub fn trust_snapshot_authority_sha256(&self) -> &Sha256Digest {
        &self.trust_snapshot_authority_sha256
    }
    pub fn observed_time_authority_sha256(&self) -> &Sha256Digest {
        &self.observed_time_authority_sha256
    }
    pub fn observed_interval(&self) -> (u64, u64) {
        (self.observed_earliest_unix_s, self.observed_latest_unix_s)
    }
    pub fn signers(&self) -> &[RoleSignerIdentity] { &self.signers }
    pub fn observation_sha256(&self) -> &Sha256Digest { &self.observation_sha256 }
}

pub fn verify_transparency_witness_observation(
    statement: TransparencyWitnessStatement,
    checkpoint: &AuthenticatedTransparencyCheckpoint,
    observed_time: &TrustedTime,
    witness_authority: &AuthorizedTrustRoleAttestation,
    witness_role_proof: &RootRoleQuorumProof,
) -> Result<VerifiedTransparencyWitnessObservation, TransparencyWitnessObservationError> {
    if statement.checkpoint_sha256() != checkpoint.checkpoint_sha256()
        || statement.tree_head_sha256() != checkpoint.statement().tree_head_sha256()
        || statement.tree_size() != checkpoint.statement().tree_size()
        || statement.root_sha256() != checkpoint.statement().root_sha256()
    {
        return Err(TransparencyWitnessObservationError::CheckpointMismatch);
    }
    if statement.observed_time_authority_sha256() != observed_time.authority_sha256()
        || statement.observed_interval() != observed_time.consensus_interval()
    {
        return Err(TransparencyWitnessObservationError::ObservedTimeMismatch);
    }
    let (_, checkpoint_latest) = checkpoint.statement().consensus_interval();
    let (observed_earliest, _) = statement.observed_interval();
    if observed_earliest < checkpoint_latest {
        return Err(TransparencyWitnessObservationError::ObservationNotDefinitelyAfterCheckpoint);
    }
    if observed_time.bindings().iter().any(|binding| {
        binding.source_artifact_sha256() == checkpoint.checkpoint_sha256()
            || binding.source_artifact_sha256() == checkpoint.statement().tree_head_sha256()
    }) {
        return Err(TransparencyWitnessObservationError::TimeDependsOnCheckpoint);
    }
    if witness_authority.role() != TrustRole::TransparencyWitness {
        return Err(TransparencyWitnessObservationError::WrongRole);
    }
    if witness_authority.root_authority_sha256() != checkpoint.root_authority_sha256()
        || observed_time.root_authority_sha256() != checkpoint.root_authority_sha256()
    {
        return Err(TransparencyWitnessObservationError::RootAuthorityMismatch);
    }
    if witness_authority.subject_sha256() != checkpoint.checkpoint_sha256() {
        return Err(TransparencyWitnessObservationError::SubjectMismatch);
    }
    if witness_authority.payload_sha256() != statement.statement_sha256() {
        return Err(TransparencyWitnessObservationError::PayloadMismatch);
    }
    if witness_authority.context_sha256() != Some(checkpoint.statement().tree_head_sha256()) {
        return Err(TransparencyWitnessObservationError::ContextMismatch);
    }
    if witness_role_proof.role() != TrustRole::TransparencyWitness {
        return Err(TransparencyWitnessObservationError::RoleProofWrongRole);
    }
    if witness_authority.role_quorum_proof_sha256() != witness_role_proof.proof_sha256() {
        return Err(TransparencyWitnessObservationError::RoleProofMismatch);
    }

    let mut signers = witness_role_proof.signers().to_vec();
    signers.sort_by(|left, right| {
        left.principal_id
            .cmp(&right.principal_id)
            .then(left.algorithm.cmp(&right.algorithm))
            .then(left.key_id.cmp(&right.key_id))
    });
    let observation_sha256 = witness_observation_digest(
        &statement,
        witness_authority.authority_sha256(),
        observed_time.authority_sha256(),
        &signers,
    );
    Ok(VerifiedTransparencyWitnessObservation {
        checkpoint_sha256: checkpoint.checkpoint_sha256().clone(),
        witness_role_authority_sha256: witness_authority.authority_sha256().clone(),
        root_authority_sha256: witness_authority.root_authority_sha256().clone(),
        trust_snapshot_authority_sha256: witness_authority.trust_snapshot_authority_sha256().clone(),
        observed_time_authority_sha256: observed_time.authority_sha256().clone(),
        observed_earliest_unix_s: statement.observed_earliest_unix_s,
        observed_latest_unix_s: statement.observed_latest_unix_s,
        statement_sha256: statement.statement_sha256,
        signers,
        observation_sha256,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TransparencyWitnessPolicy {
    pub minimum_distinct_observations: usize,
    pub minimum_distinct_principals: usize,
    pub minimum_distinct_organizations: usize,
    pub minimum_distinct_regions: usize,
    pub minimum_distinct_algorithms: usize,
    pub maximum_observations: usize,
    pub maximum_observation_delay_s: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransparencyWitnessPolicyIssue {
    ZeroObservationThreshold,
    ZeroPrincipalThreshold,
    ZeroMaximumObservations,
    MaximumObservationsTooLarge,
    ObservationThresholdExceedsMaximum,
    OrganizationThresholdExceedsPrincipals,
    RegionThresholdExceedsPrincipals,
    ZeroAlgorithmThreshold,
}

impl TransparencyWitnessPolicy {
    pub fn validate(&self) -> Result<(), Vec<TransparencyWitnessPolicyIssue>> {
        let mut issues = Vec::new();
        if self.minimum_distinct_observations == 0 {
            issues.push(TransparencyWitnessPolicyIssue::ZeroObservationThreshold);
        }
        if self.minimum_distinct_principals == 0 {
            issues.push(TransparencyWitnessPolicyIssue::ZeroPrincipalThreshold);
        }
        if self.maximum_observations == 0 {
            issues.push(TransparencyWitnessPolicyIssue::ZeroMaximumObservations);
        }
        if self.maximum_observations > MAX_WITNESS_OBSERVATIONS {
            issues.push(TransparencyWitnessPolicyIssue::MaximumObservationsTooLarge);
        }
        if self.minimum_distinct_observations > self.maximum_observations {
            issues.push(TransparencyWitnessPolicyIssue::ObservationThresholdExceedsMaximum);
        }
        if self.minimum_distinct_organizations > self.minimum_distinct_principals {
            issues.push(TransparencyWitnessPolicyIssue::OrganizationThresholdExceedsPrincipals);
        }
        if self.minimum_distinct_regions > self.minimum_distinct_principals {
            issues.push(TransparencyWitnessPolicyIssue::RegionThresholdExceedsPrincipals);
        }
        if self.minimum_distinct_algorithms == 0 {
            issues.push(TransparencyWitnessPolicyIssue::ZeroAlgorithmThreshold);
        }
        if issues.is_empty() { Ok(()) } else { Err(issues) }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum TransparencyWitnessQuorumFinding {
    InvalidPolicy,
    EmptyObservations,
    TooManyObservations,
    CheckpointMismatch,
    RootAuthorityMismatch,
    DuplicateWitnessAuthority,
    ObservationBeforeCheckpoint,
    ObservationTooLate,
    InsufficientObservations { actual: usize, required: usize },
    InsufficientPrincipals { actual: usize, required: usize },
    InsufficientOrganizations { actual: usize, required: usize },
    InsufficientRegions { actual: usize, required: usize },
    InsufficientAlgorithms { actual: usize, required: usize },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VerifiedTransparencyWitnessQuorum {
    checkpoint_sha256: Sha256Digest,
    root_authority_sha256: Sha256Digest,
    observation_sha256s: Vec<Sha256Digest>,
    principal_ids: Vec<String>,
    organization_ids: Vec<String>,
    region_ids: Vec<String>,
    quorum_sha256: Sha256Digest,
}

impl VerifiedTransparencyWitnessQuorum {
    pub fn checkpoint_sha256(&self) -> &Sha256Digest { &self.checkpoint_sha256 }
    pub fn root_authority_sha256(&self) -> &Sha256Digest { &self.root_authority_sha256 }
    pub fn observation_sha256s(&self) -> &[Sha256Digest] { &self.observation_sha256s }
    pub fn principal_ids(&self) -> &[String] { &self.principal_ids }
    pub fn organization_ids(&self) -> &[String] { &self.organization_ids }
    pub fn region_ids(&self) -> &[String] { &self.region_ids }
    pub fn quorum_sha256(&self) -> &Sha256Digest { &self.quorum_sha256 }
    pub const fn configured_directory_diversity_established(&self) -> bool { true }
    pub const fn global_independence_established(&self) -> bool { false }
    pub const fn global_log_consistency_established(&self) -> bool { false }
}

pub fn verify_transparency_witness_quorum(
    checkpoint: &AuthenticatedTransparencyCheckpoint,
    observations: &[VerifiedTransparencyWitnessObservation],
    policy: &TransparencyWitnessPolicy,
) -> Result<VerifiedTransparencyWitnessQuorum, Vec<TransparencyWitnessQuorumFinding>> {
    let mut findings = Vec::new();
    if policy.validate().is_err() {
        findings.push(TransparencyWitnessQuorumFinding::InvalidPolicy);
    }
    if observations.is_empty() {
        findings.push(TransparencyWitnessQuorumFinding::EmptyObservations);
    }
    if observations.len() > policy.maximum_observations {
        findings.push(TransparencyWitnessQuorumFinding::TooManyObservations);
    }

    let mut authority_ids = BTreeSet::new();
    let mut principal_ids = BTreeSet::new();
    let mut organization_ids = BTreeSet::new();
    let mut region_ids = BTreeSet::new();
    let mut algorithms = BTreeSet::new();
    let mut observation_sha256s = Vec::new();
    let (_, checkpoint_latest) = checkpoint.statement().consensus_interval();

    for observation in observations {
        if observation.checkpoint_sha256() != checkpoint.checkpoint_sha256() {
            findings.push(TransparencyWitnessQuorumFinding::CheckpointMismatch);
        }
        if observation.root_authority_sha256() != checkpoint.root_authority_sha256() {
            findings.push(TransparencyWitnessQuorumFinding::RootAuthorityMismatch);
        }
        if !authority_ids.insert(observation.witness_role_authority_sha256().clone()) {
            findings.push(TransparencyWitnessQuorumFinding::DuplicateWitnessAuthority);
        }
        let (observed_earliest, observed_latest) = observation.observed_interval();
        if observed_earliest < checkpoint_latest {
            findings.push(TransparencyWitnessQuorumFinding::ObservationBeforeCheckpoint);
        }
        if observed_earliest > checkpoint_latest.saturating_add(policy.maximum_observation_delay_s) {
            findings.push(TransparencyWitnessQuorumFinding::ObservationTooLate);
        }
        debug_assert!(observed_latest >= observed_earliest);
        for signer in observation.signers() {
            principal_ids.insert(signer.principal_id.clone());
            organization_ids.insert(signer.organization_id.clone());
            region_ids.insert(signer.region_id.clone());
            algorithms.insert(signer.algorithm.clone());
        }
        observation_sha256s.push(observation.observation_sha256().clone());
    }

    if authority_ids.len() < policy.minimum_distinct_observations {
        findings.push(TransparencyWitnessQuorumFinding::InsufficientObservations {
            actual: authority_ids.len(), required: policy.minimum_distinct_observations,
        });
    }
    if principal_ids.len() < policy.minimum_distinct_principals {
        findings.push(TransparencyWitnessQuorumFinding::InsufficientPrincipals {
            actual: principal_ids.len(), required: policy.minimum_distinct_principals,
        });
    }
    if organization_ids.len() < policy.minimum_distinct_organizations {
        findings.push(TransparencyWitnessQuorumFinding::InsufficientOrganizations {
            actual: organization_ids.len(), required: policy.minimum_distinct_organizations,
        });
    }
    if region_ids.len() < policy.minimum_distinct_regions {
        findings.push(TransparencyWitnessQuorumFinding::InsufficientRegions {
            actual: region_ids.len(), required: policy.minimum_distinct_regions,
        });
    }
    if algorithms.len() < policy.minimum_distinct_algorithms {
        findings.push(TransparencyWitnessQuorumFinding::InsufficientAlgorithms {
            actual: algorithms.len(), required: policy.minimum_distinct_algorithms,
        });
    }
    if !findings.is_empty() { return Err(findings); }

    observation_sha256s.sort();
    let principal_ids: Vec<_> = principal_ids.into_iter().collect();
    let organization_ids: Vec<_> = organization_ids.into_iter().collect();
    let region_ids: Vec<_> = region_ids.into_iter().collect();
    let quorum_sha256 = witness_quorum_digest(
        checkpoint, policy, &observation_sha256s, &principal_ids, &organization_ids,
        &region_ids, &algorithms,
    );
    Ok(VerifiedTransparencyWitnessQuorum {
        checkpoint_sha256: checkpoint.checkpoint_sha256().clone(),
        root_authority_sha256: checkpoint.root_authority_sha256().clone(),
        observation_sha256s,
        principal_ids,
        organization_ids,
        region_ids,
        quorum_sha256,
    })
}

fn witness_statement_digest(
    checkpoint_sha256: &Sha256Digest,
    tree_head_sha256: &Sha256Digest,
    tree_size: u64,
    root_sha256: &Sha256Digest,
    observed_time_authority_sha256: &Sha256Digest,
    earliest: u64,
    latest: u64,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(WITNESS_STATEMENT_DOMAIN);
    digest.text(checkpoint_sha256.as_str());
    digest.text(tree_head_sha256.as_str());
    digest.text(&tree_size.to_string());
    digest.text(root_sha256.as_str());
    digest.text(observed_time_authority_sha256.as_str());
    digest.text(&earliest.to_string());
    digest.text(&latest.to_string());
    digest.digest()
}

fn witness_observation_digest(
    statement: &TransparencyWitnessStatement,
    witness_role_authority_sha256: &Sha256Digest,
    observed_time_authority_sha256: &Sha256Digest,
    signers: &[RoleSignerIdentity],
) -> Sha256Digest {
    let mut digest = FramedDigest::new(WITNESS_OBSERVATION_DOMAIN);
    digest.text(statement.statement_sha256.as_str());
    digest.text(witness_role_authority_sha256.as_str());
    digest.text(observed_time_authority_sha256.as_str());
    for signer in signers {
        digest.text("signer");
        digest_signature_algorithm(&mut digest, &signer.algorithm);
        digest.text(&signer.key_id);
        digest.text(signer.verification_key_sha256.as_str());
        digest.text(&signer.principal_id);
        digest.text(&signer.organization_id);
        digest.text(&signer.region_id);
    }
    digest.digest()
}

fn witness_quorum_digest(
    checkpoint: &AuthenticatedTransparencyCheckpoint,
    policy: &TransparencyWitnessPolicy,
    observation_sha256s: &[Sha256Digest],
    principal_ids: &[String],
    organization_ids: &[String],
    region_ids: &[String],
    algorithms: &BTreeSet<SignatureAlgorithm>,
) -> Sha256Digest {
    let mut digest = FramedDigest::new(WITNESS_QUORUM_DOMAIN);
    digest.text(checkpoint.checkpoint_sha256().as_str());
    digest.text(checkpoint.root_authority_sha256().as_str());
    digest.text(&policy.minimum_distinct_observations.to_string());
    digest.text(&policy.minimum_distinct_principals.to_string());
    digest.text(&policy.minimum_distinct_organizations.to_string());
    digest.text(&policy.minimum_distinct_regions.to_string());
    digest.text(&policy.minimum_distinct_algorithms.to_string());
    digest.text(&policy.maximum_observations.to_string());
    digest.text(&policy.maximum_observation_delay_s.to_string());
    for value in observation_sha256s { digest.text(value.as_str()); }
    for value in principal_ids { digest.text(value); }
    for value in organization_ids { digest.text(value); }
    for value in region_ids { digest.text(value); }
    for algorithm in algorithms {
        digest.text("algorithm");
        digest_signature_algorithm(&mut digest, algorithm);
    }
    digest.digest()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(value: &str) -> Sha256Digest { Sha256Digest::of_bytes(value.as_bytes()) }

    #[test]
    fn signer_metadata_carries_directory_derived_diversity() {
        let a = RoleSignerIdentity {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: "a".into(), verification_key_sha256: sha("a-key"),
            principal_id: "p-a".into(), organization_id: "org-a".into(), region_id: "r-a".into(),
        };
        let b = RoleSignerIdentity {
            algorithm: SignatureAlgorithm::Ed25519,
            key_id: "b".into(), verification_key_sha256: sha("b-key"),
            principal_id: "p-b".into(), organization_id: "org-b".into(), region_id: "r-b".into(),
        };
        assert_eq!(BTreeSet::from([a.principal_id, b.principal_id]).len(), 2);
        assert_eq!(BTreeSet::from([a.organization_id, b.organization_id]).len(), 2);
        assert_eq!(BTreeSet::from([a.region_id, b.region_id]).len(), 2);
    }
}
