// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bound emergency rollback to a previously authorized promotion.
//!
//! Rollback is a new release decision, not a deletion of history. Authority is
//! limited to a prior promotion and binds the triggering incidents, target
//! artifact provenance, current regional gateway quorum, and witnessed
//! transparency checkpoint.

use crate::artifact_provenance::VerifiedArtifactProvenance;
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::region_quorum::{RegionalQuorumEvidence, digest_regional_quorum_evidence};
use crate::release_promotion::AuthorizedReleasePromotion;
use crate::threshold::VerifiedThresholdCeremony;
use crate::transparency_witness::VerifiedTransparencyWitnessQuorum;
use serde::{Deserialize, Serialize};

pub const RELEASE_ROLLBACK_SCHEMA: &str = "symthaea.fabrication.release-rollback.v1";
pub const MAX_ROLLBACK_INCIDENTS: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseRollbackEvidence {
    pub schema_version: String,
    pub rollback_sequence: u64,
    pub from_promotion_digest: Sha256Digest,
    pub from_promotion_sequence: u64,
    pub target_promotion_digest: Sha256Digest,
    pub target_promotion_sequence: u64,
    pub target_candidate_digest: Sha256Digest,
    pub target_software_version: String,
    pub target_source_tree_digest: Sha256Digest,
    pub target_artifact_set_digest: Sha256Digest,
    pub target_artifact_provenance_digest: Sha256Digest,
    pub regional_quorum_digest: Sha256Digest,
    pub transparency_checkpoint_digest: Sha256Digest,
    pub transparency_witness_quorum_digest: Sha256Digest,
    pub triggering_incident_digests: Vec<Sha256Digest>,
    pub compatibility_evidence_digest: Sha256Digest,
    pub authorized_at_unix_s: u64,
    pub expires_at_unix_s: u64,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseRollbackPolicy {
    pub maximum_authorization_duration_s: u64,
    pub minimum_triggering_incidents: usize,
    pub maximum_triggering_incidents: usize,
    pub require_target_provenance: bool,
    pub require_compatibility_evidence: bool,
}

impl Default for ReleaseRollbackPolicy {
    fn default() -> Self {
        Self {
            maximum_authorization_duration_s: 600,
            minimum_triggering_incidents: 1,
            maximum_triggering_incidents: MAX_ROLLBACK_INCIDENTS,
            require_target_provenance: true,
            require_compatibility_evidence: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReleaseRollbackError {
    UnsupportedSchema,
    InvalidPolicy,
    InvalidWindow,
    InvalidSequence,
    TargetNotPrior,
    SamePromotion,
    InvalidReason,
    MissingIncidents,
    TooManyIncidents,
    DuplicateIncident,
    NonCanonicalIncidentOrder,
    TargetProvenanceMismatch,
    RegionalQuorumMismatch,
    TransparencyWitnessMismatch,
    MissingCompatibilityEvidence,
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    RegionalQuorum(String),
    Encoding(String),
}

#[derive(Debug, Clone)]
pub struct AuthorizedReleaseRollback {
    evidence: ReleaseRollbackEvidence,
    rollback_digest: Sha256Digest,
    ceremony_digest: Sha256Digest,
}

impl AuthorizedReleaseRollback {
    pub fn evidence(&self) -> &ReleaseRollbackEvidence {
        &self.evidence
    }
    pub fn rollback_digest(&self) -> Sha256Digest {
        self.rollback_digest
    }
    pub fn ceremony_digest(&self) -> Sha256Digest {
        self.ceremony_digest
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_release_rollback_evidence(
    rollback_sequence: u64,
    current: &AuthorizedReleasePromotion,
    target: &AuthorizedReleasePromotion,
    target_provenance: &VerifiedArtifactProvenance,
    regional_quorum: &RegionalQuorumEvidence,
    witness_quorum: &VerifiedTransparencyWitnessQuorum,
    mut triggering_incident_digests: Vec<Sha256Digest>,
    compatibility_evidence_digest: Sha256Digest,
    authorized_at_unix_s: u64,
    expires_at_unix_s: u64,
    reason: impl Into<String>,
    policy: &ReleaseRollbackPolicy,
) -> Result<ReleaseRollbackEvidence, ReleaseRollbackError> {
    validate_policy(policy)?;
    if rollback_sequence == 0 {
        return Err(ReleaseRollbackError::InvalidSequence);
    }
    if current.promotion_digest() == target.promotion_digest() {
        return Err(ReleaseRollbackError::SamePromotion);
    }
    if target.evidence().promotion_sequence >= current.evidence().promotion_sequence {
        return Err(ReleaseRollbackError::TargetNotPrior);
    }
    if authorized_at_unix_s >= expires_at_unix_s
        || expires_at_unix_s.saturating_sub(authorized_at_unix_s)
            > policy.maximum_authorization_duration_s
    {
        return Err(ReleaseRollbackError::InvalidWindow);
    }
    triggering_incident_digests.sort();
    validate_incidents(&triggering_incident_digests, policy)?;
    if policy.require_target_provenance
        && target_provenance.artifact_set_digest() != target.evidence().artifact_set_digest
    {
        return Err(ReleaseRollbackError::TargetProvenanceMismatch);
    }
    if regional_quorum.gateway_consensus_digest != current.evidence().gateway_consensus_digest
        || regional_quorum.gateway_state_digest != current.evidence().gateway_state_digest
        || regional_quorum.gateway_generation != current.evidence().gateway_generation
    {
        return Err(ReleaseRollbackError::RegionalQuorumMismatch);
    }
    if witness_quorum.checkpoint_digest() != current.evidence().transparency_checkpoint_digest {
        return Err(ReleaseRollbackError::TransparencyWitnessMismatch);
    }
    if policy.require_compatibility_evidence
        && compatibility_evidence_digest == Sha256Digest([0; 32])
    {
        return Err(ReleaseRollbackError::MissingCompatibilityEvidence);
    }
    let reason = reason.into();
    validate_reason(&reason)?;
    Ok(ReleaseRollbackEvidence {
        schema_version: RELEASE_ROLLBACK_SCHEMA.into(),
        rollback_sequence,
        from_promotion_digest: current.promotion_digest(),
        from_promotion_sequence: current.evidence().promotion_sequence,
        target_promotion_digest: target.promotion_digest(),
        target_promotion_sequence: target.evidence().promotion_sequence,
        target_candidate_digest: target.evidence().candidate_digest,
        target_software_version: target.evidence().software_version.clone(),
        target_source_tree_digest: target.evidence().source_tree_digest,
        target_artifact_set_digest: target.evidence().artifact_set_digest,
        target_artifact_provenance_digest: target_provenance.provenance_digest(),
        regional_quorum_digest: digest_regional_quorum_evidence(regional_quorum)
            .map_err(|error| ReleaseRollbackError::RegionalQuorum(format!("{error:?}")))?,
        transparency_checkpoint_digest: current.evidence().transparency_checkpoint_digest,
        transparency_witness_quorum_digest: witness_quorum.witness_quorum_digest(),
        triggering_incident_digests,
        compatibility_evidence_digest,
        authorized_at_unix_s,
        expires_at_unix_s,
        reason,
    })
}

pub fn digest_release_rollback(
    evidence: &ReleaseRollbackEvidence,
) -> Result<Sha256Digest, ReleaseRollbackError> {
    validate_evidence(evidence)?;
    let bytes = serde_json::to_vec(evidence)
        .map_err(|error| ReleaseRollbackError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.release-rollback-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn authorize_release_rollback(
    evidence: ReleaseRollbackEvidence,
    ceremony: &VerifiedThresholdCeremony,
) -> Result<AuthorizedReleaseRollback, ReleaseRollbackError> {
    let rollback_digest = digest_release_rollback(&evidence)?;
    if ceremony.purpose() != "release-rollback" {
        return Err(ReleaseRollbackError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != rollback_digest {
        return Err(ReleaseRollbackError::CeremonyPayloadMismatch);
    }
    Ok(AuthorizedReleaseRollback {
        evidence,
        rollback_digest,
        ceremony_digest: ceremony.ceremony_digest(),
    })
}

fn validate_evidence(evidence: &ReleaseRollbackEvidence) -> Result<(), ReleaseRollbackError> {
    if evidence.schema_version != RELEASE_ROLLBACK_SCHEMA {
        return Err(ReleaseRollbackError::UnsupportedSchema);
    }
    if evidence.rollback_sequence == 0
        || evidence.from_promotion_sequence == 0
        || evidence.target_promotion_sequence == 0
    {
        return Err(ReleaseRollbackError::InvalidSequence);
    }
    if evidence.target_promotion_sequence >= evidence.from_promotion_sequence {
        return Err(ReleaseRollbackError::TargetNotPrior);
    }
    if evidence.from_promotion_digest == evidence.target_promotion_digest {
        return Err(ReleaseRollbackError::SamePromotion);
    }
    if evidence.authorized_at_unix_s >= evidence.expires_at_unix_s {
        return Err(ReleaseRollbackError::InvalidWindow);
    }
    if evidence.triggering_incident_digests.is_empty() {
        return Err(ReleaseRollbackError::MissingIncidents);
    }
    validate_canonical_incidents(&evidence.triggering_incident_digests)?;
    validate_reason(&evidence.reason)?;
    Ok(())
}

fn validate_incidents(
    incidents: &[Sha256Digest],
    policy: &ReleaseRollbackPolicy,
) -> Result<(), ReleaseRollbackError> {
    if incidents.len() < policy.minimum_triggering_incidents {
        return Err(ReleaseRollbackError::MissingIncidents);
    }
    if incidents.len() > policy.maximum_triggering_incidents {
        return Err(ReleaseRollbackError::TooManyIncidents);
    }
    validate_canonical_incidents(incidents)
}

fn validate_canonical_incidents(incidents: &[Sha256Digest]) -> Result<(), ReleaseRollbackError> {
    let mut previous = None;
    for digest in incidents {
        if previous == Some(*digest) {
            return Err(ReleaseRollbackError::DuplicateIncident);
        }
        if previous.is_some_and(|value| value > *digest) {
            return Err(ReleaseRollbackError::NonCanonicalIncidentOrder);
        }
        previous = Some(*digest);
    }
    Ok(())
}

fn validate_policy(policy: &ReleaseRollbackPolicy) -> Result<(), ReleaseRollbackError> {
    if policy.maximum_authorization_duration_s == 0
        || policy.minimum_triggering_incidents == 0
        || policy.maximum_triggering_incidents < policy.minimum_triggering_incidents
        || policy.maximum_triggering_incidents > MAX_ROLLBACK_INCIDENTS
    {
        return Err(ReleaseRollbackError::InvalidPolicy);
    }
    Ok(())
}

fn validate_reason(reason: &str) -> Result<(), ReleaseRollbackError> {
    if reason.trim().is_empty()
        || reason != reason.trim()
        || reason.len() > 4_096
        || reason.chars().any(char::is_control)
    {
        return Err(ReleaseRollbackError::InvalidReason);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incident_order_is_part_of_canonicality() {
        assert_eq!(
            validate_canonical_incidents(&[Sha256Digest([2; 32]), Sha256Digest([1; 32])]),
            Err(ReleaseRollbackError::NonCanonicalIncidentOrder)
        );
    }
}
