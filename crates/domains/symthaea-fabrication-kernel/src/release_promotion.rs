// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-complete, threshold-authorized release promotion.

use crate::artifact_set::{ArtifactSetError, ReleaseArtifactSet, digest_release_artifact_set};
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::gateway_membership::{GatewayMembership, digest_gateway_membership};
use crate::lease_authority::AuthorizedPartitionLease;
use crate::release_certification::CertifiedReleaseCandidate;
use crate::threshold::VerifiedThresholdCeremony;
use crate::transparency::{
    TransparencyEntry, TransparencyError, TransparencyInclusionProof, digest_transparency_entry,
    verify_transparency_inclusion,
};
use crate::transparency_checkpoint::VerifiedTransparencyCheckpoint;
use serde::{Deserialize, Serialize};

pub const RELEASE_PROMOTION_SCHEMA: &str = "symthaea.fabrication.release-promotion.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleasePromotionEvidence {
    pub schema_version: String,
    pub promotion_sequence: u64,
    pub candidate_digest: Sha256Digest,
    pub software_version: String,
    pub source_tree_digest: Sha256Digest,
    pub artifact_set_digest: Sha256Digest,
    pub gateway_state_digest: Sha256Digest,
    pub gateway_generation: u64,
    pub gateway_consensus_digest: Sha256Digest,
    pub gateway_replay_digest: Sha256Digest,
    pub membership_digest: Sha256Digest,
    pub membership_epoch: u64,
    pub partition_lease_digest: Sha256Digest,
    pub fencing_token: u64,
    pub transparency_checkpoint_digest: Sha256Digest,
    pub transparency_root_digest: Sha256Digest,
    pub transparency_tree_size: u64,
    pub transparency_entry_sequence: u64,
    pub authorized_at_unix_s: u64,
    pub expires_at_unix_s: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleasePromotionPolicy {
    pub require_no_unresolved_incidents: bool,
    pub maximum_authorization_duration_s: u64,
    pub maximum_checkpoint_age_s: u64,
}

impl Default for ReleasePromotionPolicy {
    fn default() -> Self {
        Self {
            require_no_unresolved_incidents: true,
            maximum_authorization_duration_s: 900,
            maximum_checkpoint_age_s: 300,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReleasePromotionError {
    UnsupportedSchema,
    InvalidPolicy,
    InvalidWindow,
    PromotionSequenceZero,
    CandidateExpired,
    UnresolvedIncidents,
    ArtifactSet(ArtifactSetError),
    SourceTreeMismatch,
    MembershipInvalid(String),
    MembershipInactive,
    LeaseExpired,
    LeaseMembershipMismatch,
    LeaseCandidateMismatch,
    Transparency(TransparencyError),
    TransparencyEntryMismatch,
    TransparencyCheckpointMismatch,
    TransparencyCheckpointStale,
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    Encoding(String),
}

#[derive(Debug, Clone)]
pub struct AuthorizedReleasePromotion {
    evidence: ReleasePromotionEvidence,
    promotion_digest: Sha256Digest,
    ceremony_digest: Sha256Digest,
}

impl AuthorizedReleasePromotion {
    pub fn evidence(&self) -> &ReleasePromotionEvidence {
        &self.evidence
    }
    pub fn promotion_digest(&self) -> Sha256Digest {
        self.promotion_digest
    }
    pub fn ceremony_digest(&self) -> Sha256Digest {
        self.ceremony_digest
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_release_promotion_evidence(
    promotion_sequence: u64,
    candidate: &CertifiedReleaseCandidate,
    artifact_set: &ReleaseArtifactSet,
    gateway_replay_digest: Sha256Digest,
    membership: &GatewayMembership,
    lease: &AuthorizedPartitionLease,
    transparency_entry: &TransparencyEntry,
    inclusion_proof: &TransparencyInclusionProof,
    checkpoint: &VerifiedTransparencyCheckpoint,
    authorized_at_unix_s: u64,
    expires_at_unix_s: u64,
    policy: &ReleasePromotionPolicy,
) -> Result<ReleasePromotionEvidence, ReleasePromotionError> {
    validate_policy(policy)?;
    if promotion_sequence == 0 {
        return Err(ReleasePromotionError::PromotionSequenceZero);
    }
    if authorized_at_unix_s >= expires_at_unix_s
        || expires_at_unix_s - authorized_at_unix_s > policy.maximum_authorization_duration_s
    {
        return Err(ReleasePromotionError::InvalidWindow);
    }
    let candidate_evidence = candidate.candidate();
    if authorized_at_unix_s >= candidate_evidence.expires_at_unix_s {
        return Err(ReleasePromotionError::CandidateExpired);
    }
    if policy.require_no_unresolved_incidents
        && !candidate_evidence.unresolved_incident_digests.is_empty()
    {
        return Err(ReleasePromotionError::UnresolvedIncidents);
    }
    artifact_set
        .validate()
        .map_err(ReleasePromotionError::ArtifactSet)?;
    if artifact_set.source_tree_digest != candidate_evidence.source_tree_digest {
        return Err(ReleasePromotionError::SourceTreeMismatch);
    }
    membership
        .validate()
        .map_err(|error| ReleasePromotionError::MembershipInvalid(format!("{error:?}")))?;
    if !membership.is_active_at(authorized_at_unix_s) {
        return Err(ReleasePromotionError::MembershipInactive);
    }
    let lease_evidence = lease.lease();
    if authorized_at_unix_s.saturating_mul(1_000) >= lease_evidence.expires_at_unix_ms {
        return Err(ReleasePromotionError::LeaseExpired);
    }
    let membership_digest = digest_gateway_membership(membership)
        .map_err(|error| ReleasePromotionError::MembershipInvalid(format!("{error:?}")))?;
    if lease_evidence.membership_digest != membership_digest
        || lease_evidence.membership_epoch != membership.epoch
    {
        return Err(ReleasePromotionError::LeaseMembershipMismatch);
    }
    if lease_evidence.gateway_state_digest != candidate_evidence.gateway_state_digest
        || lease_evidence.gateway_generation != candidate_evidence.gateway_generation
        || lease_evidence.gateway_consensus_digest != candidate_evidence.gateway_consensus_digest
    {
        return Err(ReleasePromotionError::LeaseCandidateMismatch);
    }
    verify_transparency_inclusion(inclusion_proof).map_err(ReleasePromotionError::Transparency)?;
    if transparency_entry.kind != "release-candidate"
        || transparency_entry.subject_digest != candidate.candidate_digest()
        || digest_transparency_entry(transparency_entry)
            .map_err(ReleasePromotionError::Transparency)?
            != inclusion_proof.leaf_digest
        || transparency_entry.sequence != inclusion_proof.leaf_index.saturating_add(1)
    {
        return Err(ReleasePromotionError::TransparencyEntryMismatch);
    }
    let checkpoint_evidence = checkpoint.checkpoint();
    if checkpoint_evidence.root_digest != inclusion_proof.root_digest
        || checkpoint_evidence.log_size != inclusion_proof.tree_size
        || inclusion_proof.root_digest != checkpoint_evidence.root_digest
    {
        return Err(ReleasePromotionError::TransparencyCheckpointMismatch);
    }
    if authorized_at_unix_s < checkpoint_evidence.issued_at_unix_s
        || authorized_at_unix_s.saturating_sub(checkpoint_evidence.issued_at_unix_s)
            > policy.maximum_checkpoint_age_s
        || authorized_at_unix_s >= checkpoint_evidence.expires_at_unix_s
    {
        return Err(ReleasePromotionError::TransparencyCheckpointStale);
    }
    Ok(ReleasePromotionEvidence {
        schema_version: RELEASE_PROMOTION_SCHEMA.into(),
        promotion_sequence,
        candidate_digest: candidate.candidate_digest(),
        software_version: candidate_evidence.software_version.clone(),
        source_tree_digest: candidate_evidence.source_tree_digest,
        artifact_set_digest: digest_release_artifact_set(artifact_set)
            .map_err(ReleasePromotionError::ArtifactSet)?,
        gateway_state_digest: candidate_evidence.gateway_state_digest,
        gateway_generation: candidate_evidence.gateway_generation,
        gateway_consensus_digest: candidate_evidence.gateway_consensus_digest,
        gateway_replay_digest,
        membership_digest,
        membership_epoch: membership.epoch,
        partition_lease_digest: lease.lease_digest(),
        fencing_token: lease_evidence.fencing_token,
        transparency_checkpoint_digest: checkpoint.checkpoint_digest(),
        transparency_root_digest: checkpoint_evidence.root_digest,
        transparency_tree_size: checkpoint_evidence.log_size,
        transparency_entry_sequence: transparency_entry.sequence,
        authorized_at_unix_s,
        expires_at_unix_s,
    })
}

pub fn digest_release_promotion(
    evidence: &ReleasePromotionEvidence,
) -> Result<Sha256Digest, ReleasePromotionError> {
    if evidence.schema_version != RELEASE_PROMOTION_SCHEMA {
        return Err(ReleasePromotionError::UnsupportedSchema);
    }
    if evidence.promotion_sequence == 0 {
        return Err(ReleasePromotionError::PromotionSequenceZero);
    }
    if evidence.authorized_at_unix_s >= evidence.expires_at_unix_s {
        return Err(ReleasePromotionError::InvalidWindow);
    }
    let bytes = serde_json::to_vec(evidence)
        .map_err(|error| ReleasePromotionError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.release-promotion-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn authorize_release_promotion(
    evidence: ReleasePromotionEvidence,
    ceremony: &VerifiedThresholdCeremony,
) -> Result<AuthorizedReleasePromotion, ReleasePromotionError> {
    let promotion_digest = digest_release_promotion(&evidence)?;
    if ceremony.purpose() != "release-promotion" {
        return Err(ReleasePromotionError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != promotion_digest {
        return Err(ReleasePromotionError::CeremonyPayloadMismatch);
    }
    Ok(AuthorizedReleasePromotion {
        evidence,
        promotion_digest,
        ceremony_digest: ceremony.ceremony_digest(),
    })
}

fn validate_policy(policy: &ReleasePromotionPolicy) -> Result<(), ReleasePromotionError> {
    if policy.maximum_authorization_duration_s == 0 || policy.maximum_checkpoint_age_s == 0 {
        return Err(ReleasePromotionError::InvalidPolicy);
    }
    Ok(())
}
