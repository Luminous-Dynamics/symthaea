// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Composed release assurance across promotion, provenance, regions, and witnesses.

use crate::artifact_provenance::VerifiedArtifactProvenance;
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::region_quorum::{RegionalQuorumEvidence, digest_regional_quorum_evidence};
use crate::release_promotion::AuthorizedReleasePromotion;
use crate::threshold::VerifiedThresholdCeremony;
use crate::transparency_witness::VerifiedTransparencyWitnessQuorum;
use serde::{Deserialize, Serialize};

pub const RELEASE_ASSURANCE_SCHEMA: &str = "symthaea.fabrication.release-assurance.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseAssuranceEvidence {
    pub schema_version: String,
    pub promotion_digest: Sha256Digest,
    pub candidate_digest: Sha256Digest,
    pub artifact_set_digest: Sha256Digest,
    pub artifact_provenance_digest: Sha256Digest,
    pub regional_quorum_digest: Sha256Digest,
    pub gateway_consensus_digest: Sha256Digest,
    pub transparency_checkpoint_digest: Sha256Digest,
    pub transparency_witness_quorum_digest: Sha256Digest,
    pub assured_at_unix_s: u64,
    pub expires_at_unix_s: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleaseAssurancePolicy {
    pub maximum_assurance_duration_s: u64,
}

impl Default for ReleaseAssurancePolicy {
    fn default() -> Self {
        Self {
            maximum_assurance_duration_s: 600,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReleaseAssuranceError {
    UnsupportedSchema,
    InvalidPolicy,
    InvalidWindow,
    PromotionExpired,
    ArtifactProvenanceMismatch,
    RegionalQuorumMismatch,
    TransparencyWitnessMismatch,
    CeremonyPurposeMismatch,
    CeremonyPayloadMismatch,
    RegionalQuorum(String),
    Encoding(String),
}

#[derive(Debug, Clone)]
pub struct AssuredReleasePromotion {
    evidence: ReleaseAssuranceEvidence,
    assurance_digest: Sha256Digest,
    ceremony_digest: Sha256Digest,
}

impl AssuredReleasePromotion {
    pub fn evidence(&self) -> &ReleaseAssuranceEvidence {
        &self.evidence
    }
    pub fn assurance_digest(&self) -> Sha256Digest {
        self.assurance_digest
    }
    pub fn ceremony_digest(&self) -> Sha256Digest {
        self.ceremony_digest
    }
}

#[allow(clippy::too_many_arguments)]
pub fn build_release_assurance_evidence(
    promotion: &AuthorizedReleasePromotion,
    provenance: &VerifiedArtifactProvenance,
    regional_quorum: &RegionalQuorumEvidence,
    witness_quorum: &VerifiedTransparencyWitnessQuorum,
    assured_at_unix_s: u64,
    expires_at_unix_s: u64,
    policy: &ReleaseAssurancePolicy,
) -> Result<ReleaseAssuranceEvidence, ReleaseAssuranceError> {
    if policy.maximum_assurance_duration_s == 0 {
        return Err(ReleaseAssuranceError::InvalidPolicy);
    }
    if assured_at_unix_s >= expires_at_unix_s
        || expires_at_unix_s.saturating_sub(assured_at_unix_s) > policy.maximum_assurance_duration_s
    {
        return Err(ReleaseAssuranceError::InvalidWindow);
    }
    if assured_at_unix_s >= promotion.evidence().expires_at_unix_s {
        return Err(ReleaseAssuranceError::PromotionExpired);
    }
    if provenance.artifact_set_digest() != promotion.evidence().artifact_set_digest {
        return Err(ReleaseAssuranceError::ArtifactProvenanceMismatch);
    }
    if regional_quorum.gateway_consensus_digest != promotion.evidence().gateway_consensus_digest
        || regional_quorum.gateway_state_digest != promotion.evidence().gateway_state_digest
        || regional_quorum.gateway_generation != promotion.evidence().gateway_generation
    {
        return Err(ReleaseAssuranceError::RegionalQuorumMismatch);
    }
    if witness_quorum.checkpoint_digest() != promotion.evidence().transparency_checkpoint_digest {
        return Err(ReleaseAssuranceError::TransparencyWitnessMismatch);
    }
    Ok(ReleaseAssuranceEvidence {
        schema_version: RELEASE_ASSURANCE_SCHEMA.into(),
        promotion_digest: promotion.promotion_digest(),
        candidate_digest: promotion.evidence().candidate_digest,
        artifact_set_digest: promotion.evidence().artifact_set_digest,
        artifact_provenance_digest: provenance.provenance_digest(),
        regional_quorum_digest: digest_regional_quorum_evidence(regional_quorum)
            .map_err(|error| ReleaseAssuranceError::RegionalQuorum(format!("{error:?}")))?,
        gateway_consensus_digest: promotion.evidence().gateway_consensus_digest,
        transparency_checkpoint_digest: promotion.evidence().transparency_checkpoint_digest,
        transparency_witness_quorum_digest: witness_quorum.witness_quorum_digest(),
        assured_at_unix_s,
        expires_at_unix_s,
    })
}

pub fn digest_release_assurance(
    evidence: &ReleaseAssuranceEvidence,
) -> Result<Sha256Digest, ReleaseAssuranceError> {
    if evidence.schema_version != RELEASE_ASSURANCE_SCHEMA {
        return Err(ReleaseAssuranceError::UnsupportedSchema);
    }
    if evidence.assured_at_unix_s >= evidence.expires_at_unix_s {
        return Err(ReleaseAssuranceError::InvalidWindow);
    }
    let bytes = serde_json::to_vec(evidence)
        .map_err(|error| ReleaseAssuranceError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.release-assurance-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn authorize_release_assurance(
    evidence: ReleaseAssuranceEvidence,
    ceremony: &VerifiedThresholdCeremony,
) -> Result<AssuredReleasePromotion, ReleaseAssuranceError> {
    let assurance_digest = digest_release_assurance(&evidence)?;
    if ceremony.purpose() != "release-assurance" {
        return Err(ReleaseAssuranceError::CeremonyPurposeMismatch);
    }
    if ceremony.payload_digest() != assurance_digest {
        return Err(ReleaseAssuranceError::CeremonyPayloadMismatch);
    }
    Ok(AssuredReleasePromotion {
        evidence,
        assurance_digest,
        ceremony_digest: ceremony.ceremony_digest(),
    })
}
