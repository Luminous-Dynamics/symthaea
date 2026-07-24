// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reproducible replay contract for federated release promotion.

use crate::artifact_set::{ArtifactSetError, ReleaseArtifactSet, digest_release_artifact_set};
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::gateway_membership::{
    GatewayMembership, GatewayMembershipError, digest_gateway_membership,
};
use crate::lease_authority::AuthorizedPartitionLease;
use crate::release_promotion::AuthorizedReleasePromotion;
use crate::rollout::AuthorizedRolloutAdvance;
use crate::transparency_checkpoint::VerifiedTransparencyCheckpoint;
use serde::{Deserialize, Serialize};

pub const PROMOTION_REPLAY_SCHEMA: &str = "symthaea.fabrication.promotion-replay.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromotionReplayContract {
    pub schema_version: String,
    pub promotion_digest: Sha256Digest,
    pub promotion_ceremony_digest: Sha256Digest,
    pub candidate_digest: Sha256Digest,
    pub source_tree_digest: Sha256Digest,
    pub artifact_set_digest: Sha256Digest,
    pub gateway_replay_digest: Sha256Digest,
    pub membership_digest: Sha256Digest,
    pub membership_epoch: u64,
    pub partition_lease_digest: Sha256Digest,
    pub lease_ceremony_digest: Sha256Digest,
    pub fencing_token: u64,
    pub transparency_checkpoint_digest: Sha256Digest,
    pub transparency_root_digest: Sha256Digest,
    pub rollout_advance_digest: Option<Sha256Digest>,
    pub rollout_ceremony_digest: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromotionReplayError {
    ArtifactSet(ArtifactSetError),
    Membership(GatewayMembershipError),
    EvidenceMismatch(&'static str),
    Encoding(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromotionReplayMismatch {
    SchemaVersion,
    Promotion,
    PromotionCeremony,
    Candidate,
    SourceTree,
    ArtifactSet,
    GatewayReplay,
    Membership,
    MembershipEpoch,
    PartitionLease,
    LeaseCeremony,
    FencingToken,
    TransparencyCheckpoint,
    TransparencyRoot,
    RolloutAdvance,
    RolloutCeremony,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromotionReplayVerificationReport {
    pub mismatches: Vec<PromotionReplayMismatch>,
}

impl PromotionReplayVerificationReport {
    pub fn reproducible(&self) -> bool {
        self.mismatches.is_empty()
    }
}

pub fn build_promotion_replay_contract(
    promotion: &AuthorizedReleasePromotion,
    artifact_set: &ReleaseArtifactSet,
    membership: &GatewayMembership,
    lease: &AuthorizedPartitionLease,
    checkpoint: &VerifiedTransparencyCheckpoint,
    rollout: Option<&AuthorizedRolloutAdvance>,
) -> Result<PromotionReplayContract, PromotionReplayError> {
    let evidence = promotion.evidence();
    let artifact_set_digest =
        digest_release_artifact_set(artifact_set).map_err(PromotionReplayError::ArtifactSet)?;
    let membership_digest =
        digest_gateway_membership(membership).map_err(PromotionReplayError::Membership)?;
    if artifact_set_digest != evidence.artifact_set_digest {
        return Err(PromotionReplayError::EvidenceMismatch("artifact set"));
    }
    if membership_digest != evidence.membership_digest
        || membership.epoch != evidence.membership_epoch
    {
        return Err(PromotionReplayError::EvidenceMismatch("membership"));
    }
    if lease.lease_digest() != evidence.partition_lease_digest
        || lease.lease().fencing_token != evidence.fencing_token
    {
        return Err(PromotionReplayError::EvidenceMismatch("partition lease"));
    }
    if checkpoint.checkpoint_digest() != evidence.transparency_checkpoint_digest
        || checkpoint.checkpoint().root_digest != evidence.transparency_root_digest
    {
        return Err(PromotionReplayError::EvidenceMismatch(
            "transparency checkpoint",
        ));
    }
    if rollout
        .is_some_and(|advance| advance.advance().promotion_digest != promotion.promotion_digest())
    {
        return Err(PromotionReplayError::EvidenceMismatch("rollout promotion"));
    }
    Ok(PromotionReplayContract {
        schema_version: PROMOTION_REPLAY_SCHEMA.into(),
        promotion_digest: promotion.promotion_digest(),
        promotion_ceremony_digest: promotion.ceremony_digest(),
        candidate_digest: evidence.candidate_digest,
        source_tree_digest: evidence.source_tree_digest,
        artifact_set_digest,
        gateway_replay_digest: evidence.gateway_replay_digest,
        membership_digest,
        membership_epoch: membership.epoch,
        partition_lease_digest: lease.lease_digest(),
        lease_ceremony_digest: lease.ceremony_digest(),
        fencing_token: lease.lease().fencing_token,
        transparency_checkpoint_digest: checkpoint.checkpoint_digest(),
        transparency_root_digest: checkpoint.checkpoint().root_digest,
        rollout_advance_digest: rollout.map(AuthorizedRolloutAdvance::advance_digest),
        rollout_ceremony_digest: rollout.map(AuthorizedRolloutAdvance::ceremony_digest),
    })
}

pub fn verify_promotion_replay_contract(
    contract: &PromotionReplayContract,
    promotion: &AuthorizedReleasePromotion,
    artifact_set: &ReleaseArtifactSet,
    membership: &GatewayMembership,
    lease: &AuthorizedPartitionLease,
    checkpoint: &VerifiedTransparencyCheckpoint,
    rollout: Option<&AuthorizedRolloutAdvance>,
) -> Result<PromotionReplayVerificationReport, PromotionReplayError> {
    let expected = build_promotion_replay_contract(
        promotion,
        artifact_set,
        membership,
        lease,
        checkpoint,
        rollout,
    )?;
    let mut mismatches = Vec::new();
    compare(
        contract.schema_version != PROMOTION_REPLAY_SCHEMA,
        PromotionReplayMismatch::SchemaVersion,
        &mut mismatches,
    );
    compare(
        contract.promotion_digest != expected.promotion_digest,
        PromotionReplayMismatch::Promotion,
        &mut mismatches,
    );
    compare(
        contract.promotion_ceremony_digest != expected.promotion_ceremony_digest,
        PromotionReplayMismatch::PromotionCeremony,
        &mut mismatches,
    );
    compare(
        contract.candidate_digest != expected.candidate_digest,
        PromotionReplayMismatch::Candidate,
        &mut mismatches,
    );
    compare(
        contract.source_tree_digest != expected.source_tree_digest,
        PromotionReplayMismatch::SourceTree,
        &mut mismatches,
    );
    compare(
        contract.artifact_set_digest != expected.artifact_set_digest,
        PromotionReplayMismatch::ArtifactSet,
        &mut mismatches,
    );
    compare(
        contract.gateway_replay_digest != expected.gateway_replay_digest,
        PromotionReplayMismatch::GatewayReplay,
        &mut mismatches,
    );
    compare(
        contract.membership_digest != expected.membership_digest,
        PromotionReplayMismatch::Membership,
        &mut mismatches,
    );
    compare(
        contract.membership_epoch != expected.membership_epoch,
        PromotionReplayMismatch::MembershipEpoch,
        &mut mismatches,
    );
    compare(
        contract.partition_lease_digest != expected.partition_lease_digest,
        PromotionReplayMismatch::PartitionLease,
        &mut mismatches,
    );
    compare(
        contract.lease_ceremony_digest != expected.lease_ceremony_digest,
        PromotionReplayMismatch::LeaseCeremony,
        &mut mismatches,
    );
    compare(
        contract.fencing_token != expected.fencing_token,
        PromotionReplayMismatch::FencingToken,
        &mut mismatches,
    );
    compare(
        contract.transparency_checkpoint_digest != expected.transparency_checkpoint_digest,
        PromotionReplayMismatch::TransparencyCheckpoint,
        &mut mismatches,
    );
    compare(
        contract.transparency_root_digest != expected.transparency_root_digest,
        PromotionReplayMismatch::TransparencyRoot,
        &mut mismatches,
    );
    compare(
        contract.rollout_advance_digest != expected.rollout_advance_digest,
        PromotionReplayMismatch::RolloutAdvance,
        &mut mismatches,
    );
    compare(
        contract.rollout_ceremony_digest != expected.rollout_ceremony_digest,
        PromotionReplayMismatch::RolloutCeremony,
        &mut mismatches,
    );
    Ok(PromotionReplayVerificationReport { mismatches })
}

pub fn digest_promotion_replay_contract(
    contract: &PromotionReplayContract,
) -> Result<Sha256Digest, PromotionReplayError> {
    let bytes = serde_json::to_vec(contract)
        .map_err(|error| PromotionReplayError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.promotion-replay-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

fn compare(
    differs: bool,
    mismatch: PromotionReplayMismatch,
    output: &mut Vec<PromotionReplayMismatch>,
) {
    if differs {
        output.push(mismatch);
    }
}
