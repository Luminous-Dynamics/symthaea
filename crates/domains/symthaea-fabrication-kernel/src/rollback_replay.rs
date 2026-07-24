// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic replay contract for release rollback and retirement evidence.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::gateway_decommission_tracker::{
    GatewayDecommissionTracker, digest_gateway_decommission_tracker,
};
use crate::release_assurance::AssuredReleasePromotion;
use crate::release_lineage::{ReleaseLineage, digest_release_lineage};
use crate::release_rollback::AuthorizedReleaseRollback;
use crate::transparency_witness_tracker::{
    TransparencyWitnessTracker, digest_transparency_witness_tracker,
};
use serde::{Deserialize, Serialize};

pub const ROLLBACK_REPLAY_SCHEMA: &str = "symthaea.fabrication.rollback-replay.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RollbackReplayContract {
    pub schema_version: String,
    pub from_assurance_digest: Sha256Digest,
    pub target_assurance_digest: Sha256Digest,
    pub rollback_digest: Sha256Digest,
    pub release_lineage_digest: Sha256Digest,
    pub release_lineage_head: Sha256Digest,
    pub decommission_tracker_digest: Sha256Digest,
    pub witness_tracker_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RollbackReplayError {
    UnsupportedSchema,
    MissingLineageHead,
    RollbackFromMismatch,
    RollbackTargetMismatch,
    Encoding(String),
    Evidence(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RollbackReplayMismatch {
    FromAssurance,
    TargetAssurance,
    Rollback,
    ReleaseLineage,
    ReleaseLineageHead,
    DecommissionTracker,
    WitnessTracker,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RollbackReplayVerificationReport {
    pub mismatches: Vec<RollbackReplayMismatch>,
}

impl RollbackReplayVerificationReport {
    pub fn is_match(&self) -> bool {
        self.mismatches.is_empty()
    }
}

pub fn build_rollback_replay_contract(
    from_assurance: &AssuredReleasePromotion,
    target_assurance: &AssuredReleasePromotion,
    rollback: &AuthorizedReleaseRollback,
    lineage: &ReleaseLineage,
    decommission_tracker: &GatewayDecommissionTracker,
    witness_tracker: &TransparencyWitnessTracker,
) -> Result<RollbackReplayContract, RollbackReplayError> {
    if rollback.evidence().from_promotion_digest != from_assurance.evidence().promotion_digest {
        return Err(RollbackReplayError::RollbackFromMismatch);
    }
    if rollback.evidence().target_promotion_digest != target_assurance.evidence().promotion_digest {
        return Err(RollbackReplayError::RollbackTargetMismatch);
    }
    let release_lineage_head = lineage
        .chain_head()
        .ok_or(RollbackReplayError::MissingLineageHead)?;
    Ok(RollbackReplayContract {
        schema_version: ROLLBACK_REPLAY_SCHEMA.into(),
        from_assurance_digest: from_assurance.assurance_digest(),
        target_assurance_digest: target_assurance.assurance_digest(),
        rollback_digest: rollback.rollback_digest(),
        release_lineage_digest: digest_release_lineage(lineage)
            .map_err(|error| RollbackReplayError::Evidence(format!("{error:?}")))?,
        release_lineage_head,
        decommission_tracker_digest: digest_gateway_decommission_tracker(decommission_tracker)
            .map_err(|error| RollbackReplayError::Evidence(format!("{error:?}")))?,
        witness_tracker_digest: digest_transparency_witness_tracker(witness_tracker)
            .map_err(|error| RollbackReplayError::Evidence(format!("{error:?}")))?,
    })
}

pub fn digest_rollback_replay_contract(
    contract: &RollbackReplayContract,
) -> Result<Sha256Digest, RollbackReplayError> {
    if contract.schema_version != ROLLBACK_REPLAY_SCHEMA {
        return Err(RollbackReplayError::UnsupportedSchema);
    }
    let bytes = serde_json::to_vec(contract)
        .map_err(|error| RollbackReplayError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.rollback-replay-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn verify_rollback_replay_contract(
    contract: &RollbackReplayContract,
    from_assurance: &AssuredReleasePromotion,
    target_assurance: &AssuredReleasePromotion,
    rollback: &AuthorizedReleaseRollback,
    lineage: &ReleaseLineage,
    decommission_tracker: &GatewayDecommissionTracker,
    witness_tracker: &TransparencyWitnessTracker,
) -> Result<RollbackReplayVerificationReport, RollbackReplayError> {
    if contract.schema_version != ROLLBACK_REPLAY_SCHEMA {
        return Err(RollbackReplayError::UnsupportedSchema);
    }
    let mut mismatches = Vec::new();
    if contract.from_assurance_digest != from_assurance.assurance_digest() {
        mismatches.push(RollbackReplayMismatch::FromAssurance);
    }
    if contract.target_assurance_digest != target_assurance.assurance_digest() {
        mismatches.push(RollbackReplayMismatch::TargetAssurance);
    }
    if contract.rollback_digest != rollback.rollback_digest() {
        mismatches.push(RollbackReplayMismatch::Rollback);
    }
    let lineage_digest = digest_release_lineage(lineage)
        .map_err(|error| RollbackReplayError::Evidence(format!("{error:?}")))?;
    if contract.release_lineage_digest != lineage_digest {
        mismatches.push(RollbackReplayMismatch::ReleaseLineage);
    }
    if contract.release_lineage_head != lineage.chain_head().unwrap_or(Sha256Digest([0; 32])) {
        mismatches.push(RollbackReplayMismatch::ReleaseLineageHead);
    }
    let decommission_digest = digest_gateway_decommission_tracker(decommission_tracker)
        .map_err(|error| RollbackReplayError::Evidence(format!("{error:?}")))?;
    if contract.decommission_tracker_digest != decommission_digest {
        mismatches.push(RollbackReplayMismatch::DecommissionTracker);
    }
    if contract.witness_tracker_digest
        != digest_transparency_witness_tracker(witness_tracker)
            .map_err(|error| RollbackReplayError::Evidence(format!("{error:?}")))?
    {
        mismatches.push(RollbackReplayMismatch::WitnessTracker);
    }
    Ok(RollbackReplayVerificationReport { mismatches })
}
