// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic replay contract for Series 12 containment authority.

use crate::containment_state::{
    ContainmentStateError, FabricationContainmentState, digest_containment_state,
};
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::gateway_tombstone_registry::digest_gateway_tombstone_registry;
use crate::post_rollback_requalification_tracker::digest_post_rollback_requalification_tracker;
use crate::rollout_revocation_tracker::digest_rollout_revocation_tracker;
use crate::signer_compromise_tracker::digest_signer_compromise_tracker;
use crate::trust::{TrustSnapshot, digest_trust_snapshot};
use crate::witness_gossip_tracker::digest_witness_gossip_tracker;
use serde::{Deserialize, Serialize};

pub const CONTAINMENT_REPLAY_SCHEMA: &str = "symthaea.fabrication.containment-replay.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContainmentReplayContract {
    pub schema_version: String,
    pub source_tree_digest: Sha256Digest,
    pub active_promotion_digest: Sha256Digest,
    pub trust_snapshot_digest: Sha256Digest,
    pub release_resilience_generation: u64,
    pub release_resilience_state_digest: Sha256Digest,
    pub containment_generation: u64,
    pub containment_state_digest: Sha256Digest,
    pub signer_compromise_tracker_digest: Sha256Digest,
    pub witness_gossip_tracker_digest: Sha256Digest,
    pub gateway_tombstone_registry_digest: Sha256Digest,
    pub rollout_revocation_tracker_digest: Sha256Digest,
    pub post_rollback_requalification_tracker_digest: Sha256Digest,
    pub created_at_unix_s: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContainmentReplayMismatch {
    SourceTree,
    ActivePromotion,
    TrustSnapshot,
    ReleaseResilienceGeneration,
    ReleaseResilienceState,
    ContainmentGeneration,
    ContainmentState,
    SignerCompromise,
    WitnessGossip,
    GatewayTombstones,
    RolloutRevocations,
    PostRollbackRequalification,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContainmentReplayError {
    UnsupportedSchema,
    InvalidContract,
    ContainmentState(ContainmentStateError),
    TrustSnapshot,
    Tracker(String),
    Encoding(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContainmentReplayVerificationReport {
    pub mismatches: Vec<ContainmentReplayMismatch>,
}

impl ContainmentReplayVerificationReport {
    pub fn exact(&self) -> bool {
        self.mismatches.is_empty()
    }
}

pub fn build_containment_replay_contract(
    source_tree_digest: Sha256Digest,
    active_promotion_digest: Sha256Digest,
    trust_snapshot: &TrustSnapshot,
    state: &FabricationContainmentState,
    created_at_unix_s: u64,
) -> Result<ContainmentReplayContract, ContainmentReplayError> {
    state
        .validate()
        .map_err(ContainmentReplayError::ContainmentState)?;
    trust_snapshot
        .validate()
        .map_err(|_| ContainmentReplayError::TrustSnapshot)?;
    if source_tree_digest == Sha256Digest([0; 32])
        || active_promotion_digest == Sha256Digest([0; 32])
        || created_at_unix_s == 0
    {
        return Err(ContainmentReplayError::InvalidContract);
    }
    Ok(ContainmentReplayContract {
        schema_version: CONTAINMENT_REPLAY_SCHEMA.into(),
        source_tree_digest,
        active_promotion_digest,
        trust_snapshot_digest: digest_trust_snapshot(trust_snapshot)
            .map_err(|_| ContainmentReplayError::TrustSnapshot)?,
        release_resilience_generation: state.release_resilience_generation,
        release_resilience_state_digest: state.release_resilience_state_digest,
        containment_generation: state.generation,
        containment_state_digest: digest_containment_state(state)
            .map_err(ContainmentReplayError::ContainmentState)?,
        signer_compromise_tracker_digest: digest_signer_compromise_tracker(
            &state.signer_compromise_tracker,
        )
        .map_err(|error| ContainmentReplayError::Tracker(format!("{error:?}")))?,
        witness_gossip_tracker_digest: digest_witness_gossip_tracker(&state.witness_gossip_tracker)
            .map_err(|error| ContainmentReplayError::Tracker(format!("{error:?}")))?,
        gateway_tombstone_registry_digest: digest_gateway_tombstone_registry(
            &state.gateway_tombstone_registry,
        )
        .map_err(|error| ContainmentReplayError::Tracker(format!("{error:?}")))?,
        rollout_revocation_tracker_digest: digest_rollout_revocation_tracker(
            &state.rollout_revocation_tracker,
        )
        .map_err(|error| ContainmentReplayError::Tracker(format!("{error:?}")))?,
        post_rollback_requalification_tracker_digest: digest_post_rollback_requalification_tracker(
            &state.post_rollback_requalification_tracker,
        )
        .map_err(|error| ContainmentReplayError::Tracker(format!("{error:?}")))?,
        created_at_unix_s,
    })
}

pub fn digest_containment_replay_contract(
    contract: &ContainmentReplayContract,
) -> Result<Sha256Digest, ContainmentReplayError> {
    validate_contract(contract)?;
    let bytes = serde_json::to_vec(contract)
        .map_err(|error| ContainmentReplayError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.containment-replay-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn verify_containment_replay_contract(
    contract: &ContainmentReplayContract,
    expected_source_tree_digest: Sha256Digest,
    expected_active_promotion_digest: Sha256Digest,
    trust_snapshot: &TrustSnapshot,
    state: &FabricationContainmentState,
) -> Result<ContainmentReplayVerificationReport, ContainmentReplayError> {
    validate_contract(contract)?;
    state
        .validate()
        .map_err(ContainmentReplayError::ContainmentState)?;
    let expected = build_containment_replay_contract(
        expected_source_tree_digest,
        expected_active_promotion_digest,
        trust_snapshot,
        state,
        contract.created_at_unix_s,
    )?;
    let mut mismatches = Vec::new();
    if contract.source_tree_digest != expected.source_tree_digest {
        mismatches.push(ContainmentReplayMismatch::SourceTree);
    }
    if contract.active_promotion_digest != expected.active_promotion_digest {
        mismatches.push(ContainmentReplayMismatch::ActivePromotion);
    }
    if contract.trust_snapshot_digest != expected.trust_snapshot_digest {
        mismatches.push(ContainmentReplayMismatch::TrustSnapshot);
    }
    if contract.release_resilience_generation != expected.release_resilience_generation {
        mismatches.push(ContainmentReplayMismatch::ReleaseResilienceGeneration);
    }
    if contract.release_resilience_state_digest != expected.release_resilience_state_digest {
        mismatches.push(ContainmentReplayMismatch::ReleaseResilienceState);
    }
    if contract.containment_generation != expected.containment_generation {
        mismatches.push(ContainmentReplayMismatch::ContainmentGeneration);
    }
    if contract.containment_state_digest != expected.containment_state_digest {
        mismatches.push(ContainmentReplayMismatch::ContainmentState);
    }
    if contract.signer_compromise_tracker_digest != expected.signer_compromise_tracker_digest {
        mismatches.push(ContainmentReplayMismatch::SignerCompromise);
    }
    if contract.witness_gossip_tracker_digest != expected.witness_gossip_tracker_digest {
        mismatches.push(ContainmentReplayMismatch::WitnessGossip);
    }
    if contract.gateway_tombstone_registry_digest != expected.gateway_tombstone_registry_digest {
        mismatches.push(ContainmentReplayMismatch::GatewayTombstones);
    }
    if contract.rollout_revocation_tracker_digest != expected.rollout_revocation_tracker_digest {
        mismatches.push(ContainmentReplayMismatch::RolloutRevocations);
    }
    if contract.post_rollback_requalification_tracker_digest
        != expected.post_rollback_requalification_tracker_digest
    {
        mismatches.push(ContainmentReplayMismatch::PostRollbackRequalification);
    }
    Ok(ContainmentReplayVerificationReport { mismatches })
}

fn validate_contract(contract: &ContainmentReplayContract) -> Result<(), ContainmentReplayError> {
    if contract.schema_version != CONTAINMENT_REPLAY_SCHEMA {
        return Err(ContainmentReplayError::UnsupportedSchema);
    }
    if contract.source_tree_digest == Sha256Digest([0; 32])
        || contract.active_promotion_digest == Sha256Digest([0; 32])
        || contract.release_resilience_generation == 0
        || contract.containment_generation == 0
        || contract.created_at_unix_s == 0
    {
        return Err(ContainmentReplayError::InvalidContract);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_rejects_zero_source_tree() {
        let contract = ContainmentReplayContract {
            schema_version: CONTAINMENT_REPLAY_SCHEMA.into(),
            source_tree_digest: Sha256Digest([0; 32]),
            active_promotion_digest: Sha256Digest([1; 32]),
            trust_snapshot_digest: Sha256Digest([2; 32]),
            release_resilience_generation: 1,
            release_resilience_state_digest: Sha256Digest([3; 32]),
            containment_generation: 1,
            containment_state_digest: Sha256Digest([4; 32]),
            signer_compromise_tracker_digest: Sha256Digest([5; 32]),
            witness_gossip_tracker_digest: Sha256Digest([6; 32]),
            gateway_tombstone_registry_digest: Sha256Digest([7; 32]),
            rollout_revocation_tracker_digest: Sha256Digest([8; 32]),
            post_rollback_requalification_tracker_digest: Sha256Digest([9; 32]),
            created_at_unix_s: 10,
        };
        assert_eq!(
            digest_containment_replay_contract(&contract),
            Err(ContainmentReplayError::InvalidContract)
        );
    }
}
