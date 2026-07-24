// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hash-chained durable state for post-rollback and compromise containment.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::gateway_tombstone_registry::{
    GatewayTombstoneRegistry, GatewayTombstoneRegistryError, digest_gateway_tombstone_registry,
};
use crate::post_rollback_requalification_tracker::{
    PostRollbackRequalificationTracker, PostRollbackRequalificationTrackingError,
    digest_post_rollback_requalification_tracker,
};
use crate::rollout_revocation_tracker::{
    RolloutRevocationTracker, RolloutRevocationTrackingError, digest_rollout_revocation_tracker,
};
use crate::signer_compromise_tracker::{
    SignerCompromiseTracker, SignerCompromiseTrackingError, digest_signer_compromise_tracker,
};
use crate::witness_gossip_tracker::{
    WitnessGossipTracker, WitnessGossipTrackingError, digest_witness_gossip_tracker,
};
use serde::{Deserialize, Serialize};

pub const CONTAINMENT_STATE_SCHEMA: &str = "symthaea.fabrication.containment-state.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FabricationContainmentState {
    pub schema_version: String,
    pub generation: u64,
    pub previous_state_digest: Option<Sha256Digest>,
    pub release_resilience_generation: u64,
    pub release_resilience_state_digest: Sha256Digest,
    pub signer_compromise_tracker: SignerCompromiseTracker,
    pub witness_gossip_tracker: WitnessGossipTracker,
    pub gateway_tombstone_registry: GatewayTombstoneRegistry,
    pub rollout_revocation_tracker: RolloutRevocationTracker,
    pub post_rollback_requalification_tracker: PostRollbackRequalificationTracker,
    pub latest_containment_replay_digest: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContainmentStateError {
    UnsupportedSchema,
    GenerationZero,
    InvalidGenesis,
    EmptyResilienceDigest,
    GenerationNotSuccessor { previous: u64, proposed: u64 },
    PreviousDigestMismatch,
    ResilienceGenerationRollback,
    ResilienceSameGenerationSubstitution,
    SignerCompromise(SignerCompromiseTrackingError),
    WitnessGossip(WitnessGossipTrackingError),
    GatewayTombstone(GatewayTombstoneRegistryError),
    RolloutRevocation(RolloutRevocationTrackingError),
    PostRollbackRequalification(PostRollbackRequalificationTrackingError),
    SignerCompromiseHistoryChanged,
    WitnessObservationHistoryChanged,
    WitnessEquivocationHistoryChanged,
    ReplayEvidenceRemoved,
    Encoding(String),
}

impl FabricationContainmentState {
    pub fn genesis(
        release_resilience_generation: u64,
        release_resilience_state_digest: Sha256Digest,
    ) -> Result<Self, ContainmentStateError> {
        let state = Self {
            schema_version: CONTAINMENT_STATE_SCHEMA.into(),
            generation: 1,
            previous_state_digest: None,
            release_resilience_generation,
            release_resilience_state_digest,
            signer_compromise_tracker: SignerCompromiseTracker::default(),
            witness_gossip_tracker: WitnessGossipTracker::default(),
            gateway_tombstone_registry: GatewayTombstoneRegistry::default(),
            rollout_revocation_tracker: RolloutRevocationTracker::default(),
            post_rollback_requalification_tracker: PostRollbackRequalificationTracker::default(),
            latest_containment_replay_digest: None,
        };
        state.validate()?;
        Ok(state)
    }

    pub fn validate(&self) -> Result<(), ContainmentStateError> {
        if self.schema_version != CONTAINMENT_STATE_SCHEMA {
            return Err(ContainmentStateError::UnsupportedSchema);
        }
        if self.generation == 0 || self.release_resilience_generation == 0 {
            return Err(ContainmentStateError::GenerationZero);
        }
        if self.release_resilience_state_digest == Sha256Digest([0; 32]) {
            return Err(ContainmentStateError::EmptyResilienceDigest);
        }
        if (self.generation == 1) != self.previous_state_digest.is_none() {
            return Err(ContainmentStateError::InvalidGenesis);
        }
        digest_signer_compromise_tracker(&self.signer_compromise_tracker)
            .map_err(ContainmentStateError::SignerCompromise)?;
        digest_witness_gossip_tracker(&self.witness_gossip_tracker)
            .map_err(ContainmentStateError::WitnessGossip)?;
        digest_gateway_tombstone_registry(&self.gateway_tombstone_registry)
            .map_err(ContainmentStateError::GatewayTombstone)?;
        digest_rollout_revocation_tracker(&self.rollout_revocation_tracker)
            .map_err(ContainmentStateError::RolloutRevocation)?;
        digest_post_rollback_requalification_tracker(&self.post_rollback_requalification_tracker)
            .map_err(ContainmentStateError::PostRollbackRequalification)?;
        Ok(())
    }

    pub fn successor(
        &self,
        release_resilience_generation: u64,
        release_resilience_state_digest: Sha256Digest,
    ) -> Result<Self, ContainmentStateError> {
        self.validate()?;
        if release_resilience_generation < self.release_resilience_generation {
            return Err(ContainmentStateError::ResilienceGenerationRollback);
        }
        if release_resilience_generation == self.release_resilience_generation
            && release_resilience_state_digest != self.release_resilience_state_digest
        {
            return Err(ContainmentStateError::ResilienceSameGenerationSubstitution);
        }
        Ok(Self {
            schema_version: CONTAINMENT_STATE_SCHEMA.into(),
            generation: self.generation.saturating_add(1),
            previous_state_digest: Some(digest_containment_state(self)?),
            release_resilience_generation,
            release_resilience_state_digest,
            signer_compromise_tracker: self.signer_compromise_tracker.clone(),
            witness_gossip_tracker: self.witness_gossip_tracker.clone(),
            gateway_tombstone_registry: self.gateway_tombstone_registry.clone(),
            rollout_revocation_tracker: self.rollout_revocation_tracker.clone(),
            post_rollback_requalification_tracker: self
                .post_rollback_requalification_tracker
                .clone(),
            latest_containment_replay_digest: self.latest_containment_replay_digest,
        })
    }
}

pub fn digest_containment_state(
    state: &FabricationContainmentState,
) -> Result<Sha256Digest, ContainmentStateError> {
    state.validate()?;
    let bytes = serde_json::to_vec(state)
        .map_err(|error| ContainmentStateError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.containment-state-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn verify_containment_state_successor(
    previous: &FabricationContainmentState,
    proposed: &FabricationContainmentState,
) -> Result<(), ContainmentStateError> {
    previous.validate()?;
    proposed.validate()?;
    if proposed.generation != previous.generation.saturating_add(1) {
        return Err(ContainmentStateError::GenerationNotSuccessor {
            previous: previous.generation,
            proposed: proposed.generation,
        });
    }
    if proposed.previous_state_digest != Some(digest_containment_state(previous)?) {
        return Err(ContainmentStateError::PreviousDigestMismatch);
    }
    if proposed.release_resilience_generation < previous.release_resilience_generation {
        return Err(ContainmentStateError::ResilienceGenerationRollback);
    }
    if proposed.release_resilience_generation == previous.release_resilience_generation
        && proposed.release_resilience_state_digest != previous.release_resilience_state_digest
    {
        return Err(ContainmentStateError::ResilienceSameGenerationSubstitution);
    }
    let old_compromises = previous.signer_compromise_tracker.records();
    let new_compromises = proposed.signer_compromise_tracker.records();
    if new_compromises.len() < old_compromises.len()
        || new_compromises[..old_compromises.len()] != old_compromises[..]
    {
        return Err(ContainmentStateError::SignerCompromiseHistoryChanged);
    }
    let old_observations = previous.witness_gossip_tracker.observations();
    let new_observations = proposed.witness_gossip_tracker.observations();
    if new_observations.len() < old_observations.len()
        || new_observations[..old_observations.len()] != old_observations[..]
    {
        return Err(ContainmentStateError::WitnessObservationHistoryChanged);
    }
    let old_equivocations = previous.witness_gossip_tracker.equivocations();
    let new_equivocations = proposed.witness_gossip_tracker.equivocations();
    if new_equivocations.len() < old_equivocations.len()
        || new_equivocations[..old_equivocations.len()] != old_equivocations[..]
    {
        return Err(ContainmentStateError::WitnessEquivocationHistoryChanged);
    }
    proposed
        .gateway_tombstone_registry
        .verify_successor_of(&previous.gateway_tombstone_registry)
        .map_err(ContainmentStateError::GatewayTombstone)?;
    proposed
        .rollout_revocation_tracker
        .verify_successor_of(&previous.rollout_revocation_tracker)
        .map_err(ContainmentStateError::RolloutRevocation)?;
    proposed
        .post_rollback_requalification_tracker
        .verify_successor_of(&previous.post_rollback_requalification_tracker)
        .map_err(ContainmentStateError::PostRollbackRequalification)?;
    if previous.latest_containment_replay_digest.is_some()
        && proposed.latest_containment_replay_digest.is_none()
    {
        return Err(ContainmentStateError::ReplayEvidenceRemoved);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genesis_and_successor_are_hash_linked() {
        let genesis = FabricationContainmentState::genesis(1, Sha256Digest([1; 32])).unwrap();
        let next = genesis.successor(2, Sha256Digest([2; 32])).unwrap();
        assert_eq!(verify_containment_state_successor(&genesis, &next), Ok(()));
    }
}
