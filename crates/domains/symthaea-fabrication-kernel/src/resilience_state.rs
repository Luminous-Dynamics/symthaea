// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hash-chained durable state for release resilience and rollback authority.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::gateway_decommission_tracker::{
    GatewayDecommissionTracker, GatewayDecommissionTrackingError,
    digest_gateway_decommission_tracker,
};
use crate::region_quorum_tracker::{RegionalQuorumTracker, RegionalQuorumTrackingError};
use crate::release_lineage::{ReleaseLineage, ReleaseLineageError, digest_release_lineage};
use crate::transparency_witness_tracker::{
    TransparencyWitnessTracker, TransparencyWitnessTrackingError,
    digest_transparency_witness_tracker,
};
use serde::{Deserialize, Serialize};

pub const RELEASE_RESILIENCE_STATE_SCHEMA: &str =
    "symthaea.fabrication.release-resilience-state.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseResilienceState {
    pub schema_version: String,
    pub generation: u64,
    pub previous_state_digest: Option<Sha256Digest>,
    pub release_lineage: ReleaseLineage,
    pub regional_quorum_tracker: RegionalQuorumTracker,
    pub transparency_witness_tracker: TransparencyWitnessTracker,
    pub gateway_decommission_tracker: GatewayDecommissionTracker,
    pub latest_rollback_replay_digest: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReleaseResilienceStateError {
    UnsupportedSchema,
    GenerationZero,
    InvalidGenesis,
    GenerationNotSuccessor { previous: u64, proposed: u64 },
    PreviousDigestMismatch,
    ReleaseLineage(ReleaseLineageError),
    RegionalQuorum(RegionalQuorumTrackingError),
    TransparencyWitness(TransparencyWitnessTrackingError),
    GatewayDecommission(GatewayDecommissionTrackingError),
    RollbackReplayRemoved,
    Encoding(String),
}

impl ReleaseResilienceState {
    pub fn genesis() -> Self {
        Self {
            schema_version: RELEASE_RESILIENCE_STATE_SCHEMA.into(),
            generation: 1,
            previous_state_digest: None,
            release_lineage: ReleaseLineage::default(),
            regional_quorum_tracker: RegionalQuorumTracker::default(),
            transparency_witness_tracker: TransparencyWitnessTracker::default(),
            gateway_decommission_tracker: GatewayDecommissionTracker::default(),
            latest_rollback_replay_digest: None,
        }
    }

    pub fn validate(&self) -> Result<(), ReleaseResilienceStateError> {
        if self.schema_version != RELEASE_RESILIENCE_STATE_SCHEMA {
            return Err(ReleaseResilienceStateError::UnsupportedSchema);
        }
        if self.generation == 0 {
            return Err(ReleaseResilienceStateError::GenerationZero);
        }
        if self.generation == 1 && self.previous_state_digest.is_some() {
            return Err(ReleaseResilienceStateError::InvalidGenesis);
        }
        if self.generation > 1 && self.previous_state_digest.is_none() {
            return Err(ReleaseResilienceStateError::InvalidGenesis);
        }
        self.release_lineage
            .validate()
            .map_err(ReleaseResilienceStateError::ReleaseLineage)?;
        self.regional_quorum_tracker
            .validate()
            .map_err(ReleaseResilienceStateError::RegionalQuorum)?;
        digest_gateway_decommission_tracker(&self.gateway_decommission_tracker)
            .map_err(ReleaseResilienceStateError::GatewayDecommission)?;
        digest_transparency_witness_tracker(&self.transparency_witness_tracker)
            .map_err(ReleaseResilienceStateError::TransparencyWitness)?;
        Ok(())
    }

    pub fn successor(&self) -> Result<Self, ReleaseResilienceStateError> {
        self.validate()?;
        Ok(Self {
            schema_version: RELEASE_RESILIENCE_STATE_SCHEMA.into(),
            generation: self.generation.saturating_add(1),
            previous_state_digest: Some(digest_release_resilience_state(self)?),
            release_lineage: self.release_lineage.clone(),
            regional_quorum_tracker: self.regional_quorum_tracker.clone(),
            transparency_witness_tracker: self.transparency_witness_tracker.clone(),
            gateway_decommission_tracker: self.gateway_decommission_tracker.clone(),
            latest_rollback_replay_digest: self.latest_rollback_replay_digest,
        })
    }
}

pub fn digest_release_resilience_state(
    state: &ReleaseResilienceState,
) -> Result<Sha256Digest, ReleaseResilienceStateError> {
    state.validate()?;
    let bytes = serde_json::to_vec(state)
        .map_err(|error| ReleaseResilienceStateError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.release-resilience-state-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn verify_release_resilience_successor(
    previous: &ReleaseResilienceState,
    proposed: &ReleaseResilienceState,
) -> Result<(), ReleaseResilienceStateError> {
    previous.validate()?;
    proposed.validate()?;
    if proposed.generation != previous.generation.saturating_add(1) {
        return Err(ReleaseResilienceStateError::GenerationNotSuccessor {
            previous: previous.generation,
            proposed: proposed.generation,
        });
    }
    if proposed.previous_state_digest != Some(digest_release_resilience_state(previous)?) {
        return Err(ReleaseResilienceStateError::PreviousDigestMismatch);
    }
    if proposed.release_lineage.events.len() < previous.release_lineage.events.len()
        || proposed.release_lineage.events[..previous.release_lineage.events.len()]
            != previous.release_lineage.events
    {
        return Err(ReleaseResilienceStateError::ReleaseLineage(
            ReleaseLineageError::ActivePromotionMismatch,
        ));
    }
    proposed
        .regional_quorum_tracker
        .verify_successor_of(&previous.regional_quorum_tracker)
        .map_err(ReleaseResilienceStateError::RegionalQuorum)?;
    proposed
        .transparency_witness_tracker
        .verify_successor_of(&previous.transparency_witness_tracker)
        .map_err(ReleaseResilienceStateError::TransparencyWitness)?;
    proposed
        .gateway_decommission_tracker
        .verify_successor_of(&previous.gateway_decommission_tracker)
        .map_err(ReleaseResilienceStateError::GatewayDecommission)?;
    if previous.latest_rollback_replay_digest.is_some()
        && proposed.latest_rollback_replay_digest.is_none()
    {
        return Err(ReleaseResilienceStateError::RollbackReplayRemoved);
    }
    let _ = digest_release_lineage(&proposed.release_lineage)
        .map_err(ReleaseResilienceStateError::ReleaseLineage)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genesis_has_stable_successor_link() {
        let genesis = ReleaseResilienceState::genesis();
        let next = genesis.successor().unwrap();
        assert_eq!(verify_release_resilience_successor(&genesis, &next), Ok(()));
    }
}
