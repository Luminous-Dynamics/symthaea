// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Persistent anti-equivocation state for verified gateway consensus results.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::gateway_consensus::VerifiedGatewayConsensus;
use crate::gateway_state::FabricationGatewayState;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const GATEWAY_CONSENSUS_TRACKER_SCHEMA: &str =
    "symthaea.fabrication.gateway-consensus-tracker.v1";
pub const MAX_GATEWAY_VOTES: usize = 1_000_000;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
struct GatewayVoteId {
    gateway_id: String,
    generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct GatewayVote {
    state_digest: Sha256Digest,
    consensus_digest: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayConsensusTracker {
    pub schema_version: String,
    latest_generation: Option<u64>,
    latest_state_digest: Option<Sha256Digest>,
    latest_consensus_digest: Option<Sha256Digest>,
    votes: BTreeMap<GatewayVoteId, GatewayVote>,
}

impl Default for GatewayConsensusTracker {
    fn default() -> Self {
        Self {
            schema_version: GATEWAY_CONSENSUS_TRACKER_SCHEMA.into(),
            latest_generation: None,
            latest_state_digest: None,
            latest_consensus_digest: None,
            votes: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayConsensusTrackingError {
    UnsupportedSchema,
    CapacityExceeded,
    StateMismatch,
    GenerationMismatch,
    GenerationRollback { latest: u64, proposed: u64 },
    GenerationGap { latest: u64, proposed: u64 },
    SameGenerationFork { generation: u64 },
    PreviousStateMismatch,
    GatewayEquivocation { gateway_id: String, generation: u64 },
    EvidenceRollback(&'static str),
    Encoding(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcceptedGatewayConsensus {
    pub generation: u64,
    pub state_digest: Sha256Digest,
    pub consensus_digest: Sha256Digest,
    pub idempotent_replay: bool,
}

impl GatewayConsensusTracker {
    pub fn validate(&self) -> Result<(), GatewayConsensusTrackingError> {
        if self.schema_version != GATEWAY_CONSENSUS_TRACKER_SCHEMA {
            return Err(GatewayConsensusTrackingError::UnsupportedSchema);
        }
        if self.votes.len() > MAX_GATEWAY_VOTES {
            return Err(GatewayConsensusTrackingError::CapacityExceeded);
        }
        if self.latest_generation.is_some()
            != (self.latest_state_digest.is_some() && self.latest_consensus_digest.is_some())
        {
            return Err(GatewayConsensusTrackingError::EvidenceRollback(
                "latest consensus fields are inconsistent",
            ));
        }
        Ok(())
    }

    pub fn accept(
        &mut self,
        state: &FabricationGatewayState,
        consensus: &VerifiedGatewayConsensus,
    ) -> Result<AcceptedGatewayConsensus, GatewayConsensusTrackingError> {
        self.validate()?;
        let state_digest = state
            .digest()
            .map_err(|_| GatewayConsensusTrackingError::StateMismatch)?;
        if consensus.state_digest() != state_digest {
            return Err(GatewayConsensusTrackingError::StateMismatch);
        }
        if consensus.generation() != state.generation {
            return Err(GatewayConsensusTrackingError::GenerationMismatch);
        }
        if let Some(latest) = self.latest_generation {
            if state.generation < latest {
                return Err(GatewayConsensusTrackingError::GenerationRollback {
                    latest,
                    proposed: state.generation,
                });
            }
            if state.generation == latest {
                if self.latest_state_digest == Some(state_digest)
                    && self.latest_consensus_digest == Some(consensus.consensus_digest())
                {
                    return Ok(AcceptedGatewayConsensus {
                        generation: state.generation,
                        state_digest,
                        consensus_digest: consensus.consensus_digest(),
                        idempotent_replay: true,
                    });
                }
                return Err(GatewayConsensusTrackingError::SameGenerationFork {
                    generation: state.generation,
                });
            }
            if state.generation != latest.saturating_add(1) {
                return Err(GatewayConsensusTrackingError::GenerationGap {
                    latest,
                    proposed: state.generation,
                });
            }
            if state.previous_state_digest != self.latest_state_digest {
                return Err(GatewayConsensusTrackingError::PreviousStateMismatch);
            }
        }
        if self.votes.len().saturating_add(consensus.gateways().len()) > MAX_GATEWAY_VOTES {
            return Err(GatewayConsensusTrackingError::CapacityExceeded);
        }
        for gateway_id in consensus.gateways() {
            let vote_id = GatewayVoteId {
                gateway_id: gateway_id.clone(),
                generation: state.generation,
            };
            let vote = GatewayVote {
                state_digest,
                consensus_digest: consensus.consensus_digest(),
            };
            if self
                .votes
                .get(&vote_id)
                .is_some_and(|existing| existing != &vote)
            {
                return Err(GatewayConsensusTrackingError::GatewayEquivocation {
                    gateway_id: gateway_id.clone(),
                    generation: state.generation,
                });
            }
        }
        for gateway_id in consensus.gateways() {
            self.votes.insert(
                GatewayVoteId {
                    gateway_id: gateway_id.clone(),
                    generation: state.generation,
                },
                GatewayVote {
                    state_digest,
                    consensus_digest: consensus.consensus_digest(),
                },
            );
        }
        self.latest_generation = Some(state.generation);
        self.latest_state_digest = Some(state_digest);
        self.latest_consensus_digest = Some(consensus.consensus_digest());
        Ok(AcceptedGatewayConsensus {
            generation: state.generation,
            state_digest,
            consensus_digest: consensus.consensus_digest(),
            idempotent_replay: false,
        })
    }

    pub fn latest_generation(&self) -> Option<u64> {
        self.latest_generation
    }
    pub fn latest_state_digest(&self) -> Option<Sha256Digest> {
        self.latest_state_digest
    }
    pub fn latest_consensus_digest(&self) -> Option<Sha256Digest> {
        self.latest_consensus_digest
    }

    pub fn digest(&self) -> Result<Sha256Digest, GatewayConsensusTrackingError> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| GatewayConsensusTrackingError::Encoding(error.to_string()))?;
        let mut hasher = Sha256::new();
        hasher.update(b"symthaea.fabrication.gateway-consensus-tracker-digest.v1\0");
        hasher.update(&bytes);
        Ok(hasher.finalize())
    }

    pub fn verify_successor_of(
        &self,
        previous: &Self,
    ) -> Result<(), GatewayConsensusTrackingError> {
        previous.validate()?;
        self.validate()?;
        for (vote_id, previous_vote) in &previous.votes {
            if self.votes.get(vote_id) != Some(previous_vote) {
                return Err(GatewayConsensusTrackingError::EvidenceRollback(
                    "gateway consensus vote disappeared or changed",
                ));
            }
        }
        if self.latest_generation < previous.latest_generation {
            return Err(GatewayConsensusTrackingError::EvidenceRollback(
                "latest gateway consensus generation regressed",
            ));
        }
        if self.latest_generation == previous.latest_generation
            && (self.latest_state_digest != previous.latest_state_digest
                || self.latest_consensus_digest != previous.latest_consensus_digest)
        {
            return Err(GatewayConsensusTrackingError::EvidenceRollback(
                "latest gateway consensus was substituted",
            ));
        }
        Ok(())
    }
}
