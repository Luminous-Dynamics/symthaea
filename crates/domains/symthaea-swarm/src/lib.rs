// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-swarm — Collective Consciousness Protocol
//!
//! Implements a P2P swarm protocol for sharing consciousness states and
//! verified math/proof records between Symthaea nodes.
//!
//! The domain messages in this module are transport-independent. The optional
//! [`networking`] module adds an authenticated Iroh gossip transport.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use symthaea_core::hdc::ContinuousHV;
use uuid::Uuid;

pub const MAX_IDENTIFIER_BYTES: usize = 256;
pub const MAX_SMTLIB_BYTES: usize = 512 * 1024;
pub const MAX_MACRO_PAYLOAD_BYTES: usize = 512 * 1024;
pub const MAX_WEIGHT_UPDATE_BYTES: usize = 2 * 1024 * 1024;
pub const MAX_CURVATURE_RESIDUALS: usize = 16_384;

/// Message containing a node's local consciousness state and morphology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmStateMsg {
    pub node_id: Uuid,
    pub platform_type: String,
    pub local_phi: f64,
    pub consciousness_hv: ContinuousHV,
    pub intent_hv: ContinuousHV,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HapticPulseMsg {
    pub node_id: Uuid,
    pub position: [f64; 4],
    pub surprise: f64,
    pub impact_vector: [f64; 4],
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmProofMsg {
    pub node_id: Uuid,
    pub label: String,
    pub smtlib2: String,
    pub proof_hv: ContinuousHV,
    pub verified: bool,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LawGossipMsg {
    pub node_id: Uuid,
    pub law_id: String,
    pub smtlib2: String,
    pub proposing_phi: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutualAidMsg {
    pub sender_id: Uuid,
    pub target_id: Uuid,
    pub tend_amount: f64,
    pub support_hv: ContinuousHV,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvatureGossipMsg {
    pub node_id: Uuid,
    pub residuals: Vec<f64>,
    pub dim: usize,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialPhiGossipMsg {
    pub node_id: Uuid,
    pub collective_phi: f64,
    pub integration_ratio: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroGossipMsg {
    pub node_id: Uuid,
    pub domain: String,
    pub payload: Vec<u8>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmMessage {
    State(SwarmStateMsg),
    HapticPulse(HapticPulseMsg),
    ProofGossip(SwarmProofMsg),
    LawGossip(LawGossipMsg),
    MacroGossip(MacroGossipMsg),
    CurvatureGossip(CurvatureGossipMsg),
    SocialPhiGossip(SocialPhiGossipMsg),
    MutualAid(MutualAidMsg),
    WeightUpdate {
        node_id: Uuid,
        target: String,
        kernel: Vec<u8>,
        proof_bytes: Vec<u8>,
        timestamp: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SwarmMessageKind {
    State,
    HapticPulse,
    ProofGossip,
    LawGossip,
    MacroGossip,
    CurvatureGossip,
    SocialPhiGossip,
    MutualAid,
    WeightUpdate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryClass {
    BestEffort,
    Durable,
}

#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum MessageValidationError {
    #[error("application node ID must not be nil")]
    NilNodeId,
    #[error("{field} exceeds the maximum encoded length of {max} bytes")]
    FieldTooLarge { field: &'static str, max: usize },
    #[error("{field} contains a non-finite value")]
    NonFinite { field: &'static str },
    #[error("{field} must be non-negative")]
    Negative { field: &'static str },
    #[error("curvature residual count exceeds {MAX_CURVATURE_RESIDUALS}")]
    TooManyResiduals,
    #[error("curvature dimension does not match residual count")]
    CurvatureDimensionMismatch,
}

impl SwarmMessage {
    pub fn kind(&self) -> SwarmMessageKind {
        match self {
            Self::State(_) => SwarmMessageKind::State,
            Self::HapticPulse(_) => SwarmMessageKind::HapticPulse,
            Self::ProofGossip(_) => SwarmMessageKind::ProofGossip,
            Self::LawGossip(_) => SwarmMessageKind::LawGossip,
            Self::MacroGossip(_) => SwarmMessageKind::MacroGossip,
            Self::CurvatureGossip(_) => SwarmMessageKind::CurvatureGossip,
            Self::SocialPhiGossip(_) => SwarmMessageKind::SocialPhiGossip,
            Self::MutualAid(_) => SwarmMessageKind::MutualAid,
            Self::WeightUpdate { .. } => SwarmMessageKind::WeightUpdate,
        }
    }

    pub fn claimed_node_id(&self) -> Uuid {
        match self {
            Self::State(msg) => msg.node_id,
            Self::HapticPulse(msg) => msg.node_id,
            Self::ProofGossip(msg) => msg.node_id,
            Self::LawGossip(msg) => msg.node_id,
            Self::MacroGossip(msg) => msg.node_id,
            Self::CurvatureGossip(msg) => msg.node_id,
            Self::SocialPhiGossip(msg) => msg.node_id,
            Self::MutualAid(msg) => msg.sender_id,
            Self::WeightUpdate { node_id, .. } => *node_id,
        }
    }

    pub fn timestamp_ms(&self) -> Option<u64> {
        match self {
            Self::State(msg) => Some(msg.timestamp),
            Self::HapticPulse(msg) => Some(msg.timestamp),
            Self::ProofGossip(msg) => Some(msg.timestamp),
            Self::LawGossip(msg) => Some(msg.timestamp),
            Self::MacroGossip(msg) => Some(msg.timestamp),
            Self::CurvatureGossip(msg) => Some(msg.timestamp),
            Self::SocialPhiGossip(msg) => Some(msg.timestamp),
            Self::MutualAid(_) => None,
            Self::WeightUpdate { timestamp, .. } => Some(*timestamp),
        }
    }

    pub fn delivery_class(&self) -> DeliveryClass {
        match self {
            Self::State(_)
            | Self::HapticPulse(_)
            | Self::MacroGossip(_)
            | Self::CurvatureGossip(_)
            | Self::SocialPhiGossip(_) => DeliveryClass::BestEffort,
            Self::ProofGossip(_)
            | Self::LawGossip(_)
            | Self::MutualAid(_)
            | Self::WeightUpdate { .. } => DeliveryClass::Durable,
        }
    }

    pub fn validate(&self) -> Result<(), MessageValidationError> {
        if self.claimed_node_id().is_nil() {
            return Err(MessageValidationError::NilNodeId);
        }

        fn bounded(
            field: &'static str,
            value: &str,
            max: usize,
        ) -> Result<(), MessageValidationError> {
            if value.len() > max {
                return Err(MessageValidationError::FieldTooLarge { field, max });
            }
            Ok(())
        }

        fn finite(field: &'static str, value: f64) -> Result<(), MessageValidationError> {
            if !value.is_finite() {
                return Err(MessageValidationError::NonFinite { field });
            }
            Ok(())
        }

        match self {
            Self::State(msg) => {
                bounded("platform_type", &msg.platform_type, MAX_IDENTIFIER_BYTES)?;
                finite("local_phi", msg.local_phi)?;
            }
            Self::HapticPulse(msg) => {
                finite("surprise", msg.surprise)?;
                for value in msg.position {
                    finite("position", value)?;
                }
                for value in msg.impact_vector {
                    finite("impact_vector", value)?;
                }
            }
            Self::ProofGossip(msg) => {
                bounded("label", &msg.label, MAX_IDENTIFIER_BYTES)?;
                bounded("smtlib2", &msg.smtlib2, MAX_SMTLIB_BYTES)?;
            }
            Self::LawGossip(msg) => {
                bounded("law_id", &msg.law_id, MAX_IDENTIFIER_BYTES)?;
                bounded("smtlib2", &msg.smtlib2, MAX_SMTLIB_BYTES)?;
                finite("proposing_phi", msg.proposing_phi)?;
                if msg.proposing_phi < 0.0 {
                    return Err(MessageValidationError::Negative {
                        field: "proposing_phi",
                    });
                }
            }
            Self::MacroGossip(msg) => {
                bounded("domain", &msg.domain, MAX_IDENTIFIER_BYTES)?;
                if msg.payload.len() > MAX_MACRO_PAYLOAD_BYTES {
                    return Err(MessageValidationError::FieldTooLarge {
                        field: "macro payload",
                        max: MAX_MACRO_PAYLOAD_BYTES,
                    });
                }
            }
            Self::CurvatureGossip(msg) => {
                if msg.residuals.len() > MAX_CURVATURE_RESIDUALS {
                    return Err(MessageValidationError::TooManyResiduals);
                }
                if msg.dim != msg.residuals.len() {
                    return Err(MessageValidationError::CurvatureDimensionMismatch);
                }
                for value in &msg.residuals {
                    finite("curvature residual", *value)?;
                }
            }
            Self::SocialPhiGossip(msg) => {
                finite("collective_phi", msg.collective_phi)?;
                finite("integration_ratio", msg.integration_ratio)?;
            }
            Self::MutualAid(msg) => {
                finite("tend_amount", msg.tend_amount)?;
                if msg.tend_amount < 0.0 {
                    return Err(MessageValidationError::Negative {
                        field: "tend_amount",
                    });
                }
            }
            Self::WeightUpdate {
                target,
                kernel,
                proof_bytes,
                ..
            } => {
                bounded("target", target, MAX_IDENTIFIER_BYTES)?;
                if kernel.len() > MAX_WEIGHT_UPDATE_BYTES {
                    return Err(MessageValidationError::FieldTooLarge {
                        field: "kernel",
                        max: MAX_WEIGHT_UPDATE_BYTES,
                    });
                }
                if proof_bytes.len() > MAX_WEIGHT_UPDATE_BYTES {
                    return Err(MessageValidationError::FieldTooLarge {
                        field: "proof_bytes",
                        max: MAX_WEIGHT_UPDATE_BYTES,
                    });
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LawVoteOutcome {
    Recorded {
        support: f64,
        threshold: f64,
        newly_ratified: bool,
    },
    StaleVote,
    UnknownVoter,
    VoteExceedsAdvertisedPhi {
        advertised_phi: f64,
        proposed_phi: f64,
    },
    ConflictingText,
    Invalid(MessageValidationError),
}

#[derive(Default, Debug, Clone)]
pub struct SwarmAggregator {
    pub peer_states: HashMap<Uuid, SwarmStateMsg>,
    pub swarm_proofs: Vec<SwarmProofMsg>,
    pub collective_laws: HashMap<String, (String, f64)>,
    pub law_votes: HashMap<String, HashMap<Uuid, f64>>,
    pub law_vote_timestamps: HashMap<String, HashMap<Uuid, u64>>,
    pub ratified_laws: HashSet<String>,
    pub haptic_map: HashMap<[i32; 3], f64>,
    pub swarm_curvature: Vec<f64>,
}

impl SwarmAggregator {
    pub fn new() -> Self { Self::default() }

    pub fn update_peer(&mut self, msg: SwarmStateMsg) {
        if !msg.local_phi.is_finite() { return; }
        match self.peer_states.get(&msg.node_id) {
            Some(existing) if existing.timestamp > msg.timestamp => {}
            _ => { self.peer_states.insert(msg.node_id, msg); }
        }
    }

    pub fn ingest_haptic_pulse(&mut self, msg: HapticPulseMsg) {
        if !msg.surprise.is_finite() { return; }
        let grid_pos = [
            msg.position[0].round() as i32,
            msg.position[1].round() as i32,
            msg.position[2].round() as i32,
        ];
        let entry = self.haptic_map.entry(grid_pos).or_insert(0.0);
        *entry = *entry * 0.7 + msg.surprise * 0.3;
    }

    pub fn ingest_curvature_gossip(&mut self, msg: CurvatureGossipMsg) {
        if msg.dim != msg.residuals.len()
            || msg.residuals.len() > MAX_CURVATURE_RESIDUALS
            || msg.residuals.iter().any(|value| !value.is_finite())
        { return; }

        if self.swarm_curvature.is_empty() {
            self.swarm_curvature = msg.residuals;
        } else {
            for (current, incoming) in self.swarm_curvature.iter_mut().zip(msg.residuals) {
                *current = (*current).max(incoming);
            }
        }
    }

    pub fn ingest_peer_proof(&mut self, msg: SwarmProofMsg) {
        if let Some(existing) = self.swarm_proofs.iter_mut()
            .find(|proof| proof.label == msg.label && proof.node_id == msg.node_id)
        {
            if msg.timestamp > existing.timestamp { *existing = msg; }
            return;
        }
        self.swarm_proofs.push(msg);
    }

    pub fn try_ingest_law_proposal(&mut self, msg: LawGossipMsg) -> LawVoteOutcome {
        if let Err(error) = SwarmMessage::LawGossip(msg.clone()).validate() {
            return LawVoteOutcome::Invalid(error);
        }

        if let Some((existing_text, _)) = self.collective_laws.get(&msg.law_id) {
            if existing_text != &msg.smtlib2 { return LawVoteOutcome::ConflictingText; }
        }

        let Some(voter_state) = self.peer_states.get(&msg.node_id) else {
            return LawVoteOutcome::UnknownVoter;
        };
        if msg.proposing_phi > voter_state.local_phi {
            return LawVoteOutcome::VoteExceedsAdvertisedPhi {
                advertised_phi: voter_state.local_phi,
                proposed_phi: msg.proposing_phi,
            };
        }

        let law_id = msg.law_id.clone();
        let law_text = msg.smtlib2.clone();
        let timestamps = self.law_vote_timestamps.entry(law_id.clone()).or_default();
        if timestamps.get(&msg.node_id).is_some_and(|accepted_at| *accepted_at >= msg.timestamp) {
            return LawVoteOutcome::StaleVote;
        }
        timestamps.insert(msg.node_id, msg.timestamp);
        self.law_votes.entry(law_id.clone()).or_default().insert(msg.node_id, msg.proposing_phi);

        let support = self.law_votes.get(&law_id).into_iter()
            .flat_map(|votes| votes.values()).copied()
            .filter(|value| value.is_finite() && *value >= 0.0).sum::<f64>();

        self.collective_laws.insert(law_id.clone(), (law_text, support));

        let total_phi = self.peer_states.values().map(|state| state.local_phi)
            .filter(|value| value.is_finite() && *value >= 0.0).sum::<f64>();
        let threshold = total_phi * 0.5;
        let ratified = total_phi > 0.0 && support >= threshold;
        let newly_ratified = ratified && self.ratified_laws.insert(law_id);

        LawVoteOutcome::Recorded { support, threshold, newly_ratified }
    }

    pub fn ingest_law_proposal(&mut self, msg: LawGossipMsg) {
        let law_id = msg.law_id.clone();
        match self.try_ingest_law_proposal(msg) {
            LawVoteOutcome::Recorded { newly_ratified: true, .. } =>
                tracing::info!(law_id, "swarm law ratified"),
            LawVoteOutcome::StaleVote => tracing::debug!(law_id, "ignored stale or replayed law vote"),
            LawVoteOutcome::UnknownVoter =>
                tracing::warn!(law_id, "rejected law vote without a current peer state"),
            LawVoteOutcome::VoteExceedsAdvertisedPhi { advertised_phi, proposed_phi } =>
                tracing::warn!(law_id, advertised_phi, proposed_phi, "rejected law vote exceeding advertised phi"),
            LawVoteOutcome::ConflictingText =>
                tracing::warn!(law_id, "rejected conflicting law text for existing law_id"),
            LawVoteOutcome::Invalid(error) =>
                tracing::warn!(law_id, %error, "rejected invalid law proposal"),
            LawVoteOutcome::Recorded { .. } => {}
        }
    }

    pub fn audit_constitutional_consistency(&self) -> Result<bool, Vec<String>> {
        let z3 = symthaea_runtime::formal::z3_bridge::Z3Bridge::new();
        let mut assertions = Vec::new();
        for (law_id, (smt, _)) in &self.collective_laws {
            assertions.push(format!("; Law: {law_id}\n{smt}"));
        }
        if let Some(core) = z3.get_unsat_core(&assertions) { Err(core) } else { Ok(true) }
    }

    pub fn reconcile_constitutional_conflict(&self, core: &[String]) -> Option<(String, String)> {
        let _z3 = symthaea_runtime::formal::z3_bridge::Z3Bridge::new();
        tracing::info!(law_count = core.len(), "reconciling constitutional conflict");

        if core.iter().any(|law| law.contains("robot_torque"))
            && core.iter().any(|law| law.contains("> 0.9"))
        {
            let harmonious_law = "(assert (=> (< available_mw 5.0) (< robot_torque 0.35)))".to_string();
            let performance_compromise = "(assert (<= robot_torque 0.85))".to_string();
            return Some(("RES-COLLAPSE-RECONCILED".into(), format!("{harmonious_law}; {performance_compromise}")));
        }
        None
    }

    pub fn hive_mind_vector(&self) -> ContinuousHV {
        if self.peer_states.is_empty() { return ContinuousHV::zero(16_384); }
        let mut hive = ContinuousHV::zero(16_384);
        for state in self.peer_states.values() {
            hive = ContinuousHV::bundle(&[&hive, &state.consciousness_hv]);
        }
        hive.normalize();
        hive
    }

    pub fn calculate_swarm_phi(&self) -> f64 {
        if self.peer_states.is_empty() { return 0.0; }
        let sum_local_phi = self.peer_states.values().map(|state| state.local_phi)
            .filter(|value| value.is_finite()).sum::<f64>();
        let avg_local_phi = sum_local_phi / self.peer_states.len() as f64;
        let hive = self.hive_mind_vector();
        let coherence = self.peer_states.values()
            .map(|state| hive.similarity(&state.consciousness_hv) as f64).sum::<f64>();
        let avg_coherence = (coherence / self.peer_states.len() as f64).max(0.0);
        (avg_local_phi * 0.7 + avg_coherence * 0.3).clamp(0.0, 1.0)
    }
}

pub mod semantic_admission;
pub mod semantic_canonical;
pub mod semantic_commit;
#[cfg(feature = "semantic-digest")]
pub mod semantic_digest;
#[cfg(feature = "semantic-digest")]
pub mod semantic_transition;
#[cfg(feature = "semantic-digest")]
pub mod semantic_evidence_digest;
#[cfg(feature = "semantic-digest")]
pub mod semantic_evidence_history;

pub mod fault;
#[cfg(feature = "networking")]
pub mod networking;
#[cfg(feature = "networking")]
pub mod direct;
#[cfg(feature = "networking")]
pub mod realtime;
#[cfg(feature = "networking")]
pub mod enrollment;
#[cfg(feature = "networking")]
pub mod readiness;
#[cfg(feature = "networking")]
pub mod symtropy;
