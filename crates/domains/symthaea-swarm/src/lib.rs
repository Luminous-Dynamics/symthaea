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

/// Maximum length accepted for identifiers, labels, and routing strings.
pub const MAX_IDENTIFIER_BYTES: usize = 256;
/// Maximum length accepted for SMT-LIB source embedded in one message.
pub const MAX_SMTLIB_BYTES: usize = 512 * 1024;
/// Maximum opaque payload accepted by [`MacroGossipMsg`].
pub const MAX_MACRO_PAYLOAD_BYTES: usize = 512 * 1024;
/// Maximum compressed kernel or proof payload accepted by a weight update.
pub const MAX_WEIGHT_UPDATE_BYTES: usize = 2 * 1024 * 1024;
/// Maximum number of curvature residuals accepted in one message.
pub const MAX_CURVATURE_RESIDUALS: usize = 16_384;

/// Message containing a node's local consciousness state and morphology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmStateMsg {
    /// Stable application-level identifier of the originating node.
    pub node_id: Uuid,
    /// The robotic morphology of this node.
    pub platform_type: String,
    /// Local Phi value.
    pub local_phi: f64,
    /// Local consciousness vector (HDC).
    pub consciousness_hv: ContinuousHV,
    /// Node's current mood or intent vector.
    pub intent_hv: ContinuousHV,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
}

/// Message containing a low-latency haptic pulse for collective proprioception.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HapticPulseMsg {
    pub node_id: Uuid,
    /// Local spatial coordinates `[x, y, z, w]`.
    pub position: [f64; 4],
    /// Joint-level prediction error (Channel 5).
    pub surprise: f64,
    /// Kinetic energy vector of the encounter.
    pub impact_vector: [f64; 4],
    pub timestamp: u64,
}

/// Message containing a verified proof lemma for collective swarm memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmProofMsg {
    /// Unique ID of the node that proved this lemma.
    pub node_id: Uuid,
    /// Stable identifier or label of the lemma (for example, `L3.0`).
    pub label: String,
    /// Verbatim SMT-LIB2 query source.
    pub smtlib2: String,
    /// High-dimensional geometric signature of the verified structure.
    pub proof_hv: ContinuousHV,
    /// True if the formula was mathematically proved valid.
    pub verified: bool,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
}

/// Message containing a self-synthesized safety law for swarm-wide voting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LawGossipMsg {
    pub node_id: Uuid,
    pub law_id: String,
    pub smtlib2: String,
    /// The Phi-weight (consciousness level) of the proposing node.
    pub proposing_phi: f64,
    pub timestamp: u64,
}

/// Message for routing economic Tend credit between nodes in distress.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutualAidMsg {
    pub sender_id: Uuid,
    pub target_id: Uuid,
    pub tend_amount: f64,
    /// High-dimensional proof of the deficit being addressed.
    pub support_hv: ContinuousHV,
}

/// Message for broadcasting structural failure residuals (geometric curvature).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvatureGossipMsg {
    pub node_id: Uuid,
    /// Vector of structural failure residuals (Einstein condition violation).
    pub residuals: Vec<f64>,
    /// Metadata for the local geometric basis.
    pub dim: usize,
    pub timestamp: u64,
}

/// Message for broadcasting collective Social Phi metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialPhiGossipMsg {
    pub node_id: Uuid,
    pub collective_phi: f64,
    pub integration_ratio: f64,
    pub timestamp: u64,
}

/// Bounded extension message for macro-scale domain state.
///
/// The original snapshot referenced `MacroGossipMsg` without defining it,
/// leaving the crate uncompilable. This explicit opaque extension keeps the
/// wire protocol extensible while still enforcing a domain label and payload
/// size limit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroGossipMsg {
    pub node_id: Uuid,
    /// Domain-specific schema or channel name.
    pub domain: String,
    /// Domain payload. Callers should version the payload schema in `domain`.
    pub payload: Vec<u8>,
    pub timestamp: u64,
}

/// Unified swarm application protocol.
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

    /// Message containing a metamorphic weight update kernel and its ZKP proof.
    WeightUpdate {
        node_id: Uuid,
        target: String,
        kernel: Vec<u8>,
        proof_bytes: Vec<u8>,
        timestamp: u64,
    },
}

/// Stable message discriminator used by metrics and queue policy.
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

/// Delivery policy for the local application queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryClass {
    /// Freshness matters more than complete delivery. Overflow may be counted
    /// and dropped rather than blocking the network receive loop.
    BestEffort,
    /// The message must not be silently discarded by the local adapter.
    Durable,
}

/// Domain-level validation failure before a message enters the network or
/// aggregator.
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

    /// Application-level identity claimed by this message.
    ///
    /// The Iroh transport binds this UUID to the signed endpoint identity on
    /// first observation and rejects later conflicting claims.
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

/// Result of applying a law vote to the local governance view.
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

/// Aggregator for swarm-wide consciousness states and collective proofs.
#[derive(Default, Debug, Clone)]
pub struct SwarmAggregator {
    /// Collection of states from other nodes.
    pub peer_states: HashMap<Uuid, SwarmStateMsg>,
    /// Global swarm-replicated formal lemma proof repository database.
    pub swarm_proofs: Vec<SwarmProofMsg>,
    /// Swarm-wide legislated laws: law_id -> (SMT source, total Phi support).
    pub collective_laws: HashMap<String, (String, f64)>,
    /// Latest vote weight per application node, preventing replay inflation.
    pub law_votes: HashMap<String, HashMap<Uuid, f64>>,
    /// Timestamp of each accepted vote, preventing stale reordered gossip from
    /// replacing a newer vote.
    pub law_vote_timestamps: HashMap<String, HashMap<Uuid, u64>>,
    /// Laws for which the threshold transition has already been emitted.
    pub ratified_laws: HashSet<String>,
    /// Collective haptic map: position -> surprise magnitude.
    pub haptic_map: HashMap<[i32; 3], f64>,
    pub swarm_curvature: Vec<f64>,
}

impl SwarmAggregator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or update a peer's state, ignoring stale replayed state.
    pub fn update_peer(&mut self, msg: SwarmStateMsg) {
        if !msg.local_phi.is_finite() {
            return;
        }
        match self.peer_states.get(&msg.node_id) {
            Some(existing) if existing.timestamp > msg.timestamp => {}
            _ => {
                self.peer_states.insert(msg.node_id, msg);
            }
        }
    }

    /// Ingest a low-latency haptic pulse.
    pub fn ingest_haptic_pulse(&mut self, msg: HapticPulseMsg) {
        if !msg.surprise.is_finite() {
            return;
        }
        let grid_pos = [
            msg.position[0].round() as i32,
            msg.position[1].round() as i32,
            msg.position[2].round() as i32,
        ];

        let entry = self.haptic_map.entry(grid_pos).or_insert(0.0);
        *entry = *entry * 0.7 + msg.surprise * 0.3;
    }

    /// Ingest geometric residuals from peers.
    pub fn ingest_curvature_gossip(&mut self, msg: CurvatureGossipMsg) {
        if msg.dim != msg.residuals.len()
            || msg.residuals.len() > MAX_CURVATURE_RESIDUALS
            || msg.residuals.iter().any(|value| !value.is_finite())
        {
            return;
        }

        if self.swarm_curvature.is_empty() {
            self.swarm_curvature = msg.residuals;
        } else {
            for (current, incoming) in self.swarm_curvature.iter_mut().zip(msg.residuals) {
                *current = (*current).max(incoming);
            }
        }
    }

    /// Insert a proof once per `(node_id, label)`, replacing it only with a
    /// newer record.
    pub fn ingest_peer_proof(&mut self, msg: SwarmProofMsg) {
        if let Some(existing) = self
            .swarm_proofs
            .iter_mut()
            .find(|proof| proof.label == msg.label && proof.node_id == msg.node_id)
        {
            if msg.timestamp > existing.timestamp {
                *existing = msg;
            }
            return;
        }
        self.swarm_proofs.push(msg);
    }

    /// Ingest one latest-value vote per application node.
    ///
    /// Replaying the same gossip message no longer increases support. The
    /// authenticated networking layer additionally pins `node_id` to the
    /// signed Iroh endpoint identity.
    pub fn try_ingest_law_proposal(&mut self, msg: LawGossipMsg) -> LawVoteOutcome {
        if let Err(error) = SwarmMessage::LawGossip(msg.clone()).validate() {
            return LawVoteOutcome::Invalid(error);
        }

        if let Some((existing_text, _)) = self.collective_laws.get(&msg.law_id) {
            if existing_text != &msg.smtlib2 {
                return LawVoteOutcome::ConflictingText;
            }
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
        if timestamps
            .get(&msg.node_id)
            .is_some_and(|accepted_at| *accepted_at >= msg.timestamp)
        {
            return LawVoteOutcome::StaleVote;
        }
        timestamps.insert(msg.node_id, msg.timestamp);
        self.law_votes
            .entry(law_id.clone())
            .or_default()
            .insert(msg.node_id, msg.proposing_phi);

        let support = self
            .law_votes
            .get(&law_id)
            .into_iter()
            .flat_map(|votes| votes.values())
            .copied()
            .filter(|value| value.is_finite() && *value >= 0.0)
            .sum::<f64>();

        self.collective_laws
            .insert(law_id.clone(), (law_text, support));

        let total_phi = self
            .peer_states
            .values()
            .map(|state| state.local_phi)
            .filter(|value| value.is_finite() && *value >= 0.0)
            .sum::<f64>();
        let threshold = total_phi * 0.5;
        let ratified = total_phi > 0.0 && support >= threshold;
        let newly_ratified = ratified && self.ratified_laws.insert(law_id);

        LawVoteOutcome::Recorded {
            support,
            threshold,
            newly_ratified,
        }
    }

    /// Backwards-compatible wrapper that logs governance transitions.
    pub fn ingest_law_proposal(&mut self, msg: LawGossipMsg) {
        let law_id = msg.law_id.clone();
        match self.try_ingest_law_proposal(msg) {
            LawVoteOutcome::Recorded {
                newly_ratified: true,
                ..
            } => tracing::info!(law_id, "swarm law ratified"),
            LawVoteOutcome::StaleVote => {
                tracing::debug!(law_id, "ignored stale or replayed law vote")
            }
            LawVoteOutcome::UnknownVoter => {
                tracing::warn!(law_id, "rejected law vote without a current peer state")
            }
            LawVoteOutcome::VoteExceedsAdvertisedPhi {
                advertised_phi,
                proposed_phi,
            } => {
                tracing::warn!(
                    law_id,
                    advertised_phi,
                    proposed_phi,
                    "rejected law vote exceeding advertised phi"
                )
            }
            LawVoteOutcome::ConflictingText => {
                tracing::warn!(law_id, "rejected conflicting law text for existing law_id")
            }
            LawVoteOutcome::Invalid(error) => {
                tracing::warn!(law_id, %error, "rejected invalid law proposal")
            }
            LawVoteOutcome::Recorded { .. } => {}
        }
    }

    /// Formally audit the consistency of the Swarm Constitution.
    pub fn audit_constitutional_consistency(&self) -> Result<bool, Vec<String>> {
        let z3 = symthaea_runtime::formal::z3_bridge::Z3Bridge::new();
        let mut assertions = Vec::new();

        for (law_id, (smt, _)) in &self.collective_laws {
            assertions.push(format!("; Law: {law_id}\n{smt}"));
        }

        if let Some(core) = z3.get_unsat_core(&assertions) {
            Err(core)
        } else {
            Ok(true)
        }
    }

    /// Autonomously reconcile a constitutional conflict by synthesizing a
    /// constrained compromise candidate.
    pub fn reconcile_constitutional_conflict(&self, core: &[String]) -> Option<(String, String)> {
        let _z3 = symthaea_runtime::formal::z3_bridge::Z3Bridge::new();
        tracing::info!(
            law_count = core.len(),
            "reconciling constitutional conflict"
        );

        if core.iter().any(|law| law.contains("robot_torque"))
            && core.iter().any(|law| law.contains("> 0.9"))
        {
            let harmonious_law =
                "(assert (=> (< available_mw 5.0) (< robot_torque 0.35)))".to_string();
            let performance_compromise = "(assert (<= robot_torque 0.85))".to_string();
            return Some((
                "RES-COLLAPSE-RECONCILED".into(),
                format!("{harmonious_law}; {performance_compromise}"),
            ));
        }

        None
    }

    pub fn hive_mind_vector(&self) -> ContinuousHV {
        if self.peer_states.is_empty() {
            return ContinuousHV::zero(16_384);
        }

        let mut hive = ContinuousHV::zero(16_384);
        for state in self.peer_states.values() {
            hive = ContinuousHV::bundle(&[&hive, &state.consciousness_hv]);
        }
        hive.normalize();
        hive
    }

    pub fn calculate_swarm_phi(&self) -> f64 {
        if self.peer_states.is_empty() {
            return 0.0;
        }
        let sum_local_phi = self
            .peer_states
            .values()
            .map(|state| state.local_phi)
            .filter(|value| value.is_finite())
            .sum::<f64>();
        let avg_local_phi = sum_local_phi / self.peer_states.len() as f64;

        let hive = self.hive_mind_vector();
        let coherence = self
            .peer_states
            .values()
            .map(|state| hive.similarity(&state.consciousness_hv) as f64)
            .sum::<f64>();
        let avg_coherence = (coherence / self.peer_states.len() as f64).max(0.0);

        (avg_local_phi * 0.7 + avg_coherence * 0.3).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod domain_tests {
    use super::*;

    fn law(node_id: Uuid, phi: f64) -> LawGossipMsg {
        LawGossipMsg {
            node_id,
            law_id: "safe-torque".into(),
            smtlib2: "(assert (< robot_torque 0.9))".into(),
            proposing_phi: phi,
            timestamp: 1,
        }
    }

    #[test]
    fn macro_message_is_now_defined_and_bounded() {
        let message = SwarmMessage::MacroGossip(MacroGossipMsg {
            node_id: Uuid::from_u128(1),
            domain: "symthaea.macro.v1".into(),
            payload: vec![0; 32],
            timestamp: 1,
        });
        assert_eq!(message.kind(), SwarmMessageKind::MacroGossip);
        assert!(message.validate().is_ok());
    }

    #[test]
    fn nil_application_node_id_is_rejected() {
        let message = SwarmMessage::MacroGossip(MacroGossipMsg {
            node_id: Uuid::nil(),
            domain: "symthaea.macro.v1".into(),
            payload: Vec::new(),
            timestamp: 1,
        });
        assert_eq!(message.validate(), Err(MessageValidationError::NilNodeId));
    }

    #[test]
    fn duplicate_law_gossip_does_not_inflate_support() {
        let node = Uuid::from_u128(1);
        let mut aggregator = SwarmAggregator::new();
        aggregator.update_peer(SwarmStateMsg {
            node_id: node,
            platform_type: "test".into(),
            local_phi: 1.0,
            consciousness_hv: ContinuousHV::zero(16_384),
            intent_hv: ContinuousHV::zero(16_384),
            timestamp: 1,
        });

        aggregator.ingest_law_proposal(law(node, 0.6));
        aggregator.ingest_law_proposal(law(node, 0.6));

        assert_eq!(aggregator.collective_laws["safe-torque"].1, 0.6);
    }

    #[test]
    fn stale_law_vote_cannot_replace_newer_vote() {
        let node = Uuid::from_u128(1);
        let mut aggregator = SwarmAggregator::new();
        aggregator.update_peer(SwarmStateMsg {
            node_id: node,
            platform_type: "test".into(),
            local_phi: 1.0,
            consciousness_hv: ContinuousHV::zero(16_384),
            intent_hv: ContinuousHV::zero(16_384),
            timestamp: 10,
        });

        let mut newer = law(node, 0.7);
        newer.timestamp = 20;
        assert!(matches!(
            aggregator.try_ingest_law_proposal(newer),
            LawVoteOutcome::Recorded { .. }
        ));

        let mut stale = law(node, 0.2);
        stale.timestamp = 19;
        assert_eq!(
            aggregator.try_ingest_law_proposal(stale),
            LawVoteOutcome::StaleVote
        );
        assert_eq!(aggregator.collective_laws["safe-torque"].1, 0.7);
    }

    #[test]
    fn law_vote_cannot_exceed_latest_advertised_phi() {
        let node = Uuid::from_u128(1);
        let mut aggregator = SwarmAggregator::new();
        aggregator.update_peer(SwarmStateMsg {
            node_id: node,
            platform_type: "test".into(),
            local_phi: 0.4,
            consciousness_hv: ContinuousHV::zero(16_384),
            intent_hv: ContinuousHV::zero(16_384),
            timestamp: 10,
        });

        assert_eq!(
            aggregator.try_ingest_law_proposal(law(node, 0.8)),
            LawVoteOutcome::VoteExceedsAdvertisedPhi {
                advertised_phi: 0.4,
                proposed_phi: 0.8,
            }
        );
    }

    #[test]
    fn conflicting_text_for_same_law_id_is_rejected() {
        let node = Uuid::from_u128(1);
        let other = Uuid::from_u128(2);
        let mut aggregator = SwarmAggregator::new();
        for node_id in [node, other] {
            aggregator.update_peer(SwarmStateMsg {
                node_id,
                platform_type: "test".into(),
                local_phi: 1.0,
                consciousness_hv: ContinuousHV::zero(16_384),
                intent_hv: ContinuousHV::zero(16_384),
                timestamp: 1,
            });
        }
        aggregator.ingest_law_proposal(law(node, 0.2));
        let mut conflicting = law(other, 0.3);
        conflicting.smtlib2 = "(assert (> robot_torque 0.9))".into();
        assert_eq!(
            aggregator.try_ingest_law_proposal(conflicting),
            LawVoteOutcome::ConflictingText
        );
    }

    #[test]
    fn stale_state_does_not_replace_newer_state() {
        let node = Uuid::from_u128(7);
        let make_state = |timestamp, phi| SwarmStateMsg {
            node_id: node,
            platform_type: "test".into(),
            local_phi: phi,
            consciousness_hv: ContinuousHV::zero(16_384),
            intent_hv: ContinuousHV::zero(16_384),
            timestamp,
        };
        let mut aggregator = SwarmAggregator::new();
        aggregator.update_peer(make_state(10, 0.8));
        aggregator.update_peer(make_state(9, 0.1));
        assert_eq!(aggregator.peer_states[&node].local_phi, 0.8);
    }
}

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
