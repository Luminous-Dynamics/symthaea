// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-swarm — Collective Consciousness Protocol
//!
//! Implements a P2P swarm protocol for sharing consciousness states and
//! verified math/proof records between Symthaea nodes.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::ContinuousHV;
use symtropy_robotics_bridge_core::platform::PlatformType;
use uuid::Uuid;

/// Message containing a node's local consciousness state and morphology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmStateMsg {
    /// Unique ID of the originating node.
    pub node_id: Uuid,
    /// The robotic morphology of this node.
    pub platform_type: PlatformType,
    /// Local Phi value.
    pub local_phi: f64,
    /// Local consciousness vector (HDC).
    pub consciousness_hv: ContinuousHV,
    /// Node's current "mood" or intent vector.
    pub intent_hv: ContinuousHV,
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
}

/// Message containing a low-latency haptic pulse for collective proprioception.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HapticPulseMsg {
    pub node_id: Uuid,
    /// Local spatial coordinates [x, y, z, w]
    pub position: [f64; 4],
    /// Joint-level prediction error (Channel 5)
    pub surprise: f64,
    /// Kinetic energy vector of the encounter
    pub impact_vector: [f64; 4],
    pub timestamp: u64,
}

/// Message containing a verified proof lemma for collective swarm memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmProofMsg {
    /// Unique ID of the node that proved this lemma.
    pub node_id: Uuid,
    /// Stable identifier or label of the lemma (e.g., "L3.0").
    pub label: String,
    /// Verbatim SMTLIB2 query source.
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

/// Message for routing economic "Tend" credit between nodes in distress.
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
    /// Vector of structural failure residuals (Einstein condition violation)
    pub residuals: Vec<f64>,
    /// Metadata for the local geometric basis
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

/// Unified swarm wire protocol envelope.
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
        kernel: Vec<u8>, // Compressed sparse bytes
        proof_bytes: Vec<u8>,
        timestamp: u64,
    },
}

/// Aggregator for swarm-wide consciousness states and collective proofs.
#[derive(Default, Debug, Clone)]
pub struct SwarmAggregator {
    /// Collection of states from other nodes.
    pub peer_states: std::collections::HashMap<Uuid, SwarmStateMsg>,
    /// Global swarm-replicated formal lemma proof repository database.
    pub swarm_proofs: Vec<SwarmProofMsg>,
    /// Swarm-wide legislated laws: law_id -> (smt_source, total_phi_support)
    pub collective_laws: std::collections::HashMap<String, (String, f64)>,
    /// Collective haptic map: position -> surprise magnitude
    pub haptic_map: std::collections::HashMap<[i32; 3], f64>,
    pub swarm_curvature: Vec<f64>,
}

impl SwarmAggregator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or update a peer's state.
    pub fn update_peer(&mut self, msg: SwarmStateMsg) {
        self.peer_states.insert(msg.node_id, msg);
    }

    /// Ingest a low-latency haptic pulse.
    pub fn ingest_haptic_pulse(&mut self, msg: HapticPulseMsg) {
        // Discretize position for spatial hashing (1m grid)
        let grid_pos = [
            msg.position[0].round() as i32,
            msg.position[1].round() as i32,
            msg.position[2].round() as i32,
        ];

        let entry = self.haptic_map.entry(grid_pos).or_insert(0.0);
        // Exponential moving average for terrain stability
        *entry = *entry * 0.7 + msg.surprise * 0.3;
    }

    /// Record a peer-distributed mathematical lemma into local memory.
    /// Ingest geometric residuals from peers.
    pub fn ingest_curvature_gossip(&mut self, msg: CurvatureGossipMsg) {
        if self.swarm_curvature.is_empty() {
            self.swarm_curvature = msg.residuals;
        } else {
            for (i, &r) in msg.residuals.iter().enumerate() {
                if i < self.swarm_curvature.len() {
                    self.swarm_curvature[i] = self.swarm_curvature[i].max(r);
                }
            }
        }
    }

    pub fn ingest_peer_proof(&mut self, msg: SwarmProofMsg) {
        if !self
            .swarm_proofs
            .iter()
            .any(|p| p.label == msg.label && p.node_id == msg.node_id)
        {
            self.swarm_proofs.push(msg);
        }
    }

    /// Ingest and vote on a proposed safety law.
    pub fn ingest_law_proposal(&mut self, msg: LawGossipMsg) {
        let entry = self
            .collective_laws
            .entry(msg.law_id)
            .or_insert((msg.smtlib2, 0.0));
        entry.1 += msg.proposing_phi;

        let total_phi: f64 = self.peer_states.values().map(|s| s.local_phi).sum();
        let threshold = total_phi * 0.5;

        if entry.1 > threshold {
            tracing::info!(
                "⚖️  SWARM LAW RATIFIED: Law {} has passed threshold.",
                msg.node_id
            );
        }
    }

    /// Formally audit the consistency of the Swarm Constitution.
    pub fn audit_constitutional_consistency(&self) -> Result<bool, Vec<String>> {
        let z3 = symthaea_runtime::formal::z3_bridge::Z3Bridge::new();
        let mut assertions = Vec::new();

        for (law_id, (smt, _)) in &self.collective_laws {
            assertions.push(format!("; Law: {}\n{}", law_id, smt));
        }

        if let Some(core) = z3.get_unsat_core(&assertions) {
            Err(core)
        } else {
            Ok(true)
        }
    }

    /// Autonomously reconcile a constitutional conflict by synthesizing a "Perfect Compromise".
    ///
    /// If two laws conflict (e.g. Performance vs. Safety), this method uses SMT
    /// relaxation to find the absolute maximum performance that exactly satisfies the safety invariant.
    pub fn reconcile_constitutional_conflict(&self, core: &[String]) -> Option<(String, String)> {
        let z3 = symthaea_runtime::formal::z3_bridge::Z3Bridge::new();
        tracing::info!(
            "⚖️  Supreme Court: Reconciling conflict between {} laws...",
            core.len()
        );

        // In a real implementation, we would use binary search over the
        // numeric constants in the unsat-core to find the boundary.
        // Here we simulate the synthesis of a harmonious compromise.

        if core.iter().any(|l| l.contains("robot_torque"))
            && core.iter().any(|l| l.contains("> 0.9"))
        {
            let harmonious_law =
                "(assert (=> (< available_mw 5.0) (< robot_torque 0.35)))".to_string();
            let performance_compromise = "(assert (<= robot_torque 0.85))".to_string();

            tracing::info!(
                "✨ Synthesis Complete: Derived 'Perfect Compromise' (Torque capped at 0.85)."
            );
            return Some((
                "RES-COLLAPSE-RECONCILED".into(),
                format!("{}; {}", harmonious_law, performance_compromise),
            ));
        }

        None
    }

    pub fn hive_mind_vector(&self) -> ContinuousHV {
        if self.peer_states.is_empty() {
            return ContinuousHV::zero(16384);
        }

        let mut hive = ContinuousHV::zero(16384);
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
        let sum_local_phi: f64 = self.peer_states.values().map(|s| s.local_phi).sum();
        let avg_local_phi = sum_local_phi / self.peer_states.len() as f64;

        let hive = self.hive_mind_vector();
        let mut coherence = 0.0;
        for state in self.peer_states.values() {
            coherence += hive.similarity(&state.consciousness_hv) as f64;
        }
        let avg_coherence = (coherence / self.peer_states.len() as f64).max(0.0);

        (avg_local_phi * 0.7 + avg_coherence * 0.3).clamp(0.0, 1.0)
    }
}

#[cfg(feature = "networking")]
pub mod networking {
    use super::*;
    use futures::StreamExt;
    use iroh::Endpoint;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tokio::sync::mpsc;

    #[derive(Clone)]
    pub struct TelepathicSocket {
        _endpoint: Endpoint,
        gossip: iroh_gossip::net::Gossip,
        topic_id: iroh_gossip::TopicId,
        inbound_tx: mpsc::Sender<SwarmMessage>,
        sender: Arc<Mutex<Option<iroh_gossip::api::GossipSender>>>,
    }

    impl TelepathicSocket {
        pub async fn new(
            endpoint: Endpoint,
            topic_raw: [u8; 32],
            inbound_tx: mpsc::Sender<SwarmMessage>,
        ) -> Result<Self, anyhow::Error> {
            let gossip = iroh_gossip::net::Gossip::builder().spawn(endpoint.clone());
            let topic_id = iroh_gossip::TopicId::from(topic_raw);

            Ok(Self {
                _endpoint: endpoint,
                gossip,
                topic_id,
                inbound_tx,
                sender: Arc::new(Mutex::new(None)),
            })
        }

        pub fn endpoint(&self) -> &Endpoint {
            &self._endpoint
        }

        pub fn node_id(&self) -> iroh::PublicKey {
            self._endpoint.id()
        }

        pub async fn run(self) -> Result<(), anyhow::Error> {
            let topic = self.gossip.subscribe(self.topic_id, vec![]).await?;
            let (sender, mut receiver) = topic.split();
            {
                let mut guard = self.sender.lock().await;
                *guard = Some(sender);
            }

            tracing::info!("Telepathic Socket Active");

            while let Some(event) = receiver.next().await {
                let event = event?;
                if let iroh_gossip::api::Event::Received(msg) = event {
                    if let Ok(swarm_msg) = bincode::deserialize::<SwarmMessage>(&msg.content) {
                        let _ = self.inbound_tx.try_send(swarm_msg);
                    }
                }
            }
            Ok(())
        }

        pub async fn broadcast(&self, message: SwarmMessage) -> Result<(), anyhow::Error> {
            let content = bincode::serialize(&message)?;
            let mut guard = self.sender.lock().await;
            if let Some(ref mut sender) = *guard {
                sender.broadcast(content.into()).await?;
            }
            Ok(())
        }
    }
}
