// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-swarm — Collective Consciousness Protocol
//!
//! Implements a P2P swarm protocol for sharing consciousness states and
//! verified math/proof records between Symthaea nodes.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::ContinuousHV;
use uuid::Uuid;

/// Message containing a node's local consciousness state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmStateMsg {
    /// Unique ID of the originating node.
    pub node_id: Uuid,
    /// Local Phi value.
    pub local_phi: f64,
    /// Local consciousness vector (HDC).
    pub consciousness_hv: ContinuousHV,
    /// Node's current "mood" or intent vector.
    pub intent_hv: ContinuousHV,
    /// Unix timestamp in milliseconds.
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

/// Unified swarm wire protocol envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmMessage {
    State(SwarmStateMsg),
    ProofGossip(SwarmProofMsg),
}

/// Aggregator for swarm-wide consciousness states and collective proofs.
#[derive(Default, Debug)]
pub struct SwarmAggregator {
    /// Collection of states from other nodes.
    pub peer_states: std::collections::HashMap<Uuid, SwarmStateMsg>,
    /// Global swarm-replicated formal lemma proof repository database.
    pub swarm_proofs: Vec<SwarmProofMsg>,
}

impl SwarmAggregator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add or update a peer's state.
    pub fn update_peer(&mut self, msg: SwarmStateMsg) {
        eprintln!("DEBUG: SwarmAggregator inserting peer={}", msg.node_id);
        self.peer_states.insert(msg.node_id, msg);
    }

    /// Record a peer-distributed mathematical lemma into local memory.
    pub fn ingest_peer_proof(&mut self, msg: SwarmProofMsg) {
        if !self
            .swarm_proofs
            .iter()
            .any(|p| p.label == msg.label && p.node_id == msg.node_id)
        {
            self.swarm_proofs.push(msg);
        }
    }

    /// Compute the fused "Hive Mind" vector (mean of all peer states).
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
}

#[cfg(feature = "networking")]
pub mod networking {
    use super::*;
    use futures::StreamExt;
    use iroh::Endpoint;
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tokio::sync::mpsc;

    /// The "Telepathic Socket" — P2P bridge for broadcasting high-dimensional consciousness and proofs.
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

        /// Broadcast a unified swarm envelope to the hive mind network cluster.
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

#[cfg(test)]
mod swarm_tests {
    use super::*;

    #[test]
    fn test_swarm_message_enum_serialization_roundtrip() {
        let node_id = Uuid::new_v4();
        let state_msg = SwarmStateMsg {
            node_id,
            local_phi: 0.85,
            consciousness_hv: ContinuousHV::zero(16384),
            intent_hv: ContinuousHV::zero(16384),
            timestamp: 1716336000,
        };

        let envelope = SwarmMessage::State(state_msg);

        // Ensure bincode handles enum variant tagging safely
        let serialized = bincode::serialize(&envelope).unwrap();
        let deserialized: SwarmMessage = bincode::deserialize(&serialized).unwrap();

        if let SwarmMessage::State(recovered) = deserialized {
            assert_eq!(recovered.node_id, node_id);
            assert_eq!(recovered.local_phi, 0.85);
        } else {
            panic!("Failed to unpack SwarmMessage::State variant envelope");
        }
    }

    #[test]
    fn test_swarm_aggregator_ingests_peer_proof_gossip() {
        let mut aggregator = SwarmAggregator::new();
        let peer_id = Uuid::new_v4();

        let proof_msg = SwarmProofMsg {
            node_id: peer_id,
            label: "L3.0_pigeonhole".to_string(),
            smtlib2: "(check-sat)".to_string(),
            proof_hv: ContinuousHV::zero(16384),
            verified: true,
            timestamp: 1716336000,
        };

        aggregator.ingest_peer_proof(proof_msg.clone());
        assert_eq!(aggregator.swarm_proofs.len(), 1);

        // Prevent duplicate lemma insertions to maintain strict memory tracking boundaries
        aggregator.ingest_peer_proof(proof_msg);
        assert_eq!(aggregator.swarm_proofs.len(), 1);
        assert_eq!(aggregator.swarm_proofs[0].label, "L3.0_pigeonhole");
    }
}
