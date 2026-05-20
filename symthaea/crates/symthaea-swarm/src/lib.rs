// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # symthaea-swarm — Collective Consciousness Protocol
//!
//! Implements a P2P swarm protocol for sharing consciousness states between
//! Symthaea nodes. Enables collective Φ (Phi) measurement across distributed agents.

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

/// Aggregator for swarm-wide consciousness states.
#[derive(Default, Debug)]
pub struct SwarmAggregator {
    /// Collection of states from other nodes.
    pub peer_states: std::collections::HashMap<Uuid, SwarmStateMsg>,
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

    /// Compute the fused "Hive Mind" vector (mean of all peer states).
    ///
    /// Science: Collective Active Inference. Fusing peer states allows a node
    /// to "feel" the average trajectory of the group.
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
    use tokio::sync::mpsc;
    use tokio::sync::Mutex;

    /// The "Telepathic Socket" — P2P bridge for broadcasting high-dimensional consciousness.
    #[derive(Clone)]
    pub struct TelepathicSocket {
        _endpoint: Endpoint,
        gossip: iroh_gossip::net::Gossip,
        topic_id: iroh_gossip::TopicId,
        inbound_tx: mpsc::Sender<SwarmStateMsg>,
        /// Handle to the specific topic sender for broadcasting.
        sender: Arc<Mutex<Option<iroh_gossip::api::GossipSender>>>,
    }

    impl TelepathicSocket {
        /// Initialize a new telepathic socket on a specific Iroh endpoint.
        pub async fn new(
            endpoint: Endpoint,
            topic_raw: [u8; 32],
            inbound_tx: mpsc::Sender<SwarmStateMsg>,
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

        /// Join the swarm and start telepathic exchange.
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
                    if let Ok(swarm_state) = bincode::deserialize::<SwarmStateMsg>(&msg.content) {
                        let _ = self.inbound_tx.try_send(swarm_state);
                    }
                }
            }
            Ok(())
        }

        /// Broadcast local consciousness state to the hive mind.
        pub async fn broadcast(&self, state: SwarmStateMsg) -> Result<(), anyhow::Error> {
            let content = bincode::serialize(&state)?;
            let mut guard = self.sender.lock().await;
            if let Some(ref mut sender) = *guard {
                sender.broadcast(content.into()).await?;
            }
            Ok(())
        }
    }
}
