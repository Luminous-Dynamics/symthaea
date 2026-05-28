// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Swarm Bridge — P2P Memetic Propagation.
//!
//! Links Broca's cognitive breakthroughs to the Iroh P2P swarm, allowing
//! for decentralized sharing of evolved semantic nuclei.

use anyhow::Result;
use std::sync::Arc;
use symthaea_core::hdc::ContinuousHV;
use symthaea_swarm::SwarmMessage;
use symthaea_swarm::SwarmProofMsg;
use uuid::Uuid;

#[derive(Clone)]
pub struct SwarmBridge {
    pub node_id: Uuid,
    #[cfg(feature = "networking")]
    pub socket: Arc<tokio::sync::Mutex<Option<symthaea_swarm::networking::TelepathicSocket>>>,
}

impl SwarmBridge {
    pub fn new() -> Self {
        Self {
            node_id: Uuid::new_v4(),
            #[cfg(feature = "networking")]
            socket: Arc::new(tokio::sync::Mutex::new(None)),
        }
    }

    /// Publish a cognitive breakthrough (Semantic Nucleus) to the swarm.
    pub async fn publish_breakthrough(&self, label: &str, nucleus: &ContinuousHV) -> Result<()> {
        println!("📡 Swarm: Publishing breakthrough '{}'...", label);
        
        let msg = SwarmProofMsg {
            node_id: self.node_id,
            label: label.to_string(),
            smtlib2: "// Broca Semantic Nucleus".to_string(),
            proof_hv: nucleus.clone(),
            verified: true,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_millis() as u64,
        };

        #[cfg(feature = "networking")]
        {
            let mut socket_locked = self.socket.lock().await;
            if let Some(ref socket) = *socket_locked {
                socket.broadcast(SwarmMessage::ProofGossip(msg)).await?;
                println!("   └─ P2P Broadcast SUCCESS via Iroh.");
            } else {
                println!("   └─ [Mock] P2P Broadcast simulated (networking disabled).");
            }
        }

        Ok(())
    }
}
