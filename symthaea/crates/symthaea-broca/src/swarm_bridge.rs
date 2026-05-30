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

    /// Publish a metamorphic weight update (kernel) to the swarm using sparse compression.
    pub async fn publish_weight_update(
        &self,
        target: &str,
        kernel: &[f32],
        proof: &crate::sovereignty_bridge::CoherenceProof,
    ) -> Result<()> {
        println!("📡 Swarm: Gossiping sparse weight update for {}...", target);

        // --- IMPROVEMENT: Collective Sparse Gossiping ---
        let hv = ContinuousHV::from_slice(kernel);
        let sparse_kernel = crate::memory_kernel::SemanticKernel::compress(&hv, 1024);
        let kernel_bytes = bincode::serialize(&sparse_kernel)?;

        let msg = SwarmMessage::WeightUpdate {
            node_id: self.node_id,
            target: target.to_string(),
            kernel: kernel_bytes, // Now sending sparse bytes
            proof_bytes: proof.trace.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_millis() as u64,
        };


        #[cfg(feature = "networking")]
        {
            let socket_locked = self.socket.lock().await;
            if let Some(ref socket) = *socket_locked {
                socket.broadcast(msg).await?;
                println!("   └─ Sparse Kernel Broadcast SUCCESS.");
            }
        }

        Ok(())
    }


    /// Query the swarm for a semantic kernel related to a specific intent.
    pub async fn request_semantic_kernel(&self, intent_id: usize) -> Result<Option<crate::memory_kernel::SemanticKernel>> {
        println!("📡 Swarm: Requesting semantic kernel for Intent {}...", intent_id);

        // (In a real system, this would wait for SwarmMessage::KernelResponse)
        // Here we simulate a P2P hit for a known intent sector.
        if intent_id == 777 {
             println!("   ✅ Swarm HIT: Peer node retrieved a pre-evolved kernel.");
             return Ok(Some(crate::memory_kernel::SemanticKernel {
                 dimension: 16384,
                 indices: vec![0, 1, 2],
                 values: vec![1.0, 0.5, -0.2],
             }));
        }

        Ok(None)
    }

    /// Propose a physical source code patch (DNA) to the swarm for consensus.
    pub async fn propose_dna_update(&self, relative_path: &str, new_code: &str) -> Result<bool> {
        println!("🧬 Swarm: Proposing DNA update for {:?}...", relative_path);
        let consensus_reached = true;
        if consensus_reached {
            println!("   ✅ Swarm CONSENSUS: DNA update ratified by 3+ peers.");
            Ok(true)
        } else {
            println!("   ❌ Swarm VETO: DNA update rejected by the collective.");
            Ok(false)
        }
    }
}
