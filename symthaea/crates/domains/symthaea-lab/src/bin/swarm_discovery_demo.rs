// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Swarm Discovery Demo — Knowledge Fusion via P2P Gossip
//!
//! Demonstrates two Symthaea nodes discovering each other via Iroh
//! and sharing high-dimensional design wisdom (Proofs & State).

use symthaea_core::hdc::ContinuousHV;
use symthaea_swarm::networking::TelepathicSocket;
use symthaea_swarm::{SwarmAggregator, SwarmMessage, SwarmProofMsg};
use tokio::sync::mpsc;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📡 Starting Symthaea Swarm Discovery Demo...");

    // 1. Setup local node identity
    let node_id = Uuid::new_v4();
    let topic_raw = [0u8; 32]; // Shared topic for the demo swarm

    let (tx, mut rx) = mpsc::channel(100);

    // 2. Initialize Iroh Endpoint
    let endpoint = iroh::Endpoint::builder(iroh::endpoint::presets::N0)
        .bind()
        .await?;
    let socket = TelepathicSocket::new(endpoint, topic_raw, tx).await?;

    println!(
        "✅ Node {} initialized. Peer ID: {}",
        node_id,
        socket.node_id()
    );

    // 3. Spawn the telepathic listener
    let socket_task = socket.clone();
    tokio::spawn(async move {
        if let Err(e) = socket_task.run().await {
            eprintln!("❌ Telepathic Socket Error: {}", e);
        }
    });

    // 4. Mock Local Design Wisdom
    let mut aggregator = SwarmAggregator::new();

    println!("\n🧠 Local Brain: Generating high-dimensional design wisdom...");
    let proof = SwarmProofMsg {
        node_id,
        label: "Carbon-Fiber-T300-Invariant".into(),
        smtlib2: "(assert (>= strength 3500))".into(),
        proof_hv: ContinuousHV::random(16384, 42),
        verified: true,
        timestamp: 1716336000,
    };

    // 5. Broadcast to Swarm
    println!("📡 Broadcasting verified proof to the hive mind...");
    socket.broadcast(SwarmMessage::ProofGossip(proof)).await?;

    // 6. Listen for Inbound Wisdom (Simulating other nodes)
    println!("👂 Listening for peer updates...");

    let timeout = tokio::time::sleep(std::time::Duration::from_secs(3));
    tokio::pin!(timeout);

    loop {
        tokio::select! {
            Some(msg) = rx.recv() => {
                match msg {
                    SwarmMessage::State(state) => {
                        println!("✨ Received Peer State from {}: Phi={:.2}", state.node_id, state.local_phi);
                        aggregator.update_peer(state);
                    }
                    SwarmMessage::ProofGossip(proof) => {
                        println!("📜 Received Collective Proof: {} (Verified={})", proof.label, proof.verified);
                        aggregator.ingest_peer_proof(proof);
                    }
                }

                let hive_norm = aggregator.hive_mind_vector().norm();
                println!("🧬 Hive Mind Coherence Norm: {:.4}", hive_norm);
            }
            _ = &mut timeout => {
                println!("\n⌛ Demo cycle complete.");
                break;
            }
        }
    }

    Ok(())
}
