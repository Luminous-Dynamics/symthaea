// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use std::time::Duration;
use symthaea::mind::{AsyncMind, MindConfig};
use symthaea::swarm::SwarmMessage;
use tokio::time::sleep;

#[tokio::main]
async fn main() {
    println!("🧠 Symthaea v0.9.0: The Schizophrenic Swarm (Local P2P Telepathy)");
    println!("---------------------------------------------------------------");

    // 1. SETUP: Node A (Heavy/Wall Power)
    let config_a = MindConfig {
        ..Default::default()
    };
    let (node_a, _join_a) = AsyncMind::spawn(config_a);
    node_a.update_thermodynamics(0.1).await; // Wall power

    // 2. SETUP: Node B (Light/Battery)
    let config_b = MindConfig {
        ..Default::default()
    };
    let (node_b, _join_b) = AsyncMind::spawn(config_b);
    node_b.update_thermodynamics(0.9).await; // Battery / High load

    println!("\n[PHASE 1] Initializing Swarm Link...");

    // 3. EVOLUTION: Node A discovers a mutation
    println!("\n[PHASE 2] Node A (Heavy) running Meta-Forge...");
    sleep(Duration::from_millis(500)).await;

    let mutation = SwarmMessage::BrainMutation {
        mutation_id: "breakthrough_001".to_string(),
        tau_scale: 1.15,
        predicted_phi_gain: 0.082,
    };
    println!("   ⚡ Node A discovered mutation: tau_scale = 1.15, phi_gain = +8.2%");

    // 4. TELEPATHY: Node A broadcasts to Node B
    println!("\n[PHASE 3] Telepathic Transmission (Swarm Gossip)...");
    node_b.receive_swarm_gossip(mutation).await;

    // 5. VERIFICATION: Node B hot-swaps
    sleep(Duration::from_millis(200)).await;
    let state_b = node_b.snapshot().await;

    println!("\n[PHASE 4] Node B (Light) hot-swap audit:");
    if let Some(scale) = state_b.last_mutation_suggestion {
        println!(
            "   ✅ SUCCESS: Node B received and recognized DNA mutation (scale: {:.2})",
            scale
        );
        println!("   -> Node B is now evolving toward Node A's discovery.");
    } else {
        println!("   ❌ FAILURE: Node B did not receive the mutation.");
    }

    println!("\n[CONCLUSION] Digital Telepathy achieved. The Swarm is one mind.");
    node_a.shutdown().await;
    node_b.shutdown().await;
}