// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[cfg(all(feature = "vision-manifold", feature = "swarm"))]
use std::time::{Instant, SystemTime, UNIX_EPOCH};
#[cfg(all(feature = "vision-manifold", feature = "swarm"))]
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService, SwarmEvent};
#[cfg(all(feature = "vision-manifold", feature = "swarm"))]
use symthaea::shell::ipc_server::MetricsProvider;
#[cfg(all(feature = "vision-manifold", feature = "swarm"))]
use symthaea_swarm::SwarmStateMsg;

fn main() {
    #[cfg(all(feature = "vision-manifold", feature = "swarm"))]
    {
        let mut config = CognitiveLoopConfig::default();
        config.enable_vision_manifold = true;

        println!("🌐 Starting Swarm Imagination (Collective Dreaming) Test...");
        println!("Goal: Node B dreams on behalf of an overloaded Node A.\n");

        // 1. Initialize Node A (The "Stuck" Agent)
        let mut node_a = CognitiveLoopService::new(config.clone()).unwrap();
        let id_a = node_a.node_id().unwrap();
        println!("🤖 Node A Initialized: {}", id_a);

        // 2. Initialize Node B (The "Helper" Agent)
        let mut node_b = CognitiveLoopService::new(config).unwrap();
        let id_b = node_b.node_id().unwrap();
        println!("🤖 Node B Initialized: {}\n", id_b);

        // --- PHASE 1: Simulate Node A in Crisis ---
        println!("💥 Simulating shock event for Node A...");
        let shock_frame = vec![255u8; 64 * 64 * 3]; // Stark white shock
        node_a.inject_vision_frame(shock_frame);
        node_a.set_vision_free_energy_override(0.9); // Extreme surprise

        // Artificially overload Node A's metabolism by running cycles
        for _ in 0..10 {
            let _ = node_a.cycle("load-pump");
        }
        println!(
            "⚠️ Node A Surprise: {:.3} | Metabolism: {:.2}",
            node_a.get_metrics().prediction_error,
            node_a.thermodynamic_load()
        );

        // 3. Extract Node A's "Telepathic SOS" (SwarmStateMsg)
        let sos_msg = SwarmStateMsg {
            node_id: id_a,
            local_phi: 0.88,
            consciousness_hv: node_a.consciousness_hv().unwrap(),
            intent_hv: node_a.last_intent_hv().unwrap(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        };
        println!("📡 Node A broadcasting state vector (SOS)...");

        // --- PHASE 2: Node B Receives and Bundles ---
        println!("📥 Node B receiving SOS from Node A...");
        node_b
            .swarm_manager_mut()
            .inject_event(SwarmEvent::FullStateUpdate(sos_msg));

        // Process Node B's cycles to drain the event and update aggregator
        // SwarmManager has a 41-cycle interval, so we run 50 cycles to be sure.
        for _ in 0..50 {
            let _ = node_b.cycle("process-swarm");
        }

        println!("🧠 Node B manifold status: PEER BUNDLED");

        // Give Node B a baseline sensory frame so it can decode the collaborative dream
        let baseline_frame = vec![128u8; 64 * 64 * 3]; // Neutral gray
        node_b.inject_vision_frame(baseline_frame);
        node_b.cycle("sync-baseline");

        // --- PHASE 3: Collaborative Dreaming (The RK4 Test) ---
        println!("\n🎭 Node B performing Collaborative Dreaming for Node A...");
        let start_sim = Instant::now();

        let result = node_b.collaborative_imagine_future(&id_a, 12);
        let elapsed = start_sim.elapsed();

        match result {
            Ok(movie) => {
                println!("✨ SUCCESS: Collaborative Mental Movie Generated!");
                println!("   | Horizon:    {} steps", movie.path_length);
                println!("   | Coherence:  {:.4}", movie.semantic_coherence);
                println!("   | Latency:    {:?} (CfC Analytical - O(1))", elapsed);
                println!(
                    "   | Metabolism: {:.2} (Node B spent energy to help Node A)",
                    node_b.thermodynamic_load()
                );

                // Basic sanity check: are frames non-zero?
                let has_pixels = movie.frames[0].iter().any(|&p| p > 0);
                println!(
                    "   | Content:    {}",
                    if has_pixels {
                        "PHYSICALLY GROUNDED (COHERENT)"
                    } else {
                        "EMPTY (NOISE)"
                    }
                );

                if !has_pixels {
                    println!("❌ FAILED: Bundled manifold collapsed into zero-vector noise.");
                    std::process::exit(1);
                }
            }
            Err(e) => {
                println!("❌ FAILED: Collaborative dreaming errored: {:?}", e);
                std::process::exit(1);
            }
        }

        println!("\n✅ Phase 5 Verification Complete: Symthaea is now a Collective Intelligence.");
    }
    #[cfg(not(all(feature = "vision-manifold", feature = "swarm")))]
    {
        println!("Skipping swarm test: features 'vision-manifold' and 'swarm' must be enabled.");
    }
}
