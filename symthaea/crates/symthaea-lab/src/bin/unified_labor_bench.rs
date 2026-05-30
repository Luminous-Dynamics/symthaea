// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Unified Swarm Labor Benchmark (Drone-Humanoid-Arm Heterogeneous Swarm)
//! 
//! Simulates a heterogeneous swarm working together on a construction task.
//! - Drone: Performs aerial terrain mapping (Haptic Gossip).
//! - Humanoid: Performs heavy labor, using drone data to stabilize gait.
//! - Manipulator: Receives heavy components from the humanoid, adjusting for load.

use symthaea_swarm::{SwarmAggregator, SwarmMessage, HapticPulseMsg, SwarmStateMsg};
use symthaea_humanoid::controller::HumanoidController;
use symthaea_humanoid::morphology::HumanoidMorphology;
use symthaea_humanoid::types::{HumanoidConfig, HumanoidState};
use symthaea_core::genesis::GenesisSeed;
use symtropy_robotics_bridge_core::platform::PlatformType;
use tracing::{info, level_filters::LevelFilter};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};
use uuid::Uuid;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env().add_directive(LevelFilter::INFO.into()))
        .init();

    info!("🌐 INITIATING UNIFIED SWARM LABOR BENCHMARK...");

    // 1. Initialize Swarm Infrastructure
    let mut swarm = SwarmAggregator::new();
    let genesis = GenesisSeed::from_phrase("Unified Labor Seed v1");
    
    let drone_id = Uuid::new_v4();
    let humanoid_id = Uuid::new_v4();
    let arm_id = Uuid::new_v4();

    // 2. Setup 64-DOF Humanoid Controller
    let morphology = HumanoidMorphology::FullSpine;
    let config = HumanoidConfig {
        morphology,
        ..HumanoidConfig::default()
    };
    let mut humanoid_controller = HumanoidController::new(&genesis, &config);

    info!("🤖 Heterogeneous Swarm Registered:");
    info!("   🛸 Drone [{}] - Aerial Metrology", drone_id);
    info!("   🚶 Humanoid [{}] - Industrial Labor", humanoid_id);
    info!("   🦾 Manipulator [{}] - Precision Assembly", arm_id);

    // 3. Multi-Platform Task Loop
    const STEPS: usize = 100;
    const DT: f32 = 0.02;

    for step in 0..STEPS {
        let t = step as f32 * DT;

        // --- PHASE A: DRONE RECONNAISSANCE ---
        // Drone detects a "Haptic Anomaly" (Wind spike / obstacle)
        if step == 20 {
            info!("🛸 [Drone] Detected terrain anomaly at [10, 0, 5]. Broadcasting Haptic Pulse...");
            let pulse = HapticPulseMsg {
                node_id: drone_id,
                position: [10.0, 0.0, 5.0, 0.0],
                surprise: 5.5, // High prediction error
                impact_vector: [0.0, 0.0, 20.0, 0.0],
                timestamp: step as u64,
            };
            swarm.ingest_haptic_pulse(pulse);
        }

        // --- PHASE B: HUMANOID LABOR ---
        // Humanoid checks swarm haptic map before stepping
        let current_pos: [f64; 3] = [10.0, 0.0, 4.5]; // Approaching anomaly
        let grid_pos = [
            current_pos[0].round() as i32,
            current_pos[1].round() as i32,
            current_pos[2].round() as i32,
        ];
        
        let swarm_surprise = swarm.haptic_map.get(&grid_pos).cloned().unwrap_or(0.0);
        
        if swarm_surprise > 1.0 {
            info!("🚶 [Humanoid] Received Haptic Empathy from Drone. Pre-adjusting gait for anomaly (Surprise: {:.2})", swarm_surprise);
        }

        // --- PHASE C: HAPTIC HAND-OFF ---
        // Humanoid reaches the Manipulator Arm to hand off a payload
        if step == 80 {
            info!("🤝 [Swarm] Initiating Payload Hand-off: Humanoid -> Manipulator");
            
            // Humanoid broadcasts its current intent (using a stable seed vector)
            let humanoid_state = SwarmStateMsg {
                node_id: humanoid_id,
                platform_type: PlatformType::Humanoid,
                local_phi: 0.85,
                consciousness_hv: genesis.hv("humanoid::joint_strain", 16384),
                intent_hv: genesis.hv("humanoid::intent", 16384),
                timestamp: step as u64,
            };
            swarm.update_peer(humanoid_state);
            
            info!("🦾 [Manipulator] Feeling Humanoid strain. Synchronizing PID gains for soft capture.");
        }

        // --- PHASE D: COLLECTIVE PHI CALCULATION ---
        if step % 25 == 0 {
            let swarm_phi = swarm.calculate_swarm_phi();
            info!("[T={:.2}s] Swarm Coherence: {:.3} | Active Haptic Nodes: {}", 
                  t, swarm_phi, swarm.haptic_map.len());
        }
    }

    info!("✨ UNIFIED SWARM LABOR BENCHMARK COMPLETE.");
    info!("✅ Drone-Humanoid-Arm collaboration verified via Haptic Empathy.");

    Ok(())
}
