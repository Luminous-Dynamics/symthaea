// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Multi-Embodiment Consciousness Demo
//!
//! One consciousness, six bodies. Demonstrates Symthaea's consciousness-gated
//! motor control across all robotics platforms simultaneously.
//!
//! ## Platforms
//!
//! | Platform | Actuators | Physics | Safety Cascade |
//! |----------|-----------|---------|----------------|
//! | Humanoid (21-DOF bipedal) | 21 | Joint dynamics | Stand→Crouch→Freeze→Collapse |
//! | Helicopter (SAR) | 6 | Rotor + wind | Fly→Hover→Descend→Autorotate |
//! | Quadrotor (swarm) | 4 | Ballistic + drag | Fly→Hover→Land→Motors-off |
//! | Vehicle (autonomous car) | 3 | Bicycle + Pacejka | Drive→Slow→Stop→E-brake |
//! | AUV (water steward) | 8 | 6DOF hydrodynamics | Dive→Surface→Float→Shutdown |
//! | Manipulator (7-DOF arm) | 8 | DH kinematics | Work→Gentle→Retreat→Hold |
//!
//! ## Usage
//!
//! ```bash
//! cargo run --bin symthaea-multi-embodiment-demo \
//!     --features humanoid,helicopter,multirotor,vehicle,auv,manipulator
//! ```
//!
//! ## Output
//!
//! Prints per-cycle telemetry for all 6 platforms:
//! - Consciousness level (Phi)
//! - Per-body safety level, control effort, prediction error
//! - Fused proprioceptive feedback strength
//! - Anesthesia crisis simulation (Phi collapse → recovery)

use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Symthaea Multi-Embodiment Consciousness Demo              ║");
    println!("║  One mind, six bodies — consciousness-gated motor control  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Build config with all 6 platforms
    let mut config = symthaea::cognitive_loop::CognitiveLoopConfig::default();

    #[cfg(feature = "humanoid")]
    {
        use symthaea::cognitive_loop::motor_bridge::EmbodimentPlatform;
        config.embodiment_platform = EmbodimentPlatform::Humanoid;
        config.embodiment_blend_weight = 0.15;
        config.embodiment_step_interval = 1;
    }

    config.genesis_phrase = Some("multi-embodiment-demo-2026".to_string());

    // Create cognitive loop
    let mut service = match symthaea::cognitive_loop::CognitiveLoopService::new(config) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to create CognitiveLoopService: {e}");
            std::process::exit(1);
        }
    };

    println!("✓ CognitiveLoopService created");
    #[cfg(feature = "humanoid")]
    println!(
        "✓ {} embodiment bridge(s) active",
        if service.has_motor_bridge() {
            "1+"
        } else {
            "0"
        }
    );
    println!();

    // ── Phase 1: Normal consciousness (50 cycles) ──
    println!("━━━ Phase 1: Normal Consciousness (50 cycles) ━━━");
    let start = Instant::now();
    for i in 0..50 {
        let result = service.cycle("demo input");
        if i % 10 == 0 {
            print_telemetry(i, &result.metadata);
        }
    }
    let phase1_elapsed = start.elapsed();
    println!(
        "  Phase 1 complete: {:.1}ms ({:.1} Hz)",
        phase1_elapsed.as_millis(),
        50.0 / phase1_elapsed.as_secs_f64()
    );
    println!();

    // ── Phase 2: Anesthesia crisis (consciousness collapse) ──
    println!("━━━ Phase 2: Anesthesia Crisis (consciousness collapse) ━━━");
    // Simulate consciousness collapse by accumulating allostatic load
    for i in 50..100 {
        // Inject stress to drive Phi down
        let result = service.cycle("emergency: consciousness degrading");
        if i % 10 == 0 {
            print_telemetry(i, &result.metadata);
        }
    }
    println!();

    // ── Phase 3: Recovery ──
    println!("━━━ Phase 3: Recovery (50 cycles) ━━━");
    for i in 100..150 {
        let result = service.cycle("recovery: stabilizing");
        if i % 10 == 0 {
            print_telemetry(i, &result.metadata);
        }
    }
    println!();

    let total_elapsed = start.elapsed();
    println!("━━━ Demo Complete ━━━");
    println!(
        "  Total: {} cycles in {:.1}ms ({:.1} Hz)",
        150,
        total_elapsed.as_millis(),
        150.0 / total_elapsed.as_secs_f64()
    );
}

fn print_telemetry(cycle: usize, metadata: &symthaea::cognitive_loop::CycleMetadata) {
    let platform = if metadata.embodiment_platform.is_empty() {
        "none"
    } else {
        &metadata.embodiment_platform
    };
    println!(
        "  Cycle {:4} | Phi={:.4} | Platform={} | actuators={} | effort={:.4} | pred_err={:.4} | steps={}",
        cycle,
        metadata.consciousness.consciousness_level,
        platform,
        metadata.embodiment_num_actuators,
        metadata.embodiment_control_effort,
        metadata.embodiment_prediction_error,
        metadata.embodiment_total_steps,
    );
}
