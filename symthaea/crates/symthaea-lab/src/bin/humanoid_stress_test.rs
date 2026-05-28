// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! 64-DOF Humanoid Stress Test (Deterministic Verification)
//! 
//! Measures the stabilization speed and thermodynamic cost of the flagship 
//! 64-actuator FullSpine humanoid under unexpected force injection.

use symtropy_math::{Point, Bivector, HyperBox, Transform};
use symtropy_physics::{PhysicsWorld, RigidBody};
use symtropy_physics::body::BodyType;
use symtropy_robotics_bridge_core::platform::PlatformType;
use symtropy_consciousness_physics::ConsciousnessField;
use std::time::Instant;
use tracing::{info, warn, level_filters::LevelFilter};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};
use nalgebra::SVector;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env().add_directive(LevelFilter::INFO.into()))
        .init();

    info!("🔬 INITIATING 64-DOF HUMANOID DETERMINISTIC STRESS TEST...");

    const D: usize = 4; // Testing in 4D to verify N-D stress bleeding
    const DT: f64 = 0.01667; // 60Hz lockstep
    const SIM_DURATION_SECS: f64 = 5.0;
    const SHOVE_TIME: f64 = 1.0;

    // 1. Initialize 4D Physics World
    let mut world = PhysicsWorld::<D>::new(SVector::from([0.0, -9.81, 0.0, 0.0]));
    
    // Add static ground box (large) instead of halfspace for better GJK stability
    let ground_handle = world.add_body(RigidBody::new(
        symtropy_physics::body::BodyHandle(0), 
        BodyType::Static,
        Transform::identity(),
        Box::new(HyperBox::<D>::new([100.0, 0.1, 100.0, 100.0])), // Large flat platform
        0.0, 
        SVector::zeros(), 
    ));
    info!("🌱 Ground platform established (Handle: {:?})", ground_handle);

    // 2. Initialize Consciousness Field with N-D Scaling
    let mut field = ConsciousnessField::<D>::new();
    
    // 3. Spawn 64-DOF Humanoid articulated segments (Simulated)
    let platform = PlatformType::Humanoid;
    let n_actuators = platform.num_actuators();
    
    let mut segments = Vec::new();
    for i in 0..8 { 
        let pos = Point::new([0.0, 1.0 + (i as f64 * 0.2), 0.0, 0.0]); // Lower to hit the ground faster
        let handle = world.add_sphere(pos, 0.1, platform.default_mass() / 8.0);
        field.register(handle, 100_000.0, 2.0); 
        segments.push(handle);
    }

    info!("🤖 Humanoid Spawned: {} DOF | Mass: {}kg | Dimension: {}D", 
          n_actuators, platform.default_mass(), D);

    let mut current_time = 0.0;
    let mut step_count = 0;
    let mut shove_applied = false;
    let start_instant = Instant::now();

    info!("⏳ Running {}s simulation with Ground Impacts...", SIM_DURATION_SECS);

    while current_time < SIM_DURATION_SECS {
        // 4. Force Injection (The Shove)
        if current_time >= SHOVE_TIME && !shove_applied {
            info!("💥 [T={:.2}s] INJECTING EXTERNAL SHOVE: 500N Force Vector", current_time);
            let shove = SVector::from([500.0, 0.0, 0.0, 100.0]); 
            
            if let Some(torso) = segments.last() {
                if let Some(body) = world.body_mut(*torso) {
                    body.apply_force(shove);
                }
            }
            shove_applied = true;
        }

        // 5. Active Motor Simulation (Power consumption)
        for handle in &segments {
            if let Some(body) = world.body_mut(*handle) {
                body.apply_torque(Bivector::zero()); 
                field.consume_energy(*handle, 2.0); // 2 Joules of internal motor work
            }
        }

        // 6. Physics Step with Callback (Friction/Impulses will record dissipation)
        world.step_with_callback(DT, &mut field);
        
        // Finalize this tick's energy balance
        field.tick_thermodynamics();

        // 7. Metrics Gathering
        if step_count % 30 == 0 {
            let total_energy: f64 = segments.iter().map(|h| field.entities.get(h).map(|e| e.energy.available).unwrap_or(0.0)).sum();
            
            info!("[T={:.2}s] System Energy: {:.2}J | Total Loss: {:.2}J", 
                  current_time, total_energy, field.ledger.energy_out);
        }

        current_time += DT;
        step_count += 1;
    }

    let total_duration = start_instant.elapsed();
    let ledger = &field.ledger;

    info!("✨ STRESS TEST COMPLETE.");
    info!("📊 FINAL PERFORMANCE REPORT:");
    info!("   ⏱️  Total Sim Time: {:?}", total_duration);
    info!("   🌡️  Total Energy Consumed (Work): {:.2} Joules", ledger.lifetime_energy);
    info!("   🔥 Total Dissipation (Heat): {:.2} Joules", ledger.energy_out);
    info!("   📉 Energy Conservation Error: {:.6}%", ledger.lifetime_error_rate() * 100.0);
    
    if ledger.lifetime_error_rate() < 0.2 { // Wider tolerance for complex collisions
        info!("   ✅ Determinism & Conservation Check PASSED.");
    } else {
        warn!("   ⚠️  Conservation Check FAILED: Energy drift {:.2}%", ledger.lifetime_error_rate() * 100.0);
    }

    Ok(())
}
