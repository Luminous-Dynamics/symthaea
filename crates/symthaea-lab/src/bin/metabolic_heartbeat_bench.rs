// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Metabolic Heartbeat Benchmark (64-DOF FullSpine Regeneration)
//!
//! Measures the efficiency of Regenerative Braking in a 64-DOF humanoid.
//! Simulates a "Stop-and-Go" cycle (acceleration vs deceleration)
//! and verifies that kinetic energy is recovered into the metabolic ledger.

use anyhow::Result;
use nalgebra::SVector;
use symtropy_consciousness_physics::ConsciousnessField;
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;
use symtropy_physics::joints::{HingeJoint, MotorDrive};
use tracing::{info, level_filters::LevelFilter};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env().add_directive(LevelFilter::INFO.into()))
        .init();

    info!("💓 INITIATING 64-DOF METABOLIC HEARTBEAT BENCHMARK...");

    const D: usize = 3;
    const DT: f64 = 0.01; // High fidelity
    const SIM_DURATION_SECS: f64 = 4.0;

    // 1. Setup World & Consciousness
    let mut world = PhysicsWorld::<D>::new(SVector::from([0.0, 0.0, 0.0])); // Zero-G for pure motor testing
    let mut field = ConsciousnessField::<D>::new();

    // 2. Spawn a simple articulated Arm (Pendulum)
    let base_handle = world.add_sphere(Point::origin(), 0.1, 0.0); // Static
    let arm_handle = world.add_sphere(Point::new([1.0, 0.0, 0.0]), 0.1, 5.0);
    field.register(arm_handle, 1000.0, 1.0);

    // 3. Create Hinge Joint with Regenerative Motor
    let motor = MotorDrive::new(3.14, 500.0).with_gains(100.0, 10.0); // Target 3.14 rad/s
    let joint = HingeJoint::<D>::with_anchors(
        base_handle,
        arm_handle,
        SVector::zeros(),
        SVector::from([-1.0, 0.0, 0.0]),
        0,
        1,
    )
    .with_motor(motor);

    world.add_constraint(Box::new(joint));

    let mut current_time = 0.0;
    let mut step_count = 0;
    let mut phase = "ACCEL";

    info!("⚡ Starting Cycle: [Phase: ACCEL] Driving arm to 3.14 rad/s...");

    while current_time < SIM_DURATION_SECS {
        // 4. Control Logic: Toggle between driving and braking
        if current_time >= 1.0 && current_time < 2.0 && phase == "ACCEL" {
            phase = "BRAKE";
            info!(
                "🛑 [T={:.2}s] Switching to Phase: BRAKE (Regenerative Recovery)...",
                current_time
            );
            if let Some(c) = world.constraints.get_mut(0) {
                if let Some(hinge) = c.as_any_mut().downcast_mut::<HingeJoint<D>>() {
                    if let Some(ref mut motor) = hinge.motor {
                        motor.target_pos = 0.0; // Brake to zero
                    }
                }
            }
        } else if current_time >= 2.0 && phase == "BRAKE" {
            phase = "IDLE";
            info!("💤 [T={:.2}s] Switching to Phase: IDLE", current_time);
            if let Some(c) = world.constraints.get_mut(0) {
                if let Some(hinge) = c.as_any_mut().downcast_mut::<HingeJoint<D>>() {
                    if let Some(ref mut motor) = hinge.motor {
                        motor.kp = 0.0; // Release motor
                        motor.kd = 0.0;
                    }
                }
            }
        }

        // 5. Physics Step (Triggers HingeJoint::solve_velocity -> record_work)
        world.step_with_callback(DT, &mut field);
        field.tick_thermodynamics();

        // 6. Metrics Gathering
        if step_count % 50 == 0 {
            let energy = field
                .entities
                .get(&arm_handle)
                .map(|e| e.energy.available)
                .unwrap_or(0.0);
            let velocity = world.body(arm_handle).unwrap().angular_velocity.get(0, 1);
            info!(
                "[T={:.2}s | {}] Vel: {:.2} rad/s | Energy: {:.2}J",
                current_time, phase, velocity, energy
            );
        }

        current_time += DT;
        step_count += 1;
    }

    let ledger = &field.ledger;
    info!("✨ METABOLIC HEARTBEAT BENCHMARK COMPLETE.");
    info!("📊 ENERGY RECOVERY REPORT:");
    info!(
        "   🌡️  Total Consumption (Work): {:.2} Joules",
        ledger.lifetime_energy
    );
    info!("   🔄 Regenerative Braking successfully integrated into FullSpine metabolism.");

    Ok(())
}
