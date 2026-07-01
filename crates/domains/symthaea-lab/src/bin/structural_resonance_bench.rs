// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Structural Resonance Benchmark (64-DOF FullSpine Humanoid)
//!
//! Tests the harmonic stability of the 11-segment cervical spine chain.
//! Simulates high-speed locomotion by applying oscillatory forces at the head
//! and measuring the propagation of kinetic energy through the spinal manifold.

use anyhow::Result;
use nalgebra::SVector;
use symtropy_consciousness_physics::ConsciousnessField;
use symtropy_math::Point;
use symtropy_physics::PhysicsWorld;
use symtropy_physics::joints::{HingeJoint, MotorDrive};
use symtropy_robotics_bridge_core::safety::JointSafetyAuthority;
use tracing::info;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env().add_directive(LevelFilter::INFO.into()))
        .init();

    info!("🦴 INITIATING 64-DOF STRUCTURAL RESONANCE BENCHMARK...");

    const D: usize = 4; // 4D physics for vibrational "bleeding" verification
    const DT: f64 = 0.01667;
    const SIM_DURATION_SECS: f64 = 5.0;
    const FREQUENCY_HZ: f64 = 5.0; // 5Hz head bobbing (running speed)

    // 1. Setup 4D World & Consciousness
    let mut world = PhysicsWorld::<D>::new(SVector::from([0.0, -9.81, 0.0, 0.0]));
    let mut field = ConsciousnessField::<D>::new();

    // 2. Spawn 11-segment Spine Chain
    let mut segments = Vec::new();
    let segment_mass = 0.2;
    let base_pos = [0.0, 1.2, 0.0, 0.0];

    // Static base (representing the upper chest)
    let base_handle = world.add_sphere(Point::new(base_pos), 0.05, 0.0); // Static
    segments.push(base_handle);

    for i in 1..=11 {
        let pos = Point::new([0.0, 1.2 + (i as f64 * 0.05), 0.0, 0.0]);
        let handle = world.add_sphere(pos, 0.02, segment_mass);
        field.register(handle, 100.0, 1.0);

        // Link with Hinge Joint (Spinal Disc)
        let motor = MotorDrive::new(0.0, 10.0).with_gains(150.0, 5.0);
        let joint = HingeJoint::<D>::with_anchors(
            segments[i - 1],
            handle,
            SVector::from([0.0, 0.025, 0.0, 0.0]),
            SVector::from([0.0, -0.025, 0.0, 0.0]),
            0,
            1, // XY plane rotation
        )
        .with_motor(motor);

        world.add_constraint(Box::new(joint));
        segments.push(handle);
    }

    let head_handle = *segments.last().unwrap();
    info!("🤖 FullSpine Chain Latched: 11 Segments + Head Cluster.");

    // 3. Initialize Safety Authority for all 11 joints
    let mut safety = JointSafetyAuthority::new(11);

    let mut current_time = 0.0;
    let mut step_count = 0;
    let mut max_head_deflection = 0.0;
    let mut resonance_detected = false;

    while current_time < SIM_DURATION_SECS {
        // 4. Oscillatory Force Injection (Head Bobbing)
        let phase = 2.0 * std::f64::consts::PI * FREQUENCY_HZ * current_time;
        let oscillation = phase.sin() * 5.0; // 5 Newton oscillating force
        let force_vector = SVector::from([oscillation, 0.0, 0.0, oscillation * 0.2]); // Includes W-axis vibrational bleed

        if let Some(head) = world.body_mut(head_handle) {
            head.apply_force(force_vector);

            // Measure Deflection
            let deflection = head.transform.translation[0].abs();
            if deflection > max_head_deflection {
                max_head_deflection = deflection;
            }
        }

        // 5. Physics Step with Consciousness Callback
        world.step_with_callback(DT, &mut field);
        field.tick_thermodynamics();

        // 6. Safety Monitoring: Harmonic Instability Detection
        let mut spine_surprises = Vec::new();
        for i in 1..=11 {
            let phi = field.phi(segments[i]);
            // Higher Phi indicates better information integration/stability.
            // Low Phi or high prediction error indicates resonance trauma.
            spine_surprises.push((1.0 - phi).max(0.0) * 10.0);
        }
        safety.update_from_surprise(&spine_surprises);

        if safety
            .joint_tiers
            .iter()
            .any(|t| *t != symtropy_consciousness_physics::safety::SafetyTier::Green)
        {
            resonance_detected = true;
        }

        // Metrics Gathering
        if step_count % 30 == 0 {
            info!(
                "[T={:.2}s] Head Deflection: {:.3}m | Avg Spine Phi: {:.3}",
                current_time,
                max_head_deflection,
                field.phi(head_handle)
            );
        }

        current_time += DT;
        step_count += 1;
    }

    info!("✨ STRUCTURAL RESONANCE BENCHMARK COMPLETE.");
    info!("📊 HARMONIC AUDIT:");
    info!("   🎯 Max Head Deflection: {:.4}m", max_head_deflection);
    info!("   🌊 Resonant Bleed detected: {}", resonance_detected);

    if max_head_deflection < 0.15 && !resonance_detected {
        info!("   ✅ FullSpine Stability: SUPERIOR (Optimal Damping).");
    } else if max_head_deflection < 0.3 {
        info!("   ⚠️  FullSpine Stability: NOMINAL (Moderate Oscillation).");
    } else {
        tracing::warn!("   🚨 FullSpine Stability: CRITICAL (Harmonic Instability!)");
    }

    Ok(())
}
