// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! # Manipulator Quantitative Benchmark
//!
//! 100 pick-place cycles with deterministic sinusoidal human trajectory.
//! Compares Adaptive Safety Gradient vs ISO/TS 15066 SSM throughput.
//!
//! ```bash
//! cd symtropy/crates/symtropy-manipulator-demo
//! cargo run --example manipulator_benchmark --release
//! ```

use std::time::Instant;
use symthaea_core::genesis::GenesisSeed;
use symthaea_manipulator::kinematics::ManipulatorKinematics;
use symthaea_manipulator::simulator::{ManipulatorPhysicsSimulator, SimpleManipulatorSimulator};
use symthaea_manipulator::types::NUM_JOINTS;
use symthaea_manipulator::encoder::ManipulatorHdcEncoder;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Manipulator Quantitative Benchmark                         ║");
    println!("║  Adaptive Safety vs ISO/TS 15066 SSM — 100 cycles           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let dt = 0.002; // 500Hz physics
    let total_steps = 50_000; // 100 seconds at 500Hz

    // Deterministic sinusoidal human trajectory
    let human_approach_period = 8.0; // seconds per approach/retreat cycle
    let human_closest = 0.4; // meters (enters workspace)
    let human_farthest = 2.0; // meters (outside workspace)

    // Pick/place targets
    let pick = [0.4, -0.3, 0.15];
    let place = [0.4, 0.3, 0.15];
    let approach_h = 0.30;

    let kinematics = ManipulatorKinematics::default_7dof();

    // ─── Adaptive arm (consciousness-gated) ───
    println!("━━━ Running Adaptive Safety arm (100s simulation) ━━━");
    let genesis = GenesisSeed::from_phrase("manipulator-benchmark");
    let mut encoder = ManipulatorHdcEncoder::new(&genesis, 32);
    let mut adaptive_sim = SimpleManipulatorSimulator::new();
    let mut adaptive_cycles = 0u32;
    let mut adaptive_phase = 0; // 0=approach pick, 1=transit, 2=approach place, 3=return
    let mut adaptive_target = [pick[0], pick[1], approach_h];

    let start = Instant::now();
    for step in 0..total_steps {
        let t = step as f64 * dt;

        // Human distance (sinusoidal)
        let human_dist = human_farthest
            - (human_farthest - human_closest)
                * ((t / human_approach_period * std::f64::consts::TAU).sin() * 0.5 + 0.5);

        // Adaptive safety: continuous gain based on human distance
        let gain = if human_dist > 1.2 {
            1.0
        } else if human_dist > 0.8 {
            0.6 + 0.4 * (human_dist - 0.8) / 0.4
        } else if human_dist > 0.5 {
            0.3 + 0.3 * (human_dist - 0.5) / 0.3
        } else {
            0.1
        };

        // IK toward current target
        let state = adaptive_sim.state();
        if let Some(q_target) = kinematics.ik_dls(
            &adaptive_target, &state.joint_angles, 0.1, 30, 0.01,
        ) {
            let mut cmd = symthaea_manipulator::ManipulatorCommand::zero();
            for i in 0..NUM_JOINTS {
                let err = q_target[i] - state.joint_angles[i];
                let vel = state.joint_velocities[i];
                cmd.joint_torques[i] = (gain as f32 * (8.0 * err - 2.0 * vel) as f32).clamp(-1.0, 1.0);
            }
            adaptive_sim.step(&cmd, dt);
        }

        // Check waypoint
        let ee = adaptive_sim.state().end_effector_position;
        let dist = ((ee[0]-adaptive_target[0]).powi(2) + (ee[1]-adaptive_target[1]).powi(2) + (ee[2]-adaptive_target[2]).powi(2)).sqrt();
        if dist < 0.02 {
            adaptive_phase = (adaptive_phase + 1) % 4;
            adaptive_target = match adaptive_phase {
                0 => [pick[0], pick[1], approach_h],
                1 => [place[0], place[1], approach_h],
                2 => [place[0], place[1], approach_h],
                _ => [pick[0], pick[1], approach_h],
            };
            if adaptive_phase == 0 {
                adaptive_cycles += 1;
            }
        }
    }
    let adaptive_time = start.elapsed();

    // ─── ISO arm (binary stop/go) ───
    println!("━━━ Running ISO/TS 15066 SSM arm (100s simulation) ━━━");
    let mut iso_sim = SimpleManipulatorSimulator::new();
    let mut iso_cycles = 0u32;
    let mut iso_phase = 0;
    let mut iso_target = [pick[0], pick[1], approach_h];
    let iso_sp = 1.0; // Protective distance (conservative)

    let start = Instant::now();
    for step in 0..total_steps {
        let t = step as f64 * dt;

        let human_dist = human_farthest
            - (human_farthest - human_closest)
                * ((t / human_approach_period * std::f64::consts::TAU).sin() * 0.5 + 0.5);

        // Binary: full speed or full stop
        let gain = if human_dist > iso_sp { 1.0 } else { 0.0 };

        let state = iso_sim.state();
        if gain > 0.0 {
            if let Some(q_target) = kinematics.ik_dls(
                &iso_target, &state.joint_angles, 0.1, 30, 0.01,
            ) {
                let mut cmd = symthaea_manipulator::ManipulatorCommand::zero();
                for i in 0..NUM_JOINTS {
                    let err = q_target[i] - state.joint_angles[i];
                    let vel = state.joint_velocities[i];
                    cmd.joint_torques[i] = ((8.0 * err - 2.0 * vel) as f32).clamp(-1.0, 1.0);
                }
                iso_sim.step(&cmd, dt);
            }
        }

        let ee = iso_sim.state().end_effector_position;
        let dist = ((ee[0]-iso_target[0]).powi(2) + (ee[1]-iso_target[1]).powi(2) + (ee[2]-iso_target[2]).powi(2)).sqrt();
        if dist < 0.02 {
            iso_phase = (iso_phase + 1) % 4;
            iso_target = match iso_phase {
                0 => [pick[0], pick[1], approach_h],
                1 => [place[0], place[1], approach_h],
                2 => [place[0], place[1], approach_h],
                _ => [pick[0], pick[1], approach_h],
            };
            if iso_phase == 0 {
                iso_cycles += 1;
            }
        }
    }
    let iso_time = start.elapsed();

    // ─── Results ───
    println!();
    println!("━━━ Results (100s simulated, sinusoidal human approach) ━━━");
    println!("  Adaptive Safety:  {} cycles in {:.2}s wall time", adaptive_cycles, adaptive_time.as_secs_f64());
    println!("  ISO/TS 15066 SSM: {} cycles in {:.2}s wall time", iso_cycles, iso_time.as_secs_f64());

    if iso_cycles > 0 {
        let advantage = (adaptive_cycles as f64 - iso_cycles as f64) / iso_cycles as f64 * 100.0;
        println!();
        if advantage > 0.0 {
            println!("  *** THROUGHPUT ADVANTAGE: +{:.1}% (Adaptive: {} vs ISO: {} cycles) ***",
                advantage, adaptive_cycles, iso_cycles);
        } else {
            println!("  Throughput: ISO leads by {:.1}%", -advantage);
        }
    }

    println!();
    println!("━━━ CSV ━━━");
    println!("metric,adaptive,iso");
    println!("cycles,{},{}", adaptive_cycles, iso_cycles);
    println!("wall_time_s,{:.3},{:.3}", adaptive_time.as_secs_f64(), iso_time.as_secs_f64());
    println!("cycles_per_100s,{},{}", adaptive_cycles, iso_cycles);
}
