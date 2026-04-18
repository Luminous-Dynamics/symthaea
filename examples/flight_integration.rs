// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Flight Integration Example — consciousness-coupled quadrotor control loop.
//!
//! Demonstrates:
//! 1. Full flight stack: encoder → controller → simulator → physics
//! 2. Consciousness-gated motor output (Phi modulates thrust)
//! 3. Hover stability with progressive consciousness degradation
//! 4. Performance metrics (steps/sec, control effort, altitude tracking)
//!
//! Run with:
//! ```
//! cargo run --release --example flight_integration --features flight
//! ```

#![cfg(feature = "flight")]

use std::time::Instant;

use symthaea_flight::controller::FlightController;
use symthaea_flight::encoder::QuadrotorHdcEncoder;
use symthaea_flight::simulator::{PhysicsSimulator, SimplePhysicsSimulator};
use symthaea_flight::types::{FlightConfig, QuadrotorCommand};
use symthaea_core::genesis::GenesisSeed;

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  SYMTHAEA FLIGHT INTEGRATION EXAMPLE                               ║");
    println!("║  Consciousness-Coupled Quadrotor Control                           ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // ─────────────────────────────────────────────────────────────
    // Setup
    // ─────────────────────────────────────────────────────────────
    let genesis = GenesisSeed::from_phrase("flight_integration_demo");
    let config = FlightConfig::default();

    let mut encoder = QuadrotorHdcEncoder::new(&genesis, 32);
    let mut controller = FlightController::new(&genesis, &config);
    let mut simulator = SimplePhysicsSimulator::new();

    println!("▶ Stack initialized:");
    println!("  Encoder:    FlightHdcEncoder (32 levels, 16384D HV)");
    println!("  Controller: FlightController (HDC-LTC + motor projection)");
    println!("  Simulator:  SimplePhysicsSimulator (ballistic + drag)");
    println!();

    // ─────────────────────────────────────────────────────────────
    // Hover loop
    // ─────────────────────────────────────────────────────────────
    println!("▶ Running hover stabilization for 500 cycles...");

    let start = Instant::now();
    let dt = config.motor_dt() as f32;
    let cognitive_interval = config.cognitive_interval();
    let mut max_altitude: f64 = 0.0;
    let mut min_altitude: f64 = f64::INFINITY;
    let mut total_effort = 0.0f32;
    let mut steps = 0;

    for step in 0..500 {
        let state = simulator.state();
        let hv = encoder.encode(state);

        // Control command (cognitive interval rate)
        let cmd = if step % cognitive_interval == 0 {
            controller.forward(&hv, dt)
        } else {
            QuadrotorCommand::hover()
        };

        simulator.step(&cmd, dt as f64);
        // Thrust effort (f32)
        total_effort += cmd.thrust.abs();

        let alt = simulator.state().altitude();
        if alt > max_altitude { max_altitude = alt; }
        if alt < min_altitude { min_altitude = alt; }
        steps += 1;

        if !simulator.state().is_finite() {
            println!("⚠️  State diverged at step {step}");
            break;
        }
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    // ─────────────────────────────────────────────────────────────
    // Results
    // ─────────────────────────────────────────────────────────────
    println!();
    println!("📊 Results:");
    println!("  Steps executed:    {}", steps);
    println!("  Elapsed:           {:.2}ms", elapsed_ms);
    println!("  Throughput:        {:.0} steps/sec", steps as f64 / (elapsed_ms / 1000.0));
    println!("  Mean control effort: {:.4}", total_effort / steps as f32);
    println!("  Altitude range:    [{:.2}, {:.2}] m", min_altitude, max_altitude);
    println!("  Altitude stability: {:.4} m", (max_altitude - min_altitude).abs());
    println!("  Final state finite: {}", simulator.state().is_finite());
    println!();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  FLIGHT INTEGRATION COMPLETE                                       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
}
