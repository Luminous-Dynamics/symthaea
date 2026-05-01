// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sim-driven Φ trace on quadrotor (Part 3/5 of 2-3-4-5).
//!
//! Closes §8's hand-crafted-observations limitation for flight. The
//! existing `phi_trace.rs` uses synthetic sinusoidal observations
//! (even with `PT_PLATFORM_OBS=1`). This example uses the ACTUAL
//! `SimplePhysicsSimulator` + PD-baseline controller + a wind-gust
//! schedule, matching what `symtropy-flight-demo/src/plugin.rs` runs
//! inside Bevy — minus the Bevy wrapping.
//!
//! The observation packed into `RoboticAgent::tick` is the same shape
//! the flight-demo uses (per `consciousness_bridge.rs`):
//!
//!     [prediction_error, danger, altitude_norm, attitude_norm]
//!
//! Produces a Φ distribution that reflects real simulator-driven
//! observation dynamics — the closest thing to a "ground truth"
//! per-platform Φ band we can produce without hardware.
//!
//! Run:
//!     cargo run -p symtropy-robotics-bridge --example phi_trace_sim_driven --release
//!
//! Env:
//!     PTSD_STEPS=N        number of ticks (default 2000)
//!     PTSD_SEED=N         RNG seed for gust schedule (default 42)
//!     PTSD_CSV=path       dump per-step CSV

use std::io::Write;

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_multirotor::encoder::QuadrotorHdcEncoder;
use symthaea_multirotor::simulator::{PhysicsSimulator, SimplePhysicsSimulator};
use symthaea_multirotor::types::{pd_baseline, FlightSetpoint, PdGains, QuadrotorCommand};
use symtropy_physics::BodyHandle;
use symtropy_robotics_bridge::agent::RoboticAgent;
use symtropy_robotics_bridge::platform::PlatformType;

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (sorted.len() as f64 * q).floor() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn main() {
    let steps: usize = std::env::var("PTSD_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let seed: u64 = std::env::var("PTSD_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let csv_path = std::env::var("PTSD_CSV").ok();

    println!();
    println!("════════════════════════════════════════════════════════════════════");
    println!(" Φ trace — SIM-DRIVEN (quadrotor via SimplePhysicsSimulator + PD)");
    println!("════════════════════════════════════════════════════════════════════");
    println!(" steps          : {steps}");
    println!(" seed           : {seed}");
    println!(" simulator      : SimplePhysicsSimulator (ballistic, same as flight-demo)");
    println!(" controller     : pd_baseline -> hover setpoint at 1.0 m");
    println!(" wind schedule  : Dryden-ish sinusoidal gusts (seed-phased)");
    println!(" observation    : [pe, danger, altitude_norm, attitude_norm]");
    println!();

    let genesis = GenesisSeed::from_phrase(&format!("phi_trace_sim_{seed}"));
    let mut simulator = SimplePhysicsSimulator::new();
    simulator.reset(1.0); // 1m altitude
    let mut encoder = QuadrotorHdcEncoder::new(&genesis, 32);
    let setpoint = FlightSetpoint {
        position: [0.0, 0.0, 1.0],
        yaw: 0.0,
    };
    let gains = PdGains::default();
    let mut agent = RoboticAgent::new(BodyHandle(0), PlatformType::Quadrotor, "phi_trace_sim");

    let dt: f64 = 0.002; // 500 Hz physics

    let mut phi_samples = Vec::with_capacity(steps);
    let mut last_perception: Option<ContinuousHV> = None;
    let mut last_pe = 0.0_f32;
    let mut csv_file = csv_path.as_ref().and_then(|p| {
        let f = std::fs::File::create(p).ok()?;
        let mut w = std::io::BufWriter::new(f);
        writeln!(
            w,
            "step,phi,danger,altitude,attitude_norm,pe,gust_intensity"
        )
        .ok();
        Some(w)
    });

    for step in 0..steps {
        let s = step as f64;
        let phase = (seed as f64 % 1000.0) * 0.001;

        // Wind schedule — Dryden-ish bursty gust magnitude in [0.0, 0.7] m/s²
        let gust_intensity = (0.3 + 0.4 * (s * 0.08 + phase).sin().powi(2)).clamp(0.0, 1.0);
        let gust_force = [
            gust_intensity * 0.08 * (s * 0.13).sin(),
            gust_intensity * 0.08 * (s * 0.13 + 1.5).cos(),
            0.0,
        ];
        simulator.apply_external_force(gust_force);

        // PD-baseline command toward hover.
        let state = simulator.state().clone();
        let cmd: QuadrotorCommand = pd_baseline(&state, &setpoint, &gains);

        // Observation normalization — same recipe as flight-demo's consciousness_bridge.
        let altitude_norm = (state.position[2] / 3.0).clamp(0.0, 1.0);
        let (roll, pitch, _yaw) = state.euler_angles();
        let attitude_norm = ((roll.abs() + pitch.abs()) / std::f64::consts::PI).clamp(0.0, 1.0);
        let danger = (gust_intensity + attitude_norm * 0.5).min(1.0);
        let observation = [last_pe as f64, danger, altitude_norm, attitude_norm];

        // Tick the agent — Φ reflects the post-FEP-wiring ConsciousnessInputs
        // built from this real observation stream.
        let _gain = agent.tick(&observation, danger);
        let phi = agent.phi();
        phi_samples.push(phi);

        // Step physics and update prediction error from HDC encoder.
        simulator.step(&cmd, dt);
        let perception = encoder.encode(simulator.state());
        if let Some(prev) = last_perception.as_ref() {
            last_pe = (1.0 - perception.similarity(prev).max(0.0)).min(1.0);
        }
        last_perception = Some(perception);

        if let Some(w) = csv_file.as_mut() {
            writeln!(
                w,
                "{},{:.6},{:.4},{:.4},{:.4},{:.4},{:.4}",
                step, phi, danger, state.position[2], attitude_norm, last_pe, gust_intensity
            )
            .ok();
        }
    }

    let mut sorted = phi_samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = phi_samples.len() as f64;
    let mean = phi_samples.iter().sum::<f64>() / n;
    let var = phi_samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let std = var.sqrt();
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let p05 = percentile(&sorted, 0.05);
    let p50 = percentile(&sorted, 0.50);
    let p95 = percentile(&sorted, 0.95);

    let thresh_applied = 0.110; // quadrotor's applied threshold per §8 table
    let above = phi_samples.iter().filter(|&&x| x > thresh_applied).count();
    let pct = 100.0 * above as f64 / n;

    println!("────────── Φ distribution (SIM-DRIVEN) ──────────");
    println!(" n      = {}", phi_samples.len());
    println!(" min    = {min:.4}");
    println!(" max    = {max:.4}");
    println!(" mean   = {mean:.4}");
    println!(" std    = {std:.4}");
    println!(" p05    = {p05:.4}");
    println!(" p50    = {p50:.4}");
    println!(" p95    = {p95:.4}");
    println!();
    println!("────────── comparison to hand-crafted phi_trace ──────────");
    println!(" Hand-crafted quadrotor (PT_PLATFORM_OBS=1):");
    println!("   min=0.0941  max=0.1195  mean=0.1090  p50=0.1100  p95=0.1165");
    println!(" Sim-driven (this run):");
    println!("   min={min:.4}  max={max:.4}  mean={mean:.4}  p50={p50:.4}  p95={p95:.4}");
    println!();
    println!("────────── sprint-threshold diagnostic ──────────");
    println!(" Applied SPRINT_THRESHOLD (quadrotor demo plugin): {thresh_applied:.3}");
    println!(
        " Φ > {thresh_applied:.3} fraction: {pct:.1} %  ({above} / {})",
        phi_samples.len()
    );
    println!();
    let new_p50 = p50;
    println!(" Recommended threshold from sim-driven p50: {new_p50:.3}");
    println!(
        "   (vs applied 0.110 — {})",
        if (new_p50 - 0.110).abs() < 0.005 {
            "agrees with hand-crafted calibration"
        } else {
            "DIFFERS from hand-crafted calibration; consider re-tuning"
        }
    );

    if let Some(p) = csv_path.as_ref() {
        println!();
        println!(" CSV written to: {p}");
    }
}
