// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sim-driven Φ trace on helicopter (SAR hover + Dryden wind).
//!
//! Part of the 5-platform extension of `phi_trace_sim_driven.rs`. Uses
//! `SimpleHelicopterSimulator` + `HelicopterCommand::hover()` + a
//! Dryden-ish wind-gust schedule, matching the flight-demo pattern
//! for the helicopter demo.
//!
//! The observation pack mirrors the helicopter-demo's consciousness_bridge:
//!
//!     [pe, danger, altitude_norm, wind_norm]
//!
//! Run:
//!     cargo run -p symtropy-robotics-bridge --example phi_trace_sim_driven_helicopter --release
//!
//! Env:
//!     PTSDH_STEPS=N  ticks (default 2000)
//!     PTSDH_SEED=N   RNG seed for wind phase (default 42)
//!     PTSDH_CSV=path dump per-step CSV

use std::io::Write;

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_helicopter::encoder::HelicopterHdcEncoder;
use symthaea_helicopter::simulator::{HelicopterPhysicsSimulator, SimpleHelicopterSimulator};
use symthaea_helicopter::types::HelicopterCommand;
use symtropy_physics::BodyHandle;
use symtropy_robotics_bridge::RoboticAgentTrait;
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
    let steps: usize = std::env::var("PTSDH_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let seed: u64 = std::env::var("PTSDH_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let csv_path = std::env::var("PTSDH_CSV").ok();

    println!();
    println!("════════════════════════════════════════════════════════════════════");
    println!(" Φ trace — SIM-DRIVEN (helicopter via SimpleHelicopterSimulator)");
    println!("════════════════════════════════════════════════════════════════════");
    println!(" steps         : {steps}");
    println!(" seed          : {seed}");
    println!(" simulator     : SimpleHelicopterSimulator (500 kg SAR class)");
    println!(" controller    : HelicopterCommand::hover (constant)");
    println!(" wind schedule : Dryden-ish bursty gusts (seed-phased)");
    println!(" observation   : [pe, danger, altitude_norm, wind_norm]");
    println!();

    const DT: f64 = 0.002;

    let genesis = GenesisSeed::from_phrase(&format!("phi_trace_sim_heli_{seed}"));
    let mut simulator = SimpleHelicopterSimulator::new();
    simulator.reset(20.0);
    let mut encoder = HelicopterHdcEncoder::new(&genesis, 32);
    let mut agent = RoboticAgent::new(
        BodyHandle(0),
        PlatformType::Helicopter,
        "phi_trace_sim_heli",
    );

    let mut phi_samples = Vec::with_capacity(steps);
    let mut last_perception: Option<ContinuousHV> = None;
    let mut last_pe = 0.0_f64;

    let mut csv_file = csv_path.as_ref().and_then(|p| {
        let f = std::fs::File::create(p).ok()?;
        let mut w = std::io::BufWriter::new(f);
        writeln!(w, "step,phi,danger,altitude,wind_norm,pe").ok();
        Some(w)
    });

    for step in 0..steps {
        let s = step as f64;
        let phase = (seed as f64 % 1000.0) * 0.001;

        // Dryden-ish bursty wind in [0.0, 0.9]
        let wind_norm = (0.25 + 0.5 * (s * 0.06 + phase).sin().powi(2)).clamp(0.0, 1.0);
        // Translate wind intensity to an external force vector (scales with mass).
        let gust_force = [
            wind_norm * 80.0 * (s * 0.11).sin(),
            wind_norm * 80.0 * (s * 0.11 + 1.2).cos(),
            0.0,
        ];
        simulator.apply_external_force(gust_force);

        let state = simulator.state().clone();
        let altitude_norm = (state.altitude() / 40.0).clamp(0.0, 1.0);
        let danger = (wind_norm + (1.0 - altitude_norm).max(0.0) * 0.3).min(1.0);
        let observation = [last_pe, danger, altitude_norm, wind_norm];

        let _gain = agent.tick(&observation, danger);
        let phi = agent.phi();
        phi_samples.push(phi);

        // Apply hover command to keep the sim stable.
        let cmd = HelicopterCommand::hover();
        simulator.step(&cmd, DT);

        let perception = encoder.encode(simulator.state());
        if let Some(prev) = last_perception.as_ref() {
            last_pe = (1.0 - perception.similarity(prev).max(0.0) as f64).clamp(0.0, 1.0);
        }
        last_perception = Some(perception);

        if let Some(w) = csv_file.as_mut() {
            writeln!(
                w,
                "{},{:.6},{:.4},{:.4},{:.4},{:.4}",
                step,
                phi,
                danger,
                state.altitude(),
                wind_norm,
                last_pe
            )
            .ok();
        }
    }

    let mut sorted = phi_samples.clone();
    sorted.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());
    let n = phi_samples.len() as f64;
    let mean = phi_samples.iter().sum::<f64>() / n;
    let var = phi_samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let std = var.sqrt();

    let thresh_applied = 0.110;
    let above = phi_samples.iter().filter(|&&x| x > thresh_applied).count();
    let pct = 100.0 * above as f64 / n;

    println!("────────── Φ distribution (SIM-DRIVEN) ──────────");
    println!(" n      = {}", phi_samples.len());
    println!(" min    = {:.4}", sorted[0]);
    println!(" max    = {:.4}", sorted[sorted.len() - 1]);
    println!(" mean   = {mean:.4}");
    println!(" std    = {std:.4}");
    println!(" p05    = {:.4}", percentile(&sorted, 0.05));
    println!(" p50    = {:.4}", percentile(&sorted, 0.50));
    println!(" p95    = {:.4}", percentile(&sorted, 0.95));
    println!();
    println!(" Applied SPRINT_THRESHOLD (helicopter demo plugin): {thresh_applied:.3}");
    println!(
        " Φ > {thresh_applied:.3} fraction: {pct:.1} %  ({above} / {})",
        phi_samples.len()
    );
    println!();
    println!(" Hand-crafted helicopter (PT_PLATFORM_OBS=1):");
    println!("   min=0.096  max=0.122  mean=0.109  p50=0.110  p95=0.119");
    println!(" Sim-driven (this run):");
    println!(
        "   min={:.4} max={:.4} mean={:.4} p50={:.4} p95={:.4}",
        sorted[0],
        sorted[sorted.len() - 1],
        mean,
        percentile(&sorted, 0.50),
        percentile(&sorted, 0.95)
    );

    if let Some(p) = csv_path.as_ref() {
        println!();
        println!(" CSV written to: {p}");
    }
}
