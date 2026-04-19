// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sim-driven Φ trace on manipulator — **idle-scenario characterization**.
//!
//! Part of the 5-platform extension of `phi_trace_sim_driven.rs`. Uses
//! `SimpleManipulatorSimulator` with zero commanded torque (arm rests
//! under gravity) + sinusoidal human obstacle. Observation pack matches
//! the benchmark's pattern:
//!
//!     [pe, danger, human_norm, effort_norm]
//!
//! # Scope note
//!
//! This example measures the **idle** scenario — manipulator at rest
//! while a human approaches. Prediction error stays low (no commanded
//! motion to mispredict), effort_norm stays ≈ 0, so Φ sits at the
//! **top** of its band (p50 ≈ 0.131).
//!
//! The canonical **active-cycle** sim-driven trace is produced by
//! `MANIP_BENCH_PHI_TRACE=1 cargo run -p symtropy-manipulator-demo
//! --example manipulator_benchmark --release`, data in
//! `data/phi_trace_40s.csv`, with p50 ≈ 0.121 under the full pick-
//! place + human-obstacle scenario.
//!
//! Both measurements are valid characterizations of different operating
//! regimes; the §8 calibration recommendation uses the active-cycle
//! number because production cobots are typically in active cycling
//! when the supervisor matters most.
//!
//! Run:
//!     cargo run -p symtropy-robotics-bridge --example phi_trace_sim_driven_manipulator --release
//!
//! Env:
//!     PTSDM_STEPS=N  ticks (default 2000)
//!     PTSDM_SEED=N   RNG seed for human-approach phase (default 42)
//!     PTSDM_CSV=path dump per-step CSV

use std::io::Write;

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_manipulator::encoder::ManipulatorHdcEncoder;
use symthaea_manipulator::kinematics::ManipulatorKinematics;
use symthaea_manipulator::simulator::{ManipulatorPhysicsSimulator, SimpleManipulatorSimulator};
use symthaea_manipulator::types::ManipulatorCommand;
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
    let steps: usize = std::env::var("PTSDM_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let seed: u64 = std::env::var("PTSDM_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let csv_path = std::env::var("PTSDM_CSV").ok();

    println!();
    println!("════════════════════════════════════════════════════════════════════");
    println!(" Φ trace — SIM-DRIVEN (manipulator via SimpleManipulatorSimulator)");
    println!("════════════════════════════════════════════════════════════════════");
    println!(" steps         : {steps}");
    println!(" seed          : {seed}");
    println!(" simulator     : SimpleManipulatorSimulator (7-DOF DH + DLS IK)");
    println!(" disturbance   : sinusoidal human approach 0.25 → 2.5 m");
    println!(" observation   : [pe, danger, human_norm, effort_norm]");
    println!();

    const DT: f64 = 0.002; // 500 Hz physics
    const COG_INTERVAL: usize = 20; // cognitive tick every 20 physics steps = 25 Hz

    let _kinematics = ManipulatorKinematics::default_7dof();
    let mut sim = SimpleManipulatorSimulator::new();
    let genesis = GenesisSeed::from_phrase(&format!("phi_trace_sim_manip_{seed}"));
    let mut encoder = ManipulatorHdcEncoder::new(&genesis, 32);
    let mut agent = RoboticAgent::new(
        BodyHandle(0),
        PlatformType::Manipulator,
        "phi_trace_sim_manip",
    );

    let mut phi_samples: Vec<f64> = Vec::with_capacity(steps / COG_INTERVAL + 1);
    let mut last_perception: Option<ContinuousHV> = None;

    let mut csv_file = csv_path.as_ref().and_then(|p| {
        let f = std::fs::File::create(p).ok()?;
        let mut w = std::io::BufWriter::new(f);
        writeln!(w, "step,phi,danger,human_dist,pe,effort_norm").ok();
        Some(w)
    });

    // Deterministic seed-dependent phase so each invocation varies.
    let phase_offset = (seed as f64 % 1000.0) * 0.001;

    for step in 0..steps {
        let t = step as f64 * DT;

        // Sinusoidal human obstacle — mirrors the benchmark's trial params.
        // Period 8s, min 0.3m, max 2.7m.
        let human_dist = 1.5 + 1.2 * ((t * std::f64::consts::TAU / 8.0) + phase_offset).sin();

        if step % COG_INTERVAL == 0 {
            let state = sim.state();
            let hv = encoder.encode(state);
            let pe = match &last_perception {
                Some(prev) => (1.0 - hv.similarity(prev).max(0.0) as f64).clamp(0.0, 1.0),
                None => 0.0,
            };
            let danger = ((2.0 - human_dist) / 2.0).clamp(0.0, 1.0);
            let human_norm = (1.0 / (1.0 + human_dist)).clamp(0.0, 1.0);
            let effort_norm = (state
                .joint_velocities
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt()
                / 5.0)
                .clamp(0.0, 1.0);
            let observation = [pe, danger, human_norm, effort_norm];

            let _gain = agent.tick(&observation, danger);
            let phi = agent.phi();
            phi_samples.push(phi);
            last_perception = Some(hv);

            if let Some(w) = csv_file.as_mut() {
                writeln!(
                    w,
                    "{},{:.6},{:.4},{:.4},{:.4},{:.4}",
                    step, phi, danger, human_dist, pe, effort_norm
                )
                .ok();
            }
        }

        // Zero-torque step; the sim drifts under gravity, which is fine
        // for a Φ-distribution measurement (we're not trying to complete
        // a cycle, just characterize the signal under representative
        // observations).
        let cmd = ManipulatorCommand::zero();
        sim.step(&cmd, DT);
    }

    let mut sorted = phi_samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = phi_samples.len() as f64;
    let mean = phi_samples.iter().sum::<f64>() / n;
    let var = phi_samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let std = var.sqrt();

    let thresh_applied = 0.125; // manipulator-benchmark's paper anchor
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
    println!(" Applied SPRINT_THRESHOLD (manipulator-benchmark anchor): {thresh_applied:.3}");
    println!(
        " Φ > {thresh_applied:.3} fraction: {pct:.1} %  ({above} / {})",
        phi_samples.len()
    );
    println!();
    println!(" Hand-crafted manipulator (PT_PLATFORM_OBS=1):");
    println!("   min=0.090  max=0.130  mean=0.113  p50=0.114  p95=0.129");
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
