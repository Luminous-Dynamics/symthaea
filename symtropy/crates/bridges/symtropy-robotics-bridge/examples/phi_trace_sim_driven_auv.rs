// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sim-driven Φ trace on AUV (6DOF hydrodynamics under rotating current).

use std::io::Write;

use symthaea_auv::encoder::AuvHdcEncoder;
use symthaea_auv::simulator::{AuvPhysicsSimulator, SimpleAuvSimulator};
use symthaea_auv::types::AuvCommand;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
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
    let steps: usize = std::env::var("PTSDA_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let seed: u64 = std::env::var("PTSDA_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let csv_path = std::env::var("PTSDA_CSV").ok();

    println!();
    println!("════════════════════════════════════════════════════════════════════");
    println!(" Φ trace — SIM-DRIVEN (AUV via SimpleAuvSimulator)");
    println!("════════════════════════════════════════════════════════════════════");
    println!(" steps         : {steps}");
    println!(" seed          : {seed}");
    println!(" simulator     : SimpleAuvSimulator (6DOF hydrodynamics)");
    println!(" disturbance   : slow rotating current + bursty chemical plume");
    println!(" observation   : [pe, danger, depth_norm, current_norm]");
    println!();

    const DT: f64 = 0.002;

    let genesis = GenesisSeed::from_phrase(&format!("phi_trace_sim_auv_{seed}"));
    let mut simulator = SimpleAuvSimulator::new();
    let mut encoder = AuvHdcEncoder::new(&genesis, 32);
    let mut agent = RoboticAgent::new(BodyHandle(0), PlatformType::Auv, "phi_trace_sim_auv");

    let mut phi_samples = Vec::with_capacity(steps);
    let mut last_perception: Option<ContinuousHV> = None;
    let mut last_pe = 0.0_f64;

    let mut csv_file = csv_path.as_ref().and_then(|p| {
        let f = std::fs::File::create(p).ok()?;
        let mut w = std::io::BufWriter::new(f);
        writeln!(w, "step,phi,danger,depth,current_norm,chemical,pe").ok();
        Some(w)
    });

    for step in 0..steps {
        let s = step as f64;
        let phase = (seed as f64 % 1000.0) * 0.001;

        // Slow rotating current: magnitude ≈ 0.4 m/s, direction rotates at 0.02 Hz
        let current_angle = s * 0.018 + phase;
        let current_norm = (0.3 + 0.3 * current_angle.sin()).clamp(0.0, 1.0);
        let current_force = [
            current_norm * 3.0 * current_angle.cos(),
            current_norm * 3.0 * current_angle.sin(),
            0.0,
        ];
        simulator.apply_external_force(current_force);

        // Bursty chemical plume
        let chemical = if (s * 0.02 + phase).sin() > 0.7 {
            0.8
        } else {
            0.1
        };

        let state = simulator.state().clone();
        let depth_norm = (state.depth_m() / 25.0).clamp(0.0, 1.0);
        let danger = (current_norm * 0.5 + chemical * 0.3).min(1.0);
        let observation = [last_pe, danger, depth_norm, current_norm];

        let _gain = agent.tick(&observation, danger);
        let phi = agent.phi();
        phi_samples.push(phi);

        // Zero-thrust: AUV drifts under current. Adequate for Φ-distribution
        // characterization under representative disturbance.
        let cmd = AuvCommand::zero();
        simulator.step(&cmd, DT);

        let perception = encoder.encode(simulator.state());
        if let Some(prev) = last_perception.as_ref() {
            last_pe = (1.0 - perception.similarity(prev).max(0.0) as f64).clamp(0.0, 1.0);
        }
        last_perception = Some(perception);

        if let Some(w) = csv_file.as_mut() {
            writeln!(
                w,
                "{},{:.6},{:.4},{:.4},{:.4},{:.4},{:.4}",
                step,
                phi,
                danger,
                state.depth_m(),
                current_norm,
                chemical,
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

    let thresh_applied = 0.130;
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
    println!(" Applied SPRINT_THRESHOLD (AUV demo plugin): {thresh_applied:.3}");
    println!(
        " Φ > {thresh_applied:.3} fraction: {pct:.1} %  ({above} / {})",
        phi_samples.len()
    );
    println!();
    println!(" Hand-crafted AUV (PT_PLATFORM_OBS=1):");
    println!("   min=0.095  max=0.135  mean=0.126  p50=0.130  p95=0.131");
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
