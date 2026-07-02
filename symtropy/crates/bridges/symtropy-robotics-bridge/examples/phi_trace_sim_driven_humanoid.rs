// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sim-driven Φ trace on humanoid (21-DOF DMC21) — **fall-and-recover scenario**.
//!
//! # Scope note
//!
//! The humanoid simulator is notoriously fragile — with zero torque or
//! untuned control, the robot falls from standing in ~2-3 seconds.
//! This example uses `HumanoidCommand::zero()` and lets the robot fall
//! under gravity, then continue laying on the ground. The resulting Φ
//! trace characterizes:
//!
//!   - First ~100 ticks: transient standing dynamics (Φ relatively high)
//!   - Middle ticks: fall event (prediction error spikes)
//!   - Later ticks: ground-pose with low prediction error (Φ saturates)
//!
//! This is the **high-risk** platform in the sim-driven-validation
//! extension. A proper measurement would use `BalanceController` to
//! hold the humanoid upright + apply graded push impulses, matching the
//! demo's scenario. That scope is deferred: the `BalanceController`
//! lives inside `symtropy-humanoid-demo/src/controller.rs` (Bevy-
//! embedded) and would need lifting out as standalone Rust. Similarly,
//! the BPTT-trained controller from `train_flight.rs` could be
//! re-derived for humanoid but requires humanoid-specific
//! `HumanoidTrainer` invocation (see `symthaea-humanoid/src/
//! training.rs`).
//!
//! For this session, we characterize the fall-scenario Φ distribution
//! as a lower-bound reference; compare to hand-crafted for template
//! validation; note the difference as expected (hand-crafted models
//! push-and-recover, sim-driven here models fall-and-rest).

use std::io::Write;

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_humanoid::encoder::HumanoidHdcEncoder;
use symthaea_humanoid::simulator::{HumanoidPhysicsSimulator, SimpleHumanoidSimulator};
use symthaea_humanoid::types::HumanoidCommand;
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
    let steps: usize = std::env::var("PTSDU_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let seed: u64 = std::env::var("PTSDU_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let csv_path = std::env::var("PTSDU_CSV").ok();

    println!();
    println!("════════════════════════════════════════════════════════════════════");
    println!(" Φ trace — SIM-DRIVEN (humanoid 21-DOF, fall-scenario)");
    println!("════════════════════════════════════════════════════════════════════");
    println!(" steps         : {steps}");
    println!(" seed          : {seed}");
    println!(" simulator     : SimpleHumanoidSimulator (DMC21)");
    println!(" controller    : HumanoidCommand::zero (robot falls under gravity)");
    println!(" disturbance   : none (gravity only)");
    println!(" observation   : [pe, danger, uprightness, push_norm=0]");
    println!();
    println!(" ⚠ Fall scenario — see doc-comment for scope. BalanceController-");
    println!("   driven measurement is deferred as follow-up work.");
    println!();

    const DT: f64 = 0.025; // 40 Hz, matching humanoid's default physics_dt

    let genesis = GenesisSeed::from_phrase(&format!("phi_trace_sim_human_{seed}"));
    let mut simulator = SimpleHumanoidSimulator::new();
    let mut encoder = HumanoidHdcEncoder::new(&genesis, 32);
    let mut agent = RoboticAgent::new(BodyHandle(0), PlatformType::Humanoid, "phi_trace_sim_human");

    let mut phi_samples = Vec::with_capacity(steps);
    let mut last_perception: Option<ContinuousHV> = None;
    let mut last_pe = 0.0_f64;

    let mut csv_file = csv_path.as_ref().and_then(|p| {
        let f = std::fs::File::create(p).ok()?;
        let mut w = std::io::BufWriter::new(f);
        writeln!(w, "step,phi,danger,uprightness,root_height,pe").ok();
        Some(w)
    });

    for step in 0..steps {
        let state = simulator.state().clone();
        let uprightness = state.uprightness().clamp(0.0, 1.0);
        let push_norm = 0.0_f64; // no perturbations in this scope
        // Danger: falling humanoid = high danger.
        let danger = ((1.0 - uprightness) * 0.65 + push_norm * 0.35).min(1.0);
        let observation = [last_pe, danger, uprightness, push_norm];

        let _gain = agent.tick(&observation, danger);
        let phi = agent.phi();
        phi_samples.push(phi);

        // Zero-torque step. Humanoid falls.
        let cmd = HumanoidCommand::zero();
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
                step, phi, danger, uprightness, state.root_height, last_pe
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

    println!("────────── Φ distribution (SIM-DRIVEN, fall-scenario) ──────────");
    println!(" n      = {}", phi_samples.len());
    println!(" min    = {:.4}", sorted[0]);
    println!(" max    = {:.4}", sorted[sorted.len() - 1]);
    println!(" mean   = {mean:.4}");
    println!(" std    = {std:.4}");
    println!(" p05    = {:.4}", percentile(&sorted, 0.05));
    println!(" p50    = {:.4}", percentile(&sorted, 0.50));
    println!(" p95    = {:.4}", percentile(&sorted, 0.95));
    println!();
    println!(" Applied SPRINT_THRESHOLD (humanoid demo plugin): {thresh_applied:.3}");
    println!(
        " Φ > {thresh_applied:.3} fraction: {pct:.1} %  ({above} / {})",
        phi_samples.len()
    );
    println!();
    println!(" Hand-crafted humanoid (PT_PLATFORM_OBS=1, push-and-recover):");
    println!("   min=0.098  max=0.131  mean=0.123  p50=0.130  p95=0.131");
    println!(" Sim-driven (this run, fall-scenario):");
    println!(
        "   min={:.4} max={:.4} mean={:.4} p50={:.4} p95={:.4}",
        sorted[0],
        sorted[sorted.len() - 1],
        mean,
        percentile(&sorted, 0.50),
        percentile(&sorted, 0.95)
    );
    println!();
    println!(" NOTE: Fall-scenario and push-and-recover scenario measure");
    println!(" different operating regimes. Direct comparability requires a");
    println!(" BalanceController-driven run — deferred to follow-up work.");

    if let Some(p) = csv_path.as_ref() {
        println!();
        println!(" CSV written to: {p}");
    }
}
