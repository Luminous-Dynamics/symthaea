// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sim-driven Φ trace on vehicle (bicycle + tire-slip + ice patches).

use std::io::Write;

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_vehicle::encoder::VehicleHdcEncoder;
use symthaea_vehicle::simulator::{BicycleModelSimulator, VehiclePhysicsSimulator};
use symthaea_vehicle::types::VehicleCommand;
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
    let steps: usize = std::env::var("PTSDV_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let seed: u64 = std::env::var("PTSDV_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let csv_path = std::env::var("PTSDV_CSV").ok();

    println!();
    println!("════════════════════════════════════════════════════════════════════");
    println!(" Φ trace — SIM-DRIVEN (vehicle via BicycleModelSimulator)");
    println!("════════════════════════════════════════════════════════════════════");
    println!(" steps         : {steps}");
    println!(" seed          : {seed}");
    println!(" simulator     : BicycleModelSimulator (bicycle + Pacejka tires)");
    println!(" controller    : constant throttle (no Stanley steering — simplified)");
    println!(" disturbance   : step-change friction (ice patches) + lateral gust");
    println!(" observation   : [pe, danger, speed_norm, slip_norm]");
    println!();

    const DT: f64 = 0.01; // 100 Hz physics for vehicle

    let genesis = GenesisSeed::from_phrase(&format!("phi_trace_sim_veh_{seed}"));
    let mut simulator = BicycleModelSimulator::new();
    let mut encoder = VehicleHdcEncoder::new(&genesis, 32);
    let mut agent = RoboticAgent::new(BodyHandle(0), PlatformType::Vehicle, "phi_trace_sim_veh");

    let mut phi_samples = Vec::with_capacity(steps);
    let mut last_perception: Option<ContinuousHV> = None;
    let mut last_pe = 0.0_f64;

    let mut csv_file = csv_path.as_ref().and_then(|p| {
        let f = std::fs::File::create(p).ok()?;
        let mut w = std::io::BufWriter::new(f);
        writeln!(w, "step,phi,danger,speed,slip,friction,pe").ok();
        Some(w)
    });

    for step in 0..steps {
        let s = step as f64;
        let phase = (seed as f64 % 1000.0) * 0.001;

        // Ice-patch schedule: step-changes in friction scale.
        let friction = if (s * 0.013 + phase).sin() > 0.3 {
            1.0
        } else {
            0.35
        };
        simulator.set_friction_scale(friction);

        // Mild lateral gust on low-friction segments.
        if friction < 0.5 {
            simulator.apply_external_force([0.0, 200.0 * (s * 0.18).sin()]);
        } else {
            simulator.apply_external_force([0.0, 0.0]);
        }

        let state = simulator.state().clone();
        let speed_norm = (state.speed / 15.0).clamp(0.0, 1.0);
        let slip_norm =
            ((state.tire_slip_front.abs() + state.tire_slip_rear.abs()) / 0.52).clamp(0.0, 1.0);
        let danger = ((1.0_f64 - friction) * 0.8 + slip_norm * 0.2).min(1.0);
        let observation = [last_pe, danger, speed_norm, slip_norm];

        let _gain = agent.tick(&observation, danger);
        let phi = agent.phi();
        phi_samples.push(phi);

        // Constant-throttle command: enough to keep speed > 0 without
        // needing full waypoint tracking.
        let cmd = VehicleCommand {
            steering: 0.0,
            throttle: 0.35,
            brake: 0.0,
        };
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
                step, phi, danger, state.speed, slip_norm, friction, last_pe
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

    let thresh_applied = 0.101;
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
    println!(" Applied SPRINT_THRESHOLD (vehicle demo plugin): {thresh_applied:.3}");
    println!(
        " Φ > {thresh_applied:.3} fraction: {pct:.1} %  ({above} / {})",
        phi_samples.len()
    );
    println!();
    println!(" Hand-crafted vehicle (PT_PLATFORM_OBS=1):");
    println!("   min=0.086  max=0.132  mean=0.112  p50=0.101  p95=0.132");
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
