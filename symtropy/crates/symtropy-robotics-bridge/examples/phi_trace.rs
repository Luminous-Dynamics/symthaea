// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Empirical signal-band measurement via `RoboticAgent::tick()`.
//!
//! Runs a `RoboticAgent` on a specified `PlatformType` for N ticks with
//! synthetic but representative observations, logging the consciousness-
//! inspired correlate (`phi`, the scalar output of
//! `MasterConsciousnessEquation`) at each tick. Used to answer the
//! question: "is the manipulator's measured Φ band [0.099, 0.145]
//! actually what the other five sprint_floor_gain adopters also
//! experience, or are we using a miscalibrated threshold because we
//! never measured?"
//!
//! This is the concrete gap-closing experiment for the
//! Φ-gated-safety paper's §8 "unverified transferability assumption".
//!
//! Usage:
//!
//!     cargo run -p symtropy-robotics-bridge --example phi_trace --release
//!
//! Env:
//!     PT_PLATFORM=quadrotor|vehicle|humanoid|manipulator|auv|helicopter
//!                             (default: quadrotor)
//!     PT_STEPS=N              number of ticks (default: 1000)
//!     PT_CSV=path             dump per-step CSV
//!     PT_SEED=N               RNG seed for observation synthesis (default: 42)

use std::io::Write;

use symtropy_physics::BodyHandle;
use symtropy_robotics_bridge::agent::RoboticAgent;
use symtropy_robotics_bridge::platform::PlatformType;

#[derive(Debug, Clone, Copy)]
struct Stats {
    n: usize,
    min: f64,
    max: f64,
    mean: f64,
    std: f64,
    p05: f64,
    p50: f64,
    p95: f64,
}

fn compute_stats(samples: &[f64]) -> Stats {
    let n = samples.len();
    if n == 0 {
        return Stats {
            n: 0,
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std: 0.0,
            p05: 0.0,
            p50: 0.0,
            p95: 0.0,
        };
    }
    let mut sorted: Vec<f64> = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = samples.iter().sum::<f64>() / n as f64;
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
    let p = |q: f64| -> f64 {
        let idx = ((n as f64) * q).floor() as usize;
        sorted[idx.min(n - 1)]
    };
    Stats {
        n,
        min: sorted[0],
        max: sorted[n - 1],
        mean,
        std: var.sqrt(),
        p05: p(0.05),
        p50: p(0.50),
        p95: p(0.95),
    }
}

fn parse_platform(s: &str) -> Option<PlatformType> {
    match s.trim().to_lowercase().as_str() {
        "quadrotor" | "flight" => Some(PlatformType::Quadrotor),
        "vehicle" => Some(PlatformType::Vehicle),
        "humanoid" => Some(PlatformType::Humanoid),
        "manipulator" => Some(PlatformType::Manipulator),
        "auv" => Some(PlatformType::Auv),
        "helicopter" => Some(PlatformType::Helicopter),
        _ => None,
    }
}

fn observation_dim(platform: PlatformType) -> usize {
    // Conservative default. `RoboticAgent::tick` accepts &[f64]; the
    // consciousness equation hashes over whatever length is provided.
    // These lengths mirror each platform demo's typical observation pack.
    match platform {
        PlatformType::Quadrotor => 4, // altitude, attitude-x, attitude-y, wind-gust
        PlatformType::Vehicle => 3,   // speed, slip, friction
        PlatformType::Humanoid => 2,  // uprightness, push
        PlatformType::Manipulator => 4, // danger, PE, effort, stiffness
        PlatformType::Auv => 4,       // depth, current, chemical, PE
        PlatformType::Helicopter => 4, // altitude, wind, attitude, PE
        _ => 4,
    }
}

/// Deterministic synthetic observation stream that covers a realistic
/// range of danger/stability — NOT a re-run of the per-platform demo's
/// scenario (that would require each demo's physics sim). We trade
/// scenario-accuracy for portability: this is enough to surface whether
/// the consciousness-equation aggregation is producing a narrow band in
/// [0.099, 0.145] for all platforms or a platform-specific distribution.
fn synth_observation(dim: usize, step: usize, seed: u64) -> (Vec<f64>, f64) {
    let mut obs = vec![0.0; dim];
    let s = step as f64;
    for (i, v) in obs.iter_mut().enumerate() {
        let phase = (seed as f64 + i as f64 * 0.7) * 0.1;
        *v = 0.5 + 0.45 * ((s * 0.03 + phase).sin() * 0.5 + (s * 0.17 + phase).cos() * 0.5);
        *v = v.clamp(0.0, 1.0);
    }
    // Danger: a second phased sinusoid, uncorrelated with observation.
    let danger_phase = seed as f64 * 0.31;
    let danger = (0.3 + 0.4 * (s * 0.05 + danger_phase).sin()).clamp(0.0, 1.0);
    (obs, danger)
}

fn main() {
    let platform_str = std::env::var("PT_PLATFORM").unwrap_or_else(|_| "quadrotor".into());
    let platform = parse_platform(&platform_str).unwrap_or_else(|| {
        eprintln!("unknown PT_PLATFORM={platform_str}, using quadrotor");
        PlatformType::Quadrotor
    });
    let steps: usize = std::env::var("PT_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    let seed: u64 = std::env::var("PT_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let csv_path = std::env::var("PT_CSV").ok();
    let dim = observation_dim(platform);

    println!();
    println!("════════════════════════════════════════════════════════════════════");
    println!(" Φ trace — empirical signal-band measurement via RoboticAgent::tick");
    println!("════════════════════════════════════════════════════════════════════");
    println!(" platform         : {:?}", platform);
    println!(" observation dim  : {}", dim);
    println!(" steps            : {}", steps);
    println!(" seed             : {}", seed);
    println!();

    let mut agent = RoboticAgent::new(BodyHandle(0), platform, "phi_trace");

    let mut phi_samples = Vec::with_capacity(steps);
    let mut csv_file = csv_path.as_ref().and_then(|p| {
        let f = std::fs::File::create(p).ok()?;
        let mut w = std::io::BufWriter::new(f);
        writeln!(w, "step,phi,danger").ok();
        Some(w)
    });

    for step in 0..steps {
        let (obs, danger) = synth_observation(dim, step, seed);
        let _gain = agent.tick(&obs, danger);
        let phi = agent.phi();
        phi_samples.push(phi);
        if let Some(w) = csv_file.as_mut() {
            writeln!(w, "{},{:.6},{:.4}", step, phi, danger).ok();
        }
    }

    let s = compute_stats(&phi_samples);
    println!("────────── Φ distribution ──────────");
    println!(" n      = {}", s.n);
    println!(" min    = {:.4}", s.min);
    println!(" max    = {:.4}", s.max);
    println!(" mean   = {:.4}", s.mean);
    println!(" std    = {:.4}", s.std);
    println!(" p05    = {:.4}", s.p05);
    println!(" p50    = {:.4}", s.p50);
    println!(" p95    = {:.4}", s.p95);
    println!();
    println!("────────── sprint-threshold diagnostic ──────────");
    let thresh = 0.125; // current SPRINT_THRESHOLD on all 6 adopters
    let above = phi_samples.iter().filter(|&&x| x > thresh).count();
    let pct = 100.0 * above as f64 / s.n as f64;
    println!(" SPRINT_THRESHOLD = 0.125 (2026-04-19 recalibration)");
    println!(" Φ > 0.125 fraction : {:.1} %  ({} / {})", pct, above, s.n);
    println!();
    println!(" Pre-FEP-wiring band (commit ≤6517226491): [0.099, 0.145]");
    println!(" Post-FEP-wiring band (commit 996750d12b+): [0.088, 0.133]");
    println!(
        " This run's range:                   [{:.4}, {:.4}]",
        s.min, s.max
    );
    println!(
        " Transferability verdict: {}",
        if s.max < 0.06 {
            "FAILS — signal band collapsed (agent untrained or observation-pack too uniform)"
        } else if (s.min - 0.099).abs() < 0.05 && (s.max - 0.145).abs() < 0.05 {
            "MATCHES manipulator band to within 0.05"
        } else {
            "DRIFTS — threshold 0.135 may be miscalibrated for this platform"
        }
    );

    if let Some(p) = csv_path.as_ref() {
        println!();
        println!("CSV written to: {p}");
    }
}
