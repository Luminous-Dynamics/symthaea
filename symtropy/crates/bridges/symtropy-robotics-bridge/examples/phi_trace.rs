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
//!     PT_PLATFORM_OBS=1       use hand-crafted platform-specific observation
//!                             generators (vs legacy shared sinusoid). Produces
//!                             genuinely different Φ distributions per platform.

use std::io::Write;

use symtropy_physics::BodyHandle;
use symtropy_robotics_bridge::RoboticAgentTrait;
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
    sorted.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());
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
/// scenario-accuracy for portability.
///
/// **Platform-aware mode** (`PT_PLATFORM_OBS=1`): each platform gets a
/// hand-crafted observation generator reflecting its actual sensor
/// dynamics. This produces genuinely different Φ distributions per
/// platform instead of dim-only variation. The generators model the
/// dominant temporal pattern of each sensor channel:
///
///   quadrotor:    altitude (stable ~0.6, small perturb) +
///                 attitude-x/y (gust-correlated spikes) +
///                 wind-gust (Dryden-ish bursty sinusoid)
///   vehicle:      speed (periodic figure-8) +
///                 slip (bursty, spikes near friction changes) +
///                 friction (step-changes during ice patches)
///   humanoid:     uprightness (slow drift under push) +
///                 push-norm (step-impulse perturbations)
///   manipulator:  danger (smooth sinusoid) +
///                 PE (rising during approach) +
///                 effort (correlated with danger) +
///                 stiffness (quasi-constant with small jitter)
///   auv:          depth (descending then holding) +
///                 current (slow rotation) +
///                 chemical (bursty plume) +
///                 PE (tracks current variation)
///   helicopter:   altitude (hover-oscillation) +
///                 wind (Dryden bursts) +
///                 attitude (recovery-correlated) +
///                 PE (rises during gusts)
///
/// **Legacy mode** (default, `PT_PLATFORM_OBS` unset): all platforms
/// receive the same sinusoidal observation stream with platform-specific
/// dimensionality only — the structural invariance test from the §8
/// original discussion.
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

/// Platform-aware observation generators. Each returns
/// `(observation_vector, danger_level)` for a given step + seed.
fn platform_observation(platform: PlatformType, step: usize, seed: u64) -> (Vec<f64>, f64) {
    let s = step as f64;
    let phase = (seed as f64 % 1000.0) * 0.001;
    let clamp01 = |x: f64| x.clamp(0.0, 1.0);
    match platform {
        PlatformType::Quadrotor => {
            // Stable altitude with mild perturbation; attitude correlated
            // with wind gusts; wind-gust bursty.
            let gust = clamp01(0.3 + 0.4 * (s * 0.08 + phase).sin().powi(2));
            let alt = clamp01(0.6 + 0.1 * (s * 0.02).sin());
            let att_x = clamp01(0.5 + 0.3 * gust * (s * 0.12).sin());
            let att_y = clamp01(0.5 + 0.3 * gust * (s * 0.12 + 1.5).sin());
            let danger = clamp01(gust * 0.7 + 0.2 * att_x);
            (vec![alt, att_x, att_y, gust], danger)
        }
        PlatformType::Vehicle => {
            // Periodic speed (figure-8); slip spikes near friction changes;
            // friction step-changes (ice patches).
            let speed = clamp01(0.5 + 0.35 * (s * 0.025).sin());
            let friction = if (s * 0.013).sin() > 0.3 { 1.0 } else { 0.35 };
            let slip = clamp01((1.0 - friction) * (0.3 + 0.5 * (s * 0.18).sin().abs()));
            let danger = clamp01((1.0 - friction) * 0.8 + slip * 0.2);
            (vec![speed, slip, friction], danger)
        }
        PlatformType::Humanoid => {
            // Uprightness drifts during push impulses; push bursts at low freq.
            let push = if ((s * 0.01 + phase) as f64).sin() > 0.85 {
                0.9
            } else {
                0.05
            };
            let uprightness = clamp01(0.9 - 0.5 * push - 0.1 * (s * 0.04).sin().abs());
            let danger = clamp01(push * 0.7 + (1.0 - uprightness) * 0.3);
            (vec![uprightness, push], danger)
        }
        PlatformType::Manipulator => {
            // Danger: smooth sinusoid (human obstacle approach);
            // PE rises during approach; effort correlates with danger;
            // stiffness quasi-constant with small jitter.
            let approach = clamp01(0.5 + 0.4 * (s * 0.05 + phase).sin());
            let pe = clamp01(0.15 + 0.3 * approach + 0.05 * (s * 0.09).cos());
            let effort = clamp01(0.4 + 0.3 * approach);
            let stiffness = clamp01(0.65 + 0.05 * (s * 0.023).sin());
            (vec![approach, pe, effort, stiffness], approach * 0.9)
        }
        PlatformType::Auv => {
            // Depth: descending then holding; current: slow rotation;
            // chemical: bursty plume; PE tracks current variation.
            let depth = clamp01(0.2 + 0.6 * (1.0 - (-s * 0.004).exp()));
            let current = clamp01(0.3 + 0.3 * (s * 0.018).sin());
            let chemical = if (s * 0.02 + phase).sin() > 0.7 {
                0.8
            } else {
                0.1
            };
            let pe = clamp01(0.2 + 0.4 * current.abs());
            let danger = clamp01(current * 0.5 + chemical * 0.3);
            (vec![depth, current, chemical, pe], danger)
        }
        PlatformType::Helicopter => {
            // Altitude: hover with oscillation; wind: Dryden bursts;
            // attitude: recovery-correlated; PE: rises during gusts.
            let wind = clamp01(0.25 + 0.5 * (s * 0.06 + phase).sin().powi(2));
            let altitude = clamp01(0.7 + 0.1 * (s * 0.025).sin() - 0.15 * wind);
            let attitude = clamp01(0.5 + 0.35 * wind * (s * 0.14).sin());
            let pe = clamp01(0.15 + 0.5 * wind);
            let danger = clamp01(wind * 0.7 + (1.0 - altitude) * 0.3);
            (vec![altitude, wind, attitude, pe], danger)
        }
        _ => {
            // Fallback: legacy synthetic stream for platforms without a
            // hand-crafted generator.
            synth_observation(observation_dim(platform), step, seed)
        }
    }
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
    let platform_obs = std::env::var("PT_PLATFORM_OBS")
        .ok()
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false);
    let dim = observation_dim(platform);

    println!();
    println!("════════════════════════════════════════════════════════════════════");
    println!(" Φ trace — empirical signal-band measurement via RoboticAgent::tick");
    println!("════════════════════════════════════════════════════════════════════");
    println!(" platform         : {:?}", platform);
    println!(" observation dim  : {}", dim);
    println!(
        " observation mode : {}",
        if platform_obs {
            "PLATFORM-AWARE (hand-crafted)"
        } else {
            "legacy synthetic"
        }
    );
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
        let (obs, danger) = if platform_obs {
            platform_observation(platform, step, seed)
        } else {
            synth_observation(dim, step, seed)
        };
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
