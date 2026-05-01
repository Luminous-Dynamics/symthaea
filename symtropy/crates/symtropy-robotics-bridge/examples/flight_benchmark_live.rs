// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Paired TierGate vs SprintFloor comparison on quadrotor using
//! **RoboticAgent-driven Φ** (vs the synthetic-signal version in
//! `symthaea-multirotor/examples/flight_benchmark.rs`).
//!
//! Complements Figure 3 of the Φ-gated-safety paper. Figure 3's
//! flight_benchmark synthesizes a sinusoidal signal in [0.05, 0.95]
//! to isolate the gating-policy choice from the cognitive pipeline.
//! This harness does the opposite — it feeds a flight-specific
//! observation schedule (altitude / attitude / wind / PE) into
//! `RoboticAgent::tick` so Φ comes from the actual
//! `MasterConsciousnessEquation` composition, matching the §6
//! manipulator benchmark's setup.
//!
//! Expected result: the advantage direction should survive
//! (SprintFloor > TierGate) because sprint-floor retains a non-zero
//! gain below threshold while tier-gate drops to 0. Magnitude may
//! differ from the synthetic Figure 3's +71.4 % because the
//! platform-aware Φ distribution on quadrotor (p50 = 0.110, per
//! §8 table) is compressed in a narrower band than the synthetic
//! [0.05, 0.95] sinusoid.
//!
//! Run:
//!     cargo run -p symtropy-robotics-bridge --example flight_benchmark_live --release
//!
//! Env:
//!     FBL_TRIALS=N        number of paired trials (default 30)
//!     FBL_STEPS=N         sim ticks per trial (default 500)
//!     FBL_SPRINT_THRESHOLD=X  sprint threshold (default 0.110, matches
//!                             the quadrotor demo plugin's applied value)
//!     FBL_FLOOR=X         floor gain (default 0.3)
//!     FBL_CSV=path        dump per-trial CSV

use std::io::Write;

use symtropy_consciousness_physics::safety::SafetyTier;
use symtropy_physics::BodyHandle;
use symtropy_robotics_bridge::agent::RoboticAgent;
use symtropy_robotics_bridge::platform::PlatformType;

/// Inlined sprint-floor mapping (see
/// `symtropy-consciousness-physics::safety::sprint_floor_gain`).
#[inline]
fn sprint_floor_gain(signal: f64, sprint_threshold: f64, floor: f64) -> f64 {
    if signal > sprint_threshold {
        1.0
    } else {
        floor
    }
}

#[derive(Debug, Clone, Copy)]
enum Policy {
    TierGate,
    SprintFloor,
}

impl Policy {
    fn gain(&self, phi: f64, sprint_threshold: f64, floor: f64) -> f64 {
        match self {
            Policy::TierGate => SafetyTier::from_phi(phi).motor_gain() as f64,
            Policy::SprintFloor => sprint_floor_gain(phi, sprint_threshold, floor),
        }
    }
}

#[derive(Debug, Clone)]
struct TrialResult {
    mean_gain: f64,
    red_fraction: f64,
}

fn trial_seed(index: usize) -> u64 {
    let mut z = (index as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Flight-platform-aware observation + danger stream — mirrors the
/// generator in `phi_trace.rs::platform_observation` for Quadrotor.
fn flight_observation(step: usize, seed: u64) -> (Vec<f64>, f64) {
    let s = step as f64;
    let phase = (seed as f64 % 1000.0) * 0.001;
    let clamp01 = |x: f64| x.clamp(0.0, 1.0);
    let gust = clamp01(0.3 + 0.4 * (s * 0.08 + phase).sin().powi(2));
    let alt = clamp01(0.6 + 0.1 * (s * 0.02).sin());
    let att_x = clamp01(0.5 + 0.3 * gust * (s * 0.12).sin());
    let att_y = clamp01(0.5 + 0.3 * gust * (s * 0.12 + 1.5).sin());
    let danger = clamp01(gust * 0.7 + 0.2 * att_x);
    (vec![alt, att_x, att_y, gust], danger)
}

fn run_trial(
    trial_idx: usize,
    steps: usize,
    policy: Policy,
    sprint_threshold: f64,
    floor: f64,
) -> TrialResult {
    let seed = trial_seed(trial_idx);
    let mut agent = RoboticAgent::new(
        BodyHandle(0),
        PlatformType::Quadrotor,
        format!("flight_bench_live_{trial_idx}"),
    );

    let mut gain_sum = 0.0_f64;
    let mut red_frames = 0_usize;

    for step in 0..steps {
        let (obs, danger) = flight_observation(step, seed);
        // Run the RoboticAgent cognitive tick — returns the default-tier gain,
        // which we ignore; we use `agent.phi()` directly to compute our own
        // policy gain.
        let _default_gain = agent.tick(&obs, danger);
        let phi = agent.phi();

        let gain = policy.gain(phi, sprint_threshold, floor);
        gain_sum += gain;
        if gain < 1e-6 {
            red_frames += 1;
        }
    }

    TrialResult {
        mean_gain: gain_sum / steps.max(1) as f64,
        red_fraction: red_frames as f64 / steps.max(1) as f64,
    }
}

fn stats(samples: &[f64]) -> (f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, var.sqrt())
}

fn main() {
    let trials: usize = std::env::var("FBL_TRIALS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let steps: usize = std::env::var("FBL_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    let sprint_threshold: f64 = std::env::var("FBL_SPRINT_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.110);
    let floor: f64 = std::env::var("FBL_FLOOR")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.3);
    let csv_path = std::env::var("FBL_CSV").ok();

    println!();
    println!("════════════════════════════════════════════════════════════════════");
    println!(" Flight benchmark (LIVE-Φ) — paired TierGate vs SprintFloor via RoboticAgent");
    println!("════════════════════════════════════════════════════════════════════");
    println!(
        " N={trials}, steps={steps}, SPRINT_THRESHOLD={sprint_threshold:.3}, FLOOR={floor:.3}"
    );
    println!(" Observation stream: flight-platform-aware (altitude/attitude/wind/PE)");
    println!(" Φ source: RoboticAgent::tick(...) → MasterConsciousnessEquation (not synthetic)");
    println!();

    let mut tier_results = Vec::with_capacity(trials);
    let mut sprint_results = Vec::with_capacity(trials);

    let mut csv_file = csv_path.as_ref().and_then(|p| {
        let f = std::fs::File::create(p).ok()?;
        let mut w = std::io::BufWriter::new(f);
        writeln!(w, "trial,policy,mean_gain,red_fraction").ok();
        Some(w)
    });

    for i in 0..trials {
        let tier = run_trial(i, steps, Policy::TierGate, sprint_threshold, floor);
        let sprint = run_trial(i, steps, Policy::SprintFloor, sprint_threshold, floor);

        if let Some(w) = csv_file.as_mut() {
            writeln!(
                w,
                "{},tier,{:.5},{:.5}",
                i, tier.mean_gain, tier.red_fraction
            )
            .ok();
            writeln!(
                w,
                "{},sprint,{:.5},{:.5}",
                i, sprint.mean_gain, sprint.red_fraction
            )
            .ok();
        }

        let advantage = if tier.mean_gain > 1e-9 {
            100.0 * (sprint.mean_gain - tier.mean_gain) / tier.mean_gain
        } else {
            f64::NAN
        };
        println!(
            "trial {:>3}: tier gain={:.3} red={:.2}  |  sprint gain={:.3} red={:.2}  adv={:+7.1}%",
            i, tier.mean_gain, tier.red_fraction, sprint.mean_gain, sprint.red_fraction, advantage
        );
        tier_results.push(tier);
        sprint_results.push(sprint);
    }

    let tier_means: Vec<f64> = tier_results.iter().map(|r| r.mean_gain).collect();
    let sprint_means: Vec<f64> = sprint_results.iter().map(|r| r.mean_gain).collect();
    let (tier_m, tier_s) = stats(&tier_means);
    let (sprint_m, sprint_s) = stats(&sprint_means);
    let paired: Vec<f64> = tier_means
        .iter()
        .zip(sprint_means.iter())
        .map(|(t, s)| if *t > 1e-9 { 100.0 * (s - t) / t } else { 0.0 })
        .collect();
    let (adv_m, adv_s) = stats(&paired);
    let n = trials as f64;
    let ci_half = 1.96 * adv_s / n.sqrt();

    println!();
    println!("════════════════════════════════════════════════════════════════════");
    println!(" Results (N = {trials})");
    println!(" Tier gate    mean gain = {:.4} ± {:.4}", tier_m, tier_s);
    println!(
        " Sprint-floor mean gain = {:.4} ± {:.4}",
        sprint_m, sprint_s
    );
    println!();
    println!(
        " Sprint-floor advantage = {:+.1} % ± {:.1}    95 % CI ≈ [{:+.1}, {:+.1}]",
        adv_m,
        adv_s,
        adv_m - ci_half,
        adv_m + ci_half
    );
    println!("════════════════════════════════════════════════════════════════════");

    if let Some(p) = csv_path.as_ref() {
        println!();
        println!("CSV written to: {p}");
    }
}
