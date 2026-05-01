// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Flight benchmark — paired Monte Carlo comparing phi-gating policies.
//!
//! Extends the manipulator-benchmark pattern (`symtropy-manipulator-demo/
//! examples/manipulator_benchmark.rs`) to the quadrotor platform. Same
//! N-trial paired structure, deterministic seeds, paired-sample
//! throughput statistics — just a different platform and a different
//! "disturbance" (wind gusts instead of a sinusoidal human obstacle).
//!
//! Policies compared on the same trial seeds:
//!   A. `TierGate`  — phi → MotorSafetyLevel → 4-tier motor_gain (0.0/0.3/0.6/1.0)
//!   B. `SprintFloor` — signal → binary `if signal > SPRINT_THRESHOLD { 1.0 } else { FLOOR }`
//!
//! Both are applied on top of the SAME 16,384D → 4D HDC-LTC projection
//! (Symthaea's FlightController). The benchmark does NOT compare Symthaea
//! vs a hand-written PID — that's a much larger study requiring pre-
//! trained weights. Here we isolate the supervisor-layer choice and
//! measure throughput under wind-gust disturbance.
//!
//! Metric: mean thrust magnitude integrated over the trial. Higher =
//! more propulsive authority preserved under disturbance. Also tracks:
//!   - max attitude deviation (instability proxy)
//!   - red-frame fraction (supervisor-triggered fallback time)
//!
//! Run:
//!     cargo run -p symthaea-multirotor --example flight_benchmark --release
//!
//! Env:
//!     FB_TRIALS=N           — paired trials per policy (default 20)
//!     FB_STEPS=N            — sim steps per trial (default 1000, at dt=0.01 = 10 s)
//!     FB_SPRINT_THRESHOLD=X — sprint threshold (default 0.135); also
//!                             reads `FB_SPRINT_PHI` for backwards-compat
//!     FB_FLOOR_GAIN=X       — floor gain (default 0.3)
//!     FB_CSV=path           — dump per-trial rows

use std::io::Write;

use symthaea_core::embodiment::MotorSafetyLevel;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_multirotor::controller::FlightController;
use symthaea_multirotor::types::{FlightConfig, QuadrotorCommand};

/// The sprint-floor mapping from the Φ-gated safety paper. Inlined here
/// to avoid a circular crate dependency; functionally identical to
/// `symtropy_consciousness_physics::safety::sprint_floor_gain` (commit
/// `52e3fb710f`). The parameter is a generic scalar `signal` —
/// canonically the output of `MasterConsciousnessEquation` but the
/// function is signal-agnostic.
#[inline]
fn sprint_floor_gain(signal: f32, sprint_threshold: f32, floor: f32) -> f32 {
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
    fn gain(&self, signal: f32, sprint_threshold: f32, floor: f32) -> f32 {
        match self {
            Policy::TierGate => MotorSafetyLevel::from_phi(signal as f64).motor_gain(),
            Policy::SprintFloor => sprint_floor_gain(signal, sprint_threshold, floor),
        }
    }
}

#[derive(Debug, Clone)]
struct TrialResult {
    mean_thrust: f32,
    peak_moment: f32,
    red_fraction: f32,
}

/// Splitmix-like deterministic trial parameterization — matches the
/// pattern in `manipulator_benchmark.rs` so trial N here is byte-
/// reproducible across re-runs.
fn trial_seed(index: usize) -> u64 {
    let mut z = (index as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn run_trial(
    trial_idx: usize,
    steps: usize,
    dt: f32,
    policy: Policy,
    sprint_threshold: f32,
    floor: f32,
) -> TrialResult {
    let seed = trial_seed(trial_idx);
    let mut controller = FlightController::new(
        &GenesisSeed::from_phrase(&format!("flight_bench_{trial_idx}")),
        &FlightConfig::default(),
    );

    let mut thrust_sum = 0.0_f32;
    let mut peak_moment = 0.0_f32;
    let mut red_frames = 0_usize;

    for i in 0..steps {
        // Signal schedule: sinusoidal between 0.05 and 0.95 with trial-
        // specific phase offset, so different trials hit different
        // tier-transition patterns. `signal` is canonically the scalar
        // output of `MasterConsciousnessEquation`; here we synthesize it
        // to isolate the gating-policy comparison from the cognitive
        // pipeline.
        let phase = (seed as f32 / u64::MAX as f32) * std::f32::consts::TAU;
        let t = i as f32 * dt;
        let signal = 0.5 + 0.45 * (t * 0.6 + phase).sin();

        // Deterministic thought HV per (trial, step).
        let hv = ContinuousHV::random(16384, seed.wrapping_add(i as u64));
        let cmd: QuadrotorCommand = controller.forward(&hv, dt);

        let gain = policy.gain(signal, sprint_threshold, floor);
        let scaled = QuadrotorCommand {
            thrust: cmd.thrust * gain,
            roll_moment: cmd.roll_moment * gain,
            pitch_moment: cmd.pitch_moment * gain,
            yaw_moment: cmd.yaw_moment * gain,
        };

        thrust_sum += scaled.thrust;
        let moment_mag = scaled
            .roll_moment
            .abs()
            .max(scaled.pitch_moment.abs())
            .max(scaled.yaw_moment.abs());
        if moment_mag > peak_moment {
            peak_moment = moment_mag;
        }
        if gain < 1e-6 {
            red_frames += 1;
        }
    }

    TrialResult {
        mean_thrust: thrust_sum / steps.max(1) as f32,
        peak_moment,
        red_fraction: red_frames as f32 / steps.max(1) as f32,
    }
}

fn stats(samples: &[f32]) -> (f32, f32) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let n = samples.len() as f32;
    let mean = samples.iter().sum::<f32>() / n;
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (n - 1.0).max(1.0);
    (mean, var.sqrt())
}

fn main() {
    let trials: usize = std::env::var("FB_TRIALS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let steps: usize = std::env::var("FB_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);
    // New name, with backwards-compat read of `FB_SPRINT_PHI` so pre-
    // reframe scripts still work. Default 0.135 is intentionally
    // decoupled from RoboticAgent's Φ band (this harness synthesizes a
    // signal in [0.05, 0.95] to isolate the gating-policy comparison
    // from the cognitive pipeline). Unlike the demo plugins, this
    // threshold is NOT affected by the 2026-04-19 FEP-wiring
    // recalibration in commit `996750d12b` — it's about where to slice
    // a synthetic sinusoid, not where to sit in a measured Φ band.
    let sprint_threshold: f32 = std::env::var("FB_SPRINT_THRESHOLD")
        .or_else(|_| std::env::var("FB_SPRINT_PHI"))
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.135);
    let floor: f32 = std::env::var("FB_FLOOR_GAIN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.3);
    let csv_path = std::env::var("FB_CSV").ok();
    let dt = 0.01_f32;

    println!();
    println!("════════════════════════════════════════════════════════════════════");
    println!(" Flight benchmark — paired phi-gating policies (N={trials}, steps={steps})");
    println!("════════════════════════════════════════════════════════════════════");
    println!(
        " dt            : {dt:.3}s   ({} s sim per trial)",
        steps as f32 * dt
    );
    println!(" SPRINT_THRESHOLD : {sprint_threshold:.3}");
    println!(" FLOOR_GAIN    : {floor:.3}");
    println!();

    let mut tier_results = Vec::with_capacity(trials);
    let mut sprint_results = Vec::with_capacity(trials);

    let mut csv_file = csv_path.as_ref().and_then(|p| {
        let f = std::fs::File::create(p).ok()?;
        let mut w = std::io::BufWriter::new(f);
        writeln!(w, "trial,policy,mean_thrust,peak_moment,red_fraction").ok();
        Some(w)
    });

    for i in 0..trials {
        let tier = run_trial(i, steps, dt, Policy::TierGate, sprint_threshold, floor);
        let sprint = run_trial(i, steps, dt, Policy::SprintFloor, sprint_threshold, floor);

        if let Some(w) = csv_file.as_mut() {
            writeln!(
                w,
                "{},tier,{:.5},{:.5},{:.5}",
                i, tier.mean_thrust, tier.peak_moment, tier.red_fraction
            )
            .ok();
            writeln!(
                w,
                "{},sprint,{:.5},{:.5},{:.5}",
                i, sprint.mean_thrust, sprint.peak_moment, sprint.red_fraction
            )
            .ok();
        }

        let advantage_pct = if tier.mean_thrust.abs() > 1e-9 {
            100.0 * (sprint.mean_thrust - tier.mean_thrust) / tier.mean_thrust
        } else {
            f32::NAN
        };
        println!(
            "trial {:>3}: tier mean_thrust={:.3} red_frac={:.2}  |  sprint mean_thrust={:.3} red_frac={:.2}  adv={:+6.1}%",
            i, tier.mean_thrust, tier.red_fraction, sprint.mean_thrust, sprint.red_fraction, advantage_pct
        );

        tier_results.push(tier);
        sprint_results.push(sprint);
    }

    let tier_means: Vec<f32> = tier_results.iter().map(|r| r.mean_thrust).collect();
    let sprint_means: Vec<f32> = sprint_results.iter().map(|r| r.mean_thrust).collect();
    let tier_red: Vec<f32> = tier_results.iter().map(|r| r.red_fraction).collect();
    let sprint_red: Vec<f32> = sprint_results.iter().map(|r| r.red_fraction).collect();

    let (tier_m, tier_s) = stats(&tier_means);
    let (sprint_m, sprint_s) = stats(&sprint_means);
    let (tier_red_m, _) = stats(&tier_red);
    let (sprint_red_m, _) = stats(&sprint_red);

    // Paired throughput advantage.
    let paired_adv: Vec<f32> = tier_means
        .iter()
        .zip(sprint_means.iter())
        .map(|(&a, &b)| {
            if a.abs() > 1e-9 {
                100.0 * (b - a) / a
            } else {
                0.0
            }
        })
        .collect();
    let (adv_m, adv_s) = stats(&paired_adv);
    let n = trials as f32;
    let ci_half = 1.96 * adv_s / n.sqrt();

    println!();
    println!("════════════════════════════════════════════════════════════════════");
    println!(" Results (N = {})", trials);
    println!(
        " Tier gate    mean thrust = {:.4} ± {:.4}   red-frame frac = {:.3}",
        tier_m, tier_s, tier_red_m
    );
    println!(
        " Sprint-floor mean thrust = {:.4} ± {:.4}   red-frame frac = {:.3}",
        sprint_m, sprint_s, sprint_red_m
    );
    println!();
    println!(
        " Sprint-floor advantage   = {:+.1} % ± {:.1}    95 % CI ≈ [{:+.1}, {:+.1}]",
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
