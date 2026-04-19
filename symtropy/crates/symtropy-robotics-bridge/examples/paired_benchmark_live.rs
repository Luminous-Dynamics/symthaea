// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Paired TierGate vs SprintFloor comparison across all 6 adopter
//! platforms, using **RoboticAgent-driven Φ** under platform-aware
//! observations.
//!
//! Generalization of `flight_benchmark_live.rs` (commit `aa1ef4e957`).
//! That example ran the paired comparison on quadrotor only and
//! produced +125.8 % advantage. This example runs it on all 6
//! platforms so §8's claim that per-platform calibration works
//! end-to-end has cross-platform empirical backing — not just the
//! quadrotor demonstration.
//!
//! Each platform is run with its currently-applied SPRINT_THRESHOLD
//! (from the §8 table), under platform-aware observations that
//! mirror each platform's demo scenario (hover + wind for quadrotor,
//! speed + ice for vehicle, etc.).
//!
//! Run:
//!     cargo run -p symtropy-robotics-bridge --example paired_benchmark_live --release
//!
//! Env:
//!     PBL_TRIALS=N   trials per platform (default 30)
//!     PBL_STEPS=N    sim ticks per trial (default 500)
//!     PBL_CSV=path   dump per-platform CSV (one row per platform)

use std::io::Write;

use symtropy_consciousness_physics::safety::SafetyTier;
use symtropy_physics::BodyHandle;
use symtropy_robotics_bridge::agent::RoboticAgent;
use symtropy_robotics_bridge::platform::PlatformType;

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

fn trial_seed(index: usize) -> u64 {
    let mut z = (index as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Platform-aware observation generators — mirror
/// `phi_trace.rs::platform_observation`.
fn platform_observation(platform: PlatformType, step: usize, seed: u64) -> (Vec<f64>, f64) {
    let s = step as f64;
    let phase = (seed as f64 % 1000.0) * 0.001;
    let clamp01 = |x: f64| x.clamp(0.0, 1.0);
    match platform {
        PlatformType::Quadrotor => {
            let gust = clamp01(0.3 + 0.4 * (s * 0.08 + phase).sin().powi(2));
            let alt = clamp01(0.6 + 0.1 * (s * 0.02).sin());
            let att_x = clamp01(0.5 + 0.3 * gust * (s * 0.12).sin());
            let att_y = clamp01(0.5 + 0.3 * gust * (s * 0.12 + 1.5).sin());
            let danger = clamp01(gust * 0.7 + 0.2 * att_x);
            (vec![alt, att_x, att_y, gust], danger)
        }
        PlatformType::Vehicle => {
            let speed = clamp01(0.5 + 0.35 * (s * 0.025).sin());
            let friction = if (s * 0.013).sin() > 0.3 { 1.0 } else { 0.35 };
            let slip = clamp01((1.0 - friction) * (0.3 + 0.5 * (s * 0.18).sin().abs()));
            let danger = clamp01((1.0 - friction) * 0.8 + slip * 0.2);
            (vec![speed, slip, friction], danger)
        }
        PlatformType::Humanoid => {
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
            let approach = clamp01(0.5 + 0.4 * (s * 0.05 + phase).sin());
            let pe = clamp01(0.15 + 0.3 * approach + 0.05 * (s * 0.09).cos());
            let effort = clamp01(0.4 + 0.3 * approach);
            let stiffness = clamp01(0.65 + 0.05 * (s * 0.023).sin());
            (vec![approach, pe, effort, stiffness], approach * 0.9)
        }
        PlatformType::Auv => {
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
            let wind = clamp01(0.25 + 0.5 * (s * 0.06 + phase).sin().powi(2));
            let altitude = clamp01(0.7 + 0.1 * (s * 0.025).sin() - 0.15 * wind);
            let attitude = clamp01(0.5 + 0.35 * wind * (s * 0.14).sin());
            let pe = clamp01(0.15 + 0.5 * wind);
            let danger = clamp01(wind * 0.7 + (1.0 - altitude) * 0.3);
            (vec![altitude, wind, attitude, pe], danger)
        }
        _ => (vec![0.5; 4], 0.3),
    }
}

fn run_trial(
    platform: PlatformType,
    trial_idx: usize,
    steps: usize,
    policy: Policy,
    sprint_threshold: f64,
    floor: f64,
) -> (f64, f64) {
    let seed = trial_seed(trial_idx);
    let mut agent = RoboticAgent::new(
        BodyHandle(0),
        platform,
        format!("paired_bench_{:?}_{}", platform, trial_idx),
    );

    let mut gain_sum = 0.0;
    let mut red_frames = 0usize;

    for step in 0..steps {
        let (obs, danger) = platform_observation(platform, step, seed);
        let _ = agent.tick(&obs, danger);
        let phi = agent.phi();
        let gain = policy.gain(phi, sprint_threshold, floor);
        gain_sum += gain;
        if gain < 1e-6 {
            red_frames += 1;
        }
    }

    (
        gain_sum / steps.max(1) as f64,
        red_frames as f64 / steps.max(1) as f64,
    )
}

fn stats(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let n = xs.len() as f64;
    let m = xs.iter().sum::<f64>() / n;
    let v = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (m, v.sqrt())
}

struct PlatformSpec {
    name: &'static str,
    pt: PlatformType,
    sprint_threshold: f64,
    floor: f64,
}

const SPECS: &[PlatformSpec] = &[
    PlatformSpec {
        name: "quadrotor",
        pt: PlatformType::Quadrotor,
        sprint_threshold: 0.110,
        floor: 0.3,
    },
    PlatformSpec {
        name: "helicopter",
        pt: PlatformType::Helicopter,
        sprint_threshold: 0.100, // post-sim-driven-validation correction (was 0.110)
        floor: 0.3,
    },
    PlatformSpec {
        name: "vehicle",
        pt: PlatformType::Vehicle,
        sprint_threshold: 0.101,
        floor: 0.3,
    },
    PlatformSpec {
        name: "manipulator",
        pt: PlatformType::Manipulator,
        sprint_threshold: 0.114,
        floor: 0.3,
    },
    PlatformSpec {
        name: "humanoid",
        pt: PlatformType::Humanoid,
        sprint_threshold: 0.130,
        floor: 0.3,
    },
    PlatformSpec {
        name: "auv",
        pt: PlatformType::Auv,
        sprint_threshold: 0.130,
        floor: 0.3,
    },
];

fn main() {
    let trials: usize = std::env::var("PBL_TRIALS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let steps: usize = std::env::var("PBL_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    let csv_path = std::env::var("PBL_CSV").ok();

    println!();
    println!("════════════════════════════════════════════════════════════════════════════");
    println!(" Paired benchmark (live-Φ) across 6 platforms — N={trials}, steps={steps}");
    println!("════════════════════════════════════════════════════════════════════════════");
    println!();
    println!(
        "{:<12} {:>9} {:>5} {:>10} {:>10} {:>10} {:>22}",
        "platform", "thresh", "floor", "tier_mean", "sprint_mn", "adv_mean", "95 % CI"
    );
    println!(
        "{:<12} {:>9} {:>5} {:>10} {:>10} {:>10} {:>22}",
        "------------",
        "---------",
        "-----",
        "----------",
        "----------",
        "----------",
        "----------------------"
    );

    let mut csv_file = csv_path.as_ref().and_then(|p| {
        let f = std::fs::File::create(p).ok()?;
        let mut w = std::io::BufWriter::new(f);
        writeln!(
            w,
            "platform,sprint_threshold,floor,tier_mean,tier_std,sprint_mean,sprint_std,adv_mean,adv_std,ci_low,ci_high,n"
        )
        .ok();
        Some(w)
    });

    for spec in SPECS {
        let mut tier_gains = Vec::with_capacity(trials);
        let mut sprint_gains = Vec::with_capacity(trials);
        for i in 0..trials {
            let (tg, _) = run_trial(
                spec.pt,
                i,
                steps,
                Policy::TierGate,
                spec.sprint_threshold,
                spec.floor,
            );
            let (sg, _) = run_trial(
                spec.pt,
                i,
                steps,
                Policy::SprintFloor,
                spec.sprint_threshold,
                spec.floor,
            );
            tier_gains.push(tg);
            sprint_gains.push(sg);
        }
        let (tier_m, tier_s) = stats(&tier_gains);
        let (sprint_m, sprint_s) = stats(&sprint_gains);
        let paired: Vec<f64> = tier_gains
            .iter()
            .zip(sprint_gains.iter())
            .map(|(t, s)| if *t > 1e-9 { 100.0 * (s - t) / t } else { 0.0 })
            .collect();
        let (adv_m, adv_s) = stats(&paired);
        let ci_half = 1.96 * adv_s / (trials as f64).sqrt();
        let ci_low = adv_m - ci_half;
        let ci_high = adv_m + ci_half;

        println!(
            "{:<12} {:>9.3} {:>5.2} {:>10.3} {:>10.3} {:>+9.1}% {:>10}",
            spec.name,
            spec.sprint_threshold,
            spec.floor,
            tier_m,
            sprint_m,
            adv_m,
            format!("[{:+.1}, {:+.1}]", ci_low, ci_high),
        );
        if let Some(w) = csv_file.as_mut() {
            writeln!(
                w,
                "{},{:.3},{:.3},{:.5},{:.5},{:.5},{:.5},{:.3},{:.3},{:.3},{:.3},{}",
                spec.name,
                spec.sprint_threshold,
                spec.floor,
                tier_m,
                tier_s,
                sprint_m,
                sprint_s,
                adv_m,
                adv_s,
                ci_low,
                ci_high,
                trials
            )
            .ok();
        }
    }

    println!();
    if let Some(p) = csv_path.as_ref() {
        println!("CSV written to: {p}");
    }
}
