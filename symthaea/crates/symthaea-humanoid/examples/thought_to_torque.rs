// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! From thought to torque (humanoid) — legibility demo for the
//! HDC-LTC → 21-DOF motor command pipeline via `HumanoidEmbodiment`.
//!
//! Analogous to `symthaea-flight/examples/thought_to_torque.rs`, but
//! for the 21-DOF bipedal humanoid. Renders the full cognition-to-
//! actuation pathway:
//!
//!   thought_hv (16,384D ContinuousHV from a cognitive-loop cycle)
//!         │
//!         ▼
//!   HumanoidEmbodiment::step(thought_hv, dt, phi)
//!         │  ─ HumanoidController evolves HDC-LTC network
//!         │  ─ bottleneck 16,384D → 64D via fixed random projection
//!         │  ─ learned linear projection 64D → 21D torques
//!         │  ─ tanh activation → HumanoidCommand per-joint torques
//!         │  ─ phi envelope: motor_gain from MotorSafetyLevel
//!         │  ─ StandingLock fallback at Red (gravity-comp hip baseline)
//!         │  ─ simulator steps physics 1 dt
//!         ▼
//!   EmbodimentResult { num_actuators=21, control_effort,
//!                      safety_level, prediction_error, ... }
//!
//! What this demonstrates: the same HDC-LTC → motor-vector projection
//! pattern that flight uses, scaled up to 21 joints, with an internal
//! simulator step per call (not pure-projection like the flight demo).
//! `HumanoidEmbodiment` is the canonical consumer of the
//! `EmbodimentBridge` trait for humanoid — wired into
//! `CognitiveLoopService::construct` in commit `03058396b5`.
//!
//! Run:
//!     cargo run -p symthaea-humanoid --example thought_to_torque --release
//!
//! Env:
//!     HTT_STEPS=N     number of ticks (default 200)
//!     HTT_CSV=path    dump per-step CSV
//!     HTT_NO_COLOR=1  plain ASCII (for pipes / CI)

use std::io::Write;

use symthaea_core::embodiment::MotorSafetyLevel;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_humanoid::embodiment::HumanoidEmbodiment;
use symthaea_humanoid::simulator::HumanoidPhysicsSimulator;

fn main() {
    let steps: usize = std::env::var("HTT_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let csv_path = std::env::var("HTT_CSV").ok();
    let use_color = std::env::var("HTT_NO_COLOR").is_err();

    banner(use_color);

    let genesis = GenesisSeed::from_phrase("humanoid_thought_to_torque");
    let mut embodiment = HumanoidEmbodiment::new(&genesis);

    let mut csv_file = csv_path.as_ref().and_then(|p| {
        let f = std::fs::File::create(p).ok()?;
        let mut w = std::io::BufWriter::new(f);
        writeln!(
            w,
            "step,phi,safety,num_actuators,control_effort,pred_error,root_height,uprightness"
        )
        .ok();
        Some(w)
    });

    let dt = 0.025_f32;
    let mut red_frames = 0usize;
    let mut green_frames = 0usize;
    let mut total_effort = 0.0f32;
    let mut min_root_height = f64::INFINITY;
    let mut max_root_height = f64::NEG_INFINITY;

    for i in 0..steps {
        let t = i as f32 / steps.max(1) as f32;
        let phi = 0.05 + 0.9 * t; // 0.05 → 0.95 linearly, exercising all tiers.
                                  // Synthetic thought HV — deterministic seed per step. In a real
                                  // cognitive-loop context, thought_hv comes from perception +
                                  // prediction + attention in `CognitiveLoopService::cycle`.
        let hv = ContinuousHV::random(16384, 2000 + i as u64);

        let result = embodiment.step(&hv, dt, phi as f64);
        total_effort += result.control_effort;
        match result.safety_level {
            MotorSafetyLevel::Red => red_frames += 1,
            MotorSafetyLevel::Green => green_frames += 1,
            _ => {}
        }

        let state = embodiment.simulator().state();
        min_root_height = min_root_height.min(state.root_height);
        max_root_height = max_root_height.max(state.root_height);
        let uprightness = state.uprightness();

        if let Some(w) = csv_file.as_mut() {
            writeln!(
                w,
                "{},{:.3},{:?},{},{:.4},{:.5},{:.4},{:.4}",
                i,
                phi,
                result.safety_level,
                result.num_actuators,
                result.control_effort,
                result.prediction_error,
                state.root_height,
                uprightness,
            )
            .ok();
        }

        let sample_every = (steps / 16).max(1);
        if i % sample_every == 0 || i + 1 == steps {
            print_frame(
                i,
                phi,
                result.safety_level,
                result.control_effort,
                result.prediction_error,
                state.root_height,
                uprightness,
                use_color,
            );
        }
    }

    summary(
        steps,
        green_frames,
        red_frames,
        total_effort / steps.max(1) as f32,
        min_root_height,
        max_root_height,
        use_color,
    );

    if let Some(p) = csv_path.as_ref() {
        println!();
        println!("CSV written to: {p}");
    }
}

fn banner(use_color: bool) {
    let (c, r) = color_codes(use_color);
    println!();
    println!("{c}──────────────────────────────────────────────────────────────{r}");
    println!("{c} From thought to torque (humanoid) — HDC-LTC → 21-DOF command {r}");
    println!("{c}──────────────────────────────────────────────────────────────{r}");
    println!(" 16,384D thought_hv  ─▶  HDC-LTC + 64D bottleneck  ─▶  21D torques");
    println!(" phi envelope        ─▶  MotorSafetyLevel          ─▶  gain × torques");
    println!(" Red tier            ─▶  StandingLock fallback (gravity-comp hip baseline)");
    println!();
    println!(
        "{:>4} {:>5} {:>7} {:>4} {:>9} {:>9} {:>7} {:>7}",
        "step", "phi", "tier", "acts", "effort", "pred_err", "root_h", "upright"
    );
    println!(
        "{:>4} {:>5} {:>7} {:>4} {:>9} {:>9} {:>7} {:>7}",
        "----", "-----", "-------", "----", "---------", "---------", "-------", "-------"
    );
}

#[allow(clippy::too_many_arguments)]
fn print_frame(
    i: usize,
    phi: f32,
    tier: MotorSafetyLevel,
    effort: f32,
    pred_err: f32,
    root_h: f64,
    uprightness: f64,
    use_color: bool,
) {
    let (tier_str, tier_color) = tier_style(tier, use_color);
    let (_, reset) = color_codes(use_color);
    println!(
        "{:>4} {:>5.2} {}{:>7}{} {:>4} {:>9.4} {:>9.5} {:>7.3} {:>7.3}",
        i, phi, tier_color, tier_str, reset, 21, effort, pred_err, root_h, uprightness
    );
}

fn summary(
    steps: usize,
    green_frames: usize,
    red_frames: usize,
    mean_effort: f32,
    min_h: f64,
    max_h: f64,
    use_color: bool,
) {
    let (c, r) = color_codes(use_color);
    let green_pct = 100.0 * green_frames as f32 / steps.max(1) as f32;
    let red_pct = 100.0 * red_frames as f32 / steps.max(1) as f32;
    println!();
    println!("{c}──────────────────────────────── summary ──────────────────────────────{r}");
    println!(" steps:            {steps}");
    println!(" green frames:     {green_frames} ({green_pct:.1} %)");
    println!(" red frames:       {red_frames} ({red_pct:.1} %)");
    println!(" mean effort:      {mean_effort:.4}");
    println!(" root height:      [{min_h:.3}, {max_h:.3}]");
    println!(" dim reduction:    16384D  →  64D  →  21D");
    println!();
}

fn tier_style(t: MotorSafetyLevel, use_color: bool) -> (&'static str, &'static str) {
    match (t, use_color) {
        (MotorSafetyLevel::Green, true) => ("GREEN", "\x1b[32m"),
        (MotorSafetyLevel::Yellow, true) => ("YELLOW", "\x1b[33m"),
        (MotorSafetyLevel::Orange, true) => ("ORANGE", "\x1b[38;5;208m"),
        (MotorSafetyLevel::Red, true) => ("RED", "\x1b[31m"),
        (MotorSafetyLevel::Green, false) => ("GREEN", ""),
        (MotorSafetyLevel::Yellow, false) => ("YELLOW", ""),
        (MotorSafetyLevel::Orange, false) => ("ORANGE", ""),
        (MotorSafetyLevel::Red, false) => ("RED", ""),
    }
}

fn color_codes(use_color: bool) -> (&'static str, &'static str) {
    if use_color {
        ("\x1b[36m", "\x1b[0m")
    } else {
        ("", "")
    }
}
