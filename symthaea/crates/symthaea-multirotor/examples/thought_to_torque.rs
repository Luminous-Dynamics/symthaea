// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! From thought to torque — a legibility demo for the HDC-LTC → motor command pipeline.
//!
//! Renders (in the terminal, no Bevy required) the full cognition-to-actuation
//! pathway that's invisible in the production `flight-demo`:
//!
//!   thought_hv (16,384D ContinuousHV from the cognitive loop)
//!         │
//!         ▼
//!   FlightController::forward(thought_hv, dt)
//!         │  ─ evolve HDC-LTC unified network dynamics
//!         │  ─ bundle + normalize final-layer output (16,384D)
//!         │  ─ learned linear projection: W @ hv + b → [raw_thrust, raw_roll, raw_pitch, raw_yaw]
//!         │  ─ activations: sigmoid(thrust) × MAX_THRUST, tanh(moments) × MAX_MOMENT
//!         ▼
//!   QuadrotorCommand { thrust: N, roll_moment: Nm, pitch_moment: Nm, yaw_moment: Nm }
//!         │
//!         ▼
//!   MotorSafetyLevel::from_phi × motor_gain() envelope (optional)
//!         │
//!         ▼
//!   actuators
//!
//! What's being demonstrated: there is NO hand-authored PD / PID / MPC /
//! CPG controller between the 16,384D thought vector and the 4 physical
//! motor outputs. The projection is learned. Consciousness-derived phi
//! scales the command in a separate envelope layer — papered against
//! ISO 21448 / SOTIF as a triggering-condition monitor, not a safety
//! certifier.
//!
//! Run with:
//!
//!     cargo run -p symthaea-multirotor --example thought_to_torque
//!
//! Optional env:
//!     TT_STEPS=200        — number of frames to simulate (default 200)
//!     TT_CSV=path         — dump per-step CSV to this file (for offline plotting)
//!     TT_NO_COLOR=1       — plain ASCII output (for pipes / CI)

use std::io::Write;

use symthaea_core::embodiment::MotorSafetyLevel;
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_multirotor::controller::FlightController;
use symthaea_multirotor::types::{FlightConfig, QuadrotorCommand};

const BAR_WIDTH: usize = 28;

fn main() {
    let steps: usize = std::env::var("TT_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let csv_path = std::env::var("TT_CSV").ok();
    let use_color = std::env::var("TT_NO_COLOR").is_err();

    banner(use_color);

    // Instantiate the controller directly — no bridge, no simulator —
    // because the point of this demo is to expose the projection, not
    // to step physics. A realistic run would feed thought_hv from a
    // cognitive-loop cycle; here we generate synthetic per-step HVs so
    // the demo is self-contained and deterministic.
    let genesis = GenesisSeed::from_phrase("thought_to_torque");
    let config = FlightConfig::default();
    let mut controller = FlightController::new(&genesis, &config);

    let mut csv_file = csv_path.as_ref().and_then(|p| {
        let f = std::fs::File::create(p).ok()?;
        let mut w = std::io::BufWriter::new(f);
        writeln!(
            w,
            "step,phi,safety,motor_gain,hv_norm,hv_meanabs,raw_thrust,raw_roll,raw_pitch,raw_yaw,thrust,roll,pitch,yaw,scaled_thrust,scaled_roll,scaled_pitch,scaled_yaw"
        )
        .ok();
        Some(w)
    });

    let dt = 0.01_f32;
    let mut red_frames = 0usize;
    let mut green_frames = 0usize;
    let mut total_magnitude = 0.0f32;
    let mut max_hv_norm = 0.0f32;

    for i in 0..steps {
        let t = i as f32 / steps.max(1) as f32;
        let phi = 0.05 + 0.9 * t; // 0.05 → 0.95 linearly
        // Synthetic thought HV — deterministic seed per step so users can
        // replicate. A real cognitive loop emits this from perception +
        // prediction + attention.
        let hv = ContinuousHV::random(16384, 1000 + i as u64);
        let hv_norm = hv.norm();
        let hv_meanabs =
            hv.as_slice().iter().map(|x| x.abs()).sum::<f32>() / hv.as_slice().len().max(1) as f32;

        // THE projection — 16,384D → 4-DOF motor command.
        let cmd: QuadrotorCommand = controller.forward(&hv, dt);

        // Phi envelope layer.
        let safety = MotorSafetyLevel::from_phi(phi as f64);
        let gain = safety.motor_gain();
        let scaled = QuadrotorCommand {
            thrust: cmd.thrust * gain,
            roll_moment: cmd.roll_moment * gain,
            pitch_moment: cmd.pitch_moment * gain,
            yaw_moment: cmd.yaw_moment * gain,
        };

        let magnitude = cmd.thrust.abs()
            + cmd.roll_moment.abs()
            + cmd.pitch_moment.abs()
            + cmd.yaw_moment.abs();
        total_magnitude += magnitude;
        if max_hv_norm < hv_norm {
            max_hv_norm = hv_norm;
        }
        match safety {
            MotorSafetyLevel::Red => red_frames += 1,
            MotorSafetyLevel::Green => green_frames += 1,
            _ => {}
        }

        if let Some(w) = csv_file.as_mut() {
            // Note: we don't expose raw_* (pre-activation) here because
            // FlightController's forward() returns only the post-activation
            // command. The raw values are an implementation detail locked
            // behind the `train_step` path.
            writeln!(
                w,
                "{},{:.3},{:?},{:.3},{:.3},{:.5},,,,,{:.3},{:.4},{:.4},{:.4},{:.3},{:.4},{:.4},{:.4}",
                i,
                phi,
                safety,
                gain,
                hv_norm,
                hv_meanabs,
                cmd.thrust,
                cmd.roll_moment,
                cmd.pitch_moment,
                cmd.yaw_moment,
                scaled.thrust,
                scaled.roll_moment,
                scaled.pitch_moment,
                scaled.yaw_moment,
            )
            .ok();
        }

        let sample_every = (steps / 16).max(1);
        if i % sample_every == 0 || i + 1 == steps {
            print_frame(i, phi, safety, gain, hv_norm, &cmd, &scaled, use_color);
        }
    }

    summary(
        steps,
        green_frames,
        red_frames,
        total_magnitude / steps.max(1) as f32,
        max_hv_norm,
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
    println!("{c} From thought to torque — HDC-LTC → 4-DOF quadrotor command  {r}");
    println!("{c}──────────────────────────────────────────────────────────────{r}");
    println!(" 16,384D thought_hv  ─▶  evolved+projected  ─▶  [thrust, roll, pitch, yaw]");
    println!(" phi envelope        ─▶  MotorSafetyLevel   ─▶  × motor_gain (Green=1.0 → Red=0.0)");
    println!();
    println!(
        "{:>4} {:>5} {:>7} {:>5} {:>9} {:>7} {:>7} {:>7} {:>7}  → scaled",
        "step", "phi", "tier", "gain", "hv_norm", "thr_N", "roll", "pitch", "yaw"
    );
    println!(
        "{:>4} {:>5} {:>7} {:>5} {:>9} {:>7} {:>7} {:>7} {:>7}",
        "----",
        "-----",
        "-------",
        "-----",
        "---------",
        "-------",
        "-------",
        "-------",
        "-------"
    );
}

#[allow(clippy::too_many_arguments)]
fn print_frame(
    i: usize,
    phi: f32,
    tier: MotorSafetyLevel,
    gain: f32,
    hv_norm: f32,
    cmd: &QuadrotorCommand,
    scaled: &QuadrotorCommand,
    use_color: bool,
) {
    let (tier_str, tier_color) = tier_style(tier, use_color);
    let (_, reset) = color_codes(use_color);
    println!(
        "{:>4} {:>5.2} {}{:>7}{} {:>5.2} {:>9.2} {:>7.3} {:>7.4} {:>7.4} {:>7.4}  →  thr={:.3} mom=({:.4},{:.4},{:.4})",
        i,
        phi,
        tier_color,
        tier_str,
        reset,
        gain,
        hv_norm,
        cmd.thrust,
        cmd.roll_moment,
        cmd.pitch_moment,
        cmd.yaw_moment,
        scaled.thrust,
        scaled.roll_moment,
        scaled.pitch_moment,
        scaled.yaw_moment,
    );
}

fn summary(
    steps: usize,
    green_frames: usize,
    red_frames: usize,
    mean_magnitude: f32,
    max_hv_norm: f32,
    use_color: bool,
) {
    let (c, r) = color_codes(use_color);
    let green_pct = 100.0 * green_frames as f32 / steps.max(1) as f32;
    let red_pct = 100.0 * red_frames as f32 / steps.max(1) as f32;
    println!();
    println!("{c}──────────────────────────────── summary ──────────────────────────────{r}");
    println!(" steps:            {}", steps);
    println!(" green frames:     {} ({:.1} %)", green_frames, green_pct);
    println!(" red frames:       {} ({:.1} %)", red_frames, red_pct);
    println!(" max  hv_norm:     {:.3}", max_hv_norm);
    println!(" mean |cmd|:       {:.4}", mean_magnitude);
    println!(" dim reduction:    16384D  →  4D   (ratio {}×)", 16384 / 4);
    println!();
    println!(
        " input bar (16384D, log-scaled): {}",
        ascii_bar(16384, use_color)
    );
    println!(
        " output bar (4D,    log-scaled): {}",
        ascii_bar(4, use_color)
    );
    println!();
}

fn ascii_bar(width: usize, use_color: bool) -> String {
    let (c, r) = color_codes(use_color);
    let log_scale = (width as f32).log2().max(1.0);
    let filled = (log_scale as usize).clamp(1, BAR_WIDTH);
    let bar = "█".repeat(filled);
    format!("{c}{bar}{r}  ({width}D)")
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
