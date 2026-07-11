// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cognition-ablation experiment: does the thought hypervector carry any
//! task-relevant signal through the (untrained, genesis-random) decode path
//! on the multirotor bridge?
//!
//! Five conditions, identical episodes (phi held at 0.8 = Green, constant HV):
//!   zero      — all-zero thought vector (pure decoder-bias behavior; note the
//!               controller's output_bias[0] = HOVER_THRUST pre-sigmoid, so
//!               the expected bias behavior is a steady climb, not a hover)
//!   random_a  — seeded random HV (seed 42)
//!   random_b  — seeded random HV (seed 1337)
//!   intent_a  — genesis-structured HV "intent::climb"
//!   intent_b  — genesis-structured HV "intent::descend"
//!
//! If divergence(intent_a, intent_b) ≈ divergence(random_a, random_b) and both
//! are tiny relative to the overall motion scale, the thought vector carries no
//! task-relevant signal (cognition ≈ bias through an untrained projection).
//!
//! Run: cargo run -p symthaea-multirotor --example cognition_ablation --release

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_multirotor::FlightController;
use symthaea_multirotor::embodiment::FlightEmbodiment;
use symthaea_multirotor::simulator::PhysicsSimulator;
use symthaea_multirotor::types::{FlightConfig, FlightSetpoint};

const DIM: usize = 16384;
const STEPS: usize = 2000;
const DT: f32 = 0.002;
const PHI: f64 = 0.8; // constant Green — motor gain 1.0, no fallback
const GENESIS_PHRASE: &str = "cognition-ablation";

struct EpisodeRecord {
    name: &'static str,
    /// Position per step (STEPS + 1 samples, includes initial pose).
    trajectory: Vec<[f64; 3]>,
    mean_position_error: f64,
    mean_control_effort: f64,
}

fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn run_episode(name: &'static str, hv: &ContinuousHV) -> EpisodeRecord {
    let genesis = GenesisSeed::from_phrase(GENESIS_PHRASE);
    let mut bridge = FlightEmbodiment::new(&genesis);
    bridge.reset(); // deterministic start at hover altitude 0.1 m
    let setpoint = FlightSetpoint::hover();
    let mut trajectory = Vec::with_capacity(STEPS + 1);
    trajectory.push(bridge.simulator().state().position);
    let mut err_sum = 0.0_f64;
    let mut effort_sum = 0.0_f64;
    for _ in 0..STEPS {
        let r = bridge.step(hv, DT, PHI);
        assert!(r.success, "simulation diverged in condition {name}");
        effort_sum += r.control_effort as f64;
        let state = bridge.simulator().state();
        trajectory.push(state.position);
        err_sum += setpoint.position_error_magnitude(state);
    }
    EpisodeRecord {
        name,
        trajectory,
        mean_position_error: err_sum / STEPS as f64,
        mean_control_effort: effort_sum / STEPS as f64,
    }
}

/// RMS pointwise distance between two equal-length trajectories.
fn trajectory_divergence(a: &EpisodeRecord, b: &EpisodeRecord) -> f64 {
    let n = a.trajectory.len();
    let sum_sq: f64 = a
        .trajectory
        .iter()
        .zip(&b.trajectory)
        .map(|(p, q)| dist(p, q).powi(2))
        .sum();
    (sum_sq / n as f64).sqrt()
}

fn main() {
    let genesis = GenesisSeed::from_phrase(GENESIS_PHRASE);
    let conditions: Vec<(&'static str, ContinuousHV)> = vec![
        ("zero", ContinuousHV::zero(DIM)),
        ("random_a", ContinuousHV::random(DIM, 42)),
        ("random_b", ContinuousHV::random(DIM, 1337)),
        (
            "intent_a",
            ContinuousHV::from_genesis(&genesis, "intent::climb", DIM),
        ),
        (
            "intent_b",
            ContinuousHV::from_genesis(&genesis, "intent::descend", DIM),
        ),
    ];

    println!("=== Multirotor cognition ablation ===");
    println!(
        "episode: {STEPS} steps @ dt={DT} (phi={PHI} Green, constant thought HV, genesis \"{GENESIS_PHRASE}\")"
    );
    println!();

    // ── Single-step command-space sensitivity (decoder isolated from sim) ──
    println!("-- Single-step motor command per condition (fresh controller, same state) --");
    let config = FlightConfig::default();
    let mut first_commands: Vec<(&'static str, [f32; 4])> = Vec::new();
    for (name, hv) in &conditions {
        let mut ctrl = FlightController::new(&genesis, &config);
        let cmd = ctrl.forward(hv, DT);
        let v = [
            cmd.thrust,
            cmd.roll_moment,
            cmd.pitch_moment,
            cmd.yaw_moment,
        ];
        println!(
            "  {name:<9} thrust={:.6} N  moments=[{:.2e}, {:.2e}, {:.2e}]",
            v[0], v[1], v[2], v[3]
        );
        first_commands.push((name, v));
    }
    println!();
    println!("-- Pairwise command L2 distance (single step, same state) --");
    for i in 0..first_commands.len() {
        for j in (i + 1)..first_commands.len() {
            let (na, va) = &first_commands[i];
            let (nb, vb) = &first_commands[j];
            let d: f32 = va
                .iter()
                .zip(vb.iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>()
                .sqrt();
            println!("  {na:<9} vs {nb:<9}: {d:.6}");
        }
    }
    println!();

    // ── Full closed-loop episodes ──
    let records: Vec<EpisodeRecord> = conditions
        .iter()
        .map(|(name, hv)| run_episode(name, hv))
        .collect();

    println!("-- Per-condition episode metrics (setpoint: hover) --");
    println!(
        "  {:<9} {:>34} {:>12} {:>12} {:>12}",
        "condition", "final position [m]", "mean err", "net climb", "mean|cmd|"
    );
    for r in &records {
        let start = r.trajectory[0];
        let final_pos = *r.trajectory.last().unwrap();
        println!(
            "  {:<9} [{:>9.4}, {:>9.4}, {:>9.4}] {:>12.5} {:>12.5} {:>12.6}",
            r.name,
            final_pos[0],
            final_pos[1],
            final_pos[2],
            r.mean_position_error,
            final_pos[2] - start[2],
            r.mean_control_effort
        );
    }
    println!();

    println!("-- Pairwise trajectory divergence (RMS, m) --");
    let mut div = vec![vec![0.0_f64; records.len()]; records.len()];
    for i in 0..records.len() {
        for j in (i + 1)..records.len() {
            let d = trajectory_divergence(&records[i], &records[j]);
            div[i][j] = d;
            println!(
                "  {:<9} vs {:<9}: {:.3e}",
                records[i].name, records[j].name, d
            );
        }
    }
    println!();

    // ── Verdict ──
    let start = records[0].trajectory[0];
    let motion_scale: f64 = records
        .iter()
        .map(|r| {
            r.trajectory
                .iter()
                .map(|p| dist(p, &start))
                .fold(0.0_f64, f64::max)
        })
        .fold(0.0_f64, f64::max);
    let d_intents = div[3][4];
    let d_randoms = div[1][2];
    let d_zero_rand = div[0][1];
    println!("-- Verdict --");
    println!("  max displacement from start (motion scale): {motion_scale:.4} m");
    println!(
        "  divergence(intent_a=climb, intent_b=descend): {d_intents:.3e} m  (ratio to scale: {:.3e})",
        d_intents / motion_scale.max(1e-12)
    );
    println!(
        "  divergence(random_a, random_b):               {d_randoms:.3e} m  (ratio to scale: {:.3e})",
        d_randoms / motion_scale.max(1e-12)
    );
    println!(
        "  divergence(zero, random_a):                   {d_zero_rand:.3e} m  (ratio to scale: {:.3e})",
        d_zero_rand / motion_scale.max(1e-12)
    );
    let intent_vs_random = d_intents / d_randoms.max(1e-12);
    println!("  intent-divergence / random-divergence: {intent_vs_random:.3}");

    // Task-axis separation: "climb" vs "descend" is a claim about net climb.
    // Raw trajectory divergence conflates chaotic amplification of microscopic
    // command differences (butterfly effect in an open-loop unstable airframe)
    // with genuine semantic control. Only separation along the *task axis*,
    // exceeding what two random seeds produce, counts as task-relevant signal.
    let net_climb =
        |r: &EpisodeRecord| -> f64 { r.trajectory.last().unwrap()[2] - r.trajectory[0][2] };
    let task_sep_intent = net_climb(&records[3]) - net_climb(&records[4]); // climb − descend
    let task_sep_random = (net_climb(&records[1]) - net_climb(&records[2])).abs();
    println!(
        "  task-axis separation (net climb: intent_a − intent_b): {task_sep_intent:+.4} m \
         (random-pair baseline: {task_sep_random:.4} m)"
    );
    let semantically_correct = task_sep_intent > 0.0; // "climb" should out-climb "descend"
    let exceeds_noise = task_sep_intent.abs() > 3.0 * task_sep_random.max(1e-3);
    if semantically_correct && exceeds_noise {
        println!(
            "  → TASK-RELEVANT SIGNAL: the climb intent out-climbs the descend intent \
             beyond the random-seed baseline. The decoder extracts semantic content."
        );
    } else if d_intents > 0.05 * motion_scale.max(1e-12) {
        println!(
            "  → CHAOS-ONLY SENSITIVITY (null on the task axis): trajectories diverge \
             because microscopic command differences (see single-step L2 ~1e-4) are \
             amplified by unstable open-loop dynamics, but opposite intents produce no \
             systematic difference in the commanded direction. The thought vector acts \
             as a random perturbation seed, not a semantic control signal — consistent \
             with an untrained genesis-random decoder (cognition ≈ noise, not intent)."
        );
    } else {
        println!(
            "  → NULL RESULT: opposite structured intents produce trajectories no more \
             different than two random seeds, and the difference is small relative to \
             overall motion. The thought vector carries no task-relevant signal through \
             the untrained decoder (cognition ≈ bias)."
        );
    }
}
