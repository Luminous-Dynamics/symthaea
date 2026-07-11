// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cognition-ablation experiment: does the thought hypervector carry any
//! task-relevant signal through the (untrained, genesis-random) decode path?
//!
//! Five conditions, identical episodes (phi held at 0.8 = Green, constant HV):
//!   zero      — all-zero thought vector (pure decoder-bias behavior)
//!   random_a  — seeded random HV (seed 42)
//!   random_b  — seeded random HV (seed 1337)
//!   intent_a  — genesis-structured HV "intent::reach_left"
//!   intent_b  — genesis-structured HV "intent::reach_right"
//!
//! If divergence(intent_a, intent_b) ≈ divergence(random_a, random_b) and both
//! are tiny relative to the overall motion scale, the thought vector carries no
//! task-relevant signal (cognition ≈ bias through an untrained projection).
//!
//! Run: cargo run -p symthaea-manipulator --example cognition_ablation --release

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::ContinuousHV;
use symthaea_manipulator::controller::ManipulatorController;
use symthaea_manipulator::embodiment::ManipulatorEmbodiment;
use symthaea_manipulator::kinematics::ManipulatorKinematics;
use symthaea_manipulator::types::ManipulatorConfig;

const DIM: usize = 16384;
const STEPS: usize = 2000;
const DT: f32 = 0.002;
const PHI: f64 = 0.8; // constant Green — motor gain 1.0, no fallback
const TARGET: [f64; 3] = [0.4, 0.2, 0.4]; // nominal reach target
const GENESIS_PHRASE: &str = "cognition-ablation";

struct EpisodeRecord {
    name: &'static str,
    /// End-effector position per step (STEPS + 1 samples, includes initial pose).
    ee_trajectory: Vec<[f64; 3]>,
    mean_control_effort: f64,
}

fn ee_from_telemetry(bridge: &ManipulatorEmbodiment, kin: &ManipulatorKinematics) -> [f64; 3] {
    let bytes = bridge.platform_telemetry_bytes();
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("telemetry JSON should parse");
    let q: Vec<f64> = v["joint_angles"]
        .as_array()
        .expect("joint_angles array")
        .iter()
        .map(|x| x.as_f64().expect("joint angle f64"))
        .collect();
    kin.end_effector_position(&q)
}

fn dist(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn run_episode(
    name: &'static str,
    hv: &ContinuousHV,
    kin: &ManipulatorKinematics,
) -> EpisodeRecord {
    let genesis = GenesisSeed::from_phrase(GENESIS_PHRASE);
    let mut bridge = ManipulatorEmbodiment::new(&genesis);
    let mut ee_trajectory = Vec::with_capacity(STEPS + 1);
    ee_trajectory.push(ee_from_telemetry(&bridge, kin));
    let mut effort_sum = 0.0_f64;
    for _ in 0..STEPS {
        let r = bridge.step(hv, DT, PHI);
        assert!(r.success, "simulation diverged in condition {name}");
        effort_sum += r.control_effort as f64;
        ee_trajectory.push(ee_from_telemetry(&bridge, kin));
    }
    EpisodeRecord {
        name,
        ee_trajectory,
        mean_control_effort: effort_sum / STEPS as f64,
    }
}

/// RMS pointwise distance between two equal-length EE trajectories.
fn trajectory_divergence(a: &EpisodeRecord, b: &EpisodeRecord) -> f64 {
    let n = a.ee_trajectory.len();
    let sum_sq: f64 = a
        .ee_trajectory
        .iter()
        .zip(&b.ee_trajectory)
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
            ContinuousHV::from_genesis(&genesis, "intent::reach_left", DIM),
        ),
        (
            "intent_b",
            ContinuousHV::from_genesis(&genesis, "intent::reach_right", DIM),
        ),
    ];

    println!("=== Manipulator cognition ablation ===");
    println!(
        "episode: {STEPS} steps @ dt={DT} (phi={PHI} Green, constant thought HV, genesis \"{GENESIS_PHRASE}\")"
    );
    println!();

    // ── Single-step command-space sensitivity (decoder isolated from sim) ──
    // Fresh controller per condition (identical genesis → identical initial
    // LTC state); one forward() from the same state.
    println!("-- Single-step motor command per condition (fresh controller, same state) --");
    let config = ManipulatorConfig::default();
    let mut first_commands: Vec<(&'static str, Vec<f32>)> = Vec::new();
    for (name, hv) in &conditions {
        let mut ctrl = ManipulatorController::new(&genesis, &config);
        let cmd = ctrl.forward(hv, DT);
        let mut v: Vec<f32> = cmd.joint_torques.to_vec();
        v.push(cmd.gripper);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!(
            "  {name:<9} |cmd|={norm:.6}  torques={:?}",
            cmd.joint_torques.map(|t| (t * 1e4).round() / 1e4)
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
    let kin = ManipulatorKinematics::default();
    let records: Vec<EpisodeRecord> = conditions
        .iter()
        .map(|(name, hv)| run_episode(name, hv, &kin))
        .collect();

    println!("-- Per-condition episode metrics --");
    println!(
        "  {:<9} {:>34} {:>12} {:>12} {:>12}",
        "condition", "final EE position [m]", "displ [m]", "d(target)", "mean|cmd|"
    );
    let start = records[0].ee_trajectory[0];
    for r in &records {
        let final_ee = *r.ee_trajectory.last().unwrap();
        let mean_displacement: f64 = r.ee_trajectory.iter().map(|p| dist(p, &start)).sum::<f64>()
            / r.ee_trajectory.len() as f64;
        let d_target = dist(&final_ee, &TARGET);
        println!(
            "  {:<9} [{:>9.5}, {:>9.5}, {:>9.5}] {:>12.6} {:>12.6} {:>12.6}",
            r.name,
            final_ee[0],
            final_ee[1],
            final_ee[2],
            mean_displacement,
            d_target,
            r.mean_control_effort
        );
    }
    println!();

    println!("-- Pairwise EE-trajectory divergence (RMS, m) --");
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
    // Motion scale: mean EE displacement from the initial pose across all conditions.
    let motion_scale: f64 = records
        .iter()
        .map(|r| {
            r.ee_trajectory
                .iter()
                .map(|p| dist(p, &start))
                .fold(0.0_f64, f64::max)
        })
        .fold(0.0_f64, f64::max);
    let d_intents = div[3][4]; // intent_a vs intent_b
    let d_randoms = div[1][2]; // random_a vs random_b
    let d_zero_rand = div[0][1]; // zero vs random_a
    println!("-- Verdict --");
    println!("  max EE displacement from start (motion scale): {motion_scale:.4} m");
    println!(
        "  divergence(intent_a, intent_b):  {d_intents:.3e} m  (ratio to scale: {:.3e})",
        d_intents / motion_scale.max(1e-12)
    );
    println!(
        "  divergence(random_a, random_b):  {d_randoms:.3e} m  (ratio to scale: {:.3e})",
        d_randoms / motion_scale.max(1e-12)
    );
    println!(
        "  divergence(zero, random_a):      {d_zero_rand:.3e} m  (ratio to scale: {:.3e})",
        d_zero_rand / motion_scale.max(1e-12)
    );
    let intent_vs_random = d_intents / d_randoms.max(1e-12);
    println!("  intent-divergence / random-divergence: {intent_vs_random:.3}");

    // Task-axis separation: "reach_left" vs "reach_right" is a claim about the
    // lateral (y) direction of motion. Raw trajectory divergence conflates
    // chaotic/state-drift amplification of microscopic command differences with
    // genuine semantic control. Only separation along the task axis, exceeding
    // the random-seed baseline, counts as task-relevant signal.
    let final_y = |r: &EpisodeRecord| -> f64 { r.ee_trajectory.last().unwrap()[1] };
    let task_sep_intent = final_y(&records[3]) - final_y(&records[4]); // left − right
    let task_sep_random = (final_y(&records[1]) - final_y(&records[2])).abs();
    println!(
        "  task-axis separation (final EE y: intent_left − intent_right): {task_sep_intent:+.6} m \
         (random-pair baseline: {task_sep_random:.6} m)"
    );
    let exceeds_noise = task_sep_intent.abs() > 3.0 * task_sep_random.max(1e-6);
    if exceeds_noise {
        println!(
            "  → TASK-RELEVANT SIGNAL: opposite reach intents separate laterally beyond \
             the random-seed baseline. The decoder extracts semantic content."
        );
    } else if d_intents > 0.05 * motion_scale.max(1e-12) {
        println!(
            "  → CHAOS-ONLY SENSITIVITY (null on the task axis): trajectories differ \
             because microscopic command differences are amplified by the dynamics, but \
             opposite intents produce no systematic lateral separation. The thought \
             vector acts as a random perturbation seed, not a semantic control signal — \
             consistent with an untrained genesis-random decoder."
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
