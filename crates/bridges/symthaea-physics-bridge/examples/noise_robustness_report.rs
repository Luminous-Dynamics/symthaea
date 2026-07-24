// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Prints the full noise-robustness sweep table: at each measurement-noise
//! level, does the true invariant still score better than plausible decoys
//! -- with and without a smoothing front end -- for two showcase systems.
//!
//! Run: `cargo run -p symthaea-physics-bridge --example noise_robustness_report`

use symthaea_physics_bridge::noise_robustness::systems::{
    harmonic_decoys, harmonic_rhs, harmonic_trajectory, harmonic_truth, kepler_decoys, kepler_rhs,
    kepler_trajectory, kepler_truth,
};
use symthaea_physics_bridge::noise_robustness::{SweepRow, noise_sweep};

const NOISE_LEVELS: &[f64] = &[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5];

fn print_table(system: &str, raw: &[SweepRow], smoothed: &[SweepRow]) {
    println!("\n=== {system} ===");
    println!(
        "{:>10} | {:>8} {:>14} {:>14} | {:>8} {:>14} {:>14}",
        "noise", "raw rank", "truth var", "best decoy", "sm rank", "truth var", "best decoy"
    );
    for (r, s) in raw.iter().zip(smoothed.iter()) {
        println!(
            "{:>10.3} | {:>8} {:>14.3e} {:>14.3e} | {:>8} {:>14.3e} {:>14.3e}",
            r.noise_level,
            r.rank,
            r.truth_variance,
            r.best_decoy_variance,
            s.rank,
            s.truth_variance,
            s.best_decoy_variance
        );
    }
}

fn main() {
    let seed = 42;

    let harmonic_traj = harmonic_trajectory(1000, 0.01);
    let harmonic_truth_expr = harmonic_truth();
    let harmonic_decoy_exprs = harmonic_decoys();
    let raw = noise_sweep(
        ("x^2+v^2", &harmonic_truth_expr),
        &harmonic_decoy_exprs,
        harmonic_rhs,
        &harmonic_traj,
        &["x", "v"],
        NOISE_LEVELS,
        None,
        seed,
    );
    let smoothed = noise_sweep(
        ("x^2+v^2", &harmonic_truth_expr),
        &harmonic_decoy_exprs,
        harmonic_rhs,
        &harmonic_traj,
        &["x", "v"],
        NOISE_LEVELS,
        Some(5),
        seed,
    );
    print_table("Harmonic Oscillator (truth: x^2 + v^2)", &raw, &smoothed);

    let kepler_traj = kepler_trajectory(2000, 0.001);
    let kepler_truth_expr = kepler_truth();
    let kepler_decoy_exprs = kepler_decoys();
    let raw = noise_sweep(
        ("E", &kepler_truth_expr),
        &kepler_decoy_exprs,
        kepler_rhs,
        &kepler_traj,
        &["x", "y", "vx", "vy"],
        NOISE_LEVELS,
        None,
        seed,
    );
    let smoothed = noise_sweep(
        ("E", &kepler_truth_expr),
        &kepler_decoy_exprs,
        kepler_rhs,
        &kepler_traj,
        &["x", "y", "vx", "vy"],
        NOISE_LEVELS,
        Some(5),
        seed,
    );
    print_table(
        "Kepler Two-Body (truth: E = 1/2 v^2 - 1/r)",
        &raw,
        &smoothed,
    );

    println!(
        "\nrank=1 means the true invariant still scored best among the decoys at that noise level."
    );
    println!("This tests the fitness function's discrimination power under noise -- a necessary");
    println!("condition for the full GP search to succeed, not a full end-to-end noisy-GP run.");
}
