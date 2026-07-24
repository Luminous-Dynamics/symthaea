// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Demonstrates the mathematical content of the weak equivalence principle:
//! gravitational trajectories don't depend on the test particle's mass, while
//! trajectories under an ordinary (non-gravitational) force do.
//!
//! Run: `cargo run -p symthaea-gravcraft --example equivalence_principle_check`

use symthaea_gravcraft::equivalence_principle::{
    EOT_WASH_ETA_BOUND, MICROSCOPE_ETA_BOUND, simulate_external_force_trajectory,
    simulate_gravity_trajectory, trajectory_divergence,
};

fn main() {
    let pos = [7_000_000.0, 0.0, 0.0];
    let vel = [0.0, 7_500.0, 0.0];
    let earth_mass = 5.972e24;

    println!("--- Gravity: does trajectory depend on test-particle mass? ---");
    let masses = [1.0, 1_000.0, 1.0e6, 1.0e12];
    let reference = simulate_gravity_trajectory(pos, vel, earth_mass, masses[0], 2000, 1.0);
    for &m in &masses[1..] {
        let traj = simulate_gravity_trajectory(pos, vel, earth_mass, m, 2000, 1.0);
        let divergence = trajectory_divergence(&reference, &traj);
        println!(
            "  mass = {:>10.1e} kg vs {:.1e} kg -> divergence = {:.3e} m",
            masses[0], m, divergence
        );
    }
    println!("  (all divergences should be at the numerical-integration noise floor)\n");

    println!("--- Contrast: constant external force (non-gravitational) ---");
    let force = [1.0, 0.0, 0.0];
    let reference =
        simulate_external_force_trajectory([0.0; 3], [0.0; 3], force, masses[0], 2000, 1.0);
    for &m in &masses[1..] {
        let traj = simulate_external_force_trajectory([0.0; 3], [0.0; 3], force, m, 2000, 1.0);
        let divergence = trajectory_divergence(&reference, &traj);
        println!(
            "  mass = {:>10.1e} kg vs {:.1e} kg -> divergence = {:.3e} m",
            masses[0], m, divergence
        );
    }
    println!("  (these SHOULD diverge substantially — that's the control)\n");

    println!("--- What real experiments actually bound (not this simulation) ---");
    println!(
        "  Eöt-Wash 2008 (ground, Be-Ti):     eta ~ {:.2e}",
        EOT_WASH_ETA_BOUND
    );
    println!(
        "  MICROSCOPE 2022 (satellite, Ti-Pt): eta ~ {:.2e}",
        MICROSCOPE_ETA_BOUND
    );
    println!(
        "  Improvement factor: {:.1}x",
        EOT_WASH_ETA_BOUND / MICROSCOPE_ETA_BOUND
    );
}
