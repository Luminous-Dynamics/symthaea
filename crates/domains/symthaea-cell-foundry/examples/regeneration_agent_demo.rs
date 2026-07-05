// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Regeneration Agent Demo
//!
//! Demonstrates the opt-in active-inference regeneration agent
//! (`NeuralOrganoid::set_fep_regeneration_enabled`, see
//! `crate::regeneration_agent`) against the legacy flat-rate rule it can
//! replace. Runs the same amputation-recovery trajectory twice (legacy vs.
//! FEP-enabled) and reports real discrepancy-vs-day trajectories for both.
//!
//! Unlike this crate's other demos (`packing_demo`, `ion_channel_demo`,
//! `anatomical_compiler_demo`), this one does **not** assert a winner.
//! Phases 1-3's improvements were guaranteed by construction (real physics
//! separates overlapping spheres; real conductances pull toward real
//! reversal potentials). Whether a freshly-initialized, briefly-trained
//! active-inference agent measurably beats a flat 25%/day proliferation
//! rate within one short regeneration episode is a genuinely open empirical
//! question -- this demo reports the real numbers honestly, whichever way
//! they land.
//!
//! Run: cargo run -p symthaea-cell-foundry --example regeneration_agent_demo

use symthaea_cell_foundry::morphogenetic_consciousness::NeuralOrganoid;

/// Matches `crate::bioelectric::MORPHOLOGY_CONVERGENCE_TOLERANCE` (a
/// private constant) -- the RMS discrepancy below which a wound is
/// considered healed.
const CONVERGENCE_TOLERANCE: f64 = 0.05;

fn run(
    seed: u64,
    cells: usize,
    maturation_days: u32,
    recovery_days: u32,
    fep_enabled: bool,
) -> Vec<(u32, f64)> {
    let mut organoid = NeuralOrganoid::new(cells, seed);
    for _ in 0..maturation_days {
        organoid.advance_day();
    }
    organoid.capture_target_morphology();
    organoid.amputate(0.6, 2.0);
    organoid.set_fep_regeneration_enabled(fep_enabled);

    let mut trace = vec![(0u32, organoid.morphology_discrepancy().unwrap_or(1.0))];
    for day in 1..=recovery_days {
        organoid.advance_day();
        trace.push((day, organoid.morphology_discrepancy().unwrap_or(1.0)));
    }
    trace
}

fn day_converged(trace: &[(u32, f64)]) -> Option<u32> {
    trace
        .iter()
        .find(|(_, d)| *d <= CONVERGENCE_TOLERANCE)
        .map(|(day, _)| *day)
}

fn print_trace(label: &str, trace: &[(u32, f64)], sample_every: u32) {
    println!("  {label}:");
    for &(day, d) in trace {
        if day % sample_every == 0 {
            println!("    day {day:3}: discrepancy = {d:.4}");
        }
    }
    match day_converged(trace) {
        Some(day) => {
            println!("  -> converged (discrepancy <= {CONVERGENCE_TOLERANCE}) at day {day}")
        }
        None => println!(
            "  -> did not converge within {} days (final discrepancy = {:.4})",
            trace.last().unwrap().0,
            trace.last().unwrap().1
        ),
    }
}

fn main() {
    let seed = 7;
    let cells = 150;
    let maturation_days = 20;
    let recovery_days = 60;

    println!(
        "Running the same amputation-recovery trajectory twice ({cells} cells, seed={seed}, \
         {maturation_days}d maturation, {recovery_days}d recovery)...\n"
    );

    println!("Legacy dynamics (flat 25%/day proliferation-boost rate):");
    let legacy = run(seed, cells, maturation_days, recovery_days, false);
    print_trace("legacy", &legacy, 5);
    println!();

    println!("FEP-driven dynamics (active-inference-selected proliferation-boost multiplier):");
    let fep = run(seed, cells, maturation_days, recovery_days, true);
    print_trace("fep_driven", &fep, 5);
    println!();

    let legacy_final = legacy.last().unwrap().1;
    let fep_final = fep.last().unwrap().1;
    println!(
        "Final discrepancy: legacy={legacy_final:.4}, fep_driven={fep_final:.4} \
         ({})",
        if fep_final < legacy_final {
            "FEP-driven regeneration reached a lower final discrepancy"
        } else if fep_final > legacy_final {
            "legacy flat-rate regeneration reached a lower final discrepancy"
        } else {
            "no meaningful difference"
        }
    );
    println!(
        "This is reported as an honest empirical result, not a predetermined \
         conclusion -- see module docs for why."
    );
}
