// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Basal Cognition / Equifinality Experiments
//!
//! Runs the equifinality test suite from `symthaea_cell_foundry::experiments`
//! against a single captured target morphology: three amputations of
//! different sizes (all leaving surviving template tissue) plus one
//! whole-organoid Vmem scramble (which destroys the surviving template
//! entirely — an honest boundary case, not expected to recover under this
//! model; see `crate::experiments` module docs for why).
//!
//! Prints a results table and writes full trajectories to
//! `basal_cognition_results.json` next to the binary's working directory,
//! for downstream analysis/visualization.
//!
//! Run: cargo run -p symthaea-cell-foundry --example basal_cognition_experiments

use symthaea_cell_foundry::experiments::{
    Perturbation, build_radial_bipolar_template, run_equifinality_experiment,
};

fn main() {
    println!("======================================================");
    println!("   Basal Cognition / Equifinality Experiments");
    println!("======================================================");
    println!("  Levin's signature test of collective intelligence:");
    println!("  does damaged tissue recover the SAME target pattern");
    println!("  regardless of how it was damaged -- and does that");
    println!("  recovery specifically require open gap junctions?");
    println!("======================================================");
    println!();

    let seed = 11;
    let cells = 200;
    let maturation_days = 20;
    let boundary_r = 0.2;
    let recovery_days = 40;

    println!(
        "Building template: {cells} cells, seed={seed}, {maturation_days}d maturation, \
         bipolar boundary at r={boundary_r} (outer=hyperpolarized, inner=depolarized)..."
    );
    let template = build_radial_bipolar_template(seed, cells, maturation_days, boundary_r);
    println!("  template cells: {}", template.field.num_cells());
    println!();

    let perturbations = [
        Perturbation::Amputate {
            min_r: 1.1,
            max_r: 2.0,
        },
        Perturbation::Amputate {
            min_r: 0.95,
            max_r: 2.0,
        },
        Perturbation::Amputate {
            min_r: 0.8,
            max_r: 2.0,
        },
        Perturbation::ScrambleVmem { seed: 999 },
    ];

    println!(
        "Running {} perturbations x 2 permeability conditions x {recovery_days} recovery days...",
        perturbations.len()
    );
    let result = run_equifinality_experiment(&template, &perturbations, recovery_days, false);
    println!();

    println!("Perturbation                | Permeability | Final Discrepancy");
    println!("-----------------------------|--------------|-------------------");
    for c in &result.conditions {
        println!(
            "{:29}| {:12} | {:.4}",
            c.perturbation_label, c.gap_junction_permeability, c.final_discrepancy
        );
    }

    println!();
    println!("=== Equifinality Metrics ===");
    let (open_mean, blocked_mean) = result.mean_final_by_permeability();
    println!("Mean final discrepancy (open):    {open_mean:.4}");
    println!("Mean final discrepancy (blocked): {blocked_mean:.4}");
    println!(
        "Open beats blocked:               {}",
        result.open_beats_blocked()
    );
    println!(
        "Open-run spread (max-min):        {:.4}  (equifinality: same target, different damage)",
        result.open_run_spread()
    );
    println!(
        "Blocked-run spread (max-min):     {:.4}",
        result.blocked_run_spread()
    );

    println!();
    println!("=== Note on the scramble_vmem condition (no homing) ===");
    println!("Scrambling destroys the ENTIRE surviving template, so recovery via");
    println!("neighbour-propagation alone (which reads surviving neighbours' Vmem, not an");
    println!("independent per-cell positional identity) is not expected -- open and");
    println!("blocked should look similar there. See experiments.rs module docs.");

    let json = serde_json::to_string_pretty(&result).expect("serialize result");
    std::fs::write("basal_cognition_results.json", &json).expect("write results json");
    println!();
    println!(
        "Full trajectories written to basal_cognition_results.json ({} bytes)",
        json.len()
    );

    // ---- Positional homing: does a second, non-electrical memory channel
    // rescue recovery from total scramble? ----
    println!();
    println!("======================================================");
    println!("   Positional Homing: rescuing total-scramble recovery");
    println!("======================================================");
    println!("  Same scramble as above, but now comparing WITHOUT vs. WITH");
    println!("  positional homing (a per-cell pull toward the target's value");
    println!("  at that cell's own position -- NOT gated by gap junctions).");
    println!();

    let scramble = [Perturbation::ScrambleVmem { seed: 555 }];
    let without_homing = run_equifinality_experiment(&template, &scramble, recovery_days, false);
    let with_homing = run_equifinality_experiment(&template, &scramble, recovery_days, true);

    println!("Condition                        | Final Discrepancy");
    println!("----------------------------------|-------------------");
    for (label, result) in [
        ("no homing, open", &without_homing),
        ("homing, open", &with_homing),
    ] {
        let c = result
            .conditions
            .iter()
            .find(|c| c.gap_junction_permeability > 0.0)
            .unwrap();
        println!("{label:34}| {:.4}", c.final_discrepancy);
    }
    for (label, result) in [
        ("no homing, blocked", &without_homing),
        ("homing, blocked", &with_homing),
    ] {
        let c = result
            .conditions
            .iter()
            .find(|c| c.gap_junction_permeability == 0.0)
            .unwrap();
        println!("{label:34}| {:.4}", c.final_discrepancy);
    }

    let json2 = serde_json::to_string_pretty(&(without_homing.clone(), with_homing.clone()))
        .expect("serialize homing result");
    std::fs::write("basal_cognition_homing_results.json", &json2).expect("write homing json");
    println!();
    println!(
        "Homing comparison written to basal_cognition_homing_results.json ({} bytes)",
        json2.len()
    );
}
