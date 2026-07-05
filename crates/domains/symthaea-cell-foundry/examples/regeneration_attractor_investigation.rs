// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Regeneration Attractor Investigation
//!
//! `regeneration_learning_demo.rs` found that repeated amputation without
//! full inter-episode recovery time drives this tissue's morphology
//! discrepancy to a fixed ceiling within the first 2-3 episodes, after
//! which subsequent cut size stops mattering -- a real, reproducible
//! finding, left as an open question about *why* it happens.
//!
//! `TargetMorphology::discrepancy()` is a blend of two components (see
//! `TargetMorphology::discrepancy_components`, added for exactly this
//! investigation): the Vmem spatial-pattern RMS and the cell-type-
//! composition RMS. This experiment reruns the same repeated-amputation
//! setup, but logs both components separately every episode instead of
//! only the blended scalar, to see which one is actually driving the
//! plateau -- Vmem pattern drift, composition drift, or both.
//!
//! Run: cargo run -p symthaea-cell-foundry --example regeneration_attractor_investigation

use symthaea_cell_foundry::build_radial_bipolar_template;

const EPISODE_DAYS: u32 = 60;
const BOUNDARY_R: f32 = 0.2;
const AMPUTATION_CONFIGS: [(f32, f32); 4] = [(0.5, 2.0), (0.65, 2.0), (0.8, 2.0), (0.95, 2.0)];
const CONFIG_LABELS: [&str; 4] = ["large", "medium-large", "medium", "small"];
const NUM_CYCLES: u32 = 3;

/// Per episode: (config index, Vmem-pattern component, composition
/// component). Both components are already normalized/RMS values in
/// `[0, 1]`-ish ranges -- see `TargetMorphology::discrepancy_components`.
fn run_episodes(seed: u64, cells: usize, maturation_days: u32) -> Vec<(usize, f64, f64)> {
    let mut organoid = build_radial_bipolar_template(seed, cells, maturation_days, BOUNDARY_R);

    let mut results = Vec::new();
    for _cycle in 0..NUM_CYCLES {
        for (config_idx, &(min_r, max_r)) in AMPUTATION_CONFIGS.iter().enumerate() {
            organoid.amputate(min_r, max_r);
            for _ in 0..EPISODE_DAYS {
                organoid.advance_day();
            }
            let (vmem_component, composition_component) = organoid
                .target_morphology
                .as_ref()
                .expect("captured by build_radial_bipolar_template")
                .discrepancy_components(&organoid.field);
            results.push((config_idx, vmem_component, composition_component));
        }
    }
    results
}

fn print_episodes(results: &[(usize, f64, f64)]) {
    println!(
        "  {:>4} {:>14} {:>14} {:>14}",
        "ep.", "cut", "vmem component", "composition component"
    );
    for (i, (config_idx, vmem_c, comp_c)) in results.iter().enumerate() {
        println!(
            "  {:>4} {:>14} {:>14.4} {:>14.4}",
            i + 1,
            CONFIG_LABELS[*config_idx],
            vmem_c,
            comp_c
        );
    }
}

/// (first occurrence, last occurrence) of a component, per config.
fn per_config_trend(
    results: &[(usize, f64, f64)],
    pick: impl Fn(&(usize, f64, f64)) -> f64,
) -> Vec<(f64, f64)> {
    (0..AMPUTATION_CONFIGS.len())
        .map(|config_idx| {
            let occurrences: Vec<f64> = results
                .iter()
                .filter(|(c, _, _)| *c == config_idx)
                .map(&pick)
                .collect();
            (*occurrences.first().unwrap(), *occurrences.last().unwrap())
        })
        .collect()
}

fn main() {
    let seed = 11;
    let cells = 150;
    let maturation_days = 20;

    println!(
        "Running {} amputation-recovery episodes ({cells} cells, seed={seed}, \
         {maturation_days}d maturation, {EPISODE_DAYS}d recovery window per episode), \
         logging discrepancy's two components separately...\n",
        AMPUTATION_CONFIGS.len() as u32 * NUM_CYCLES
    );

    let results = run_episodes(seed, cells, maturation_days);
    print_episodes(&results);

    println!();
    println!("Per-configuration trend, Vmem-pattern component (1st -> last occurrence):");
    for (config_idx, (first, last)) in per_config_trend(&results, |(_, v, _)| *v)
        .into_iter()
        .enumerate()
    {
        println!(
            "    {:>12} cut: {first:.4} -> {last:.4} ({})",
            CONFIG_LABELS[config_idx],
            if last < first {
                "improved"
            } else if last > first {
                "got worse"
            } else {
                "no change"
            }
        );
    }

    println!();
    println!("Per-configuration trend, composition component (1st -> last occurrence):");
    for (config_idx, (first, last)) in per_config_trend(&results, |(_, _, c)| *c)
        .into_iter()
        .enumerate()
    {
        println!(
            "    {:>12} cut: {first:.4} -> {last:.4} ({})",
            CONFIG_LABELS[config_idx],
            if last < first {
                "improved"
            } else if last > first {
                "got worse"
            } else {
                "no change"
            }
        );
    }

    println!();
    println!(
        "This decomposition is the honest empirical answer to \"why does the plateau \
         happen\" -- whichever component (Vmem pattern, composition, or both) shows the \
         same climb-then-flatten pattern the blended discrepancy showed in \
         regeneration_learning_demo.rs is the one actually responsible."
    );
}
