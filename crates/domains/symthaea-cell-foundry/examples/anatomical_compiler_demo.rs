// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Anatomical Compiler Demo
//!
//! Demonstrates `search_intervention` (see `crate::anatomical_compiler`):
//! given an independently-constructed target morphology -- built on a
//! *separate* organoid, never grown by the one being searched over -- this
//! searches for gap-junction/ion-channel intervention parameters that drive
//! a *fresh, undifferentiated* organoid toward that target. This is Levin's
//! "given a desired anatomy, find the intervention" framing, distinct from
//! this crate's existing equifinality/dose-response experiments (which test
//! recovery from a hand-designed *perturbation* of a pattern the organoid
//! already had).
//!
//! Run: cargo run -p symthaea-cell-foundry --example anatomical_compiler_demo

use symthaea_cell_foundry::bioelectric::{VMEM_DEPOLARIZED, VMEM_HYPERPOLARIZED};
use symthaea_cell_foundry::morphogenetic_consciousness::NeuralOrganoid;
use symthaea_cell_foundry::{TargetMorphology, search_intervention};

/// An independent target pattern: a spherical hyperpolarized "core"
/// surrounded by a depolarized "shell" -- built on its own organoid, never
/// grown by the one the search operates on.
fn build_independent_target(seed: u64, cells: usize) -> TargetMorphology {
    let mut source = NeuralOrganoid::new(cells, seed);
    source.impose_vmem_pattern(|p| {
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if r < 0.3 {
            VMEM_HYPERPOLARIZED
        } else {
            VMEM_DEPOLARIZED
        }
    });
    source.capture_target_morphology();
    source.target_morphology.expect("just captured")
}

/// Naive baseline: gap junctions closed, no homing pull, K+ channels fully
/// blocked -- the tissue has no mechanism to move toward the target at all,
/// beyond whatever its unguided default development does on its own.
fn naive_baseline_discrepancy(
    template: &NeuralOrganoid,
    target: &TargetMorphology,
    days: u32,
) -> f64 {
    let mut organoid = template.clone();
    organoid.set_gap_junction_permeability(0.0);
    organoid.set_positional_homing(true);
    organoid.set_positional_homing_rate(0.0);
    organoid.set_gap_junction_diffusion_rate(0.0);
    organoid.set_ion_channel_model_enabled(true);
    organoid.set_potassium_channel_block(0.0);
    organoid.target_morphology = Some(target.clone());
    for _ in 0..days {
        organoid.advance_day();
    }
    organoid.morphology_discrepancy().unwrap_or(1.0)
}

fn main() {
    let cells = 100;
    let recovery_days = 25;
    let num_particles = 12;
    let max_iters = 15;

    println!("Building an independent target morphology (seed=99, {cells} cells)...");
    let target = build_independent_target(99, cells);

    println!("Building a fresh, undifferentiated organoid to search over (seed=1)...");
    let template = NeuralOrganoid::new(cells, 1);

    println!("Naive baseline (no gap-junction coupling, no homing, K+ blocked)...");
    let naive = naive_baseline_discrepancy(&template, &target, recovery_days);
    println!("  naive baseline discrepancy: {naive:.4}");

    println!(
        "Searching for an intervention ({num_particles} particles x {max_iters} iterations, \
         {recovery_days} recovery days per evaluation)..."
    );
    let (best_params, best_discrepancy) =
        search_intervention(&template, &target, recovery_days, num_particles, max_iters);

    println!();
    println!("Best intervention found:");
    println!("  gap_junction_permeability = {:.4}", best_params[0]);
    println!("  positional_homing_rate    = {:.4}", best_params[1]);
    println!("  gap_junction_diffusion_rate = {:.4}", best_params[2]);
    println!("  potassium_channel_block   = {:.4}", best_params[3]);
    println!("  resulting discrepancy     = {best_discrepancy:.4}");
    println!();
    println!(
        "Naive baseline: {naive:.4}  ->  Found intervention: {best_discrepancy:.4} \
         ({:.1}% reduction)",
        (1.0 - best_discrepancy / naive.max(1e-9)) * 100.0
    );
    assert!(
        best_discrepancy < naive,
        "search should find an intervention that beats the naive no-mechanism \
         baseline: naive={naive}, best={best_discrepancy}"
    );
}
