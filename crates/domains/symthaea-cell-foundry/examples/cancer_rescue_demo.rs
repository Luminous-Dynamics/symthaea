// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cancer Rescue Demo
//!
//! Combines two pieces of this crate that have never been used together
//! before: the cancer-as-bioelectric-defection model
//! (`NeuralOrganoid::induce_local_defection`) and the evolutionary
//! anatomical compiler (`search_intervention`, Phase 3). Every existing
//! use of `search_intervention` starts the search from a fresh,
//! undifferentiated organoid reaching toward an *arbitrary* target. This
//! demo asks a different, more clinically-flavored question: given a
//! *healthy, already-patterned* organoid that has since developed a
//! permanent cancer-like region, can a systemic bioelectric intervention
//! (gap-junction permeability, positional homing, ion-channel state)
//! elsewhere in the tissue still pull the *overall* pattern back toward
//! the original healthy target -- even though the defected cells
//! themselves can never be cured?
//!
//! **Why "never cured" isn't a bug in this demo.** Once
//! `defected[i]` is set, nothing in this crate ever clears it (confirmed
//! during Phase 4 research: no reversal/remission mechanism exists at
//! all, and daughters of defected cells inherit the flag). This mirrors
//! real Levin-lab findings on bioelectric-mediated tumor normalization:
//! restoring normal Vmem/gap-junction signaling in the surrounding tissue
//! can suppress a tumor's growth and downstream effects on the organism's
//! overall pattern, without genetically altering the tumor cells
//! themselves. This demo is the first time this crate tests that specific
//! framing computationally.
//!
//! Run: cargo run -p symthaea-cell-foundry --example cancer_rescue_demo

use symthaea_cell_foundry::morphogenetic_consciousness::NeuralOrganoid;
use symthaea_cell_foundry::{TargetMorphology, build_radial_bipolar_template, search_intervention};

fn naive_baseline_discrepancy(
    template: &NeuralOrganoid,
    target: &TargetMorphology,
    days: u32,
) -> f64 {
    let mut organoid = template.clone();
    organoid.target_morphology = Some(target.clone());
    for _ in 0..days {
        organoid.advance_day();
    }
    organoid.morphology_discrepancy().unwrap_or(1.0)
}

fn main() {
    let seed = 33;
    let cells = 150;
    let maturation_days = 20;
    let boundary_r = 0.2;
    // Defected cells proliferate unconditionally at DEFECTION_PROLIFERATION_RATE
    // = 0.35/day (vs. 0.10/day for normal tissue), so a small defected
    // population compounds fast: ~5 cells at day 0 reaches ~100 by day 10
    // and would approach MAX_CELLS=10,000 well before day 25. An earlier
    // draft of this demo used a 5-day establish + 25-day recovery window
    // (30 days total) and became impractically slow across the search's
    // ~100+ evaluations once the tissue ballooned toward that cap, with
    // the dense O(n^2) gap-junction/connectivity matrices dominating cost.
    // These shorter windows keep the defected population, and therefore
    // the whole tissue, comfortably below a few hundred cells throughout.
    let defection_establish_days = 3;
    let recovery_days = 10;
    let num_particles = 8;
    let max_iters = 8;

    println!(
        "Building a healthy, patterned organoid ({cells} cells, seed={seed}) and \
         capturing its state as the target to recover..."
    );
    let mut organoid = build_radial_bipolar_template(seed, cells, maturation_days, boundary_r);
    let target = organoid
        .target_morphology
        .clone()
        .expect("captured by build_radial_bipolar_template");

    println!("Inducing a small local defection (cancer-analog) near the tissue's core...");
    // r < 0.15 was empirically too small a region to reliably contain any
    // cells at this density (learned earlier this same session while
    // building advanced_experiments.rs's cancer-as-defection section: at
    // ~150-230 cells in a [-1,1]^3 cube, r < 0.15 captures well under 1
    // expected cell). 0.5 gives a robust handful.
    let marked = organoid.induce_local_defection(|p| {
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        r < 0.5
    });
    println!("  marked {marked} cells as defected");

    println!(
        "Letting the defection establish itself for {defection_establish_days} days \
         before any intervention..."
    );
    for _ in 0..defection_establish_days {
        organoid.advance_day();
    }
    println!(
        "  defected cell count after establishment: {}",
        organoid.defected_cell_count()
    );

    let template = organoid;

    println!("\nNaive baseline (default parameters, no intervention)...");
    let naive = naive_baseline_discrepancy(&template, &target, recovery_days);
    println!("  naive baseline discrepancy: {naive:.4}");

    println!(
        "\nSearching for a systemic intervention ({num_particles} particles x {max_iters} \
         iterations, {recovery_days} recovery days per evaluation)..."
    );
    let (best_params, best_discrepancy) =
        search_intervention(&template, &target, recovery_days, num_particles, max_iters);

    println!("\nBest intervention found:");
    println!("  gap_junction_permeability   = {:.4}", best_params[0]);
    println!("  positional_homing_rate      = {:.4}", best_params[1]);
    println!("  gap_junction_diffusion_rate = {:.4}", best_params[2]);
    println!("  potassium_channel_block     = {:.4}", best_params[3]);
    println!("  resulting discrepancy       = {best_discrepancy:.4}");

    println!(
        "\nNaive baseline: {naive:.4}  ->  Found intervention: {best_discrepancy:.4} \
         ({:.1}% reduction)",
        (1.0 - best_discrepancy / naive.max(1e-9)) * 100.0
    );
    println!(
        "The defected cells themselves are never cured by this (no mechanism in this \
         crate reverses `defected[i]` once set) -- what's being tested is whether \
         systemic bioelectric intervention elsewhere in the tissue can compensate for \
         a permanently-present cancer-like region well enough to pull the *overall* \
         tissue pattern back toward the healthy target. Reported honestly regardless \
         of which way it comes out."
    );
}
