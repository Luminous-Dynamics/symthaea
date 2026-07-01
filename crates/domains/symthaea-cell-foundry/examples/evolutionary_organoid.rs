// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evolutionary Organoid Comparison
//!
//! Runs multiple organoid simulations with different seeds to explore
//! the landscape of consciousness emergence trajectories.
//!
//! Run: cargo run -p symthaea-cell-foundry --example evolutionary_organoid

use symthaea_cell_foundry::organoid_pipeline::{OrganoidPipeline, OrganoidPipelineConfig};

fn main() {
    println!("Evolutionary Organoid Comparison");
    println!("Running 10 organoids with different seeds...");
    println!();

    println!("Seed | Final Day | Peak Phi  | Cells | Neurons | Ethics Halt");
    println!("-----|-----------|-----------|-------|---------|------------");

    let mut best_seed = 0u64;
    let mut best_phi = 0.0f64;

    for seed in 0..10u64 {
        let config = OrganoidPipelineConfig {
            initial_cells: 50,
            max_cells: 3_000,
            seed,
            ethics_phi_threshold: 0.3,
            target_day: 150,
            enable_lfp_recording: false,
            morphogenetic_steps_per_day: 2,
        };

        let mut pipeline = OrganoidPipeline::new(config);
        let result = pipeline.run();

        if result.peak_phi > best_phi {
            best_phi = result.peak_phi;
            best_seed = seed;
        }

        println!(
            "{:4} | {:9} | {:9.6} | {:5} | {:7} | {}",
            seed,
            result.final_day,
            result.peak_phi,
            result.cell_count,
            result.neuron_count,
            if result.halted_by_ethics { "YES" } else { "no" },
        );
    }

    println!();
    println!("Best seed: {} (peak Phi = {:.6})", best_seed, best_phi);
}
