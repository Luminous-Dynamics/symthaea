// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bioelectric Regeneration Demo
//!
//! Grows an organoid to maturity, captures its cell-type-by-shell pattern as
//! a "target morphology," amputates the outer shell, then regrows it twice —
//! once with gap junctions open (control) and once pharmacologically blocked
//! (`gap_junction_permeability = 0.0`, the model's analog of Levin's
//! octanol/lanthanum experiments). Both runs share the exact same amputated
//! starting state and RNG seed; the only difference is whether the
//! bioelectric (Vmem) signal can propagate. Watch the discrepancy-from-target
//! columns: the open-gap-junction run recovers the target pattern faster.
//!
//! Also demonstrates, at the single-cell level, that identical activator
//! concentration and identical gene expression can differentiate into
//! *different* lineages purely as a function of bioelectric coupling state —
//! see `bioelectric::gap_junction_blockade_changes_differentiation_outcome`
//! in the crate's test suite for the isolated mechanism this builds on.
//!
//! Run: cargo run -p symthaea-cell-foundry --example regeneration_demo

use symthaea_cell_foundry::morphogenetic_consciousness::NeuralOrganoid;

fn main() {
    println!("======================================================");
    println!("   Bioelectric Regeneration Demo (Levin-style)");
    println!("======================================================");
    println!("  Grow -> capture target morphology -> amputate -> regrow");
    println!("  Comparing: gap junctions OPEN vs. BLOCKED");
    println!("======================================================");
    println!();

    let mut organoid = NeuralOrganoid::new(150, 42);
    println!("Growing organoid to maturity (60 days)...");
    for _ in 0..60 {
        organoid.advance_day();
    }
    println!(
        "  day {}: {} cells, {:.1}% neurons, stage {:?}",
        organoid.developmental_day,
        organoid.field.num_cells(),
        organoid.neuron_fraction() * 100.0,
        organoid.stage
    );

    organoid.capture_target_morphology();
    println!();
    println!("Target morphology captured. Amputating outer shell (r in [0.6, 2.0))...");
    let removed = organoid.amputate(0.6, 2.0);
    println!("  removed {removed} cells");

    let mut control = organoid.clone();
    let mut blocked = organoid.clone();
    blocked.set_gap_junction_permeability(0.0);

    println!();
    println!("Regenerating for 60 days (open vs. blocked gap junctions)...");
    println!();
    println!("Day  | Cells (open) | Discrepancy (open) | Cells (blocked) | Discrepancy (blocked)");
    println!(
        "-----|---------------|---------------------|------------------|----------------------"
    );

    for day in 1..=60 {
        control.advance_day();
        blocked.advance_day();

        if day % 5 == 0 || day == 60 {
            let d_control = control.morphology_discrepancy().unwrap_or(0.0);
            let d_blocked = blocked.morphology_discrepancy().unwrap_or(0.0);
            println!(
                "{:4} | {:13} | {:19.4} | {:16} | {:22.4}",
                day,
                control.field.num_cells(),
                d_control,
                blocked.field.num_cells(),
                d_blocked,
            );
        }
    }

    println!();
    println!("=== Results ===");
    println!(
        "Final discrepancy (open):    {:.4}",
        control.morphology_discrepancy().unwrap_or(0.0)
    );
    println!(
        "Final discrepancy (blocked): {:.4}",
        blocked.morphology_discrepancy().unwrap_or(0.0)
    );
    println!(
        "Still regenerating (open):    {}",
        control.is_regenerating()
    );
    println!(
        "Still regenerating (blocked): {}",
        blocked.is_regenerating()
    );

    println!();
    println!("=== Single-cell mechanism ===");
    println!("(Same activator level, same gene expression — only bioelectric");
    println!(" coupling differs. See bioelectric.rs module docs for citations.)");
    for permeability in [1.0f32, 0.0] {
        use symthaea_cell_foundry::bioelectric::BioelectricState;
        use symthaea_cell_foundry::morphogenetic_consciousness::{
            MorphogeneticField, OrganoidCellType,
        };

        let mut field = MorphogeneticField::new(1, 1);
        field.activator[0] = 1.6;
        field.cells[0].cell_type = OrganoidCellType::Progenitor;
        field.cells[0].gene_expression = vec![0.35, 0.35, 0.35, 0.35, 0.5, 0.5, 0.5, 0.5];
        field.bioelectric = BioelectricState::new(1, 1);
        field.bioelectric.vmem[0] = -1.0; // hyperpolarized
        field.bioelectric.gap_junction_permeability = permeability;
        field.differentiate();

        println!(
            "  gap_junction_permeability = {permeability:.1} -> fate = {:?}",
            field.cells[0].cell_type
        );
    }
}
