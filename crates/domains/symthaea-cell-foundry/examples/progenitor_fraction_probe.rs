// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Progenitor Fraction Probe
//!
//! `regeneration_agent_efe_investigation.rs` found that the regeneration
//! agent's proliferation-boost mechanism never had any eligible cells
//! (wound-boundary AND still progenitor-type) to act on in the standard
//! 20-day-maturation scenario every other regeneration demo in this crate
//! uses. This checks whether that's because progenitors are already rare
//! tissue-wide by day 20 (in which case no amputation timing/location
//! would help), or whether it's specific to which cells end up near a
//! wound (in which case an earlier amputation might still find progenitors
//! nearby).
//!
//! Run: cargo run -p symthaea-cell-foundry --example progenitor_fraction_probe

use symthaea_cell_foundry::morphogenetic_consciousness::NeuralOrganoid;

fn main() {
    let seed = 10_000u64;
    let cells = 150;
    let max_day = 40;
    let check_every = 2;

    println!(
        "Tracking cell-type composition over {max_day} days ({cells} cells, seed={seed})...\n"
    );
    println!(
        "{:>4} {:>6} {:>12} {:>10} {:>14} {:>10}",
        "day", "cells", "progenitor", "neural_p", "neuron/glial", "undiff"
    );

    let mut organoid = NeuralOrganoid::new(cells, seed);
    for day in 0..=max_day {
        if day % check_every == 0 {
            let n = organoid.field.num_cells();
            let mut progenitor = 0;
            let mut neural_progenitor = 0;
            let mut neuron_glial = 0;
            let mut undiff = 0;
            for i in 0..n {
                let ct = organoid.field.cells[i].cell_type;
                if ct.is_progenitor() {
                    progenitor += 1;
                } else if ct.is_neuron() || ct.is_glial() {
                    neuron_glial += 1;
                } else {
                    undiff += 1;
                }
                let _ = &mut neural_progenitor; // NeuralProgenitor counted under is_progenitor()
            }
            println!(
                "{day:>4} {n:>6} {progenitor:>12} {neural_progenitor:>10} {neuron_glial:>14} {undiff:>10}"
            );
        }
        if day < max_day {
            organoid.advance_day();
        }
    }

    println!();
    println!(
        "If 'progenitor' is already near-zero well before day 20, the fix isn't about \
         WHERE the wound is -- it's about amputating earlier, before the tissue-wide \
         progenitor pool is depleted by ordinary differentiation."
    );
}
