// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Differentiation Threshold Tuning Probe
//!
//! `progenitor_fraction_probe.rs` found the tissue-wide progenitor count
//! crashes from 150 to 0 within just 8 days of ordinary maturation (150 ->
//! 11 -> 3 -> 1 -> 0 at days 0/2/4/6), which is why every regeneration
//! demo's standard 20-day maturation window leaves the regeneration agent's
//! proliferation-boost mechanism with zero eligible (wound-boundary AND
//! still-progenitor) cells to ever act on -- see
//! `regeneration_agent_efe_investigation.rs` and
//! `regeneration_statistical_replication.rs`.
//!
//! This probes the new opt-in `differentiation_threshold_multiplier`
//! (`NeuralOrganoid::set_differentiation_threshold_multiplier`, default
//! `1.0` = today's exact behavior) across several values, tracking the same
//! progenitor-count-over-time curve `progenitor_fraction_probe.rs` used, to
//! find a value that keeps a meaningful progenitor fraction alive out to
//! day 20+ instead of collapsing by day 8.
//!
//! Run: cargo run -p symthaea-cell-foundry --example differentiation_threshold_tuning

use symthaea_cell_foundry::morphogenetic_consciousness::NeuralOrganoid;

const SEED: u64 = 10_000;
const CELLS: usize = 150;
const MAX_DAY: u32 = 40;
const CHECK_EVERY: u32 = 4;
const MULTIPLIERS: [f32; 6] = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0];

fn progenitor_count(organoid: &NeuralOrganoid) -> usize {
    (0..organoid.field.num_cells())
        .filter(|&i| organoid.field.cells[i].cell_type.is_progenitor())
        .count()
}

fn main() {
    println!(
        "Tracking progenitor count over {MAX_DAY} days across differentiation-threshold \
         multipliers {MULTIPLIERS:?} ({CELLS} cells, seed={SEED})...\n"
    );

    for &multiplier in &MULTIPLIERS {
        let mut organoid = NeuralOrganoid::new(CELLS, SEED);
        organoid.set_differentiation_threshold_multiplier(multiplier);

        print!("multiplier={multiplier:>5.1}x  ");
        for day in 0..=MAX_DAY {
            if day % CHECK_EVERY == 0 {
                let p = progenitor_count(&organoid);
                print!("d{day}={p:>3} ");
            }
            if day < MAX_DAY {
                organoid.advance_day();
            }
        }
        let final_p = progenitor_count(&organoid);
        let n = organoid.field.num_cells();
        println!(
            "  final={final_p}/{n} ({:.1}%)",
            100.0 * final_p as f64 / n.max(1) as f64
        );
    }

    println!();
    println!(
        "Goal: a multiplier where the progenitor fraction stays meaningfully nonzero out to \
         day 20+ (the standard maturation window every regeneration demo uses), without \
         breaking the tissue's ability to differentiate at all over a much longer horizon."
    );
}
