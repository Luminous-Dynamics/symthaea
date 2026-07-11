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
//! `1.0` = today's exact behavior).
//!
//! **First finding, and a design correction made because of it.** The
//! initial version of this multiplier only scaled `differentiate()`'s
//! neural/glial activator thresholds (`1.5`/`0.8`), leaving the separate
//! `a < 0.3` reversion-to-undifferentiated branch untouched (the original
//! plan explicitly called that branch "a different concern"). Running this
//! probe against that first version showed the fix plateaued identically
//! from 2x through 12x -- because `is_progenitor()` excludes
//! `Undifferentiated`, and the untouched reversion branch was the actual
//! dominant depletion path, not the neural/glial thresholds. That original
//! scoping call was wrong; corrected by also scaling the reversion
//! threshold to `0.3 / multiplier` (so it gets *harder* to trigger as the
//! multiplier grows, same direction as the other two), which is what this
//! file now measures.
//!
//! **Second finding: this is a blunt instrument, not a graceful slowdown.**
//! At `multiplier=1.0` (default), the tissue's activator field naturally
//! spikes high enough in the first ~10 days to differentiate nearly the
//! whole population into neuron/glial. At `multiplier >= 8.0`, the required
//! activator level (`0.8 * multiplier` and up) is apparently never reached
//! by this tissue's Turing dynamics at all -- neuron+glial count stays at
//! *exactly* 0 for the entire probed window, not just delayed. Progenitors
//! still eventually crash to 0 too (just later: `multiplier=12.0` reaches
//! 0 by day 40 instead of day 8), because cells get stuck oscillating
//! between `Progenitor` and `Undifferentiated` without ever crossing
//! either scaled threshold. So a high multiplier doesn't proportionally
//! slow lineage commitment -- it effectively suspends it for as long as
//! the tissue's activator field stays within the now-unreachable band.
//!
//! **Why `12.0` is still a reasonable choice despite that.** The actual
//! downstream need (`regeneration_statistical_replication_v2.rs`'s
//! eligible-cell gap) only requires *some* progenitor cells to exist near a
//! wound boundary during a realistic 20-60 day recovery window -- it
//! doesn't require the rest of the tissue to keep differentiating on
//! schedule during that same window. `multiplier=12.0` keeps a real,
//! nonzero progenitor population out to day 30 (27 at day 20, 7 at day
//! 30) which is exactly what that gap needs, at the cost of an honestly-
//! documented side effect (no further neuron/glial differentiation during
//! that window) that doesn't matter for this specific use case.
//!
//! Run: cargo run -p symthaea-cell-foundry --example differentiation_threshold_tuning

use symthaea_cell_foundry::morphogenetic_consciousness::NeuralOrganoid;

const SEED: u64 = 10_000;
const CELLS: usize = 150;
const MAX_DAY: u32 = 40;
const CHECK_EVERY: u32 = 4;
const MULTIPLIERS: [f32; 4] = [1.0, 8.0, 12.0, 16.0];

fn progenitor_count(organoid: &NeuralOrganoid) -> usize {
    (0..organoid.field.num_cells())
        .filter(|&i| organoid.field.cells[i].cell_type.is_progenitor())
        .count()
}

fn differentiated_count(organoid: &NeuralOrganoid) -> usize {
    (0..organoid.field.num_cells())
        .filter(|&i| {
            let ct = organoid.field.cells[i].cell_type;
            ct.is_neuron() || ct.is_glial()
        })
        .count()
}

fn main() {
    println!(
        "Tracking progenitor and neuron+glial counts over {MAX_DAY} days across \
         differentiation-threshold multipliers {MULTIPLIERS:?} ({CELLS} cells, \
         seed={SEED})...\n"
    );

    for &multiplier in &MULTIPLIERS {
        let mut organoid = NeuralOrganoid::new(CELLS, SEED);
        organoid.set_differentiation_threshold_multiplier(multiplier);

        println!("--- multiplier={multiplier:>5.1}x ---");
        let mut snapshots: Vec<(u32, usize, usize, usize)> = Vec::new();
        print!("  progenitor:   ");
        for day in 0..=MAX_DAY {
            if day % CHECK_EVERY == 0 {
                let p = progenitor_count(&organoid);
                let d = differentiated_count(&organoid);
                let n = organoid.field.num_cells();
                print!("d{day}={p:>3} ");
                snapshots.push((day, p, d, n));
            }
            if day < MAX_DAY {
                organoid.advance_day();
            }
        }
        println!();
        print!("  neuron+glial: ");
        for (day, _, d, _) in &snapshots {
            print!("d{day}={d:>3} ");
        }
        println!();
        let (_, final_p, final_d, final_n) = *snapshots.last().unwrap();
        println!("  final: progenitor={final_p}/{final_n}, neuron+glial={final_d}/{final_n}\n");
    }

    println!(
        "Chosen value for downstream use (regeneration_statistical_replication_v2-style \
         scenarios): multiplier=12.0 -- keeps a real progenitor population alive out to \
         day 30, which is what a 20-60 day recovery window actually needs."
    );
}
