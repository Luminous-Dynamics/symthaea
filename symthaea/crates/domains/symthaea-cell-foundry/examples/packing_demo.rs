// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Packing Demo
//!
//! Demonstrates the opt-in real-physics packing correction pass
//! (`crate::packing`, `NeuralOrganoid::set_packing_enabled`) against the
//! problem it exists to fix: daughter cells spawn at their parent's
//! position plus a small jitter, and nothing in the base model ever
//! separates them afterward.
//!
//! Runs the same seeded proliferation trajectory twice — packing disabled
//! vs. enabled — and reports, per day, the minimum pairwise distance
//! across all cells (the "most overlapping pair" statistic). With packing
//! disabled this stays pathologically small as new overlapping daughters
//! accumulate; with packing enabled it stays bounded away from zero.
//!
//! Run: cargo run -p symthaea-cell-foundry --example packing_demo

use symthaea_cell_foundry::morphogenetic_consciousness::NeuralOrganoid;

fn distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Minimum pairwise distance across all cells — the statistic packing is
/// meant to keep bounded away from zero.
fn min_pairwise_distance(organoid: &NeuralOrganoid) -> f32 {
    let n = organoid.field.num_cells();
    let mut min_d = f32::INFINITY;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = distance(
                &organoid.field.cells[i].position,
                &organoid.field.cells[j].position,
            );
            min_d = min_d.min(d);
        }
    }
    min_d
}

fn run(seed: u64, days: u32, packing_enabled: bool) -> Vec<(u32, usize, f32)> {
    let mut organoid = NeuralOrganoid::new(60, seed);
    organoid.set_packing_enabled(packing_enabled);
    let mut trace = vec![(
        0,
        organoid.field.num_cells(),
        min_pairwise_distance(&organoid),
    )];
    for day in 1..=days {
        organoid.advance_day();
        trace.push((
            day,
            organoid.field.num_cells(),
            min_pairwise_distance(&organoid),
        ));
    }
    trace
}

fn main() {
    let seed = 1234;
    let days = 20;

    println!("Running {days}-day proliferation trace with packing disabled...");
    let disabled = run(seed, days, false);
    println!("Running the same trace with packing enabled...");
    let enabled = run(seed, days, true);

    println!();
    println!(
        "{:>4} {:>8} {:>14} {:>8} {:>14}",
        "day", "cells", "min_d (off)", "cells", "min_d (on)"
    );
    for ((day, n_off, d_off), (_, n_on, d_on)) in disabled.iter().zip(enabled.iter()) {
        println!("{day:>4} {n_off:>8} {d_off:>14.5} {n_on:>8} {d_on:>14.5}");
    }

    let final_off = disabled.last().unwrap().2;
    let final_on = enabled.last().unwrap().2;
    println!();
    println!(
        "Final min pairwise distance: {final_off:.5} (packing off) vs. {final_on:.5} (packing on)"
    );
    assert!(
        final_on >= final_off,
        "packing should never leave cells more overlapped than the unpacked baseline"
    );
    println!("Packing keeps freshly-divided daughters from staying permanently overlapped.");
}
