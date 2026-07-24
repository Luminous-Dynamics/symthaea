// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Defines "more generative" the way the review that led to
//! `motif_foundry` asked for: not a vibe, a measurement. Compares the
//! EXISTING hand-authored-bank + combinatorial-variety approach
//! ([`HookCell::generate_with`]) against the new procedural
//! [`motif_foundry`] generator, on the SAME metrics, over the SAME number
//! of draws, using the SAME canonical (transform-invariant) fingerprint --
//! so "unique ideas produced" is directly comparable between the two.
//!
//! Run: `cargo run --example motif_foundry_diversity_census -p symthaea-music-theory`

use symthaea_music_theory::{
    FoundryConfig, HookCell, Style, canonical_fingerprint, foundry_diversity_report,
    is_valid_candidate,
};

const DRAWS: u64 = 2000;
const METER_BEATS: f64 = 4.0;

fn census_existing_bank(
    dna: &symthaea_music_theory::spec::MelodicDna,
    seeds: std::ops::Range<u64>,
) {
    let mut canonical_seen: Vec<Vec<i32>> = Vec::new();
    let mut exact_seen: Vec<Vec<i32>> = Vec::new();
    let total = (seeds.end - seeds.start) as u32;
    for seed in seeds {
        let cell = HookCell::generate_with(dna, seed, METER_BEATS);
        let exact: Vec<i32> = cell.notes.windows(2).map(|w| w[1].0 - w[0].0).collect();
        if !exact_seen.contains(&exact) {
            exact_seen.push(exact);
        }
        let canonical = canonical_fingerprint(&cell);
        if !canonical_seen.contains(&canonical) {
            canonical_seen.push(canonical);
        }
    }
    println!(
        "  draws={total}  unique_exact={}  unique_canonical={}  \
         canonical_duplicate_rate={:.1}%",
        exact_seen.len(),
        canonical_seen.len(),
        100.0 * (1.0 - canonical_seen.len() as f64 / total as f64)
    );
}

fn main() {
    let config = FoundryConfig::default();

    println!("=== EXISTING approach: hand-authored bank + combinatorial variety ===\n");
    println!("Classic shared pool (every style with empty melody.hook_contours):");
    census_existing_bank(
        &symthaea_music_theory::spec::MelodicDna::default(),
        0..DRAWS,
    );

    for style in [Style::March, Style::Tango, Style::Nocturne] {
        println!("{style:?}'s own DNA pool:");
        census_existing_bank(&style.spec().melody, 0..DRAWS);
    }

    println!("\n=== NEW approach: motif_foundry procedural generation ===\n");
    let report = foundry_diversity_report(0..DRAWS, &config);
    println!(
        "  draws={}  valid={} ({:.1}% pass the SAME identity predicates)  \
         unique_exact={}  unique_canonical={}  canonical_duplicate_rate={:.1}%",
        report.candidates_generated,
        report.candidates_valid,
        100.0 * report.validity_rate(),
        report.unique_exact,
        report.unique_canonical,
        100.0 * report.canonical_duplicate_rate()
    );

    println!("\n=== Sanity: a hand-picked hook and a foundry candidate share one bar ===");
    let sample = symthaea_music_theory::generate_candidate(7, &config);
    println!(
        "  foundry seed 7: {:?}  valid={}  canonical_fingerprint={:?}",
        sample.notes,
        is_valid_candidate(&sample),
        canonical_fingerprint(&sample)
    );
}
