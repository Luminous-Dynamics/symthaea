// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Extends `motif_foundry_diversity_census`'s 3-style spot-check to all 29
//! styles in the crate, and reports each style's `config_for_dna`-derived
//! (style-conditioned, not generic) foundry diversity alongside its
//! existing hand-authored-pool diversity. This is deliberately a
//! MEASUREMENT tool, not a decision: no style preset in this crate flips
//! `use_procedural_foundry` on. It exists to give a grounded, ranked menu
//! for a future listening-test pilot -- "these N styles have the smallest
//! existing motif pools" is a measured fact, not invented taste, whereas
//! deciding to enable procedural generation for any specific style IS a
//! taste decision, deliberately left to a listening pass.
//!
//! Run: `cargo run --example motif_foundry_style_survey -p symthaea-music-theory`

use symthaea_music_theory::{FoundryConfig, Style, canonical_fingerprint, config_for_dna};

const DRAWS: u64 = 2000;
const METER_BEATS: f64 = 4.0;

const ALL_STYLES: [Style; 29] = [
    Style::Classical,
    Style::Waltz,
    Style::Folk,
    Style::Cinematic,
    Style::Playful,
    Style::Nocturne,
    Style::March,
    Style::Lullaby,
    Style::ModalFolk,
    Style::Fugue,
    Style::Passacaglia,
    Style::Tango,
    Style::Celtic,
    Style::Blues,
    Style::Impressionism,
    Style::SacredChoral,
    Style::Minimalism,
    Style::JazzBallad,
    Style::BaroqueSuite,
    Style::ProgFolk,
    Style::Ambient,
    Style::Sonata,
    Style::RenaissancePolyphony,
    Style::AfroCuban,
    Style::Flamenco,
    Style::BossaNova,
    Style::Opera,
    Style::IrishTraditional,
    Style::HindustaniInspired,
];

/// Existing hand-authored-pool canonical duplicate rate, using the SAME
/// technique `motif_foundry_diversity_census` uses (not the foundry --
/// the classic `HookCell::generate_with` combinatorial-variety path).
fn existing_duplicate_rate(dna: &symthaea_music_theory::spec::MelodicDna) -> f32 {
    let mut canonical_seen: Vec<Vec<i32>> = Vec::new();
    for seed in 0..DRAWS {
        let cell = symthaea_music_theory::HookCell::generate_with(dna, seed, METER_BEATS);
        let canonical = canonical_fingerprint(&cell);
        if !canonical_seen.contains(&canonical) {
            canonical_seen.push(canonical);
        }
    }
    1.0 - canonical_seen.len() as f32 / DRAWS as f32
}

fn main() {
    let mut rows: Vec<(Style, f32, f32, FoundryConfig)> = Vec::new();
    for style in ALL_STYLES {
        let dna = style.spec().melody;
        let existing_dup = existing_duplicate_rate(&dna);
        let config = config_for_dna(&dna);
        let foundry_report = symthaea_music_theory::foundry_diversity_report(0..DRAWS, &config);
        rows.push((
            style,
            existing_dup,
            foundry_report.canonical_duplicate_rate(),
            config,
        ));
    }

    // Worst (highest existing duplicate rate = least diverse today) first
    // -- the grounded ranking a listening-test pilot would start from.
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));

    println!(
        "{:<20} {:>14} {:>14} {:>10} {:>10}",
        "style", "existing_dup%", "foundry_dup%", "len_range", "max_leap"
    );
    for (style, existing_dup, foundry_dup, config) in &rows {
        println!(
            "{:<20} {:>13.1}% {:>13.1}% {:>10?} {:>10}",
            format!("{style:?}"),
            existing_dup * 100.0,
            foundry_dup * 100.0,
            config.length_range,
            config.max_leap_degrees
        );
    }

    let mean_existing: f32 = rows.iter().map(|r| r.1).sum::<f32>() / rows.len() as f32;
    let mean_foundry: f32 = rows.iter().map(|r| r.2).sum::<f32>() / rows.len() as f32;
    println!(
        "\nmean existing canonical duplicate rate across all {} styles: {:.1}%",
        rows.len(),
        mean_existing * 100.0
    );
    println!(
        "mean foundry (style-conditioned) canonical duplicate rate:  {:.1}%",
        mean_foundry * 100.0
    );
}
