// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! The user's own "define success at three levels" directive
//! (2026-07-24): level 1 (motif diversity) was measured by
//! `motif_foundry_diversity_census`/`motif_foundry_style_survey`. This
//! measures levels 2 and 3 -- "do complete pieces remain distinguishable?"
//! and "does increased diversity preserve the intended style?" -- for
//! three pilot styles spanning the survey's foundry-headroom range
//! (Flamenco: most headroom 29.2%; March: real, already listening-test-
//! validated; Nocturne: least headroom 99.6%, the honest stress case).
//!
//! Deliberately reuses TRUSTED, already-production infrastructure rather
//! than inventing new measurement code: `explorer::novelty_within` (the
//! same function `compose()`'s real pre-render novelty floor already
//! calls) for song-level diversity, and `rhythmic_identity_report`/
//! `melodic_contour_report` (this session's own listening-test-driven
//! tools) for style-identity preservation.
//!
//! No style preset is changed by this example -- it only clones each
//! style's spec locally and flips `use_procedural_foundry` on the clone.
//!
//! Run: `cargo run --example motif_foundry_song_level_diversity -p symthaea-music-theory`

use symthaea_music_theory::explorer::novelty_within;
use symthaea_music_theory::{
    MusicalIntent, Style, compose_with_spec, melodic_contour_report, rhythmic_identity_report,
};

const SEEDS: u64 = 20;

fn mean(xs: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = xs.fold((0.0, 0usize), |(s, n), x| (s + x, n + 1));
    if n == 0 { 0.0 } else { sum / n as f64 }
}

fn survey_style(style: Style) {
    let classic_spec = style.spec();
    let mut foundry_spec = classic_spec.clone();
    foundry_spec.melody.use_procedural_foundry = true;
    let intent = MusicalIntent::default();
    let seeds: Vec<u64> = (0..SEEDS).collect();

    println!("=== {style:?} ===");
    if !classic_spec.texture.hook_cell {
        // Real finding from the first run of this example (2026-07-24):
        // Flamenco sets `texture.hook_cell = false`, so its composed
        // output NEVER actually splices a hook in -- the "song diversity"
        // numbers below still measure real hook differences (identity_of
        // reads HookCell::generate_with unconditionally), but they do NOT
        // reach this style's real composed pieces. Flagged loudly rather
        // than silently producing a misleading diversity win.
        println!(
            "  NOTE: texture.hook_cell = false for {style:?} -- the hook \
             is generated but never spliced into composed output. Song \
             diversity numbers below reflect the hook alone, NOT this \
             style's actual pieces (see style-identity numbers, which \
             should come out identical)."
        );
    }

    // Level 2: song diversity. novelty_within is the SAME function
    // compose()'s real novelty floor calls -- each seed's distance to its
    // nearest batch neighbor, per channel.
    let classic_novelty = novelty_within(&classic_spec, &intent, &seeds);
    let foundry_novelty = novelty_within(&foundry_spec, &intent, &seeds);
    println!(
        "  song diversity (mean nearest-neighbor distance over {} seeds):",
        SEEDS
    );
    println!(
        "    melodic:  classic={:.3}  foundry={:.3}",
        mean(classic_novelty.iter().map(|n| n.melodic)),
        mean(foundry_novelty.iter().map(|n| n.melodic))
    );
    println!(
        "    rhythmic: classic={:.3}  foundry={:.3}",
        mean(classic_novelty.iter().map(|n| n.rhythmic)),
        mean(foundry_novelty.iter().map(|n| n.rhythmic))
    );
    println!(
        "    overall:  classic={:.3}  foundry={:.3}",
        mean(classic_novelty.iter().map(|n| n.overall)),
        mean(foundry_novelty.iter().map(|n| n.overall))
    );

    // Level 3: style identity preserved. Compose full pieces for both
    // paths, average the SAME real, already-validated identity metrics
    // this session's listening-test work used to characterize March vs
    // Tango -- reported honestly, not pass/fail gated (how much
    // divergence is acceptable is a taste call, left to a listening
    // pass).
    let classic_scores: Vec<_> = seeds
        .iter()
        .map(|&s| compose_with_spec(&MusicalIntent { seed: s, ..intent }, &classic_spec))
        .collect();
    let foundry_scores: Vec<_> = seeds
        .iter()
        .map(|&s| compose_with_spec(&MusicalIntent { seed: s, ..intent }, &foundry_spec))
        .collect();

    let classic_rhythm: Vec<_> = classic_scores
        .iter()
        .map(rhythmic_identity_report)
        .collect();
    let foundry_rhythm: Vec<_> = foundry_scores
        .iter()
        .map(rhythmic_identity_report)
        .collect();
    let classic_contour: Vec<_> = classic_scores
        .iter()
        .map(|s| melodic_contour_report(s, 8).full_piece)
        .collect();
    let foundry_contour: Vec<_> = foundry_scores
        .iter()
        .map(|s| melodic_contour_report(s, 8).full_piece)
        .collect();

    println!("  style identity (mean over {} full pieces):", SEEDS);
    println!(
        "    strong_beat_onset_ratio:  classic={:.3}  foundry={:.3}",
        mean(
            classic_rhythm
                .iter()
                .map(|r| r.strong_beat_onset_ratio as f64)
        ),
        mean(
            foundry_rhythm
                .iter()
                .map(|r| r.strong_beat_onset_ratio as f64)
        )
    );
    println!(
        "    syncopation_score:        classic={:.3}  foundry={:.3}",
        mean(classic_rhythm.iter().map(|r| r.syncopation_score as f64)),
        mean(foundry_rhythm.iter().map(|r| r.syncopation_score as f64))
    );
    println!(
        "    mean_abs_interval_semis:  classic={:.3}  foundry={:.3}",
        mean(
            classic_contour
                .iter()
                .map(|c| c.mean_abs_interval_semitones as f64)
        ),
        mean(
            foundry_contour
                .iter()
                .map(|c| c.mean_abs_interval_semitones as f64)
        )
    );
    println!(
        "    large_leap_ratio:         classic={:.3}  foundry={:.3}",
        mean(classic_contour.iter().map(|c| c.large_leap_ratio as f64)),
        mean(foundry_contour.iter().map(|c| c.large_leap_ratio as f64))
    );
    println!(
        "    direction_change_ratio:   classic={:.3}  foundry={:.3}",
        mean(
            classic_contour
                .iter()
                .map(|c| c.direction_change_ratio as f64)
        ),
        mean(
            foundry_contour
                .iter()
                .map(|c| c.direction_change_ratio as f64)
        )
    );
    println!();
}

fn main() {
    // Flamenco is included deliberately despite `texture.hook_cell =
    // false` (see the NOTE printed for it) -- a real, useful negative
    // case showing the survey's config_for_dna headroom ranking can be
    // misleading for a style whose hook never reaches composed output.
    // Cinematic is the genuine highest-headroom pilot WITH hook_cell on.
    for style in [
        Style::Flamenco,
        Style::Cinematic,
        Style::March,
        Style::Nocturne,
    ] {
        survey_style(style);
    }
}
