// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Investigates a real finding from a 2026-07-24 blind listening test:
//! two March clips (from `listening_test_v3`, base seed
//! `1784881817292537469`) were misheard as Tango, while two others were
//! confidently and correctly heard as March, and two Tango clips were
//! confidently and correctly heard as Tango. This reconstructs which
//! (style, arousal tier, seed) each numbered clip actually was — by
//! replicating `listening_test.rs`'s exact seed/shuffle derivation, not
//! guessing — composes each, and runs `rhythmic_identity_report` on the
//! realized melody to test the hypothesis: are the two misheard March
//! clips measurably more Tango-like in beat placement than the two clean
//! March controls?
//!
//! Run: `cargo run --example march_tango_rhythm_probe -p symthaea-music-theory`

use symthaea_music_theory::{MusicalIntent, Style, compose_with_spec, rhythmic_identity_report};

const STYLES: [Style; 8] = [
    Style::Tango,
    Style::Nocturne,
    Style::March,
    Style::Blues,
    Style::Minimalism,
    Style::Flamenco,
    Style::SacredChoral,
    Style::Ambient,
];
const SEEDS_PER_STYLE: usize = 4;
const BASE_SEED: u64 = 1784881817292537469;

/// Verbatim copy of `listening_test.rs::shuffle` -- must match exactly to
/// reconstruct the real clip ordering.
fn shuffle<T>(items: &mut [T], mut state: u64) {
    for i in (1..items.len()).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = ((state >> 33) as usize) % (i + 1);
        items.swap(i, j);
    }
}

fn main() {
    let mut clips: Vec<(Style, usize, u64)> = Vec::new();
    for style in STYLES {
        for k in 0..SEEDS_PER_STYLE {
            clips.push((
                style,
                k,
                BASE_SEED
                    .wrapping_add(11)
                    .wrapping_add((k as u64).wrapping_mul(17)),
            ));
        }
    }
    shuffle(&mut clips, BASE_SEED ^ 0xC1A5_51F1);

    // clip_NN.wav is 1-indexed into this shuffled order.
    let targets: [(usize, &str); 6] = [
        (7, "March, confidently correct"),
        (12, "March, confidently correct"),
        (8, "March, misheard as Tango"),
        (14, "March, misheard as Tango"),
        (6, "Tango, confidently correct"),
        (19, "Tango, confidently correct"),
    ];

    println!("base_seed = {BASE_SEED}\n");
    for (clip_num, label) in targets {
        let (style, k, seed) = clips[clip_num - 1];
        assert!(
            matches!(style, Style::March | Style::Tango),
            "clip_{clip_num:02} was expected to be March/Tango, reconstructed as {style:?} -- \
             mapping is wrong, do not trust the report below"
        );
        let (arousal, energy, valence) = match k {
            0 => (0.15, 0.25, 0.5),
            1 => (0.5, 0.5, 0.0),
            2 => (0.85, 0.85, -0.5),
            _ => (0.5, 0.6, 0.5),
        };
        let intent = MusicalIntent {
            seed,
            valence,
            arousal,
            energy,
            ..Default::default()
        };
        let score = compose_with_spec(&intent, &style.spec());
        let r = rhythmic_identity_report(&score);
        println!("clip_{clip_num:02}  {style:?} (tier k={k}, seed={seed})  -- {label}");
        println!(
            "  strong_beat_onset_ratio:   {:.3}",
            r.strong_beat_onset_ratio
        );
        println!(
            "  weak_beat_onset_ratio:     {:.3}",
            r.weak_beat_onset_ratio
        );
        println!("  anticipation_ratio:        {:.3}", r.anticipation_ratio);
        println!("  syncopation_score:         {:.3}", r.syncopation_score);
        println!(
            "  phrase_final_downbeat_ratio: {:.3}",
            r.phrase_final_downbeat_ratio
        );
        println!(
            "  long_short_on_strong_beat: {}   long_short_anticipations: {}",
            r.long_short_on_strong_beat, r.long_short_anticipations
        );
        println!();
    }
}
