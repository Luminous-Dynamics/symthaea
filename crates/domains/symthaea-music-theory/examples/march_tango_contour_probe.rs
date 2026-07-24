// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Follow-up to `march_tango_rhythm_probe.rs`: that probe FALSIFIED the
//! beat-placement hypothesis for a real listening-test confusion (two
//! March clips misheard as Tango). The listener's own stated reason for
//! both misses was "angular leaps" -- this probe measures interval size,
//! direction, and leap-recovery behavior on the SAME six real V3 clips
//! instead, testing whether contour (not rhythm) is the actual mechanism.
//!
//! Mode/valence/arousal are printed as METADATA beside each report, not
//! folded into the contour calculation itself -- so a mode effect (e.g.
//! clip 14's minor-mode tier) and a genuine contour effect can be told
//! apart rather than conflated into one number.
//!
//! Run: `cargo run --example march_tango_contour_probe -p symthaea-music-theory`

use symthaea_music_theory::{MusicalIntent, Style, compose_with_spec, melodic_contour_report};

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

fn mode_label(valence: f32) -> &'static str {
    if valence >= 0.0 { "major" } else { "minor" }
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
            "clip_{clip_num:02} was expected to be March/Tango, reconstructed as {style:?}"
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
        let spec = style.spec();
        let score = compose_with_spec(&intent, &spec);
        // Tango pins its own mode (HarmonicMinor) regardless of valence;
        // March's `mode: None` maps valence -> major/minor directly. Print
        // what the piece actually resolved to, not just the intent.
        let mode = match spec.mode {
            Some(m) => format!("{m:?} (style-pinned)"),
            None => mode_label(valence).to_string(),
        };
        let r = melodic_contour_report(&score, 8);
        println!(
            "clip_{clip_num:02}  {style:?} (tier k={k}, mode={mode}, arousal={arousal})  -- {label}"
        );
        for (section, rep) in [
            ("opening (8 bars)", r.opening),
            ("full piece", r.full_piece),
        ] {
            println!("  [{section}]");
            println!("    note_count:                 {}", rep.note_count);
            println!(
                "    mean_abs_interval_semitones: {:.2}",
                rep.mean_abs_interval_semitones
            );
            println!(
                "    large_leap_ratio:           {:.3}",
                rep.large_leap_ratio
            );
            println!("    octave_leap_count:          {}", rep.octave_leap_count);
            println!(
                "    direction_change_ratio:     {:.3}",
                rep.direction_change_ratio
            );
            println!(
                "    leap_reversal_ratio:        {:.3}",
                rep.leap_reversal_ratio
            );
            println!(
                "    leap_recovery_ratio:        {:.3}",
                rep.leap_recovery_ratio
            );
            println!(
                "    minor_second_ratio:         {:.3}",
                rep.minor_second_ratio
            );
            println!(
                "    registral_span_semitones:   {}",
                rep.registral_span_semitones
            );
        }
        println!();
    }
}
