// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generalized version of the one-off `clip14_minor_second_trace`: prints
//! any `listening_test_v3` clip's actual opening-melody notes (onset,
//! pitch, interval, emphasis) so a measured contour/rhythm anomaly can be
//! traced back to its real cause instead of guessed at. Reconstructs the
//! clip's (style, seed) exactly as `march_tango_contour_probe.rs` does.
//!
//! Run: `cargo run --example clip_note_trace -p symthaea-music-theory -- <clip_number> [opening_bars]`

use symthaea_music_theory::score::VoiceRole;
use symthaea_music_theory::{MusicalIntent, Style, compose_with_spec};

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
    let args: Vec<String> = std::env::args().collect();
    let clip_num: usize = args
        .get(1)
        .expect("usage: clip_note_trace <clip_number> [opening_bars]")
        .parse()
        .expect("clip_number must be a positive integer");
    let opening_bars: f64 = args
        .get(2)
        .map(|s| s.parse().expect("opening_bars must be a number"))
        .unwrap_or(8.0);

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

    let (style, k, seed) = clips[clip_num - 1];
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
    let mut melody: Vec<_> = score
        .notes
        .iter()
        .filter(|n| n.role == VoiceRole::Melody)
        .collect();
    melody.sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));

    let cutoff = score.meter as f64 * opening_bars;
    println!(
        "clip_{clip_num:02}  {style:?} (tier k={k}, arousal={arousal}, valence={valence}, seed={seed})"
    );
    println!("opening window: {opening_bars} bars = {cutoff} beats\n");
    let mut prev: Option<(f64, u8)> = None;
    for n in &melody {
        let onset = n.onset.beats();
        if onset >= cutoff {
            break;
        }
        let midi = n.pitch.midi();
        if let Some((_, prev_midi)) = prev {
            let iv = midi as i32 - prev_midi as i32;
            let tag = match iv.abs() {
                1 => "  <-- minor second",
                n if n >= 5 && n < 12 => "  <-- large leap",
                12.. => "  <-- OCTAVE LEAP",
                _ => "",
            };
            println!(
                "  onset={onset:6.2}  dur={:5.2}  midi={midi:3}  interval={iv:+3}  emphasis={:?}{tag}",
                n.duration.beats(),
                n.emphasis
            );
        } else {
            println!(
                "  onset={onset:6.2}  dur={:5.2}  midi={midi:3}  (first note)  emphasis={:?}",
                n.duration.beats(),
                n.emphasis
            );
        }
        prev = Some((onset, midi));
    }
}
