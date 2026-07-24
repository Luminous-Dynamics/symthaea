// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fast (no Kokoro synthesis needed) inspection of the syllable/note
//! timing `KokoroSingingEngine::render()` actually places into the output
//! buffer -- chasing the "most of the audio is silent, some parts very
//! sped up or slowed down" report after the phase-vocoder norm-floor
//! hypothesis was empirically ruled out (relaxing it barely moved the
//! measured silence fraction). Constructs the exact same
//! `VocalPerformance` `sing_with_kokoro` would, using the exact same
//! melody-compose-then-truncate steps as
//! `kokoro_singing_intelligibility_gate.rs`, and prints each syllable's
//! placed start/end time directly -- if there's a large gap before the
//! first syllable or between syllables, that's the real mechanism, not a
//! DSP artifact.
//!
//! Registered as `[[bin]]` (same symthaea-humanoid-broken-at-HEAD reason
//! as the other kokoro_singing_* tools).
//!
//! ```bash
//! nix develop -c cargo run --bin kokoro_singing_timing_probe --features "singing,voice-tts"
//! ```

use symthaea::voice::singing_engine::VocalPerformance;
use symthaea_muse::{MusicalIntent, MusicalState, Style};

const PHRASES: &[&str] = &[
    "hello world",
    "the sun rises over the valley",
    "I am singing a real melody",
    "consciousness shapes every note",
    "a quiet morning walk",
];

fn main() -> anyhow::Result<()> {
    let state = MusicalState::default();

    for (i, lyrics) in PHRASES.iter().enumerate() {
        let n_words = lyrics.split_whitespace().count();
        let bars = (n_words as f32 / 2.0).ceil().max(1.0) as usize;
        let intent = MusicalIntent {
            valence: 0.0,
            arousal: 0.4,
            energy: 0.5,
            bars,
            seed: i as u64,
            ..MusicalIntent::default()
        };
        let mut melody = symthaea_muse::theory_realize::compose_and_perform_melody(
            &intent,
            Style::Classical,
            &state,
        );
        let full_len = melody.len();
        let target_notes = symthaea_muse::singing_bridge::syllable_count(lyrics).max(1);

        println!("\n=== [{i}] {lyrics:?} ===");
        println!(
            "  composed melody: {full_len} notes, target_notes (syllable_count)={target_notes}"
        );
        println!("  full composed melody note timing (start_time, duration, freq):");
        for (n, note) in melody.iter().enumerate() {
            println!(
                "    note[{n:>2}]  start={:6.3}s  dur={:5.3}s  end={:6.3}s  freq={:6.1}Hz",
                note.start_time,
                note.duration,
                note.start_time + note.duration,
                note.frequency
            );
        }

        melody.truncate(target_notes);

        let performance = match VocalPerformance::from_melody(lyrics, &melody, "en") {
            Ok(p) => p,
            Err(e) => {
                println!("  VocalPerformance::from_melody failed: {e}");
                continue;
            }
        };

        println!(
            "  {} syllables placed after truncation to {target_notes} notes:",
            performance.syllables.len()
        );
        for (s, syl) in performance.syllables.iter().enumerate() {
            let placed_start = (syl.note.start_time - syl.consonant_advance_s).max(0.0);
            println!(
                "    syllable[{s}] {:?}  note.start={:6.3}s  note.dur={:5.3}s  placed_start={:6.3}s  end_time_s={:6.3}s  melisma_notes={}",
                syl.lyric,
                syl.note.start_time,
                syl.note.duration,
                placed_start,
                syl.end_time_s(),
                syl.melisma_notes.len()
            );
        }
    }

    Ok(())
}
