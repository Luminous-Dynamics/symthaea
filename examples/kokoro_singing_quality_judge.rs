// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Direct answer to "can you use something to judge the audio?": this
//! codebase already has a purpose-built singing-quality judge
//! (`singing_quality::analyze_vocal_stem`) that nobody had used yet on
//! the Kokoro pipeline -- unlike `analyze_render_cleanliness` (generic
//! click/clipping checks with no ground truth) or Whisper WER (measures
//! intelligibility, not musicality, and needs a live ASR worker), this
//! one compares the ACTUAL sung pitch and timing against the INTENDED
//! melody note-by-note: pitch accuracy in cents, voiced/unvoiced
//! detection, onset timing, duration accuracy. It rolls everything into
//! a single `objective_pass` bool plus the per-dimension breakdown that
//! failed it. Its own doc comment is honest that "human ratings remain
//! mandatory" -- this is a second opinion, not a replacement for actually
//! listening, but it's a much more informative one than pure audio
//! cleanliness.
//!
//! Renders the same 5-phrase set as the rest of this investigation
//! through the exact same path `sing_with_kokoro` uses (including the
//! 2026-07-22 melody-timeline-normalization fix), but keeps the
//! intermediate `VocalPerformance`/`VocalStem` around instead of
//! discarding them, since `analyze_vocal_stem` needs both.
//!
//! Registered as `[[bin]]` (same symthaea-humanoid-broken-at-HEAD reason
//! as the other kokoro_singing_* tools).
//!
//! ```bash
//! ORT_DYLIB_PATH=/path/to/libonnxruntime.so \
//! nix develop -c cargo run --bin kokoro_singing_quality_judge --features "singing,voice-tts"
//! ```

use symthaea::voice::kokoro_singing::KokoroSingingEngine;
use symthaea::voice::singing_engine::{SingingVoiceEngine, VocalPerformance};
use symthaea::voice::singing_quality::analyze_vocal_stem;
use symthaea_muse::{MusicalIntent, MusicalState, Note, Style};

const PHRASES: &[&str] = &[
    "hello world",
    "the sun rises over the valley",
    "I am singing a real melody",
    "consciousness shapes every note",
    "a quiet morning walk",
];

/// Same offset-normalization `sing_with_kokoro` applies internally --
/// duplicated here (rather than calling `sing_with_kokoro` directly)
/// because that function only returns raw PCM, not the `VocalPerformance`
/// `analyze_vocal_stem` needs as ground truth.
fn normalize_melody_timeline(melody: &[Note]) -> Vec<Note> {
    let offset = melody
        .iter()
        .map(|note| note.start_time)
        .fold(f32::MAX, f32::min)
        .max(0.0);
    if offset > 0.0 {
        melody
            .iter()
            .map(|note| Note {
                start_time: note.start_time - offset,
                ..*note
            })
            .collect()
    } else {
        melody.to_vec()
    }
}

fn main() -> anyhow::Result<()> {
    let state = MusicalState::default();
    let voice = "af_heart";

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
        let target_notes = symthaea_muse::singing_bridge::syllable_count(lyrics).max(1);
        melody.truncate(target_notes);
        let melody = normalize_melody_timeline(&melody);

        let performance = match VocalPerformance::from_melody(lyrics, &melody, "en") {
            Ok(p) => p,
            Err(e) => {
                println!("[{i}] {lyrics:?}: VocalPerformance::from_melody failed: {e}");
                continue;
            }
        };
        let mut engine = match KokoroSingingEngine::load(voice) {
            Ok(e) => e,
            Err(e) => {
                println!("[{i}] {lyrics:?}: engine load failed: {e}");
                continue;
            }
        };
        let stem = match engine.render(&performance) {
            Ok(s) => s,
            Err(e) => {
                println!("[{i}] {lyrics:?}: render failed: {e}");
                continue;
            }
        };

        let m = analyze_vocal_stem(&performance, &stem);
        println!(
            "\n=== [{i}] {lyrics:?} ===  ({:.2}s audio)",
            stem.samples.len() as f32 / stem.sample_rate as f32
        );
        println!(
            "  voiced_frame_fraction={:.3}  voiced_unvoiced_error_rate={:.3}",
            m.voiced_frame_fraction, m.voiced_unvoiced_error_rate
        );
        println!(
            "  median_pitch_error_cents={:.1}  p95_pitch_error_cents={:.1}  stable_pitch_p95_cents={:.1}  transition_pitch_p95_cents={:.1}",
            m.median_pitch_error_cents,
            m.p95_pitch_error_cents,
            m.stable_pitch_p95_cents,
            m.transition_pitch_p95_cents
        );
        println!(
            "  median_onset_error_ms={:.1}  duration_error_ms={:.1}",
            m.median_onset_error_ms, m.duration_error_ms
        );
        println!(
            "  pitch_tracking_pass={}  transition_pitch_pass={}  timing_pass={}  physical_stability_pass={}  render_cleanliness_pass={}",
            m.pitch_tracking_pass,
            m.transition_pitch_pass,
            m.timing_pass,
            m.physical_stability_pass,
            m.render_cleanliness_pass
        );
        println!("  >>> OBJECTIVE_PASS={} <<<", m.objective_pass);
    }

    Ok(())
}
