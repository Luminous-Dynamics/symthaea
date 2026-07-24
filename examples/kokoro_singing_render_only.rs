// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Same phrase set and melody composition as
//! `kokoro_singing_intelligibility_gate.rs`, but skips the (slow, and on
//! this occasion apparently hung) Whisper transcription step entirely --
//! for fast iteration on the RENDER side (the offset-normalization fix)
//! without waiting on the ASR worker.
//!
//! Registered as `[[bin]]` (same symthaea-humanoid-broken-at-HEAD reason
//! as the other kokoro_singing_* tools).
//!
//! ```bash
//! nix develop -c cargo run --bin kokoro_singing_render_only --features "singing,voice-tts" -- <out-dir>
//! ```

use std::path::PathBuf;

use symthaea::voice::kokoro_singing::sing_with_kokoro;
use symthaea_muse::{MusicalIntent, MusicalState, Style};

const PHRASES: &[&str] = &[
    "hello world",
    "the sun rises over the valley",
    "I am singing a real melody",
    "consciousness shapes every note",
    "a quiet morning walk",
];

fn save_wav(path: &std::path::Path, samples: &[f32], sample_rate: u32) -> anyhow::Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        writer.write_sample((s * 32767.0).clamp(-32768.0, 32767.0) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let out_dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "audio_output/kokoro_singing_render_only".to_string()),
    );
    std::fs::create_dir_all(&out_dir)?;

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

        match sing_with_kokoro(lyrics, &melody, voice) {
            Ok((audio, sample_rate)) if !audio.is_empty() => {
                let secs = audio.len() as f32 / sample_rate as f32;
                let path = out_dir.join(format!("{i:02}_{}.wav", lyrics.replace(' ', "_")));
                save_wav(&path, &audio, sample_rate)?;
                println!("  [{i}] {lyrics:?} -> {secs:.2}s -> {}", path.display());
            }
            Ok(_) => println!("  [{i}] {lyrics:?}: empty audio"),
            Err(e) => println!("  [{i}] {lyrics:?}: failed: {e}"),
        }
    }

    Ok(())
}
