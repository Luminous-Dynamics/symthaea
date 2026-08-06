// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Track A smoke test: has anyone ever actually listened to this crate's
//! audio output? (Answer before this file existed: no.)
//!
//! Drives `VocalTractPipeline::tick_phoneme` through ARPAbet phoneme
//! sequences for three short words (genesis-random weights, no training),
//! collects the resulting `FormantFrame` trajectory, and renders it to real
//! PCM audio via the crate's own legacy formant vocoder
//! (`speech::vocoder::synthesize`) -- the only audio-producing backend this
//! crate has (the Series 23 physical renderer is orphaned/out of scope; see
//! `speech.rs` module doc). Writes one .wav per phrase plus a debug CSV of
//! the frame trajectory. Output goes to /tmp, never into the repo.
//!
//! ```bash
//! cargo run -p symthaea-vocal-tract --example track_a_smoke_test --features hound
//! ```

use std::fs;
use std::io::Write;

use symthaea_core::genesis::GenesisSeed;
use symthaea_vocal_tract::encoder::VoiceCognitiveState;
use symthaea_vocal_tract::pipeline::VocalTractPipeline;
use symthaea_vocal_tract::speech;
use symthaea_vocal_tract::types::FormantFrame;

const SAMPLE_RATE: u32 = 22_050;
const DT: f32 = 0.005; // 200 Hz motor tick
const FRAMES_PER_PHONEME: usize = 10; // 10 * 5ms = 50ms per phoneme

/// Three short, clearly-articulated words, spelled out in the ARPAbet
/// symbols `phonetics::canonical_arpabet_symbol` accepts (verified against
/// source before writing this): every symbol below is one of
/// AA/AE/AH/AO/AW/AY/EH/ER/EY/IH/IY/OW/OY/UH/UW (vowel), P/B/T/D/K/G (stop),
/// F/V/TH/DH/S/Z/SH/ZH/HH (fricative), M/N/NG (nasal), L/R (liquid),
/// W/Y (glide), CH/JH (affricate).
const PHRASES: &[(&str, &[&str])] = &[
    ("hello", &["HH", "AH", "L", "OW"]),
    ("cat", &["K", "AE", "T"]),
    ("goodnight", &["G", "UH", "D", "N", "AY", "T"]),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = "/tmp/vocal-tract-track-a-smoketest";
    fs::create_dir_all(out_dir)?;

    println!("=== symthaea-vocal-tract Track A smoke test ===");
    println!(
        "Audio backend in use: speech::vocoder::synthesize (legacy formant \
         cascade synthesizer -- 3 StableResonators + glottal/noise excitation). \
         This IS a real audio-producing path, verified present in speech.rs."
    );
    println!("Sample rate: {SAMPLE_RATE} Hz, output dir: {out_dir}\n");

    for (name, phonemes) in PHRASES {
        let genesis = GenesisSeed::from_phrase(&format!("track-a-smoke::{name}"));
        let mut pipeline = VocalTractPipeline::new(&genesis);
        let state = VoiceCognitiveState::default();

        let mut frames: Vec<FormantFrame> = Vec::new();
        for phoneme in *phonemes {
            for _ in 0..FRAMES_PER_PHONEME {
                let frame = pipeline.tick_phoneme(&state, None, DT, Some(phoneme));
                frames.push(frame);
            }
        }

        let duration_s = frames.len() as f32 * DT;
        println!(
            "[{name}] phonemes={:?} frames={} duration={:.3}s",
            phonemes,
            frames.len(),
            duration_s
        );

        // Render real PCM via the legacy formant vocoder.
        let samples = speech::vocoder::synthesize(&frames, SAMPLE_RATE);
        let non_silent = samples.iter().any(|s| s.abs() > 1e-4);
        let peak = samples.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        println!(
            "  -> synthesized {} PCM samples ({:.3}s at {SAMPLE_RATE}Hz), \
             non_silent={non_silent}, peak_amplitude={peak:.4}",
            samples.len(),
            samples.len() as f32 / SAMPLE_RATE as f32
        );

        #[cfg(feature = "hound")]
        {
            let wav_path = format!("{out_dir}/{name}.wav");
            symthaea_vocal_tract::metrics::save_wav(&wav_path, &samples, SAMPLE_RATE)?;
            println!("  -> wrote {wav_path}");
        }
        #[cfg(not(feature = "hound"))]
        {
            println!(
                "  -> 'hound' feature not enabled: skipping .wav write. \
                 Re-run with --features hound to actually write audio to disk."
            );
        }

        // Always write the debug FormantFrame trajectory as CSV, independent
        // of the audio backend, so the raw articulatory data is inspectable
        // even if no audio feature is enabled.
        let csv_path = format!("{out_dir}/{name}_frames.csv");
        let mut f = fs::File::create(&csv_path)?;
        writeln!(f, "time,f1,f2,f3,b1,b2,b3,f0,energy,voicing")?;
        for frame in &frames {
            writeln!(
                f,
                "{},{},{},{},{},{},{},{},{},{}",
                frame.time,
                frame.f1,
                frame.f2,
                frame.f3,
                frame.b1,
                frame.b2,
                frame.b3,
                frame.f0,
                frame.energy,
                frame.voicing
            )?;
        }
        println!("  -> wrote {csv_path}\n");
    }

    println!(
        "Done. This is the first time this crate's synthesized audio has been written to disk."
    );
    Ok(())
}
