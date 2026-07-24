// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Quick spike (SYMTHAEA_SINGING_PLAN_2026-07-18.md, post-Phase-6 pivot): after
//! listening to the formant-vocoder singing WAVs, the user rejected that voice
//! outright ("not pleasant") and asked whether pitch-shifted Kokoro (a real
//! pretrained neural TTS, already verified 0% WER for speech) sounds better as
//! a singing voice, BEFORE committing to a full melody-alignment
//! implementation.
//!
//! This is deliberately the crudest possible test of that question: synthesize
//! one phrase once via Kokoro at its natural pitch, then produce a handful of
//! resample-based pitch-shifted variants spanning roughly the range a real
//! short melody would need. Resample-based shifting changes duration along
//! with pitch (the classic "chipmunk" tradeoff) — that's fine for this spike,
//! which is judging TIMBRE quality across a pitch range, not melody fidelity.
//! If the timbre holds up, a real implementation needs a proper time-preserving
//! pitch shift (phase vocoder, e.g. via the `rustfft`/`rubato` crates already
//! in this workspace) plus per-word/syllable alignment to bend each syllable
//! onto its own note — neither attempted here.
//!
//! ```bash
//! nix develop -c cargo run --example kokoro_singing_spike --features voice-tts
//! ```

use anyhow::Result;
use symthaea::voice::{KokoroConfig, KokoroEngine, save_wav};

const PHRASE: &str = "hello world";

/// Same pattern as `examples/voice_roundtrip_wer.rs::linear_resample`, reused
/// here as a resample-based pitch shifter: treating the input as if it were
/// recorded at `orig_rate * ratio` and resampling it down to `orig_rate`
/// compresses (ratio > 1, higher pitch, shorter) or stretches (ratio < 1,
/// lower pitch, longer) the waveform in time -- pitch and duration move
/// together, unlike a phase vocoder.
fn linear_resample(input: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || input.is_empty() {
        return input.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let output_len = ((input.len() as f64) * ratio).round().max(1.0) as usize;
    (0..output_len)
        .map(|i| {
            let src = i as f64 / ratio;
            let idx = src as usize;
            let frac = (src - idx as f64) as f32;
            match (input.get(idx), input.get(idx + 1)) {
                (Some(&a), Some(&b)) => a * (1.0 - frac) + b * frac,
                (Some(&a), None) => a,
                _ => 0.0,
            }
        })
        .collect()
}

fn pitch_shift(samples: &[f32], orig_rate: u32, semitones: f32) -> Vec<f32> {
    let ratio = 2.0_f32.powf(semitones / 12.0);
    linear_resample(samples, (orig_rate as f32 * ratio) as u32, orig_rate)
}

fn main() -> Result<()> {
    println!("Kokoro singing spike: synthesizing {PHRASE:?}...");
    let config = KokoroConfig::default();
    let mut engine = KokoroEngine::load(config)
        .ok_or_else(|| anyhow::anyhow!("Kokoro engine failed to load"))?;
    let sample_rate = engine.sample_rate();

    let samples = engine
        .synthesize(PHRASE, None)
        .ok_or_else(|| anyhow::anyhow!("Kokoro synthesis returned no audio"))?;
    println!(
        "  Synthesized {} samples ({:.2}s) at {sample_rate}Hz",
        samples.len(),
        samples.len() as f32 / sample_rate as f32
    );

    let out_dir = "audio_output/kokoro_singing_spike_2026-07-18";
    std::fs::create_dir_all(out_dir)?;

    // A spread roughly matching a real short melody's range: a fifth down to
    // an octave up, plus the unshifted baseline for comparison.
    let shifts: &[(&str, f32)] = &[
        ("00_baseline_natural", 0.0),
        ("01_down_fifth", -7.0),
        ("02_down_third", -4.0),
        ("03_up_third", 4.0),
        ("04_up_fifth", 7.0),
        ("05_up_octave", 12.0),
    ];

    for (name, semitones) in shifts {
        let shifted = if *semitones == 0.0 {
            samples.clone()
        } else {
            pitch_shift(&samples, sample_rate, *semitones)
        };
        let path = format!("{out_dir}/{name}.wav");
        save_wav(&shifted, sample_rate, &path)?;
        println!(
            "  [{semitones:+.0} semitones] {:.2}s -> {path}",
            shifted.len() as f32 / sample_rate as f32
        );
    }

    println!("\nDone. Listen and compare against the formant-vocoder WAVs in");
    println!("audio_output/singing_intelligibility_2026-07-18/.");
    Ok(())
}
