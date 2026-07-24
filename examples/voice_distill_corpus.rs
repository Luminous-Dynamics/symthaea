// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Generate the Kokoro-teacher distillation corpus (voice plan, 2026-07-16).
//!
//! Synthesizes each training sentence with the verified Kokoro engine
//! (0% round-trip WER, `VOICE_KOKORO_VERIFICATION_2026-07-16.json`) and
//! writes `NN.wav` + a `manifest.tsv` (`NN.wav<TAB>text`) into `--out`.
//!
//! The training set is DISJOINT from the 15-sentence evaluation gate in
//! `voice_roundtrip_wer.rs` / `voice_oracle_wer.rs` — the gate must never
//! appear in training data, or the WER comparison is leakage.
//!
//! ```bash
//! ORT_DYLIB_PATH=... cargo run --example voice_distill_corpus \
//!     --features voice-tts -- --out /path/to/corpus
//! ```

use std::path::PathBuf;
use std::time::Instant;

use symthaea::voice::{KokoroConfig, KokoroEngine};

/// 60 phonetically diverse training sentences. Everyday vocabulary plus
/// deliberate coverage of clusters, fricatives, nasals, diphthongs, and
/// varied sentence lengths. Disjoint from the evaluation gate.
const TRAIN_SENTENCES: &[&str] = &[
    "the sun rises over the quiet valley",
    "she counted twelve bright stars tonight",
    "fresh bread smells wonderful in the morning",
    "the river bends around the old stone bridge",
    "children laugh while playing in the garden",
    "a gentle breeze moves through the tall grass",
    "he placed the blue cup on the wooden table",
    "the train arrives at seven every evening",
    "autumn leaves drift slowly to the ground",
    "the library keeps rare books on the third floor",
    "thunder rolled across the distant mountains",
    "her voice carried clearly through the hall",
    "the baker shapes the dough with steady hands",
    "small boats rock gently in the harbor",
    "the clock on the wall ticks past midnight",
    "winter mornings begin with frost on the glass",
    "the teacher explained the lesson twice",
    "a curious fox watched from the tree line",
    "the engine hummed as the ship left port",
    "warm soup tastes better on a cold day",
    "the photographer waited for perfect light",
    "green apples hang low on the branch",
    "the musician tuned her violin carefully",
    "storm clouds gathered above the empty field",
    "the letter arrived three weeks too late",
    "soft rain fell against the window all night",
    "the mechanic checked the brakes and the oil",
    "wild horses ran along the western ridge",
    "the chef added pepper and a pinch of salt",
    "moonlight spread across the silver lake",
    "the students finished their projects early",
    "a narrow path leads down to the shore",
    "the farmer planted rows of yellow corn",
    "quiet footsteps echoed in the long corridor",
    "the artist mixed orange paint with red",
    "heavy snow closed the mountain road again",
    "the doctor listened to every question",
    "fireflies blinked above the summer meadow",
    "the carpenter measured the plank twice",
    "old friends met for coffee near the station",
    "the judge read the verdict in a calm voice",
    "sparrows gathered crumbs beneath the bench",
    "the pilot checked the weather before takeoff",
    "fresh strawberries filled the wooden basket",
    "the professor wrote equations on the board",
    "waves crashed against the rocky point",
    "the tailor stitched the hem with white thread",
    "distant church bells rang through the fog",
    "the gardener trimmed the hedge by hand",
    "hot tea steamed in the porcelain pot",
    "the swimmer crossed the bay before noon",
    "lanterns glowed along the festival street",
    "the editor cut the second paragraph",
    "brave explorers mapped the frozen coast",
    "the plumber fixed the leak under the sink",
    "ripe oranges dropped from the heavy branch",
    "the singer practiced scales every morning",
    "long shadows stretched across the courtyard",
    "the banker locked the vault at five",
    "crickets chirped beyond the open door",
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
    let args: Vec<String> = std::env::args().collect();
    let out = PathBuf::from(
        args.iter()
            .position(|a| a == "--out")
            .and_then(|i| args.get(i + 1))
            .ok_or_else(|| anyhow::anyhow!("--out <dir> required"))?,
    );
    std::fs::create_dir_all(&out)?;

    // Teacher voice selection (v2.5 probe, 2026-07-17): LPC F1 estimation is
    // biased on high-f0 female voices (harmonic spacing exceeds formant
    // bandwidth — the measured vowel space collapsed to F1 286-516 with the
    // default af_heart). A male teacher (f0 ~110-130Hz) tests that
    // hypothesis directly: e.g. --voice-file voices/am_michael.bin
    let voices_filename = args
        .iter()
        .position(|a| a == "--voice-file")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| KokoroConfig::default().voices_filename);
    println!("teacher voice: {voices_filename}");

    let config = KokoroConfig {
        voices_filename,
        ..KokoroConfig::default()
    };
    let mut engine = KokoroEngine::load(config)
        .ok_or_else(|| anyhow::anyhow!("Kokoro engine failed to load (ORT_DYLIB_PATH set?)"))?;
    let sr = engine.sample_rate();

    let mut manifest = String::new();
    let start = Instant::now();
    let mut ok = 0usize;
    for (i, sentence) in TRAIN_SENTENCES.iter().enumerate() {
        let t = Instant::now();
        match engine.synthesize(sentence, None) {
            Some(audio) if !audio.is_empty() => {
                let name = format!("{i:02}.wav");
                save_wav(&out.join(&name), &audio, sr)?;
                manifest.push_str(&format!("{name}\t{sentence}\n"));
                ok += 1;
                println!(
                    "  [{i:2}] {:.1}s audio in {:.1}s: {sentence:?}",
                    audio.len() as f32 / sr as f32,
                    t.elapsed().as_secs_f32()
                );
            }
            _ => println!("  [{i:2}] SYNTHESIS FAILED: {sentence:?}"),
        }
    }
    std::fs::write(out.join("manifest.tsv"), manifest)?;
    println!(
        "\nCorpus: {ok}/{} sentences in {:.0}s -> {}",
        TRAIN_SENTENCES.len(),
        start.elapsed().as_secs_f32(),
        out.display()
    );
    Ok(())
}
