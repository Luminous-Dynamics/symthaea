// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Composition-vs-rendering ablation, arm 2 of the Muse diversity truth
//! plan (Phase 1): **one score × N performance dressings**.
//!
//! The style-ID listening test (`bin/listening_test.rs`) asks whether the
//! *composition* carries style identity when rendering is held fixed. This
//! example asks the complementary question: how much identity does the
//! *rendering* carry when the composition is held fixed?
//!
//! One `(MusicalIntent, CompositionSpec, seed)` is frozen — Classical spec,
//! seed 42, default-ish intent — and composed EXACTLY ONCE (composition is
//! deterministic). The same symbolic score is then realized under
//! contrasting performance dressings:
//!
//! - (a) native renderer, default `MusicalState`;
//! - (b) `MusicalState` extremes (consciousness 0.1 vs 0.9);
//! - (c) swing 0.50 vs 0.63 (`spec.texture.swing`) — swing is applied at
//!   the PERFORMANCE layer (see `TextureSpec::swing`'s docs: "notation
//!   stays straight, the player swings"), so this arm re-*performs* the
//!   same symbolic score with different off-beat timing; it never
//!   re-composes. Note the Classical spec's default swing is already 0.50,
//!   so the swing-0.50 arm doubles as a determinism check against (a);
//! - (d) a swapped ensemble (different instruments, same notes);
//! - (e) optionally, a FluidSynth soundfont render of the same performance
//!   (`midi_export::export_performance_midi` + `fluid_render`) when the
//!   environment provides `SYMTHAEA_SOUNDFONT` + fluidsynth; skipped (not
//!   failed) otherwise.
//!
//! WAVs are written blind (shuffled clip numbers) with an answer-key JSON,
//! plus per-arm cheap descriptive stats (peak, RMS, zero-crossing rate) so
//! the arms are verifiably different files, and a README explaining the
//! blind-listening protocol.
//!
//! Usage:
//! ```bash
//! cargo run --release -p symthaea-muse --features theory \
//!     --example render_ablation
//! # optional fluidsynth arm:
//! SYMTHAEA_SOUNDFONT=/path/to/FluidR3_GM2-2.sf2 \
//! SYMTHAEA_FLUIDSYNTH=$(command -v fluidsynth) \
//! cargo run --release -p symthaea-muse --features theory \
//!     --example render_ablation
//! ```

use std::io::Write as _;
use std::path::Path;

use symthaea_muse::theory_realize::realize_with_spec;
use symthaea_muse::{AudioData, MusicalState};
use symthaea_music_theory::{MusicalIntent, Style, compose_with_spec};

const SAMPLE_RATE: u32 = 44_100;
const SEED: u64 = 42;

struct Arm {
    label: &'static str,
    /// One-line description for the answer key.
    what: &'static str,
    wav: Vec<u8>,
    stats: Stats,
}

#[derive(Clone, Copy)]
struct Stats {
    peak: f32,
    rms: f32,
    /// Zero-crossing rate: crossings per sample, mono mix.
    zcr: f32,
    seconds: f32,
}

fn main() {
    let out_dir = Path::new("audio_output/render_ablation");
    std::fs::create_dir_all(out_dir).expect("create output dir");

    // The ONE frozen composition: Classical spec, seed 42, default-ish
    // intent. Composed exactly once — every arm below re-performs this
    // same symbolic score.
    let spec = Style::Classical.spec();
    let intent = MusicalIntent {
        seed: SEED,
        ..Default::default()
    };
    let score = compose_with_spec(&intent, &spec);
    println!(
        "Composed once: Classical spec, seed {SEED} — {} notes, tempo {:.0} bpm, meter {}/4.",
        score.notes.len(),
        score.tempo_bpm,
        score.meter
    );

    let mut arms: Vec<Arm> = Vec::new();
    let push_native = |arms: &mut Vec<Arm>,
                       label: &'static str,
                       what: &'static str,
                       spec: &symthaea_music_theory::CompositionSpec,
                       state: &MusicalState| {
        let comp = realize_with_spec(&score, spec, SEED, state, SAMPLE_RATE);
        let stats = audio_stats(&comp.audio, SAMPLE_RATE);
        let wav = symthaea_muse::export::wav_bytes(&comp).expect("wav encode");
        arms.push(Arm {
            label,
            what,
            wav,
            stats,
        });
    };

    // (a) native default.
    let default_state = MusicalState::default();
    push_native(
        &mut arms,
        "native_default",
        "native renderer, default MusicalState, spec as authored (swing 0.50)",
        &spec,
        &default_state,
    );

    // (b) MusicalState extremes: consciousness 0.1 vs 0.9.
    let low = MusicalState {
        consciousness_level: 0.1,
        ..Default::default()
    };
    let high = MusicalState {
        consciousness_level: 0.9,
        ..Default::default()
    };
    push_native(
        &mut arms,
        "state_consciousness_0.1",
        "native renderer, MusicalState consciousness_level = 0.1",
        &spec,
        &low,
    );
    push_native(
        &mut arms,
        "state_consciousness_0.9",
        "native renderer, MusicalState consciousness_level = 0.9",
        &spec,
        &high,
    );

    // (c) swing 0.50 vs 0.63. Performance-layer timing only: the symbolic
    // score is NOT recomposed, the same notes are re-performed with a
    // different off-beat position. swing_0.50 should be byte-identical to
    // native_default (the Classical spec's default swing is 0.50) — a
    // built-in determinism check.
    let mut spec_straight = spec.clone();
    spec_straight.texture.swing = 0.50;
    let mut spec_shuffle = spec.clone();
    spec_shuffle.texture.swing = 0.63;
    push_native(
        &mut arms,
        "swing_0.50",
        "native renderer, default state, swing pinned 0.50 (straight; determinism check vs native_default)",
        &spec_straight,
        &default_state,
    );
    push_native(
        &mut arms,
        "swing_0.63",
        "native renderer, default state, swing 0.63 (shuffle) — same notes, re-performed timing",
        &spec_shuffle,
        &default_state,
    );

    // (d) swapped ensemble: same notes, different instruments. The
    // Classical pool is strings+piano; swap to a mallet/organ/bass trio.
    let mut spec_swapped = spec.clone();
    spec_swapped.ensemble_pool = vec![["marimba".into(), "organ".into(), "upright_bass".into()]];
    push_native(
        &mut arms,
        "ensemble_swapped",
        "native renderer, default state, ensemble swapped to marimba/organ/upright_bass",
        &spec_swapped,
        &default_state,
    );

    // (e) optional FluidSynth arm — same performance layer, soundfont
    // timbres. Skipped (not failed) when the env lacks fluidsynth or
    // SYMTHAEA_SOUNDFONT.
    if symthaea_muse::fluid_render::available().is_some() {
        let midi_path =
            std::env::temp_dir().join(format!("render_ablation_{}_{SEED}.mid", std::process::id()));
        let exported = symthaea_muse::midi_export::export_performance_midi(
            &score,
            &spec,
            SEED,
            &default_state,
            &midi_path,
        );
        match exported {
            Ok(()) => {
                match symthaea_muse::fluid_render::render_midi_to_wav(&midi_path, SAMPLE_RATE, None)
                {
                    Some(wav) => {
                        let stats = wav_stats(&wav);
                        arms.push(Arm {
                            label: "fluidsynth_default",
                            what: "FluidSynth soundfont render of the same performance MIDI (default state, spec as authored)",
                            wav,
                            stats,
                        });
                    }
                    None => eprintln!("fluidsynth arm SKIPPED: render failed or came back silent"),
                }
            }
            Err(e) => eprintln!("fluidsynth arm SKIPPED: MIDI export failed: {e}"),
        }
        let _ = std::fs::remove_file(&midi_path);
    } else {
        println!(
            "fluidsynth arm SKIPPED: SYMTHAEA_SOUNDFONT/fluidsynth not available in this environment."
        );
    }

    // Blind assignment: deterministic LCG Fisher-Yates shuffle (same
    // machinery as bin/listening_test.rs) so a set is reproducible.
    let mut order: Vec<usize> = (0..arms.len()).collect();
    shuffle(&mut order, 0xAB1A_7104);

    println!("\nPer-arm stats (arms must be verifiably different files):");
    println!(
        "  {:<26} {:>8} {:>8} {:>8} {:>8} {:>10}",
        "arm", "peak", "rms", "zcr", "secs", "wav bytes"
    );
    let mut key_lines = Vec::new();
    for (clip_no, &arm_idx) in order.iter().enumerate() {
        let arm = &arms[arm_idx];
        let name = format!("clip_{:02}.wav", clip_no + 1);
        std::fs::write(out_dir.join(&name), &arm.wav).expect("write clip");
        println!(
            "  {:<26} {:>8.4} {:>8.4} {:>8.4} {:>8.1} {:>10}",
            arm.label,
            arm.stats.peak,
            arm.stats.rms,
            arm.stats.zcr,
            arm.stats.seconds,
            arm.wav.len()
        );
        key_lines.push(format!(
            "  {{\"clip\": \"{name}\", \"arm\": \"{}\", \"what\": \"{}\", \
             \"peak\": {:.4}, \"rms\": {:.4}, \"zcr\": {:.4}, \"seconds\": {:.1}}}",
            arm.label, arm.what, arm.stats.peak, arm.stats.rms, arm.stats.zcr, arm.stats.seconds
        ));
    }

    let key = format!(
        "{{\n  \"note\": \"listen BEFORE reading; one frozen Classical/seed-{SEED} score, re-performed\",\n\
         \"clips\": [\n{}\n  ]\n}}\n",
        key_lines.join(",\n")
    );
    std::fs::write(out_dir.join("answer_key.json"), key).expect("write answer key");

    let mut readme = std::fs::File::create(out_dir.join("README.md")).expect("readme");
    writeln!(
        readme,
        "# Render ablation — does rendering carry identity?\n\n\
         Every clip in this directory is THE SAME COMPOSITION: one symbolic\n\
         score (Classical spec, seed {SEED}) composed exactly once, then\n\
         re-performed/re-rendered under contrasting dressings. Clip numbers\n\
         are shuffled; DO NOT open `answer_key.json` until you've listened.\n\n\
         ## Protocol\n\n\
         1. Listen to all clips blind, in any order.\n\
         2. For each PAIR of clips, note: do these sound like the same\n\
            piece? The same piece played differently? Different pieces?\n\
         3. Then open the answer key and check which dressing each clip is.\n\n\
         ## What each comparison answers\n\n\
         - **default vs consciousness 0.1 vs 0.9**: does the renderer's\n\
           `MusicalState` (Muse's live emotion/consciousness inputs) audibly\n\
           change a fixed score — i.e. is any measured 'diversity' from\n\
           state actually rendering dressing, not composition?\n\
         - **swing 0.50 vs 0.63**: does performance-layer timing alone\n\
           (same notes, shuffled off-beats) read as a different piece or\n\
           the same piece swung? (This arm re-performs, never re-composes:\n\
           swing is applied at the performance layer.) swing 0.50 should be\n\
           IDENTICAL to the default arm — a determinism check.\n\
         - **ensemble swapped**: do different instruments on identical\n\
           notes read as a new piece? If listeners call this 'a different\n\
           piece', timbre — not composition — is carrying identity.\n\
         - **fluidsynth (if present)**: does a soundfont render of the same\n\
           performance keep or change the piece's identity?\n\n\
         If clips here sound as different from each other as clips from the\n\
         style-ID listening test (different COMPOSITIONS), then rendering is\n\
         carrying most of the perceived identity and the composition layer\n\
         still needs work. If they all clearly read as one piece dressed\n\
         differently, the composed thought is what's being perceived.\n\n\
         Per-clip descriptive stats (peak, RMS, zero-crossing rate) live in\n\
         `answer_key.json` — they prove the files are physically different\n\
         (or identical, for the determinism-check pair), independent of ears."
    )
    .unwrap();

    println!(
        "\nWrote {} clips + README.md + answer_key.json to {}",
        arms.len(),
        out_dir.display()
    );
}

/// Deterministic shuffle (LCG Fisher-Yates), same as bin/listening_test.rs.
fn shuffle<T>(items: &mut [T], mut state: u64) {
    for i in (1..items.len()).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = ((state >> 33) as usize) % (i + 1);
        items.swap(i, j);
    }
}

/// Cheap descriptive stats over a composition's audio (mono mix).
fn audio_stats(audio: &AudioData, sample_rate: u32) -> Stats {
    let mono: Vec<f32> = match audio {
        AudioData::I16(v) => v.iter().map(|&s| s as f32 / 32768.0).collect(),
        AudioData::F32(v) => v.clone(),
        AudioData::StereoF32(v) => v.iter().map(|[l, r]| 0.5 * (l + r)).collect(),
    };
    stats_from_mono(&mono, sample_rate)
}

/// Stats for an in-memory WAV (the FluidSynth arm), decoded via hound.
fn wav_stats(wav: &[u8]) -> Stats {
    let reader = hound::WavReader::new(std::io::Cursor::new(wav)).expect("parse wav");
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.expect("sample") as f32 / max)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.expect("sample"))
            .collect(),
    };
    let mono: Vec<f32> = samples
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect();
    stats_from_mono(&mono, spec.sample_rate)
}

fn stats_from_mono(mono: &[f32], sample_rate: u32) -> Stats {
    let n = mono.len().max(1);
    let peak = mono.iter().fold(0.0f32, |acc, s| acc.max(s.abs()));
    let rms = (mono.iter().map(|s| s * s).sum::<f32>() / n as f32).sqrt();
    let crossings = mono
        .windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count();
    Stats {
        peak,
        rms,
        zcr: crossings as f32 / n as f32,
        seconds: mono.len() as f32 / sample_rate as f32,
    }
}
