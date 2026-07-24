// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! P2.3-style release gate for the SINGING path (SYMTHAEA_SINGING_PLAN_2026-07-18.md
//! Phase 3): unlike speech, there is no existing "sung lyrics" reference corpus lying
//! around on disk. This is the cheaper of the two options that plan names —
//! **self-consistency**: sing a phrase to a real composed melody, transcribe the
//! result, and compare the transcript back to the ORIGINAL lyrics that were fed in.
//! Weaker evidence than an independently-authored reference corpus, but it reuses the
//! exact scoring/worker infrastructure already proven this session
//! (`examples/fleurs_release_gate.rs`) and catches gross intelligibility regressions
//! in the sing_to_pcm -> FormantVocoder chain.
//!
//! Every phrase is sung at a fixed, neutral consciousness state (not varied) —
//! consciousness-driven melody variation was already covered by SING-1's unit tests;
//! this eval isolates "does the composed-and-sung path stay intelligible" from "does
//! it vary with state."
//!
//! ```bash
//! SYMTHAEA_FASTER_WHISPER_MODEL_PATH=/srv/luminous-dynamics/models/communication/faster-whisper-large-v3 \
//! nix develop -c cargo run --example singing_intelligibility_gate --features "singing,voice-tts" -- \
//!     [--worker communication/worker/run_faster_whisper_nixos.sh] [--report /path/to/report.json] \
//!     [--audio-dir /path/to/dir]
//! ```
//!
//! `--worker` also accepts `communication/worker/run_whisper_nixos.sh` (the
//! HF `transformers` backend, with `SYMTHAEA_WHISPER_MODEL_PATH` and
//! `SYMTHAEA_PROVIDER_ID=whisper-large-v3-local` set instead) to cross-check
//! against a different Whisper implementation. `--audio-dir` writes each
//! phrase's sung PCM as a real WAV file — see
//! SYMTHAEA_SINGING_PLAN_2026-07-18.md's closing assessment: every
//! intelligibility measurement here has so far been automated ASR
//! self-consistency; nobody has actually listened to the audio.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde::Serialize;
use symthaea::voice::vocoder::FormantVocoder;
use symthaea_communication::human::{LocalJsonlProvider, WorkerPolicy};
use symthaea_communication::metrics::{normalize_for_scoring, word_error_rate};
use symthaea_muse::{MusicalIntent, MusicalState, Style};

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Same char-level reuse of `word_error_rate` as `fleurs_release_gate.rs` — one
/// scoring implementation, two granularities (voice plan Phase 4 WER-dedup rule).
fn char_error_rate(reference: &[char], hypothesis: &[char]) -> f32 {
    let ref_words: Vec<String> = reference.iter().map(|c| c.to_string()).collect();
    let hyp_words: Vec<String> = hypothesis.iter().map(|c| c.to_string()).collect();
    word_error_rate(&ref_words.join(" "), &hyp_words.join(" ")) as f32
}

/// Same pattern as `examples/voice_roundtrip_wer.rs::linear_resample`.
/// Whisper's feature extractor is fixed at 16kHz; `faster_whisper_worker.py`
/// resamples defensively on the Python side, but `whisper_worker.py` (HF
/// transformers) does NOT and hard-errors on a rate mismatch -- resampling
/// here in Rust, once, before sending to EITHER worker keeps the input
/// identical for a fair cross-check between backends (SYMTHAEA_SINGING_PLAN
/// Phase 6, cross-checking the other Whisper backend).
const WHISPER_SAMPLE_RATE: u32 = 16_000;

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

/// Same pattern as `examples/voice_roundtrip_wer.rs::save_wav` — 16-bit PCM
/// mono WAV, for a human to actually listen to (see module docs: this eval
/// had, until now, only ever measured intelligibility via automated ASR
/// self-consistency).
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

/// A small, deliberately simple phrase set: common vocabulary (avoids testing G2P
/// robustness on exotic words, which is orthogonal to "does singing itself degrade
/// intelligibility"), mixing monosyllabic and multi-syllable words so SING-2's
/// syllabifier is genuinely exercised, not just the monosyllabic-per-note path.
const PHRASES: &[&str] = &[
    "hello world",
    "the sun rises over the valley",
    "I am singing a real melody",
    "consciousness shapes every note",
    "a quiet morning walk",
];

#[derive(Serialize)]
struct PhraseResult {
    lyrics: String,
    hypothesis: String,
    wer: f32,
    cer: f32,
    melody_notes: usize,
    audio_secs: f32,
    transcribe_ms: u128,
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let worker = PathBuf::from(
        arg_value(&args, "--worker")
            .unwrap_or_else(|| "communication/worker/run_faster_whisper_nixos.sh".to_string()),
    );
    let report_path = arg_value(&args, "--report");
    // Everything in this eval so far has been automated ASR self-
    // consistency; nobody has actually LISTENED to the audio (see
    // SYMTHAEA_SINGING_PLAN_2026-07-18.md's closing assessment). Optionally
    // write each phrase's sung PCM to a real WAV file for a human check.
    let audio_dir = arg_value(&args, "--audio-dir").map(PathBuf::from);
    if let Some(dir) = &audio_dir {
        std::fs::create_dir_all(dir)?;
    }

    println!(
        "Singing intelligibility gate (self-consistency): {} phrases",
        PHRASES.len()
    );

    let mut policy =
        WorkerPolicy::allow_one(&worker).map_err(|e| anyhow::anyhow!("worker policy: {e}"))?;
    // Same generous timeout as fleurs_release_gate.rs -- first request pays the nix
    // env build + model load under this session's heavy concurrent-session load.
    policy.response_timeout = Duration::from_secs(1800);
    let provider_id =
        std::env::var("SYMTHAEA_PROVIDER_ID").unwrap_or_else(|_| "faster-whisper-local".into());
    let mut provider = LocalJsonlProvider::spawn(provider_id, &worker, &[], policy)
        .map_err(|e| anyhow::anyhow!("spawn worker: {e}"))?;

    // Fixed, neutral, deterministic consciousness state for every phrase -- see
    // module docs for why this eval isolates intelligibility from state-variation.
    let state = MusicalState::default();

    let mut results: Vec<PhraseResult> = Vec::with_capacity(PHRASES.len());
    let start_all = Instant::now();

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
        // Cap to the real syllable count -- see symthaea-repl.rs's /sing
        // handler for the full rationale (this eval is what discovered the
        // uncapped-melody sustain-drone pathology in the first place).
        let target_notes = symthaea_muse::singing_bridge::syllable_count(lyrics).max(1);
        melody.truncate(target_notes);
        let mut vocoder = FormantVocoder::new();
        let audio = symthaea::voice::singing::sing_to_pcm(lyrics, &melody, &mut vocoder);
        if audio.is_empty() {
            println!("  [{i}] {lyrics:?}: empty audio, skipping");
            continue;
        }
        // FormantVocoder defaults to 24kHz (same convention as the REPL's /sing).
        let audio_secs = audio.len() as f32 / 24000.0;

        if let Some(dir) = &audio_dir {
            let path = dir.join(format!("{i:02}_{}.wav", lyrics.replace(' ', "_")));
            save_wav(&path, &audio, 24000)?;
            println!("  [{i}] wrote {}", path.display());
        }

        let t = Instant::now();
        let resampled = linear_resample(&audio, 24_000, WHISPER_SAMPLE_RATE);
        let preserved = symthaea::voice::transcribe_samples(
            &mut provider,
            resampled,
            WHISPER_SAMPLE_RATE,
            Some("en"),
        )?;
        let transcribe_ms = t.elapsed().as_millis();

        let reference_norm = normalize_for_scoring(lyrics);
        let hypothesis_norm = normalize_for_scoring(&preserved.original);
        let wer = word_error_rate(&reference_norm, &hypothesis_norm) as f32;
        let ref_chars: Vec<char> = reference_norm.chars().collect();
        let hyp_chars: Vec<char> = hypothesis_norm.chars().collect();
        let cer = char_error_rate(&ref_chars, &hyp_chars);

        println!(
            "  [{i}] {lyrics:?} -> {:?}  WER {:>6.1}% CER {:>6.1}%  ({} notes, {audio_secs:.1}s, {transcribe_ms}ms)",
            preserved.original,
            wer * 100.0,
            cer * 100.0,
            melody.len()
        );

        results.push(PhraseResult {
            lyrics: lyrics.to_string(),
            hypothesis: preserved.original,
            wer,
            cer,
            melody_notes: melody.len(),
            audio_secs,
            transcribe_ms,
        });
    }

    fn mean(xs: impl Iterator<Item = f32> + Clone) -> f32 {
        let (sum, n) = xs.fold((0.0f32, 0usize), |(s, n), x| (s + x, n + 1));
        if n == 0 { 0.0 } else { sum / n as f32 }
    }
    let overall_wer = mean(results.iter().map(|r| r.wer));
    let overall_cer = mean(results.iter().map(|r| r.cer));
    println!(
        "\nOverall: {} phrases, mean WER {:.1}%, mean CER {:.1}%, total {:.0}s",
        results.len(),
        overall_wer * 100.0,
        overall_cer * 100.0,
        start_all.elapsed().as_secs_f32()
    );

    if let Some(path) = report_path {
        #[derive(Serialize)]
        struct Report<'a> {
            eval: &'a str,
            method: &'a str,
            worker: String,
            n_phrases: usize,
            overall_wer: f32,
            overall_cer: f32,
            phrases: &'a [PhraseResult],
        }
        let report = Report {
            eval: "singing_intelligibility_gate",
            method: "self_consistency",
            worker: worker.display().to_string(),
            n_phrases: results.len(),
            overall_wer,
            overall_cer,
            phrases: &results,
        };
        std::fs::write(&path, serde_json::to_string_pretty(&report)?)?;
        println!("Report written to {path}");
    }

    Ok(())
}
