// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! P2.3 release-gate run: commit a real Whisper transcription-quality
//! bundle against FLEURS, closing the gap the voice plan flagged — "pilot
//! summaries (en WER 4.8%/CER 2.2% normalized) exist only as untracked
//! local artifacts; the whole evidence-gate machinery has never produced a
//! committed passing bundle."
//!
//! Reads the already-materialized FLEURS en/af test set
//! (`communication/pilot/run/fleurs-en-af-test/samples.jsonl`, 200 samples,
//! real reference transcripts, 16kHz mono WAVs already on disk — no
//! resampling needed), transcribes every sample through the
//! `faster_whisper_worker.py` JSONL provider (CTranslate2, 4-8x faster than
//! the HF `transformers` large-v3 path used earlier this session), and
//! scores against the SAME normalized WER/CER authority used everywhere
//! else in this codebase (`symthaea-communication::metrics`).
//!
//! ```bash
//! SYMTHAEA_FASTER_WHISPER_MODEL_PATH=/srv/luminous-dynamics/models/communication/faster-whisper-large-v3 \
//! cargo run --example fleurs_release_gate --features voice-tts -- \
//!     [--manifest communication/pilot/run/fleurs-en-af-test/samples.jsonl] \
//!     [--worker communication/worker/run_faster_whisper_nixos.sh] \
//!     [--report /path/to/report.json] [--limit N]
//! ```

use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use symthaea::voice::transcribe_samples;
use symthaea_communication::human::{LocalJsonlProvider, WorkerPolicy};
use symthaea_communication::metrics::{normalize_for_scoring, word_error_rate};

#[derive(Deserialize)]
struct FleursSample {
    id: String,
    language: String,
    audio_path: String,
    reference_transcript: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct SampleResult {
    id: String,
    language: String,
    reference: String,
    hypothesis: String,
    wer: f32,
    cer: f32,
    transcribe_ms: u128,
}

fn arg_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let manifest_path =
        PathBuf::from(arg_value(&args, "--manifest").unwrap_or_else(|| {
            "communication/pilot/run/fleurs-en-af-test/samples.jsonl".to_string()
        }));
    let worker = PathBuf::from(
        arg_value(&args, "--worker")
            .unwrap_or_else(|| "communication/worker/run_faster_whisper_nixos.sh".to_string()),
    );
    let report_path = arg_value(&args, "--report");
    let limit: Option<usize> = arg_value(&args, "--limit").map(|s| s.parse()).transpose()?;
    // Checkpoint: every completed SampleResult is appended here as a JSON
    // line immediately (not buffered to the end). A prior run crashed on a
    // provider timeout at sample 197/200 under heavy system load and lost
    // all 198 completed transcriptions because the report was only written
    // at the very end (2026-07-18) -- this makes that failure mode resumable
    // instead of a full ~8h re-run.
    let checkpoint_path = PathBuf::from(
        arg_value(&args, "--checkpoint")
            .unwrap_or_else(|| "fleurs_gate_checkpoint.jsonl".to_string()),
    );

    let manifest_text = std::fs::read_to_string(&manifest_path)
        .map_err(|e| anyhow::anyhow!("reading {}: {e}", manifest_path.display()))?;
    let mut samples: Vec<FleursSample> = manifest_text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| anyhow::anyhow!("parsing manifest line: {e}")))
        .collect::<anyhow::Result<_>>()?;
    if let Some(n) = limit {
        samples.truncate(n);
    }
    println!(
        "FLEURS release gate: {} samples from {}",
        samples.len(),
        manifest_path.display()
    );

    // Resume: preload any results already in the checkpoint file and skip
    // those sample IDs. Existing checkpoint entries are kept verbatim; we
    // only compute what's missing.
    let mut results: Vec<SampleResult> = Vec::with_capacity(samples.len());
    let mut done_ids: HashSet<String> = HashSet::new();
    if checkpoint_path.exists() {
        let existing = std::fs::read_to_string(&checkpoint_path)?;
        for line in existing.lines().filter(|l| !l.trim().is_empty()) {
            match serde_json::from_str::<SampleResult>(line) {
                Ok(r) => {
                    done_ids.insert(r.id.clone());
                    results.push(r);
                }
                Err(e) => eprintln!("warning: skipping unparsable checkpoint line: {e}"),
            }
        }
        println!(
            "Resuming from checkpoint {}: {} samples already done",
            checkpoint_path.display(),
            results.len()
        );
    }
    let mut checkpoint_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&checkpoint_path)?;

    let mut policy =
        WorkerPolicy::allow_one(&worker).map_err(|e| anyhow::anyhow!("worker policy: {e}"))?;
    // Generous: the nix env build + first model load happen on the very
    // first request (model loads once, at import time) and were observed
    // exceeding 120s under this session's heavy concurrent-session load —
    // same "first request pays model load" pattern as the Whisper worker
    // used throughout this session (voice_roundtrip_wer.rs uses 900s).
    // Bumped 900s -> 1800s (2026-07-18): a real run under sustained
    // load ~20-28 (concurrent CUDA build + other sessions) saw one sample
    // take 706s and the next exceed 900s outright, aborting the whole run.
    policy.response_timeout = Duration::from_secs(1800);
    // Must match faster_whisper_worker.py's self-reported default
    // (SYMTHAEA_PROVIDER_ID env var, defaults to "faster-whisper-local").
    let provider_id =
        std::env::var("SYMTHAEA_PROVIDER_ID").unwrap_or_else(|_| "faster-whisper-local".into());
    let mut provider = LocalJsonlProvider::spawn(provider_id, &worker, &[], policy)
        .map_err(|e| anyhow::anyhow!("spawn worker: {e}"))?;

    let start_all = Instant::now();
    for (i, sample) in samples.iter().enumerate() {
        if done_ids.contains(&sample.id) {
            continue;
        }
        let mut reader = hound::WavReader::open(&sample.audio_path)
            .map_err(|e| anyhow::anyhow!("opening {}: {e}", sample.audio_path))?;
        let spec = reader.spec();
        let audio: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
            hound::SampleFormat::Int => {
                let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .samples::<i16>()
                    .map(|s| s.map(|v| v as f32 * scale))
                    .collect::<Result<_, _>>()?
            }
        };

        let t = Instant::now();
        let preserved = transcribe_samples(
            &mut provider,
            audio,
            spec.sample_rate,
            Some(&sample.language),
        )?;
        let transcribe_ms = t.elapsed().as_millis();

        let reference_norm = normalize_for_scoring(&sample.reference_transcript);
        let hypothesis_norm = normalize_for_scoring(&preserved.original);
        let wer = word_error_rate(&reference_norm, &hypothesis_norm) as f32;
        // Character error rate: same normalized strings, word_error_rate's
        // edit-distance machinery applied over chars instead of words —
        // reuse via per-char "words" (single-char tokens) so there is only
        // one scoring implementation in play, matching the WER-dedup rule
        // from the voice plan's Phase 4.
        let ref_chars: Vec<char> = reference_norm.chars().collect();
        let hyp_chars: Vec<char> = hypothesis_norm.chars().collect();
        let cer = char_error_rate(&ref_chars, &hyp_chars);

        println!(
            "  [{i:3}/{:3}] {} WER {:>6.1}% CER {:>6.1}%  ({transcribe_ms}ms)",
            samples.len(),
            sample.language,
            wer * 100.0,
            cer * 100.0
        );

        let result = SampleResult {
            id: sample.id.clone(),
            language: sample.language.clone(),
            reference: sample.reference_transcript.clone(),
            hypothesis: preserved.original,
            wer,
            cer,
            transcribe_ms,
        };
        writeln!(checkpoint_file, "{}", serde_json::to_string(&result)?)?;
        checkpoint_file.flush()?;
        results.push(result);
    }

    fn mean(xs: impl Iterator<Item = f32> + Clone) -> f32 {
        let (sum, n) = xs.fold((0.0f32, 0usize), |(s, n), x| (s + x, n + 1));
        if n == 0 { 0.0 } else { sum / n as f32 }
    }

    let overall_wer = mean(results.iter().map(|r| r.wer));
    let overall_cer = mean(results.iter().map(|r| r.cer));
    println!(
        "\nOverall: {} samples, mean WER {:.1}%, mean CER {:.1}%, total {:.0}s",
        results.len(),
        overall_wer * 100.0,
        overall_cer * 100.0,
        start_all.elapsed().as_secs_f32()
    );

    let mut languages: Vec<&str> = results.iter().map(|r| r.language.as_str()).collect();
    languages.sort_unstable();
    languages.dedup();
    for lang in &languages {
        let lang_wer = mean(
            results
                .iter()
                .filter(|r| r.language == *lang)
                .map(|r| r.wer),
        );
        let lang_cer = mean(
            results
                .iter()
                .filter(|r| r.language == *lang)
                .map(|r| r.cer),
        );
        let n = results.iter().filter(|r| r.language == *lang).count();
        println!(
            "  {lang}: n={n} WER {:.1}% CER {:.1}%",
            lang_wer * 100.0,
            lang_cer * 100.0
        );
    }

    if let Some(path) = report_path {
        #[derive(serde::Serialize)]
        struct Report<'a> {
            eval: &'a str,
            worker: String,
            manifest: String,
            n_samples: usize,
            overall_wer: f32,
            overall_cer: f32,
            per_language: Vec<(String, usize, f32, f32)>,
            samples: &'a [SampleResult],
        }
        let per_language = languages
            .iter()
            .map(|lang| {
                let n = results.iter().filter(|r| &r.language == lang).count();
                let w = mean(
                    results
                        .iter()
                        .filter(|r| &r.language == lang)
                        .map(|r| r.wer),
                );
                let c = mean(
                    results
                        .iter()
                        .filter(|r| &r.language == lang)
                        .map(|r| r.cer),
                );
                (lang.to_string(), n, w, c)
            })
            .collect();
        let report = Report {
            eval: "fleurs_release_gate",
            worker: worker.display().to_string(),
            manifest: manifest_path.display().to_string(),
            n_samples: results.len(),
            overall_wer,
            overall_cer,
            per_language,
            samples: &results,
        };
        std::fs::write(&path, serde_json::to_string_pretty(&report)?)?;
        println!("Report written to {path}");
    }

    Ok(())
}

/// Character error rate via the same Levenshtein/word_error_rate machinery,
/// applied to single-character tokens instead of words — one scoring
/// implementation, two granularities (voice plan Phase 4 WER-dedup rule).
fn char_error_rate(reference: &[char], hypothesis: &[char]) -> f32 {
    let ref_words: Vec<String> = reference.iter().map(|c| c.to_string()).collect();
    let hyp_words: Vec<String> = hypothesis.iter().map(|c| c.to_string()).collect();
    word_error_rate(&ref_words.join(" "), &hyp_words.join(" ")) as f32
}
